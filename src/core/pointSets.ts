// Point-set variations (JWildfire's DrawFunc family, the `_js` turtle fractals, dla_wf, …): each builds a list of
// primitives on the CPU at init — points, line segments, triangles, n-gons — that the kernel samples per point,
// exactly the way DrawFunc.plotBlur / plotLine / plotTriangle / plotPolygon (nBlur) and the turtles' getPoint do.
// The lists are packed into one storage buffer (binding 13, `pset`) as 12-float records; each instance's hidden
// slots carry [record base, record count] (codegen data hook, flag 'pset'). Some kernels use the same buffer as a
// plain table (scrambly's permutation): a record is then simply 12 consecutive numbers.
//
// Record: [kind, color, a, b, c, d, e, f, g, h, i, j]
//   kind 0 point      a,b = x,y                                      (plotBlur: radius = amount·rnd)
//   kind 1 line       a,b,c,d = x1,y1,x2,y2; e = thickness            (plotLine: random distance along, ± thickness/2)
//   kind 2 triangle   a..f = the three corners                         (uniform barycentric)
//   kind 3 ngon       a,b = pos; c = sides; d = scale; e,f = cos/sin(angle); g = fill (hole ratio)   (nBlur randXY)
//   kind 4 dot        a,b = x,y; c = thickness                         (Brownian pDot: ± thickness/2 square)
export const PSET_STRIDE = 12;

export interface PointSet { data: Float32Array; count: number }
export type PointSetBuilder = (params: Record<string, number>) => PointSet;

export class PointSetWriter {
  private arr: number[] = [];
  count = 0;
  private push(kind: number, color: number, ...v: number[]) {
    const rec = new Array(PSET_STRIDE).fill(0);
    rec[0] = kind; rec[1] = color;
    for (let i = 0; i < v.length && i + 2 < PSET_STRIDE; i++) rec[i + 2] = v[i];
    this.arr.push(...rec); this.count++;
  }
  point(x: number, y: number, color = 0) { this.push(0, color, x, y); }
  line(x1: number, y1: number, x2: number, y2: number, thickness = 0, color = 0) { this.push(1, color, x1, y1, x2, y2, thickness); }
  triangle(x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, color = 0) { this.push(2, color, x1, y1, x2, y2, x3, y3); }
  /** JWildfire Ngon(sides, scale, angle°, pos, color, fill) */
  ngon(x: number, y: number, sides: number, scale: number, angleDeg: number, fill: number, color = 0) {
    this.push(3, color, x, y, Math.max(3, Math.round(sides)), scale, Math.cos(angleDeg * Math.PI / 180), Math.sin(angleDeg * Math.PI / 180), fill);
  }
  dot(x: number, y: number, thickness: number, color = 0) { this.push(4, color, x, y, thickness); }
  /** raw numbers (tables): `n` values per record, padded */
  raw(values: number[]) { for (let i = 0; i < values.length; i += PSET_STRIDE) { const rec = values.slice(i, i + PSET_STRIDE); while (rec.length < PSET_STRIDE) rec.push(0); this.arr.push(...rec); this.count++; } }
  done(): PointSet { return { data: Float32Array.from(this.arr), count: this.count }; }
}

/** JWildfire's `_js` Turtle (DragonFunc & co.): goForward records a segment (oldx, oldy) → (x, y). */
export class Turtle {
  private x: number; private y: number; private a: number;
  readonly segs: number[] = [];
  constructor(x0 = 0, y0 = 0, a0 = 0) { this.x = x0; this.y = y0; this.a = a0; }
  turnLeft(deg: number) { this.a += deg; }
  goForward(step: number) {
    const ox = this.x, oy = this.y;
    this.x += step * Math.cos(this.a * Math.PI / 180); this.y += step * Math.sin(this.a * Math.PI / 180);
    this.segs.push(ox, oy, this.x, this.y);
  }
  /** all recorded segments as lines of `thickness` */
  into(w: PointSetWriter, thickness: number) { for (let i = 0; i < this.segs.length; i += 4) w.line(this.segs[i], this.segs[i + 1], this.segs[i + 2], this.segs[i + 3], thickness); }
}

/** JWildfire MarsagliaRandomGenerator (seeded point clouds such as dla_wf must reproduce JWildfire's sequence). */
export class Marsaglia {
  private u = 12244355; private v = 34384;
  constructor(seed?: number) { if (seed !== undefined) this.randomize(seed); }
  /** randomize(long seed): u = (int)(seed << 16); v = (int)(seed << 16) >> 16 */
  randomize(seed: number) { const s = (seed << 16) | 0; this.u = s; this.v = s >> 16; }
  /** v = 36969·(v & 65535) + (v >> 16); u = 18000·(u & 65535) + (u >> 16); rnd = (v << 16) + u; |rnd| / 0x7fffffff — Java int arithmetic */
  random(): number {
    this.v = (Math.imul(36969, this.v & 65535) + (this.v >> 16)) | 0;
    this.u = (Math.imul(18000, this.u & 65535) + (this.u >> 16)) | 0;
    const rnd = ((this.v << 16) + this.u) | 0;
    const res = rnd / 0x7fffffff;
    return res < 0 ? -res : res;
  }
}

const builders = new Map<string, PointSetBuilder>();
export function registerPointSet(name: string, b: PointSetBuilder) { builders.set(name, b); }
export const hasPointSet = (name: string) => builders.has(name);

/** The point-set key of a variation instance (name + the params its builder reads), undefined for others. */
export function pointSetKeyFor(vi: { name: string; params: Record<string, number> }): string | undefined {
  if (!builders.has(vi.name)) return undefined;
  return `${vi.name}#${JSON.stringify(vi.params)}`;
}
const cache = new Map<string, PointSet>();
/** Build (or fetch from cache) the set for a key. Synchronous: builders run on the main thread at setFlame. */
export function pointSetFor(key: string): PointSet {
  let ps = cache.get(key);
  if (!ps) {
    const hash = key.indexOf('#');
    const name = key.slice(0, hash);
    const params = JSON.parse(key.slice(hash + 1)) as Record<string, number>;
    const t0 = performance.now();
    try { ps = builders.get(name)!(params); }
    catch (e) { console.warn(`${name}: point set failed (${(e as Error).message}); rendering nothing`); ps = { data: new Float32Array(0), count: 0 }; }
    const ms = performance.now() - t0;
    if (ms > 200) console.info(`${name}: ${ps.count} primitives in ${ms.toFixed(0)} ms`);
    if (cache.size > 64) cache.delete(cache.keys().next().value!);
    cache.set(key, ps);
  }
  return ps;
}
/** Where each set sits in the renderer's buffer (set by the renderer when it packs). */
export const pointSetLayout = new Map<string, { base: number; count: number }>();
type XFormLike = { variations: { name: string; params: Record<string, number> }[]; preVariations?: { name: string; params: Record<string, number> }[]; postVariations?: { name: string; params: Record<string, number> }[] };
export function flamePointSetKeys(flame: { layers: { xforms: XFormLike[]; final: XFormLike | null; moreFinals: XFormLike[] }[] }): string[] {
  const keys = new Set<string>();
  for (const ly of flame.layers) for (const x of [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals])
    for (const list of [x.preVariations ?? [], x.variations, x.postVariations ?? []]) for (const vi of list) { const k = pointSetKeyFor(vi); if (k) keys.add(k); }
  return [...keys];
}

// ---------------------------------------------------------------- builders
// dragon_js (DragonFunc): the dragon curve of order `level`, unit steps, 90° turns, from the origin facing +x
registerPointSet('dragon_js', (P) => {
  const level = Math.max(0, Math.min(16, Math.round(P.level ?? 2)));
  const t = new Turtle(0, 0, 0);
  const dragon = (n: number): void => { if (n === 0) t.goForward(1); else { dragon(n - 1); t.turnLeft(90); nogard(n - 1); } };
  const nogard = (n: number): void => { if (n === 0) t.goForward(1); else { dragon(n - 1); t.turnLeft(-90); nogard(n - 1); } };
  dragon(level);
  const w = new PointSetWriter(); t.into(w, (P.line_thickness ?? 0.5) / 100); return w.done();
});

// sunflower (SunFlowersFunc): nPoints n-gons on a Fibonacci spiral; colour = the radial fraction
registerPointSet('sunflower', (P) => {
  const nPoints = Math.round(Math.min(1000, Math.max(10, P.nPoints ?? 500)));
  const shape = Math.round(Math.min(20, Math.max(3, P.shape ?? 10)));
  const scale = P.scale ?? 0.02, angle = P.angle ?? 180, fill = P['F. filling'] ?? 0, invert = Math.round(P.invert ?? 0);
  const ang = angle * (3 - Math.sqrt(5));
  const rmax = Math.sqrt(nPoints + 1) / 30;
  const w = new PointSetWriter();
  for (let i = 0; i < nPoints; i++) {
    const r = Math.sqrt(i + 1) / 30;
    const t = (i + 1) * ang * Math.PI / 180;
    const x = r * Math.cos(t) / rmax, y = r * Math.sin(t) / rmax;
    let sc = 1 - r / rmax; if (invert === 1) sc = r / rmax;
    w.ngon(x, y, shape, scale * sc, 0, fill, sc);
  }
  return w.done();
});

// scrambly (ScramblyFunc): the cell permutation table, as 12-per-record raw ints
registerPointSet('scrambly', (P) => {
  const MX_L = 25;
  let LL = Math.abs(Math.round(P.l ?? 3)); if (LL < 3) LL = 3; else if (LL > MX_L) LL = MX_L;
  const LL2 = LL * LL, seed = Math.round(P.seed ?? 51), byrows = Math.round(P.byrows ?? 0);
  const mx = new Int32Array(LL2);
  const randflip = (idxmin: number, idxmax: number, sd: number) => {
    let prn = 1;
    for (let j = idxmin; ; j++) {
      prn = (Math.imul(prn, 1103515245) + 12345) | 0;
      prn = ((prn & 0xFFFF0000) | ((prn << 8) & 0xFF00) | ((prn >> 8) & 0x00FF)) | 0;
      prn = (prn & 4) !== 0 ? (prn - sd) | 0 : (prn ^ sd) | 0;
      prn = prn < 0 ? -prn : prn;
      let ridx = 1 + j;
      if (idxmax > ridx) ridx += prn % (idxmax - ridx); else break;
      const i = mx[ridx]; mx[ridx] = mx[j]; mx[j] = i;
    }
  };
  if (seed >= 0 && seed <= 50) { for (let j = 0; j < LL2; j++) mx[j] = (seed + j + 1) % LL2; }
  else {
    for (let j = 0; j < LL2; j++) mx[j] = j;
    if (byrows === 0) randflip(0, LL2, seed); else for (let j = 0; j < LL; j++) randflip(LL * j, LL * (1 + j), seed + j);
  }
  const w = new PointSetWriter(); w.raw(Array.from(mx)); return w.done();
});

// dla_wf (DLAWFFunc): diffusion-limited aggregation on a buffer_size² grid from a seeded Marsaglia sequence; the
// occupied cells become points (scaled to `scale`, optionally jittered on a circle of radius `jitter`)
registerPointSet('dla_wf', (P) => {
  const bufferSize = Math.max(16, Math.min(2000, Math.round(P.buffer_size ?? 800)));
  const maxIter = Math.max(1, Math.min(200000, Math.round(P.max_iter ?? 6000)));
  const seed = Math.round(P.seed ?? 666), scale = P.scale ?? 10, jitter = Math.max(0.01, P.jitter ?? 0.01);
  const jitterRadius = Math.max(Math.min(1, jitter), 0);
  // calculate()
  const rng = new Marsaglia(); rng.randomize(seed);
  const centre = Math.floor(bufferSize / 2), size2 = bufferSize - 2;
  const q = new Uint8Array(bufferSize * bufferSize);
  const at = (i: number, j: number) => q[i * bufferSize + j];
  q[centre * bufferSize + centre] = 1;
  let r1 = 3, r2 = 3 * r1;
  for (let i = 0; i < maxIter; i++) {
    const phi = 2 * Math.PI * rng.random();
    let ci = centre + Math.trunc(r1 * Math.cos(phi) + 0.5);
    let cj = centre + Math.trunc(r1 * Math.sin(phi) + 0.5);
    let qt = 0;
    let guard = 0;
    while (qt === 0) {
      if (++guard > 4_000_000) return { data: new Float32Array(0), count: 0 };
      let rr = rng.random(); rr += rr; rr += rr;
      switch (Math.trunc(rr)) { case 0: ci++; break; case 1: cj--; break; case 2: ci--; break; default: cj++; }
      if (ci < 1 || ci > size2 || cj < 1 || cj > size2) { qt = 1; i--; }
      else {
        const sum = at(ci - 1, cj) + at(ci + 1, cj) + at(ci, cj - 1) + at(ci, cj + 1);
        const r3 = Math.hypot(ci - centre, cj - centre);
        if (sum !== 0) { q[ci * bufferSize + cj] = 1; qt = 1; if (r3 > r1) { r1 = r3; r2 = 2.1 * r1; } }
        else if (r3 > r2) { qt = 1; i--; }
      }
    }
  }
  // getPoints(): the occupied cells, one jitter random per cell (drawn for every cell when jitter > 0)
  const jr = new Marsaglia(); jr.randomize(seed);
  const w = new PointSetWriter();
  for (let i = 0; i < bufferSize; i++) for (let j = 0; j < bufferSize; j++) {
    const aRnd = jitterRadius > 1e-9 ? jr.random() : 0;
    if (q[i * bufferSize + j] !== 0) {
      let x = (i - centre) / bufferSize * scale, y = (j - centre) / bufferSize * scale;
      if (jitterRadius > 1e-9) { const alpha = aRnd * 2 * Math.PI; x += jitterRadius * Math.cos(alpha); y += jitterRadius * Math.sin(alpha); }
      w.point(x, y);
    }
  }
  return w.done();
});

/** java.util.Random (48-bit LCG), for builders that seed one: Brownian's midpoint displacement. */
export class JavaRandom {
  private seed: bigint;
  private static readonly MULT = BigInt('0x5DEECE66D');
  private static readonly MASK = (BigInt(1) << BigInt(48)) - BigInt(1);
  constructor(seed: number) { this.seed = (BigInt(Math.trunc(seed)) ^ JavaRandom.MULT) & JavaRandom.MASK; }
  private next(bits: number): number { this.seed = (this.seed * JavaRandom.MULT + BigInt(11)) & JavaRandom.MASK; return Number(this.seed >> BigInt(48 - bits)); }
  nextDouble(): number { return (this.next(26) * 134217728 + this.next(27)) / 9007199254740992; }
}

// ---- the `_js` turtle family: line segments; the kernel draws a random point along one (plotLine, ± line_thickness)
// or, with probability show_points/(show_lines+show_points), a dot of radius point_thickness at its first end
const turtleLines = (segs: number[], thickness: number): PointSet => { const w = new PointSetWriter(); for (let i = 0; i < segs.length; i += 4) w.line(segs[i], segs[i + 1], segs[i + 2], segs[i + 3], thickness); return w.done(); };
const level = (P: Record<string, number>, dflt: number, max: number) => Math.max(0, Math.min(max, Math.round(P.level ?? dflt)));

// brownian_js (BrownianFunc.Draw2D.midpoint): recursive midpoint displacement with gaussian offsets, java.util.Random(seed)
registerPointSet('brownian_js', (P) => {
  const lvl = Math.max(1, Math.min(15, Math.round(P.level ?? 10))), variation = P.variation ?? 3;
  const seedP = Math.round(P.seed ?? 0);
  const rnd = new JavaRandom(seedP === 0 ? Date.now() : seedP);
  const uniform = (a: number, b: number) => a + rnd.nextDouble() * (b - a);
  const gaussian = () => { let r, x, y; do { x = uniform(-1, 1); y = uniform(-1, 1); r = x * x + y * y; } while (r >= 1 || r === 0); return x * Math.sqrt(-2 * Math.log(r) / r); };
  const segs: number[] = [];
  const midpoint = (x0: number, y0: number, x1: number, y1: number, v: number, n: number): void => {
    if (n === 0) { segs.push(x0, y0, x1, y1); return; }
    const xm = 0.5 * (x0 + x1) + Math.sqrt(v) * gaussian();
    const ym = 0.5 * (y0 + y1) + Math.sqrt(v) * gaussian();
    midpoint(x0, y0, xm, ym, v / 2.7, n - 1); midpoint(xm, ym, x1, y1, v / 2.7, n - 1);
  };
  midpoint(0, 0, 0, 0, variation / Math.sqrt(2), lvl);
  return turtleLines(segs, (P.line_thickness ?? 0.5) / 100);
});

// snowflake_wf (SnowflakeWFFunc.Snowflake): Reiter's hexagonal cellular automaton — a seeded background level with
// noise, a frozen centre, max_iter steps of freeze (receptive cells and their six neighbours) + diffusion — rendered onto a
// stretched canvas (values above threshold), then every lit canvas cell becomes a point (x from the row, y from the column,
// as in JWildfire), intensity = value − threshold as the record colour; one Marsaglia jitter random per canvas cell.
registerPointSet('snowflake_wf', (P) => {
  const W = Math.max(4, Math.min(512, Math.round(P.buffer_size ?? 128))), maxIter = Math.max(0, Math.min(20000, Math.round(P.max_iter ?? 500)));
  const bg = P.bg_freeze_level ?? 0.5, fgSpeed = P.fg_freeze_speed ?? 0.0005, diff = P.diffusion_speed ?? 0.01, asym = P.diffusion_asymmetry ?? 1;
  const noise = P.rnd_bg_noise ?? 0.25, threshold = P.threshold ?? 0.65, seed = Math.round(P.seed ?? 12345), scale = P.scale ?? 1;
  const jitterRadius = Math.max(Math.min(1, P.jitter ?? 0.001), 0);
  const STRETCH = 1.5 / 1.7321, EPS = 1e-8;
  const H = W + W - 1, N = W * H + (W - 1) * (H - 1);
  const nbOff = [-W - (W - 1), -(W - 1), W, W + (W - 1), W - 1, -W];
  const rng = new Marsaglia(); rng.randomize(seed);
  const flake = new Float64Array(N);
  for (let i = 0; i < N; i++) flake[i] = bg + (noise - 2 * noise * rng.random());
  flake[Math.floor(N / 2)] = 1;
  const nonRec = new Float64Array(N), rec = new Float64Array(N), tmp = new Float64Array(N);
  const tFreeze = fgSpeed / 1000, tDiff = diff / 1000 + 1;
  const cW = 0.5 * asym / tDiff, nbW = (1 * tDiff - cW) / 6;
  for (let it = 0; it < maxIter; it++) {
    tmp.set(flake); rec.fill(0);
    for (let i = 0; i < N; i++) {
      if (flake[i] >= 1) {
        nonRec[i] = 0; rec[i] = flake[i];
        for (let j = 0; j < 6; j++) { const nb = i + nbOff[j]; if (nb >= 0 && nb < N) { tmp[nb] = 0; rec[nb] = flake[nb] > 0 ? flake[nb] + tFreeze : flake[nb]; } }
      }
    }
    // diffusionPart: nonRec[i] = tmp[i]·cW + Σ valid neighbours tmp[i + off]·nbW — each fixed offset is valid on one index range
    for (let i = 0; i < N; i++) nonRec[i] = tmp[i] * cW;
    for (let j = 0; j < 6; j++) { const off = nbOff[j]; const lo = Math.max(0, -off), hi = Math.min(N, N - off); for (let i = lo; i < hi; i++) nonRec[i] += tmp[i + off] * nbW; }
    for (let i = 0; i < N; i++) flake[i] = nonRec[i] + rec[i];
  }
  // renderSnowflake(): canvas[y][x], later writes win
  const cw = Math.floor(2 * W * STRETCH), ch = H;
  const canvas = new Float32Array(cw * ch);
  const put = (value: number, x: number, y: number) => { if (value > threshold + EPS && y >= 0 && y < ch && x >= 0 && x < cw) canvas[y * cw + x] = Math.fround(value - threshold); };
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    const base = (W + W - 1) * i + j;
    put(flake[base], Math.trunc((2 * j) * STRETCH + 0.5), i);
    const i1 = base - W + 1, i2 = base + W;
    if (i1 >= 0 && i1 < N && i2 >= 0 && i2 < N) put((flake[i1] + flake[i2]) * 0.5, Math.trunc((2 * j + 1) * STRETCH + 0.5), i);
  }
  // getPoints()
  const jr = new Marsaglia(); jr.randomize(seed);
  const w = new PointSetWriter();
  for (let i = 0; i < ch; i++) for (let j = 0; j < cw; j++) {
    const aRnd = jitterRadius > EPS ? jr.random() : 0;
    const v = canvas[i * cw + j];
    if (v > EPS) {
      let x = (i - ch * 0.5) * scale / ch, y = (j - cw * 0.5) * scale / cw * STRETCH;
      if (jitterRadius > EPS) { const alpha = aRnd * 2 * Math.PI; x += jitterRadius * Math.cos(alpha); y += jitterRadius * Math.sin(alpha); }
      w.point(x, y, v);
    }
  }
  return w.done();
});

// htree_js (HtreeFunc.draw): an H of `size` at the origin, four half-size H-trees at its tips, `level` deep
registerPointSet('htree_js', (P) => {
  const segs: number[] = []; const size = P.size ?? 2;
  const drawH = (x: number, y: number, s: number) => { const x0 = x - s / 2, x1 = x + s / 2, y0 = y - s / 2, y1 = y + s / 2; segs.push(x0, y0, x0, y1, x1, y0, x1, y1, x0, y, x1, y); };
  const draw = (n: number, x: number, y: number, s: number): void => { if (n === 0) return; drawH(x, y, s); const x0 = x - s / 2, x1 = x + s / 2, y0 = y - s / 2, y1 = y + s / 2; draw(n - 1, x0, y0, s / 2); draw(n - 1, x0, y1, s / 2); draw(n - 1, x1, y0, s / 2); draw(n - 1, x1, y1, s / 2); };
  draw(level(P, 2, 9), 0, 0, size);
  return turtleLines(segs, (P.line_thickness ?? 0.5) / 100);
});

// koch_js (KochFunc.koch): the Koch curve, step 0.5, from the origin facing +x
registerPointSet('koch_js', (P) => {
  const t = new Turtle(0, 0, 0);
  const koch = (n: number, step: number): void => { if (n === 0) { t.goForward(step); return; } koch(n - 1, step); t.turnLeft(60); koch(n - 1, step); t.turnLeft(-120); koch(n - 1, step); t.turnLeft(60); koch(n - 1, step); };
  koch(level(P, 2, 8), 0.5);
  return turtleLines(t.segs, (P.line_thickness ?? 0.5) / 100);
});

// tree_js (TreeFunc.tree): a trunk of length 2 straight up, each branch bends and forks (branch_angle, branch_ratio)
registerPointSet('tree_js', (P) => {
  const segs: number[] = [];
  const bend = (P.bend_angle ?? 0) * Math.PI / 180, branch = (P.branch_angle ?? 0) * Math.PI / 180, ratio = P.branch_ratio ?? 0;
  const tree = (n: number, x: number, y: number, a: number, r: number): void => {
    const cx = x + Math.cos(a) * r, cy = y + Math.sin(a) * r;
    segs.push(x, y, cx, cy);
    if (n === 0) return;
    tree(n - 1, cx, cy, a + bend - branch, r * ratio); tree(n - 1, cx, cy, a + bend + branch, r * ratio); tree(n - 1, cx, cy, a + bend, r * (1 - ratio));
  };
  tree(level(P, 2, 12), 0, 0, Math.PI / 2, 2);
  return turtleLines(segs, (P.line_thickness ?? 0.5) / 100);
});

// hilbert_js (HilbertFunc.Hilbert.draw_hilbert): the Hilbert curve of order `level`, unit steps
registerPointSet('hilbert_js', (P) => {
  const t = new Turtle(0, 0, 0);
  const h = (n: number): void => { if (n === 0) return; t.turnLeft(90); tr(n - 1); t.goForward(1); t.turnLeft(-90); h(n - 1); t.goForward(1); h(n - 1); t.turnLeft(-90); t.goForward(1); tr(n - 1); t.turnLeft(90); };
  const tr = (n: number): void => { if (n === 0) return; t.turnLeft(-90); h(n - 1); t.goForward(1); t.turnLeft(90); tr(n - 1); t.goForward(1); tr(n - 1); t.turnLeft(90); t.goForward(1); h(n - 1); t.turnLeft(-90); };
  h(level(P, 2, 9));
  return turtleLines(t.segs, (P.line_thickness ?? 0.5) / 100);
});
