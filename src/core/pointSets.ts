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
export type PointSetBuilder = (params: Record<string, number>, res?: Record<string, string>) => PointSet;

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
  ngon(x: number, y: number, sides: number, scale: number, angleDeg: number, fill: number, color = 0, rgb?: [number, number, number]) {
    this.push(3, color, x, y, Math.max(3, Math.round(sides)), scale, Math.cos(angleDeg * Math.PI / 180), Math.sin(angleDeg * Math.PI / 180), fill, ...(rgb ?? []));
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
export function pointSetKeyFor(vi: { name: string; params: Record<string, number>; res?: Record<string, string> }): string | undefined {
  if (!builders.has(vi.name)) return undefined;
  const key = `${vi.name}#${JSON.stringify(vi.params)}#${JSON.stringify(vi.res ?? {})}`;
  if (!keyArgs.has(key)) keyArgs.set(key, { params: vi.params, res: vi.res });
  return key;
}
const keyArgs = new Map<string, { params: Record<string, number>; res?: Record<string, string> }>();
const cache = new Map<string, PointSet>();
/** Build (or fetch from cache) the set for a key. Synchronous: builders run on the main thread at setFlame. */
export function pointSetFor(key: string): PointSet {
  let ps = cache.get(key);
  if (!ps) {
    const hash = key.indexOf('#');
    const name = key.slice(0, hash);
    const args = keyArgs.get(key) ?? { params: JSON.parse(key.slice(hash + 1, key.lastIndexOf('#'))) as Record<string, number> };
    const t0 = performance.now();
    try { ps = builders.get(name)!(args.params, args.res); }
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
  /** java.util.Random.nextInt(bound) */
  nextInt(bound: number): number {
    let r = this.next(31); const m = bound - 1;
    if ((bound & m) === 0) return Number((BigInt(bound) * BigInt(r)) >> BigInt(31));
    for (let u = r; u - (r = u % bound) + m < 0 || u - r + m > 2147483647; u = this.next(31));
    return r;
  }
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

// ---- the rest of the DrawFunc family and the seeded tables (2026-08-23) ----
const limit = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const ilimit = (v: number, lo: number, hi: number) => Math.trunc(limit(v, lo, hi));

// szubieta (SZubietaFunc): a width×height grid of 20-gons coloured by an integer bit pattern (CircleSquares / Square Tile)
registerPointSet('szubieta', (P) => {
  const W = ilimit(P.width ?? 128, 32, 256), H = ilimit(P.height ?? 128, 32, 256), type = ilimit(P.type ?? 0, 0, 1), scale = limit(P.scale ?? 0.5, 0, 1);
  const w = new PointSetWriter();
  for (let i = 0; i < W; i++) for (let j = 0; j < H; j++) {
    let x = type === 0 ? ((i * i - 2 * (i | j) + j * j) | 0) % 255 : (i & ((j - 2 * (i ^ j) + j) | 0) & i) % 256;
    x = Math.abs(x);
    w.ngon(i - W / 2, j - H / 2, 20, scale, 0, 0, x / 255);
  }
  return w.done();
});

// gpattern (GPatternFunc): rows of n-gons (the side counts cycle through the 'string' resource), java.util.Random(seed) jitter
registerPointSet('gpattern', (P, R) => {
  const polys = ((R?.string ?? '3,4,5,6,5,4,3').split(',')).map((s) => { const v = parseInt(s, 10); return !Number.isFinite(v) || v > 20 || v < 3 ? 3 : v; });
  const seed = Math.trunc(P.seed ?? 10000), width = limit(P.width ?? 4, 0, 10), height = limit(P.height ?? 2, 0, 10), lineheight = limit(P.lineheight ?? 0.15, 0.01, 1.5);
  const angle = P.angle ?? 0, randPos = P.randPos ?? 0, randSize = limit(P.randSize ?? 0, -1, 1), fill = ilimit(P.fill ?? 1, 0, 1), outline = ilimit(P.ouline ?? 0, 0, 1);
  const color = limit(P['fill color'] ?? 0, 0, 1), speedcolor = limit(P['fill color speed'] ?? 1, 0, 1), outlinecolor = limit(P['outline color'] ?? 0, 0, 1);
  const rnd = new JavaRandom(seed);
  const random = (r1: number, r2: number) => r1 + (r2 - r1) * rnd.nextDouble();
  const w = new PointSetWriter();
  if (polys.length === 0) return w.done();
  const polyWidth = lineheight;
  for (let posj = 0; posj <= width; posj += polyWidth) {
    let y = 0;
    for (let polyCntr = 0; y <= height; polyCntr++) {
      const x = posj + random(-randPos, randPos);
      const r = Math.fround(polyWidth / 2) + random(-randSize, randSize);
      if (outline === 1) w.ngon(x, y, polys[polyCntr], r, angle, outline, outlinecolor);
      if (fill === 1) { const cc = ((color + speedcolor * rnd.nextDouble()) % 1 + 1) % 1; w.ngon(x, y, polys[polyCntr], r, angle, 0, cc); }
      if (polyCntr > polys.length - 2) polyCntr = -1;
      y += lineheight;
    }
  }
  return w.done();
});

// curliecue (CurliecueFunc): the curlicue trajectory (s = java.util.Random(seed).nextDouble()) as points (blur radius =
// scale) / lines / n-gons, mirrored by the symmetry mode; (float) casts where the Java has them
registerPointSet('curliecue', (P) => {
  const size = ilimit(P.Points ?? 500, 1, 10000), seed = Math.trunc(P.seed ?? 1000), type = ilimit(P.type ?? 0, 0, 1), sides = ilimit(P.sides ?? 4, 3, 20);
  const scale = limit(P.scale ?? 0.02, 0, 1), sym = ilimit(P.symmetry ?? 1, 1, 4), ecce = limit(P.eccentricity ?? 0, 0, 1), showlines = ilimit(P.showlines ?? 1, 0, 1);
  const rnd = new JavaRandom(seed); const s = rnd.nextDouble();
  const w = new PointSetWriter(); const F = Math.fround;
  let theta = 0, phi = 0, x0 = ecce, y0 = ecce;
  for (let i = 0; i < size; i++) {
    const x1 = x0 + 0.01 * Math.cos(phi), y1 = y0 + 0.01 * Math.sin(phi), c = i / size;
    const pt = (x: number, y: number) => { if (type === 0) w.point(x, y, c); else w.ngon(x, y, sides, scale, 0, 0, c); };
    const ln = (a: number, b: number, d: number, e: number) => { if (showlines === 1) w.line(F(a), F(b), F(d), F(e), 0, c); };
    if (sym === 1) {
      if (type === 0) { pt(F(x0), F(-y0)); pt(F(x1), F(-y1)); } else { pt(x0, -y0); pt(x1, -y1); }
      ln(x0, -y0, x1, -y1);
    } else if (sym === 2) {
      pt(x0, -y0); pt(x1, -y1); pt(-x0, y0); pt(-x1, y1); pt(y0, x0); pt(y1, x1); pt(-y0, -x0); pt(-y1, -x1);
      if (type === 0) { ln(-y0, -x0, -y1, -x1); ln(y0, x0, y1, x1); ln(-x0, y0, -x1, y1); ln(x0, -y0, x1, -y1); }
      else { ln(x0, -y0, x1, -y1); ln(-x0, y0, -x1, y1); ln(y0, x0, y1, x1); ln(-y0, -x0, -y1, -x1); }
    } else if (sym === 3) {
      pt(x0, -y0); pt(x1, -y1); pt(-x0, y0); pt(-x1, y1); pt(-x0, -y0); pt(-x1, -y1); pt(x0, y0); pt(x1, y1);
      if (type === 0) { ln(x0, y0, x1, y1); ln(-x0, -y0, -x1, -y1); ln(-x0, y0, -x1, y1); ln(x0, -y0, x1, -y1); }
      else { ln(x0, -y0, x1, -y1); ln(-x0, y0, -x1, y1); ln(-x0, -y0, -x1, -y1); ln(x0, y0, x1, y1); }
    } else {
      pt(x0, -y0); pt(x1, -y1); pt(-x0, y0); pt(-x1, y1); pt(y0, x0); pt(y1, x1); pt(-y0, -x0); pt(-y1, -x1);
      pt(x0, y0); pt(x1, y1); pt(-x0, -y0); pt(-x1, -y1); pt(y0, -x0); pt(y1, -x1); pt(-y0, x0); pt(-y1, x1);
      ln(-y0, x0, -y1, x1); ln(y0, -x0, y1, -x1); ln(-x0, -y0, -x1, -y1); ln(x0, y0, x1, y1); ln(-y0, -x0, -y1, -x1); ln(y0, x0, y1, x1); ln(-x0, y0, -x1, y1); ln(x0, -y0, x1, -y1);
    }
    x0 = x1; y0 = y1;
    phi = (theta + phi) % (2 * Math.PI);
    theta = (theta + 2 * Math.PI * s) % (2 * Math.PI);
  }
  return w.done();
});

// gosperisland_js (GosperIslandFunc): six Gosper curves of order `level` around (9/32, 1/8); 6·7^level unit segments,
// so the level is capped at 6 here (JWildfire allows 10: 1.7 billion segments)
registerPointSet('gosperisland_js', (P) => {
  const n = ilimit(Math.trunc((P.level ?? 2) + 0.5), 1, 6), ANGLE = 19.106605350869094394;
  const t = new Turtle(9 / 32, 1 / 8, 0);
  const gosper = (k: number): void => { if (k === 0) { t.goForward(1); return; } t.turnLeft(-ANGLE); gosper(k - 1); t.turnLeft(60); gosper(k - 1); t.turnLeft(-60); gosper(k - 1); t.turnLeft(ANGLE); };
  for (let i = 0; i < 6; i++) { gosper(n); t.turnLeft(60); }
  return turtleLines(t.segs, (P.line_thickness ?? 0.5) / 100);
});

// rsquares_js (RsquaresFunc): a square of side 1 at (0.5, 0.5), four squares of side/2.2 at its corners, `level` deep
registerPointSet('rsquares_js', (P) => {
  const segs: number[] = [];
  const sq = (x: number, y: number, s: number) => { const h = s / 2; segs.push(x - h, y - h, x - h, y + h, x - h, y + h, x + h, y + h, x + h, y + h, x + h, y - h, x + h, y - h, x - h, y - h); };
  const draw = (n: number, x: number, y: number, s: number): void => { if (n === 0) return; sq(x, y, s); const r = 2.2; draw(n - 1, x - s / 2, y - s / 2, s / r); draw(n - 1, x - s / 2, y + s / 2, s / r); draw(n - 1, x + s / 2, y - s / 2, s / r); draw(n - 1, x + s / 2, y + s / 2, s / r); };
  draw(ilimit(Math.trunc((P.level ?? 2) + 0.5), 1, 9), 0.5, 0.5, 1);
  return turtleLines(segs, (P.line_thickness ?? 0.5) / 100);
});

// arctruchet (ArcTruchetFunc): the per-tile rotation table, java.util.Random(seed) (first draw discarded), as a raw table
registerPointSet('arctruchet', (P) => {
  const rows = ilimit(P.TilesPerRow ?? 10, 1, 100), cols = ilimit(P.TilesPerColumn ?? 10, 1, 100), seed = ilimit(P.seed ?? 10000, 0, 100000);
  const rnd = new JavaRandom(seed); rnd.nextDouble();
  const tilt = new Array<number>(rows * cols).fill(0);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) tilt[j * rows + i] = Math.trunc(rnd.nextDouble() * 4);
  const w = new PointSetWriter(); w.raw(tilt); return w.done();
});

// ---- the mandala rotator maps (js/mandala/RotatorMap, Mandala2Func): Minsky/Givens integer rotations on a grid of
// cells; every cell traces its orbit until it returns (or max_n_steps), the orbit's cells are stamped with that step
// count. `wrapLoop` = Mandala2Func's loop condition (keeps tracing while wrap_range > 0), RotatorMap stops at max.
const ROT_MAX = 20000;
const WOBBLE_PICKS = 9, WRAP_DAT = [0, 12, 8, 6, 4, 3, 2, 1.5, 1], SKEW_DAT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 23, 24, 25, 31, 32, 36, 47, 48, 63, 64];
function rotatorTrace(width: number, height: number, num: number, denom: number, minsky: boolean, wobble: number, wrap: number, extraHskew: number, wrapLoop: boolean, onVisit?: (px: number, py: number, n: number) => void): Int32Array {
  const showBase = -Math.trunc(width / 2), biggest = Math.max(width, height);
  let size = wrap === 0 ? biggest * 2 : Math.trunc(biggest * wrap);
  size |= 1;
  let base = Math.trunc(size / 2);
  if (base + (showBase + biggest) > size) base = -(showBase + biggest) + size;
  let hskew1: Int32Array | null = null, vskew = new Int32Array(0), hskew2 = new Int32Array(0), smin = 0, smax = 0;
  const adjust = (v: number): number => {
    if (wrap === 0) { while (base + v < 0 || base + v >= size) { size *= 2; base = Math.trunc(size / 2); hskew1 = null; } }
    if (!hskew1) {
      smin = -base; smax = smin + size - 1;
      hskew1 = new Int32Array(size); vskew = new Int32Array(size); hskew2 = new Int32Array(size);
      const lowangle = num / ((1 + wobble) * denom), hiangle = num / ((1 - wobble) * denom), wf = 1 / 512;
      for (let i = -base; i + base < size; i++) {
        const x = i * 2 * Math.PI * wf;
        const angle = Math.PI * 2 * (lowangle - (Math.cos(x) - 1) / 2 * (hiangle - lowangle));
        if (minsky) { const eps = Math.sqrt(2 * (1 - Math.cos(angle))); hskew1[i + base] = -Math.round(eps * i + 1e-8) + extraHskew; vskew[i + base] = Math.round(eps * i + 1e-8); hskew2[i + base] = 0; }
        else { const s = Math.sin(angle), c = (Math.cos(angle) - 1) / s; hskew1[i + base] = Math.round(c * i); vskew[i + base] = Math.round(s * i + 1e-8); hskew2[i + base] = hskew1[i + base] + extraHskew; }
      }
    }
    return v - size * Math.floor((base + v) / size);
  };
  adjust(0);
  const steps = new Int32Array(width * height);
  const offs = -showBase, minX = showBase, maxX = minX + width - 1, minY = showBase, maxY = minY + height - 1;
  let px = 0, py = 0;
  const step = () => { for (let d = 0; d < denom; d++) { px += hskew1![py + base]; if (px < smin || px > smax) px = adjust(px); py += vskew[px + base]; if (py < smin || py > smax) py = adjust(py); px += hskew2[py + base]; if (px < smin || px > smax) px = adjust(px); } };
  for (let x = minX; x <= maxX; x++) for (let y = minY; y <= maxY; y++) {
    if (steps[(x + offs) * height + (y + offs)] !== 0) continue;
    px = x; py = y;
    let n: number, guard = 0;
    for (n = 1; n <= ROT_MAX || (wrapLoop && wrap > 0); n++) { step(); if (px === x && py === y) break; if (n > ROT_MAX) n = ROT_MAX; if (++guard > 40 * ROT_MAX) { n = ROT_MAX; break; } }
    px = x; py = y;
    for (let k = 1; k <= n; k++) {
      if (px >= minX && px <= maxX && py >= minY && py <= maxY) {
        const idx = (px + offs) * height + (py + offs);
        if (steps[idx] !== 0) break;
        onVisit?.(px, py, n);
        steps[idx] = n;
      }
      step(); if (px === x && py === y) break;
    }
  }
  return steps;
}
/** java.awt.Color.HSBtoRGB (8-bit ints), float arithmetic as in Java */
export function javaHSBtoRGB(hue: number, sat: number, bri: number): [number, number, number] {
  const F = Math.fround;
  if (sat === 0) { const v = Math.trunc(F(bri * 255 + 0.5)); return [v, v, v]; }
  const h = F((hue - Math.floor(hue)) * 6), f = F(h - Math.floor(h));
  const p = F(bri * (1 - sat)), q = F(bri * (1 - sat * f)), t = F(bri * (1 - sat * (1 - f)));
  let c: [number, number, number];
  switch (Math.trunc(h)) { case 0: c = [bri, t, p]; break; case 1: c = [q, bri, p]; break; case 2: c = [p, bri, t]; break; case 3: c = [p, q, bri]; break; case 4: c = [t, p, bri]; break; case 5: c = [bri, p, q]; break; default: c = [0, 0, 0]; }
  return c.map((v) => Math.trunc(F(v * 255 + 0.5))) as [number, number, number];
}
/** java.awt.Color.RGBtoHSB's hue (float arithmetic) */
export function javaRGBtoHue(r: number, g: number, b: number): number {
  const F = Math.fround;
  const cmax = Math.max(r, g, b), cmin = Math.min(r, g, b);
  if (cmax === 0 || cmax === cmin) return 0;
  const d = F(cmax - cmin), redc = F((cmax - r) / d), greenc = F((cmax - g) / d), bluec = F((cmax - b) / d);
  let h = r === cmax ? F(bluec - greenc) : g === cmax ? F(F(2 + redc) - bluec) : F(F(4 + greenc) - redc);
  h = F(h / 6); if (h < 0) h = F(h + 1);
  return h;
}
/** The mandala colour maps by orbit step count: mode 0 = Color(n%91/90, n%123/122, n%17/16), 1 = HSB(frac(log n), .85, 1),
 *  2 = HSB(frac(n / 6.333333), .85, 1); n = max_n_steps + 1 is black. Returns the 8-bit RGB and its RGBtoHSB hue. */
export function mandalaColor(n: number, mode: number): { rgb: [number, number, number]; hue: number } {
  const F = Math.fround;
  let rgb: [number, number, number];
  if (n >= ROT_MAX + 1) rgb = [0, 0, 0];
  else if (mode === 1) { const z = F(Math.log(n)); rgb = javaHSBtoRGB(F(z - Math.floor(z)), 0.85, 1); }
  else if (mode === 2) { const z = F(n / 6.333333); rgb = javaHSBtoRGB(F(z - Math.floor(z)), 0.85, 1); }
  else rgb = [F((n % 91) / 90), F((n % 123) / 122), F((n % 17) / 16)].map((v) => Math.trunc(F(v * 255 + 0.5))) as [number, number, number];
  return { rgb, hue: javaRGBtoHue(rgb[0], rgb[1], rgb[2]) };
}
const mandalaParams = (P: Record<string, number>, wmin: number, wmax: number) => {
  const width = ilimit(P.width ?? (wmax === 1000 ? 300 : 500), wmin, wmax);
  // JWildfire's setParameter order (num, then denom) with its "2·num must not be a multiple of denom" bumps
  let n = Math.max(1, P.num ?? 1); if ((n * 2) % 3 === 0) n++; const num = ilimit(n, 1, 15);
  let d = Math.max(3, P.denom ?? 3); if ((num * 2) % d === 0) d++; const denom = ilimit(d, 3, 16);
  const minsky = ilimit(P.minsky ?? 0, 0, 1) === 1;
  const wp = ilimit(P.wobble ?? 0, 0, WOBBLE_PICKS) % WOBBLE_PICKS, wrp = ilimit(P.wrap_range ?? 0, 0, WRAP_DAT.length) % WRAP_DAT.length, hp = ilimit(P.hskew ?? 0, 0, SKEW_DAT.length) % SKEW_DAT.length;
  return { width, num, denom, minsky, wobble: wp === 0 ? 0 : Math.pow(10, -6 + 0.5 * wp), wrap: WRAP_DAT[wrp], extra: SKEW_DAT[hp] };
};
// mandala (MandalaFunc + RotatorMap): one unit 4-gon (scale = size) per orbit cell; colour slot = the hue of the
// HSB(frac(log n), .85, 1) colour (color id 1), slots 9..11 its RGB in 0..1 (color id 0)
registerPointSet('mandala', (P) => {
  const m = mandalaParams(P, 100, 1000), size = limit(P.size ?? 0.6, 0, 1);
  const w = new PointSetWriter(); const cache = new Map<number, { rgb: [number, number, number]; hue: number }>();
  rotatorTrace(m.width, m.width, m.num, m.denom, m.minsky, m.wobble, m.wrap, m.extra, false, (px, py, n) => {
    let c = cache.get(n); if (!c) { c = mandalaColor(n, 1); cache.set(n, c); }
    w.ngon(px, py, 4, size, 0, 0, c.hue, [c.rgb[0] / 255, c.rgb[1] / 255, c.rgb[2] / 255]);
  });
  return w.done();
});
// mandala2 (Mandala2Func): the per-cell colour as a raw table ([x][y] → x·width + y): the hue of the mode-0 colour for
// color id 0, else the 8-bit RGB packed into one float (r·65536 + g·256 + b)
registerPointSet('mandala2', (P) => {
  const m = mandalaParams(P, 100, 2000), cid = ilimit(P['color id'] ?? 0, 0, 3);
  const steps = rotatorTrace(m.width, m.width, m.num, m.denom, m.minsky, m.wobble, m.wrap, m.extra, true);
  const cache = new Map<number, number>();
  const tab = Array.from(steps, (n) => { let v = cache.get(n); if (v === undefined) { const c = mandalaColor(n, cid === 1 ? 1 : cid === 2 ? 2 : 0); v = cid === 0 ? c.hue : c.rgb[0] * 65536 + c.rgb[1] * 256 + c.rgb[2]; cache.set(n, v); } return v; });
  const w = new PointSetWriter(); w.raw(tab); return w.done();
});

// nsudoku (NSudokuFunc): `level` nested sudoku boards of n-gons. JWildfire's board comes from an unseeded
// java.util.Random (a new one per cell) and a static generator shared by every instance — never reproducible — so
// both draw from java.util.Random(1000) here (the generator's 81 makeHoles draws included); the row-transform
// machinery (tSudoku: 1296 permutation rows, Compare / updateTotal / peek) is exact.
registerPointSet('nsudoku', (P) => {
  const level = ilimit(P.Level ?? 5, 1, 9), thickness = 1 - limit(P.thickness ?? 0.5, 0, 1), size = limit(P.size ?? 0.5, 0.3, 1), angle = P.angle ?? 0, type = ilimit(P.type ?? 4, 3, 6);
  const boardRnd = new JavaRandom(1000), rnd = new JavaRandom(1000);
  const board: number[][] = Array.from({ length: 9 }, () => new Array(9).fill(0));
  const legal = (x: number, y: number, c: number) => {
    for (let i = 0; i < 9; i++) if (c === board[x][i]) return false;
    for (let i = 0; i < 9; i++) if (c === board[i][y]) return false;
    const cx = x > 2 ? (x > 5 ? 6 : 3) : 0, cy = y > 2 ? (y > 5 ? 6 : 3) : 0;
    for (let i = cx; i < 10 && i < cx + 3; i++) for (let j = cy; j < 10 && j < cy + 3; j++) if (c === board[i][j]) return false;
    return true;
  };
  const nextCell = (x: number, y: number): boolean => {
    const toCheck = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    for (let i = toCheck.length - 1; i > 0; i--) { const cur = boardRnd.nextInt(i); const tmp = toCheck[cur]; toCheck[cur] = toCheck[i]; toCheck[i] = tmp; }
    for (let i = 0; i < toCheck.length; i++) {
      if (legal(x, y, toCheck[i])) {
        board[x][y] = toCheck[i];
        let nx = x, ny = y;
        if (x === 8) { if (y === 8) return true; nx = 0; ny = y + 1; } else nx = x + 1;
        if (nextCell(nx, ny)) return true;
      }
    }
    board[x][y] = 0; return false;
  };
  nextCell(0, 0);
  for (let i = 0; i < 81; i++) rnd.nextDouble(); // makeHoles(0)
  // tSudoku.M
  const perm = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]];
  let total: number[][] = [];
  for (let n = 0; n < 6; n++) for (let m = 0; m < 6; m++) for (let l = 0; l < 6; l++) for (let j = 0; j < 6; j++) {
    const c = [...perm[n], ...perm[m], ...perm[l], ...perm[j]];
    const row = c.slice();
    for (let i = 0; i < 9; i++) row[i] = c[i] + 3 * c[i > 5 ? 11 : i > 2 ? 10 : 9] - 3;
    total.push(row);
  }
  const compare = (tot: number[][], x: number[]) => tot.map((r) => { let k = 0; for (let j = 0; j < x.length; j++) if (r[j] === x[j]) k++; return k; });
  const vec = (s: number[][]) => { const v = new Array(81).fill(0); for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) v[9 * i + j] = s[j][i]; return v; };
  const w = new PointSetWriter();
  const poly = (color: number[], scale: number) => {
    for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) {
      const radius = size * (scale / level) / 9, area = (1 - 1 / scale) === 0 ? 0 : thickness;
      w.ngon(i / 9 - 0.5, j / 9 - 0.5, type, radius, angle, area, color[9 * i + j] / 9);
    }
  };
  let cmp: number[] = [];
  for (let n = level; n > 0; n--) {
    let sv: number[];
    if (n === level) { const tran = [1, 2, 3, 4, 5, 6, 7, 8, 9]; cmp = compare(total, tran); sv = vec(board); }
    else {
      if (total.length === 0) break;
      const tran = total[Math.trunc(rnd.nextDouble() * total.length)].slice(0, 9);
      const ns = Array.from({ length: 9 }, (_, i) => board[tran[i] - 1].slice());
      cmp = compare(total, tran); sv = vec(ns);
    }
    poly(sv, n);
    total = total.filter((_, i) => cmp[i] === 0);
  }
  return w.done();
});

// natural_foam (NaturalFoamFunc): the bubble table [x, y, z, radius] from java.util.Random(seed)
registerPointSet('natural_foam', (P) => {
  const density = ilimit(P.density ?? 50, 0, 2000), spread = P.spread ?? 1, minR = P.minRadius ?? 0.1, maxR = P.maxRadius ?? 0.9;
  const rnd = new JavaRandom(Math.trunc(P.seed ?? 123));
  const t: number[] = [];
  for (let i = 0; i < density; i++) { const x = (2 * rnd.nextDouble() - 1) * spread, y = (2 * rnd.nextDouble() - 1) * spread, z = (2 * rnd.nextDouble() - 1) * spread, r = minR + (maxR - minR) * rnd.nextDouble(); t.push(x, y, z, r); }
  const w = new PointSetWriter(); w.raw(t); return w.done();
});

// neuron3D (Neuron3DFunc.PerlinNoise): the seed-shuffled 512-entry permutation table (java.util.Random(seed).nextInt)
registerPointSet('neuron3D', (P) => {
  const rnd = new JavaRandom(Math.trunc(P.seed ?? 12345));
  const perm = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) { const k = rnd.nextInt(i + 1); const t = perm[k]; perm[k] = perm[i]; perm[i] = t; }
  const w = new PointSetWriter(); w.raw([...perm, ...perm]); return w.done();
});

// ---- sunvoroni (SunflowerVoroniFunc + megamu.mesh.Voronoi + csk.taprats.geometry.Triangulate): the Voronoi diagram of the
// sunflower points, bounded by megamu's three far points (±8000); cell outlines as lines, cells ear-clipped into triangles.
// megamu gets the Delaunay faces from QuickHull3D on the lifted points; a Bowyer–Watson insertion into the far triangle
// gives the same face set (general position), circumcentres by the same formula, (float) casts where megamu has them.
// Unknown without the quickhull jar: QuickHull's face order, which picks each region's starting vertex and so the
// ear-clipping decomposition and the per-triangle random colours (mode 2) — regions here start at the incident
// triangle with the smallest angle and run counter-clockwise (megamu's come out counter-clockwise too).
export function voronoiOfSunflower(nPoints: number, angle: number): { points: [number, number][]; edges: [number, number, number, number][]; regions: [number, number][][] } {
  const F = Math.fround;
  const ang = angle * (3 - Math.sqrt(5)), rmax = Math.sqrt(nPoints + 1) / 30;
  const pts: [number, number][] = [];
  for (let i = 0; i < nPoints; i++) { const r = Math.sqrt(i + 1) / 30, t = (i + 1) * ang * Math.PI / 180; pts.push([F(r * Math.cos(t) / rmax), F(r * Math.sin(t) / rmax)]); }
  const all: [number, number][] = [...pts, [-8000, 0], [8000, 8000], [8000, -8000]];
  const n = nPoints, fA = n, fB = n + 1, fC = n + 2;
  // Bowyer–Watson into the far triangle
  type Tri = { a: number; b: number; c: number; cx: number; cy: number; r2: number; alive: boolean };
  const mk = (a: number, b: number, c: number): Tri => {
    const [ax, ay] = all[a], [bx, by] = all[b], [cx0, cy0] = all[c];
    const d = 2 * (ax * (by - cy0) + bx * (cy0 - ay) + cx0 * (ay - by));
    const ux = ((ax * ax + ay * ay) * (by - cy0) + (bx * bx + by * by) * (cy0 - ay) + (cx0 * cx0 + cy0 * cy0) * (ay - by)) / d;
    const uy = ((ax * ax + ay * ay) * (cx0 - bx) + (bx * bx + by * by) * (ax - cx0) + (cx0 * cx0 + cy0 * cy0) * (bx - ax)) / d;
    return { a, b, c, cx: ux, cy: uy, r2: (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy), alive: true };
  };
  let tris: Tri[] = [mk(fA, fB, fC)];
  for (let p = 0; p < n; p++) {
    const [px, py] = all[p];
    const bad: Tri[] = [];
    for (const t of tris) if (t.alive && (px - t.cx) * (px - t.cx) + (py - t.cy) * (py - t.cy) < t.r2) bad.push(t);
    const edgeCount = new Map<string, [number, number]>();
    for (const t of bad) { t.alive = false; for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]] as [number, number][]) { const k = u < v ? u + ':' + v : v + ':' + u; if (edgeCount.has(k)) edgeCount.delete(k); else edgeCount.set(k, [u, v]); } }
    for (const [u, v] of edgeCount.values()) tris.push(mk(u, v, p));
    if (tris.length > 4000) tris = tris.filter((t) => t.alive);
  }
  tris = tris.filter((t) => t.alive);
  // megamu's dual points (circumcentres by its own formula, in double)
  const dual: [number, number][] = tris.map((t) => {
    const [x0, y0] = all[t.a], [x1, y1] = all[t.b], [x2, y2] = all[t.c];
    const v1x = 2 * (x1 - x0), v1y = 2 * (y1 - y0), v1z = x0 * x0 - x1 * x1 + y0 * y0 - y1 * y1;
    const v2x = 2 * (x2 - x0), v2y = 2 * (y2 - y0), v2z = x0 * x0 - x2 * x2 + y0 * y0 - y2 * y2;
    const tx = v1y * v2z - v1z * v2y, ty = v1z * v2x - v1x * v2z, tz = v1x * v2y - v1y * v2x;
    return [tx / tz, ty / tz];
  });
  // edges between triangles sharing an edge
  const byEdge = new Map<string, number[]>();
  tris.forEach((t, i) => { for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]] as [number, number][]) { const k = u < v ? u + ':' + v : v + ':' + u; (byEdge.get(k) ?? byEdge.set(k, []).get(k)!).push(i); } });
  const edges: [number, number, number, number][] = [];
  for (const [, ts] of byEdge) if (ts.length === 2) { const [i, j] = ts[0] > ts[1] ? ts : [ts[1], ts[0]]; edges.push([F(dual[i][0]), F(dual[i][1]), F(dual[j][0]), F(dual[j][1])]); }
  // regions: the incident triangles of each point, counter-clockwise by angle around the point
  const incident: number[][] = Array.from({ length: n }, () => []);
  tris.forEach((t, i) => { for (const v of [t.a, t.b, t.c]) if (v < n) incident[v].push(i); });
  const regions = incident.map((list, p) => {
    const [px, py] = pts[p];
    const ordered = list.map((i) => ({ i, a: Math.atan2(dual[i][1] - py, dual[i][0] - px) })).sort((u, v) => u.a - v.a);
    return ordered.map((o) => [F(dual[o.i][0]), F(dual[o.i][1])] as [number, number]);
  });
  return { points: pts, edges, regions };
}
/** csk.taprats.geometry.Triangulate.Process (ear clipping, Snip / InsideTriangle as in the Java) → flat triangle list */
export function earClip(contour: [number, number][]): [number, number][] {
  const n = contour.length; if (n < 3) return [];
  const area = (c: [number, number][]) => { let A = 0; for (let p = c.length - 1, q = 0; q < c.length; p = q++) A += c[p][0] * c[q][1] - c[q][0] * c[p][1]; return A * 0.5; };
  const V = new Array<number>(n);
  if (0 < area(contour)) for (let v = 0; v < n; v++) V[v] = v; else for (let v = 0; v < n; v++) V[v] = n - 1 - v;
  const inside = (Ax: number, Ay: number, Bx: number, By: number, Cx: number, Cy: number, Px: number, Py: number) => {
    const ax = Cx - Bx, ay = Cy - By, bx = Ax - Cx, by = Ay - Cy, cx = Bx - Ax, cy = By - Ay;
    const apx = Px - Ax, apy = Py - Ay, bpx = Px - Bx, bpy = Py - By, cpx = Px - Cx, cpy = Py - Cy;
    return ax * bpy - ay * bpx >= 0 && bx * cpy - by * cpx >= 0 && cx * apy - cy * apx >= 0;
  };
  const snip = (u: number, v: number, w: number, nv: number) => {
    const [Ax, Ay] = contour[V[u]], [Bx, By] = contour[V[v]], [Cx, Cy] = contour[V[w]];
    if (1e-10 > (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)) return false;
    for (let p = 0; p < nv; p++) { if (p === u || p === v || p === w) continue; const [Px, Py] = contour[V[p]]; if (inside(Ax, Ay, Bx, By, Cx, Cy, Px, Py)) return false; }
    return true;
  };
  const out: [number, number][] = [];
  let nv = n, count = 2 * nv;
  for (let v = nv - 1; nv > 2;) {
    if (0 >= count--) break;
    let u = v; if (nv <= u) u = 0;
    v = u + 1; if (nv <= v) v = 0;
    let w = v + 1; if (nv <= w) w = 0;
    if (snip(u, v, w, nv)) {
      out.push(contour[V[u]], contour[V[v]], contour[V[w]]);
      for (let s = v, t = v + 1; t < nv; s++, t++) V[s] = V[t];
      nv--; count = 2 * nv;
    }
  }
  return out;
}
registerPointSet('sunvoroni', (P) => {
  const nPoints = ilimit(P.nPoints ?? 50, 10, 1000), angle = limit(P.angle ?? 180, 0, 360), colormode = ilimit(P['color mode'] ?? 0, 0, 2);
  let outline = ilimit(P.outline ?? 0, 0, 1); const fill = ilimit(P.fill ?? 1, 0, 1), outlinecolor = limit(P['outline color'] ?? 0.5, 0, 1);
  if (outline === 0 && fill === 0) outline = 1;
  const { edges, regions } = voronoiOfSunflower(nPoints, angle);
  const w = new PointSetWriter();
  if (outline === 1) for (const e of edges) w.line(e[0], e[1], e[2], e[3], 0, outlinecolor);
  if (fill === 1) {
    const area = (c: [number, number][]) => { let A = 0; for (let p = c.length - 1, q = 0; q < c.length; p = q++) A += c[p][0] * c[q][1] - c[q][0] * c[p][1]; return A * 0.5; };
    regions.forEach((reg, i) => {
      const a = area(reg); const rnd = new JavaRandom(Math.trunc(a));
      let color = colormode === 0 ? ((1 + a) % 1 + 1) % 1 : colormode === 1 ? i / regions.length : 0;
      if (colormode === 0) color = (1 + a) - Math.trunc(1 + a); // Java fmod keeps the sign
      const t = earClip(reg);
      for (let j = 0; j + 2 < t.length; j += 3) { if (colormode === 2) color = rnd.nextDouble(); w.triangle(t[j][0], t[j][1], t[j + 1][0], t[j + 1][1], t[j + 2][0], t[j + 2][1], color); }
    });
  }
  return w.done();
});

// curliecue2 (CurliecueFunc2): the transform ignores its input and advances one trajectory per call — x += 0.001·cos φ,
// φ += θ, θ += 2π·s (s = java.util.Random(seed).nextDouble()) — so a JWildfire thread draws the first Q steps of the
// curlicue, Q = its iteration count (≈ 2 million for a 512 px / quality-100 render). Our walkers are short-lived, so the
// trajectory is tabulated instead (2^20 steps, [N, 0, x1, y1, …]) and every point samples a uniform step.
export const CURLIECUE2_STEPS = 1 << 20;
registerPointSet('curliecue2', (P) => {
  const s = new JavaRandom(Math.trunc(P.seed ?? 1000)).nextDouble();
  const N = CURLIECUE2_STEPS, tab = new Float32Array(2 + 2 * N);
  tab[0] = N;
  let x0 = 0, y0 = 0, theta = 0, phi = 0;
  for (let k = 0; k < N; k++) {
    x0 += 0.001 * Math.cos(phi); y0 += 0.001 * Math.sin(phi);
    phi = (theta + phi) % (2 * Math.PI); theta = (theta + 2 * Math.PI * s) % (2 * Math.PI);
    tab[2 + 2 * k] = x0; tab[3 + 2 * k] = y0;
  }
  const count = Math.ceil(tab.length / PSET_STRIDE), data = new Float32Array(count * PSET_STRIDE); data.set(tab);
  return { data, count };
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
