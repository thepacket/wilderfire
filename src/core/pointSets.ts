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
