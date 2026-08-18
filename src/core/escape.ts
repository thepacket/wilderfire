// Escape-time fractal layer: a per-pixel iteration z ← f(z, c) with a bailout,
// coloured from the escape data through a gradient. Rendered by src/gpu/escapeRenderer.ts.

import type { RGB } from './flame';
import { compileFormula, compileFormulaDS } from './formula';
import { bitsForZoom, centreFixed, fromNumber, fxAdd, toDecimal, toNumber } from './bigfloat';

/** Built-in iteration formulas (custom = the `custom` expression). Mode chooses what the pixel is:
 *  Mandelbrot: c = pixel, z₀ = seed; Julia: z₀ = pixel, c = the constant. */
export const ESCAPE_FORMULAS = {
  mandelbrot: { label: 'Mandelbrot z^p + c', wgsl: 'cpow(z, vec2f(P.power, 0.0)) + c', ds: 'dc_add(dc_pow(z, dc_f(P.power, 0.0)), c)', power: true },
  burningship: { label: 'Burning Ship', wgsl: 'csqr(abs(z)) + c', ds: 'dc_add(dc_sqr(dc_abs2(z)), c)', power: false },
  tricorn: { label: 'Tricorn (Mandelbar)', wgsl: 'csqr(cconj(z)) + c', ds: 'dc_add(dc_sqr(dc_conj(z)), c)', power: false },
  celtic: { label: 'Celtic', wgsl: 'vec2f(abs(z.x * z.x - z.y * z.y), 2.0 * z.x * z.y) + c', ds: 'dc_add(DC(ds_abs(ds_sub(ds_mul(z.x, z.x), ds_mul(z.y, z.y))), ds_mulf(ds_mul(z.x, z.y), 2.0)), c)', power: false },
  perpendicular: { label: 'Perpendicular Burning Ship', wgsl: 'vec2f(z.x * z.x - z.y * z.y, -2.0 * abs(z.x) * z.y) + c', ds: 'dc_add(DC(ds_sub(ds_mul(z.x, z.x), ds_mul(z.y, z.y)), ds_mulf(ds_mul(ds_abs(z.x), z.y), -2.0)), c)', power: false },
  lambda: { label: 'Lambda c·z·(1−z)', wgsl: 'cmul(c, cmul(z, vec2f(1.0, 0.0) - z))', ds: 'dc_mul(c, dc_mul(z, dc_sub(dc_f(1.0, 0.0), z)))', power: false },
  phoenix: { label: 'Phoenix z² + Re(c) + Im(c)·z₋₁', wgsl: 'csqr(z) + vec2f(c.x, 0.0) + c.y * zprev', ds: 'dc_add(dc_add(dc_sqr(z), dc_re(c)), dc_mulf(zprev, ds_hi(c.y)))', power: false, prev: true },
  magnet: { label: 'Magnet I ((z²+c−1)/(2z+c−2))²', wgsl: 'csqr(cdiv(csqr(z) + c - vec2f(1.0, 0.0), 2.0 * z + c - vec2f(2.0, 0.0)))', ds: 'dc_sqr(dc_div(dc_sub(dc_add(dc_sqr(z), c), dc_f(1.0, 0.0)), dc_sub(dc_add(dc_mulf(z, 2.0), c), dc_f(2.0, 0.0))))', power: false },
  newton: { label: 'Newton z^p − 1', wgsl: 'z - cdiv(cpow(z, vec2f(P.power, 0.0)) - vec2f(1.0, 0.0), P.power * cpow(z, vec2f(P.power - 1.0, 0.0)))', ds: 'dc_sub(z, dc_div(dc_sub(dc_pow(z, dc_f(P.power, 0.0)), dc_f(1.0, 0.0)), dc_mulf(dc_pow(z, dc_f(P.power - 1.0, 0.0)), P.power)))', power: true, convergent: true },
  nova: { label: 'Nova z − (z^p−1)/(p z^(p−1)) + c', wgsl: 'z - cdiv(cpow(z, vec2f(P.power, 0.0)) - vec2f(1.0, 0.0), P.power * cpow(z, vec2f(P.power - 1.0, 0.0))) + c', ds: 'dc_add(dc_sub(z, dc_div(dc_sub(dc_pow(z, dc_f(P.power, 0.0)), dc_f(1.0, 0.0)), dc_mulf(dc_pow(z, dc_f(P.power - 1.0, 0.0)), P.power))), c)', power: true, convergent: true },
  custom: { label: 'Custom formula', wgsl: '', ds: '', power: true },
} as const;
export type EscapeFormula = keyof typeof ESCAPE_FORMULAS;

export const OUTSIDE_COLORINGS = ['smooth', 'iterations', 'exp-smooth', 'orbit-trap', 'distance', 'angle', 'solid'] as const;
export const INSIDE_COLORINGS = ['solid', 'orbit-trap', 'final-mag', 'final-angle', 'exp-smooth'] as const;
export const TRAP_SHAPES = ['point', 'cross', 'lines', 'circle', 'square', 'ring'] as const;
export const TRANSFERS = ['linear', 'sqrt', 'cuberoot', 'log'] as const;
/** arithmetic tier: f32 (fast, ~1e4× zoom), double-single (~1e11×), perturbation (reference orbit + rebasing, unlimited
 *  for z^p + c); auto picks by zoom */
export const PRECISIONS = ['auto', 'f32', 'ds', 'perturb'] as const;
export type Precision = typeof PRECISIONS[number];

export interface EscapeColoring {
  outside: typeof OUTSIDE_COLORINGS[number];
  inside: typeof INSIDE_COLORINGS[number];
  /** gradient cycles per unit of the colouring value */
  density: number;
  /** gradient phase 0..1 */
  offset: number;
  transfer: typeof TRANSFERS[number];
  insideColor: RGB;
  solidColor: RGB;
  /** alpha of the inside / of the outside colouring (0 = the layers below show through there) */
  insideAlpha: number;
  outsideAlpha: number;
  trap: { shape: typeof TRAP_SHAPES[number]; x: number; y: number; size: number; /** minimum distance over the orbit (else the last iteration's) */ min: boolean };
}

export interface EscapeLayerData {
  formula: EscapeFormula;
  /** custom formula source (z, c, pixel, n, p1..p4) */
  custom: string;
  mode: 'mandelbrot' | 'julia';
  power: number;
  /** z₀ in Mandelbrot mode */
  seed: [number, number];
  /** c in Julia mode */
  c: [number, number];
  /** free parameters p1..p4 of custom formulas */
  params: [[number, number], [number, number], [number, number], [number, number]];
  maxIter: number;
  bailout: number;
  coloring: EscapeColoring;
  palette: RGB[];
  /** view: centre + zoom (1 = the classic ±2 frame in the smaller dimension) + rotation (rad) */
  centerX: number;
  centerY: number;
  /** the exact centre as decimal strings once the zoom is past f64 (deep views); kept in step by the nav helpers */
  centerHi?: [string, string];
  zoom: number;
  rotation: number;
  /** supersampling grid per pixel (1..3) */
  antialias: number;
  precision: Precision;
}

export function defaultEscape(palette: RGB[]): EscapeLayerData {
  return {
    formula: 'mandelbrot', custom: 'z^2 + c', mode: 'mandelbrot', power: 2, seed: [0, 0], c: [-0.75, 0.11],
    params: [[0, 0], [0, 0], [0, 0], [0, 0]], maxIter: 250, bailout: 128,
    coloring: { outside: 'smooth', inside: 'solid', density: 0.05, offset: 0, transfer: 'linear', insideColor: [0, 0, 0], solidColor: [1, 1, 1], insideAlpha: 1, outsideAlpha: 1, trap: { shape: 'point', x: 0, y: 0, size: 1, min: true } },
    palette: palette.map((c) => [...c] as RGB),
    centerX: -0.5, centerY: 0, zoom: 1, rotation: 0, antialias: 1, precision: 'auto',
  };
}

const num = (v: unknown, d: number) => (typeof v === 'number' && isFinite(v) ? v : d);
const pair = (v: unknown, d: [number, number]): [number, number] => (Array.isArray(v) && v.length === 2 ? [num(v[0], d[0]), num(v[1], d[1])] : d);
const rgb = (v: unknown, d: RGB): RGB => (Array.isArray(v) && v.length === 3 ? [num(v[0], d[0]), num(v[1], d[1]), num(v[2], d[2])] : d);
const oneOf = <T extends string>(v: unknown, list: readonly T[], d: T): T => ((list as readonly string[]).includes(v as string) ? (v as T) : d);

export function normalizeEscape(obj: any, fallbackPalette: RGB[]): EscapeLayerData {
  const d = defaultEscape(fallbackPalette);
  if (!obj || typeof obj !== 'object') return d;
  const col = obj.coloring ?? {};
  const pal: RGB[] = Array.isArray(obj.palette) && obj.palette.length === 256 ? obj.palette.map((c: unknown, i: number) => rgb(c, d.palette[i])) : d.palette;
  return {
    formula: oneOf(obj.formula, Object.keys(ESCAPE_FORMULAS) as EscapeFormula[], d.formula),
    custom: typeof obj.custom === 'string' ? obj.custom : d.custom,
    mode: obj.mode === 'julia' ? 'julia' : 'mandelbrot',
    power: num(obj.power, d.power), seed: pair(obj.seed, d.seed), c: pair(obj.c, d.c),
    params: [0, 1, 2, 3].map((i) => pair(obj.params?.[i], [0, 0])) as EscapeLayerData['params'],
    maxIter: Math.max(1, Math.min(20000, Math.round(num(obj.maxIter, d.maxIter)))), bailout: Math.max(1e-6, num(obj.bailout, d.bailout)),
    coloring: {
      outside: oneOf(col.outside, OUTSIDE_COLORINGS, d.coloring.outside), inside: oneOf(col.inside, INSIDE_COLORINGS, d.coloring.inside),
      density: num(col.density, d.coloring.density), offset: num(col.offset, 0), transfer: oneOf(col.transfer, TRANSFERS, 'linear'),
      insideColor: rgb(col.insideColor, d.coloring.insideColor), solidColor: rgb(col.solidColor, d.coloring.solidColor),
      insideAlpha: Math.min(1, Math.max(0, num(col.insideAlpha, 1))), outsideAlpha: Math.min(1, Math.max(0, num(col.outsideAlpha, 1))),
      trap: { shape: oneOf(col.trap?.shape, TRAP_SHAPES, 'point'), x: num(col.trap?.x, 0), y: num(col.trap?.y, 0), size: num(col.trap?.size, 1), min: col.trap?.min !== false },
    },
    palette: pal,
    centerX: num(obj.centerX, d.centerX), centerY: num(obj.centerY, d.centerY), zoom: Math.max(1e-30, num(obj.zoom, 1)), rotation: num(obj.rotation, 0),
    ...(Array.isArray(obj.centerHi) && obj.centerHi.length === 2 && obj.centerHi.every((v: unknown) => typeof v === 'string') ? { centerHi: [obj.centerHi[0], obj.centerHi[1]] as [string, string] } : {}),
    antialias: Math.max(1, Math.min(3, Math.round(num(obj.antialias, 1)))),
    precision: oneOf(obj.precision, PRECISIONS, 'auto'),
  };
}

// ---- precision tiers ----
/** integer power z^p + c is what the perturbation path knows */
export function perturbable(e: EscapeLayerData): boolean {
  return e.formula === 'mandelbrot' && Math.abs(e.power - Math.round(e.power)) < 1e-9 && e.power >= 2 && e.power <= 8;
}
/** the tier actually rendered */
export function escapeTier(e: EscapeLayerData): 'f32' | 'ds' | 'perturb' {
  if (e.precision === 'perturb') return perturbable(e) ? 'perturb' : 'ds';
  if (e.precision !== 'auto') return e.precision;
  if (e.zoom < 3e3) return 'f32';
  if (e.zoom > 5e9 && perturbable(e)) return 'perturb';
  return 'ds';
}
/** zoom past which the f64 centre no longer places a pixel (the exact centre lives in centerHi from here on) */
export const DEEP_ZOOM = 1e8;

/** Move the view centre by a world offset (f64 deltas are all a pan/zoom step needs — the exact centre keeps
 *  every digit as decimal strings once the view is deep). */
export function escapeMoveCentre(e: EscapeLayerData, dx: number, dy: number) {
  if (e.zoom > DEEP_ZOOM || e.centerHi) {
    const P = bitsForZoom(e.zoom);
    const [cx, cy] = centreFixed(e.centerX, e.centerY, e.centerHi, P);
    const nx = fxAdd(cx, fromNumber(dx, P)), ny = fxAdd(cy, fromNumber(dy, P));
    const digits = Math.ceil(P * 0.30103) + 2;
    e.centerHi = [toDecimal(nx, digits), toDecimal(ny, digits)];
    e.centerX = toNumber(nx); e.centerY = toNumber(ny);
    if (e.zoom <= DEEP_ZOOM) delete e.centerHi; // back in f64 land
  } else {
    e.centerX += dx; e.centerY += dy;
  }
}
/** Set the centre from user input (decimal strings or numbers). */
export function escapeSetCentre(e: EscapeLayerData, x: string, y: string) {
  const P = bitsForZoom(e.zoom);
  const [cx, cy] = centreFixed(0, 0, [x, y], P);
  e.centerX = toNumber(cx); e.centerY = toNumber(cy);
  if (e.zoom > DEEP_ZOOM) { const digits = Math.ceil(P * 0.30103) + 2; e.centerHi = [toDecimal(cx, digits), toDecimal(cy, digits)]; }
  else delete e.centerHi;
}
/** The centre as decimal strings for display (deep: the exact ones). */
export function escapeCentreText(e: EscapeLayerData): [string, string] {
  return e.centerHi ?? [String(e.centerX), String(e.centerY)];
}

/** What decides the compiled shader (everything else is a uniform): formula, custom text, mode, colourings, trap shape, AA, tier. */
export function escapeSignature(e: EscapeLayerData): string {
  return [e.formula, e.formula === 'custom' ? e.custom : '', e.mode, e.coloring.outside, e.coloring.inside, e.coloring.trap.shape, e.coloring.trap.min ? 'min' : 'last', e.coloring.transfer, e.antialias, escapeTier(e), escapeTier(e) === 'perturb' ? Math.round(e.power) : ''].join('|');
}

/** The formula's WGSL — f32 and DS forms (custom formulas compile here; a broken one falls back to z²+c and returns the error). */
export function escapeFormulaWgsl(e: EscapeLayerData): { wgsl: string; ds: string; prev: boolean; convergent: boolean; error?: string } {
  const def = ESCAPE_FORMULAS[e.formula];
  if (e.formula !== 'custom') return { wgsl: def.wgsl, ds: def.ds, prev: 'prev' in def && !!def.prev, convergent: 'convergent' in def && !!def.convergent };
  try { return { wgsl: compileFormula(e.custom), ds: compileFormulaDS(e.custom), prev: false, convergent: false }; }
  catch (err) { return { wgsl: 'csqr(z) + c', ds: 'dc_add(dc_sqr(z), c)', prev: false, convergent: false, error: (err as Error).message }; }
}
