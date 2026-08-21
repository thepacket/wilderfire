// JWildfire spatial filters (FilterHolder / FilterKernel), CPU side.
// FilterHolder builds an N×N kernel from a radius: N = int(2·support·r)+1, made
// odd; coefficient index ii = ((2i+1)/N − 1)·adjust with adjust = support·N/fw;
// weights are normalised to 1. All 18 FilterKernelType kernels are here (Mitchell-smooth
// b = 0.42, c = 0.29 is JWildfire's default; the SinePow family is popular in the wild).
// LogDensityFilter filters colours with the primary kernel and, for "sharpening"
// kernels (Mitchell, Lanczos, CatRom, Blackman, Hamming, Hanning), the intensity with a gaussian of radius 0.75.

// colour kernel at [0..), intensity kernel at [128..); the adaptive (MITCHELL_SINEPOW) kernels
// follow at [256..) low-density, [384..) smoothing, [512..) detail
export const FILT_FLOATS = 640;
export const FILT_ADAPT_LOW = 256;
export const FILT_ADAPT_SMOOTH = 384;
export const FILT_ADAPT_DETAIL = 512;
export const FILT_INTENSITY_OFFSET = 128;
const MAX_N = 11;

/** JWildfire FilterKernelType names (render/filter/*FilterKernel.java); MITCHELL_SINEPOW is JWildfire's adaptive mode, treated as MITCHELL_SMOOTH. */
export const FILTER_KERNELS = ['MITCHELL_SMOOTH', 'MITCHELL', 'GAUSSIAN', 'SINEPOW5', 'SINEPOW10', 'SINEPOW15', 'BSPLINE', 'BELL', 'BLACKMAN', 'BOX', 'CATROM', 'HAMMING', 'HANNING', 'HERMITE', 'LANCZOS2', 'LANCZOS3', 'QUADRATIC', 'TRIANGLE', 'MITCHELL_SINEPOW'] as const;
export type FilterKernel = typeof FILTER_KERNELS[number];

/** Old model values / loose input → a JWildfire kernel name. */
export function normFilterKernel(v: unknown): FilterKernel {
  if (typeof v !== 'string') return 'MITCHELL_SMOOTH';
  if (v === 'mitchell') return 'MITCHELL_SMOOTH';
  if (v === 'gaussian') return 'GAUSSIAN';
  const u = v.toUpperCase();
  return (FILTER_KERNELS as readonly string[]).includes(u) ? (u as FilterKernel) : 'MITCHELL_SMOOTH';
}

const sinc = (x: number) => { x *= Math.PI; return x !== 0 ? Math.sin(x) / x : 1; };
function mitchell(t: number, b: number, c: number): number {
  const tt = t * t;
  if (t < 0) t = -t;
  if (t < 1) return ((12 - 9 * b - 6 * c) * (t * tt) + (-18 + 12 * b + 6 * c) * tt + (6 - 2 * b)) / 6;
  if (t < 2) return ((-b - 6 * c) * (t * tt) + (6 * b + 30 * c) * tt + (-12 * b - 48 * c) * t + (8 * b + 24 * c)) / 6;
  return 0;
}
function sinePow(x: number, p: number): number {
  const r = Math.acos(2 * Math.exp(p * Math.log10(x)) * 2 - 1) / Math.PI; // JWildfire SinePowFilterKernel (log10 of x ≤ 0 → NaN/−∞ → handled)
  return Number.isFinite(r) ? r : 0;
}

/** FilterKernel.getFilterCoeff for every JWildfire kernel (t may be negative for the asymmetric ones). */
export function kernelCoeff(kernel: FilterKernel, t: number): number {
  switch (kernel) {
    case 'GAUSSIAN': return Math.exp(-2 * t * t) * Math.sqrt(2 / Math.PI);
    case 'MITCHELL': return mitchell(t, 1 / 3, 1 / 3);
    case 'MITCHELL_SMOOTH': case 'MITCHELL_SINEPOW': return mitchell(t, 0.42, 0.29);
    case 'SINEPOW5': return sinePow(t, 5);
    case 'SINEPOW10': return sinePow(t, 10);
    case 'SINEPOW15': return sinePow(t, 15);
    case 'BSPLINE': { if (t < 0) t = -t; if (t < 1) { const tt = t * t; return 0.5 * tt * t - tt + 2 / 3; } if (t < 2) { t = 2 - t; return (t * t * t) / 6; } return 0; }
    case 'BELL': { if (t < 0) t = -t; if (t < 0.5) return 0.75 - t * t; if (t < 1.5) { t -= 1.5; return 0.5 * t * t; } return 0; }
    case 'BLACKMAN': return sinc(t) * (0.42 + 0.5 * Math.cos(Math.PI * t) + 0.08 * Math.cos(2 * Math.PI * t));
    case 'BOX': return t > -0.5 && t <= 0.5 ? 1 : 0;
    case 'CATROM': {
      const x = t;
      if (x < -2) return 0;
      if (x < -1) return 0.5 * (4 + x * (8 + x * (5 + x)));
      if (x < 0) return 0.5 * (2 + x * x * (-5 - 3 * x));
      if (x < 1) return 0.5 * (2 + x * x * (-5 + 3 * x));
      if (x < 2) return 0.5 * (4 + x * (-8 + x * (5 - x)));
      return 0;
    }
    case 'HAMMING': return sinc(t) * (0.54 + 0.46 * Math.cos(Math.PI * t));
    case 'HANNING': return sinc(t) * (0.5 + 0.5 * Math.cos(Math.PI * t));
    case 'HERMITE': { if (t < 0) t = -t; return t < 1 ? (2 * t - 3) * t * t + 1 : 0; }
    case 'LANCZOS2': { if (t < 0) t = -t; return t < 2 ? sinc(t / 2) * (sinc(t) * sinc(t / 2)) : 0; }
    case 'LANCZOS3': { if (t < 0) t = -t; return t < 3 ? sinc(t / 3) * (sinc(t) * sinc(t / 3)) : 0; }
    case 'QUADRATIC': { const x = t; if (x < -1.5) return 0; if (x < -0.5) return 0.5 * (x + 1.5) * (x + 1.5); if (x < 0.5) return 0.75 - x * x; if (x < 1.5) return 0.5 * (x - 1.5) * (x - 1.5); return 0; }
    case 'TRIANGLE': { if (t < 0) t = -t; return t < 1 ? 1 - t : 0; }
  }
}

/** FilterKernel.getSpatialSupport */
export function kernelSupport(kernel: FilterKernel): number {
  switch (kernel) {
    case 'GAUSSIAN': case 'BELL': case 'QUADRATIC': return 1.5;
    case 'MITCHELL': case 'MITCHELL_SMOOTH': case 'MITCHELL_SINEPOW': case 'BSPLINE': case 'CATROM': case 'LANCZOS2': return 2;
    case 'LANCZOS3': return 3;
    case 'BOX': return 0.5;
    default: return 1; // SINEPOW*, BLACKMAN, HAMMING, HANNING, HERMITE, TRIANGLE
  }
}

/** FilterKernelType.isSharpening: these kernels filter the colours only; the intensity gets a gaussian 0.75 (LogDensityFilter). */
export function kernelSharpening(kernel: FilterKernel): boolean {
  return kernel === 'MITCHELL' || kernel === 'MITCHELL_SMOOTH' || kernel === 'MITCHELL_SINEPOW' || kernel === 'BLACKMAN' || kernel === 'CATROM' || kernel === 'HAMMING' || kernel === 'HANNING' || kernel === 'LANCZOS2' || kernel === 'LANCZOS3';
}

/** Kernel size for a radius (0 = off). Mirrors FilterKernel.getFilterSize with oversample 1. */
export function kernelSize(radius: number, kernel: FilterKernel): number {
  if (radius < 1e-6) return 0;
  const fw = Math.floor(2 * kernelSupport(kernel) * radius);
  if (fw <= 0) return 0;
  let n = fw + 1;
  if (n % 2 === 0) n++;
  return Math.min(n, MAX_N);
}

/** Row-major N×N normalised weights (empty when N = 0). */
export function kernelWeights(radius: number, kernel: FilterKernel): { n: number; w: Float32Array } {
  const n = kernelSize(radius, kernel);
  if (!n) return { n: 0, w: new Float32Array(0) };
  const support = kernelSupport(kernel);
  const fw = Math.floor(2 * support * radius);
  const adjust = (support * n) / fw;
  const w = new Float32Array(n * n);
  let sum = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const ii = ((2 * i + 1) / n - 1) * adjust;
      const jj = ((2 * j + 1) / n - 1) * adjust;
      const v = kernelCoeff(kernel, Math.sqrt(ii * ii + jj * jj));
      w[j * n + i] = v;
      sum += v;
    }
  }
  if (sum > 0) for (let k = 0; k < n * n; k++) w[k] /= sum;
  return { n, w };
}

/** JWildfire FilterHolder kernel in RASTER cells for an oversampled raster (solid rendering applies it
 *  cell by cell): N = int(2·os·support·r)+1 made odd, weights normalised to sum os² (LogDensityFilter
 *  divides by os² again). n = 0 when the radius is 0. */
export function solidFilterWeights(radius: number, kernel: FilterKernel, os: number, maxN = 25): { n: number; w: Float32Array<ArrayBuffer> } {
  if (radius < 1e-6) return { n: 0, w: new Float32Array(new ArrayBuffer(0)) };
  const support = kernelSupport(kernel);
  const fw = Math.floor(2 * os * support * radius);
  let n = fw + 1;
  if (n % 2 === 0) n++;
  n = Math.min(n, maxN);
  const adjust = fw > 0 ? (support * n) / fw : 0;
  const w = new Float32Array(new ArrayBuffer(n * n * 4));
  let sum = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const ii = ((2 * i + 1) / n - 1) * adjust;
      const jj = ((2 * j + 1) / n - 1) * adjust;
      const v = kernelCoeff(kernel, Math.sqrt(ii * ii + jj * jj));
      w[i * n + j] = v;
      sum += v;
    }
  }
  if (sum > 0) for (let k = 0; k < n * n; k++) w[k] = (w[k] / sum) * os * os;
  return { n, w };
}

/** JWildfire's adaptive kernel (MITCHELL_SINEPOW) picks a kernel per pixel: SINEPOW10 at 1.5×
 *  the radius where the density is below `filter_low_density`, SINEPOW10 at 1× where the Scharr
 *  edge response is below `filter_sharpness`, and the Mitchell-smooth primary at 0.75× on detail.
 *  The radius they scale is JWildfire's filterRadiusA = filter + 0.25; below 0.5 it stays primary. */
export function isAdaptiveKernel(kernel: FilterKernel): boolean {
  return kernel === 'MITCHELL_SINEPOW';
}

/** Both kernels packed for the tonemap's `sfilt` buffer (plus the adaptive trio when in use). */
export function buildSpatialFilters(radius: number, kernel: FilterKernel): {
  weights: Float32Array<ArrayBuffer>; nc: number; ni: number; key: string;
  adaptive: boolean; nLow: number; nSmooth: number; nDetail: number;
} {
  const weights = new Float32Array(new ArrayBuffer(FILT_FLOATS * 4));
  const c = kernelWeights(radius, kernel);
  const i = kernelSharpening(kernel) ? kernelWeights(0.75, 'GAUSSIAN') : c;
  weights.set(c.w, 0);
  weights.set(i.w, FILT_INTENSITY_OFFSET);
  const rA = radius + 0.25;
  const adaptive = isAdaptiveKernel(kernel) && rA >= 0.5;
  let nLow = 0, nSmooth = 0, nDetail = 0;
  if (adaptive) {
    const low = kernelWeights(1.5 * rA, 'SINEPOW10');
    const smooth = kernelWeights(1.0 * rA, 'SINEPOW10');
    const detail = kernelWeights(0.75 * rA, 'MITCHELL_SMOOTH');
    weights.set(low.w, FILT_ADAPT_LOW);
    weights.set(smooth.w, FILT_ADAPT_SMOOTH);
    weights.set(detail.w, FILT_ADAPT_DETAIL);
    nLow = low.n; nSmooth = smooth.n; nDetail = detail.n;
  }
  return {
    weights, nc: c.n, ni: radius < 1e-6 ? 0 : i.n, key: `${kernel}:${radius.toFixed(4)}`,
    adaptive, nLow, nSmooth, nDetail,
  };
}

/** The 1-D factor of a JWildfire gaussian FilterHolder kernel (os 1): the 2-D kernel exp(−2(ii²+jj²)) normalised
 *  over its N×N window equals the outer product of these normalised 1-D weights. n = 0 when the radius is 0. */
export function gaussianFilter1D(radius: number, maxN: number): { n: number; w: Float32Array<ArrayBuffer> } {
  if (radius < 1e-6) return { n: 0, w: new Float32Array(new ArrayBuffer(0)) };
  const support = kernelSupport('GAUSSIAN');
  const fw = Math.floor(2 * support * radius);
  let n = fw + 1;
  if (n % 2 === 0) n++;
  n = Math.min(n, maxN);
  const adjust = fw > 0 ? (support * n) / fw : 0;
  const w = new Float32Array(new ArrayBuffer(n * 4));
  let sum = 0;
  for (let i = 0; i < n; i++) { const ii = ((2 * i + 1) / n - 1) * adjust; w[i] = Math.exp(-2 * ii * ii); sum += w[i]; }
  for (let i = 0; i < n; i++) w[i] /= sum;
  return { n, w };
}
