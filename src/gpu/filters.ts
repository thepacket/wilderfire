// JWildfire spatial filters (FilterHolder / FilterKernel), CPU side.
// FilterHolder builds an N×N kernel from a radius: N = int(2·support·r)+1, made
// odd; coefficient index ii = ((2i+1)/N − 1)·adjust with adjust = support·N/fw;
// weights are normalised to 1. Mitchell-smooth (b = 0.42, c = 0.29, support 2)
// is JWildfire's default; the flam3 gaussian has support 1.5. LogDensityFilter
// filters colours with the primary kernel and, for "sharpening" kernels
// (Mitchell), the intensity with a gaussian of radius 0.75.

export const FILT_FLOATS = 256; // colour kernel at [0..), intensity kernel at [128..)
export const FILT_INTENSITY_OFFSET = 128;
const MAX_N = 11;

export type FilterKernel = 'mitchell' | 'gaussian';

export function kernelCoeff(kernel: FilterKernel, t: number): number {
  if (kernel === 'gaussian') return Math.exp(-2 * t * t) * Math.sqrt(2 / Math.PI);
  const b = 0.42, c = 0.29, tt = t * t;
  if (t < 1) return (((12 - 9 * b - 6 * c) * (t * tt)) + ((-18 + 12 * b + 6 * c) * tt) + (6 - 2 * b)) / 6;
  if (t < 2) return (((-b - 6 * c) * (t * tt)) + ((6 * b + 30 * c) * tt) + ((-12 * b - 48 * c) * t) + (8 * b + 24 * c)) / 6;
  return 0;
}

export const kernelSupport = (kernel: FilterKernel) => (kernel === 'gaussian' ? 1.5 : 2.0);

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

/** Both kernels packed for the tonemap's `sfilt` buffer. */
export function buildSpatialFilters(radius: number, kernel: FilterKernel): { weights: Float32Array<ArrayBuffer>; nc: number; ni: number; key: string } {
  const weights = new Float32Array(new ArrayBuffer(FILT_FLOATS * 4));
  const c = kernelWeights(radius, kernel);
  const i = kernel === 'mitchell' ? kernelWeights(0.75, 'gaussian') : c;
  weights.set(c.w, 0);
  weights.set(i.w, FILT_INTENSITY_OFFSET);
  return { weights, nc: c.n, ni: radius < 1e-6 ? 0 : i.n, key: `${kernel}:${radius.toFixed(4)}` };
}

/** The 1-D factor of a JWildfire gaussian FilterHolder kernel (os 1): the 2-D kernel exp(−2(ii²+jj²)) normalised
 *  over its N×N window equals the outer product of these normalised 1-D weights. n = 0 when the radius is 0. */
export function gaussianFilter1D(radius: number, maxN: number): { n: number; w: Float32Array<ArrayBuffer> } {
  if (radius < 1e-6) return { n: 0, w: new Float32Array(new ArrayBuffer(0)) };
  const support = kernelSupport('gaussian');
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
