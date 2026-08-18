// Arbitrary-precision numbers for deep zooms: BigInt fixed point (value = m · 2^-P) with decimal-string
// storage in the model (`centerXHi`), plus the reference-orbit iteration z ← z^p + c that perturbation
// rendering needs. Only what the escape layers use — add, multiply, decimal I/O, f64 conversion.

export interface Fixed { m: bigint; P: number }

/** bits of fraction needed to place a point to ~1e-3 pixel at zoom `zoom` (ppu ≈ 400·zoom) */
export function bitsForZoom(zoom: number): number {
  return Math.max(64, (Math.ceil(Math.log2(Math.max(zoom, 1)) + 24 + 32) + 7) & ~7);
}

const TEN = 10n;

/** decimal string → fixed point (P fraction bits). Accepts "-1.234e-5", ".5", "3". */
export function fromDecimal(s: string, P: number): Fixed {
  const t = s.trim();
  const m = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(t);
  if (!m || (!m[2] && !m[3])) throw new Error(`not a number: "${s}"`);
  const neg = m[1] === '-';
  const intPart = m[2] || '0', frac = m[3] || '';
  const exp10 = m[4] ? parseInt(m[4]) : 0;
  // value = (intPart·10^len(frac) + frac) · 10^(exp10 − len(frac))
  let digits = BigInt(intPart + frac);
  let e10 = exp10 - frac.length;
  // to fixed point: v · 2^P = digits · 10^e10 · 2^P
  let v: bigint;
  if (e10 >= 0) v = digits * TEN ** BigInt(e10) << BigInt(P);
  else {
    // divide by 10^-e10 with rounding: (digits · 2^P) / 10^k
    const den = TEN ** BigInt(-e10);
    const num = digits << BigInt(P);
    v = (num + den / 2n) / den;
  }
  return { m: neg ? -v : v, P };
}

/** fixed point → decimal string with `digits` fraction digits (rounded) */
export function toDecimal(x: Fixed, digits: number): string {
  const neg = x.m < 0n;
  const a = neg ? -x.m : x.m;
  const scaled = (a * TEN ** BigInt(digits) + (1n << BigInt(x.P - 1))) >> BigInt(x.P);
  const s = scaled.toString().padStart(digits + 1, '0');
  const ip = s.slice(0, s.length - digits), fp = s.slice(s.length - digits).replace(/0+$/, '');
  return (neg ? '-' : '') + ip + (fp ? '.' + fp : '');
}

export function fromNumber(v: number, P: number): Fixed {
  // exact for f64: split into mantissa/exponent
  if (!isFinite(v)) throw new Error('not finite');
  if (v === 0) return { m: 0n, P };
  const [mant, exp] = frexp(v); // v = mant · 2^exp, 0.5 ≤ |mant| < 1
  const mi = BigInt(Math.round(mant * 2 ** 53));
  const shift = P + exp - 53;
  return { m: shift >= 0 ? mi << BigInt(shift) : mi >> BigInt(-shift), P };
}

export function toNumber(x: Fixed): number {
  const neg = x.m < 0n; const a = neg ? -x.m : x.m;
  const bits = a.toString(2).length;
  // keep 60 significant bits
  const drop = Math.max(0, bits - 60);
  const top = Number(a >> BigInt(drop));
  const v = top * 2 ** (drop - x.P);
  return neg ? -v : v;
}

function frexp(v: number): [number, number] {
  if (v === 0) return [0, 0];
  const e = Math.floor(Math.log2(Math.abs(v))) + 1;
  let m = v / 2 ** e;
  // guard rounding at powers of two
  if (Math.abs(m) >= 1) return [m / 2, e + 1];
  if (Math.abs(m) < 0.5) return [m * 2, e - 1];
  return [m, e];
}

export const fxAdd = (a: Fixed, b: Fixed): Fixed => ({ m: a.m + rescale(b, a.P).m, P: a.P });
export const fxSub = (a: Fixed, b: Fixed): Fixed => ({ m: a.m - rescale(b, a.P).m, P: a.P });
export const fxMul = (a: Fixed, b: Fixed): Fixed => ({ m: (a.m * rescale(b, a.P).m) >> BigInt(a.P), P: a.P });
export function rescale(a: Fixed, P: number): Fixed {
  if (a.P === P) return a;
  return { m: P > a.P ? a.m << BigInt(P - a.P) : a.m >> BigInt(a.P - P), P };
}

/** the exact centre of a deep view: decimal strings when set, else the f64 fields */
export function centreFixed(cx: number, cy: number, hi: [string, string] | undefined, P: number): [Fixed, Fixed] {
  if (hi) { try { return [fromDecimal(hi[0], P), fromDecimal(hi[1], P)]; } catch { /* fall through */ } }
  return [fromNumber(cx, P), fromNumber(cy, P)];
}

/** Reference orbit z_{n+1} = z_n^p + c (integer p ≥ 2) in fixed point, as float pairs (hi, lo) per component,
 *  up to `maxIter` steps or escape (|z| > bailout). Returns the orbit (x_hi, x_lo, y_hi, y_lo per step, step 0 = z₀)
 *  and how many steps it has. */
export function referenceOrbit(z0: [Fixed, Fixed], c: [Fixed, Fixed], power: number, maxIter: number, bailout: number): { data: Float32Array; n: number; escaped: boolean } {
  const P = z0[0].P;
  const PB = BigInt(P);
  const out = new Float32Array((maxIter + 1) * 4);
  let x = z0[0].m, y = z0[1].m;
  const cx = rescale(c[0], P).m, cy = rescale(c[1], P).m;
  const bail2 = bailout * bailout;
  const put = (i: number, xv: bigint, yv: bigint) => {
    const xf = toNumber({ m: xv, P }), yf = toNumber({ m: yv, P });
    const xh = Math.fround(xf), yh = Math.fround(yf);
    out[i * 4] = xh; out[i * 4 + 1] = Math.fround(xf - xh); out[i * 4 + 2] = yh; out[i * 4 + 3] = Math.fround(yf - yh);
    return xf * xf + yf * yf;
  };
  put(0, x, y);
  let n = 0;
  let escaped = false;
  const p = Math.max(2, Math.round(power));
  for (let i = 1; i <= maxIter; i++) {
    // z^p by repeated multiplication (p small)
    let rx = x, ry = y;
    for (let k = 1; k < p; k++) {
      const nx = (rx * x - ry * y) >> PB;
      const ny = (rx * y + ry * x) >> PB;
      rx = nx; ry = ny;
    }
    x = rx + cx; y = ry + cy;
    n = i;
    const r2 = put(i, x, y);
    if (r2 > bail2 * 4 || !isFinite(r2)) { escaped = true; break; }
  }
  return { data: out.subarray(0, (n + 1) * 4), n, escaped };
}
