import { describe, it, expect } from 'vitest';
import { kernelSize, kernelWeights, buildSpatialFilters, kernelCoeff, FILT_INTENSITY_OFFSET } from '../src/gpu/filters';
import { TONEMAP_WGSL } from '../src/gpu/codegen';

/** JWildfire LogScaleCalculator, as implemented by TONEMAP_WGSL's logScaled(). */
function jwfIntensity(hits: number, quality: number, area: number, brightness: number, contrast = 1, ldb = 0.24) {
  const k1 = 2 * contrast * brightness;
  const k2 = 1 / (contrast * area * quality);
  const glow = (ldb * k2 * area) / (hits + 1);
  return k1 * Math.log10(1 + hits * k2) + glow;
}

describe('JWildfire log scale (reference numbers from JWildfire itself)', () => {
  // KCheck.java on synth.flame: ppu 128 on 512² → area 16, q 300, brightness 4 → k1 = 8, k2 = 1/4800
  it('reproduces LogScaleCalculator k1/k2 and intensities', () => {
    const I = (hits: number) => jwfIntensity(hits, 300, 16, 4);
    expect(I(771)).toBeCloseTo(0.5175364, 5);
    expect(I(1543)).toBeCloseTo(0.9684283, 5);
    expect(I(5000)).toBeCloseTo(2.4798789, 5);
    expect(I(1)).toBeCloseTo(0.0011237, 6);
  });
  it('is zoom-invariant: hits scale with ppu², area with 1/ppu²', () => {
    // zooming in ×2: 4× more pixels per world unit → hits/pixel ÷4, area ÷4 → same intensity
    expect(jwfIntensity(1000, 300, 16, 4, 1, 0)).toBeCloseTo(jwfIntensity(250, 300, 4, 4, 1, 0), 9); // (the tiny glow term is not invariant)
  });
  it('the shader carries the same formula and JWildfire palette scale', () => {
    expect(TONEMAP_WGSL).toContain('k1 * 0.43429448 * log(1.0 + d / (contrast * T.jw.x))'); // log10 via ln
    expect(TONEMAP_WGSL).toMatch(/ccol \+ T\.bg\.rgb \* \(1\.0 - alpha\)/);              // colour + bg·(1−alpha), not mix()
    expect(TONEMAP_WGSL).toContain('erf1(');                                              // DeCalculator acceptance test
  });
});

describe('spatial filter kernels (FilterHolder)', () => {
  it('sizes match JWildfire: mitchell 0.5 → 3, 0.75 → 5; gaussian 0.75 → 3; 0 → off', () => {
    expect(kernelSize(0.5, 'mitchell')).toBe(3);
    expect(kernelSize(0.75, 'mitchell')).toBe(5);
    expect(kernelSize(0.75, 'gaussian')).toBe(3);
    expect(kernelSize(0, 'mitchell')).toBe(0);
    expect(kernelSize(0.2, 'mitchell')).toBe(0); // fw = 0
  });
  it('weights are normalised, symmetric and centre-peaked', () => {
    for (const k of ['mitchell', 'gaussian'] as const) {
      const { n, w } = kernelWeights(0.75, k);
      let sum = 0; for (const v of w) sum += v;
      expect(sum).toBeCloseTo(1, 6);
      const c = (n - 1) / 2;
      expect(w[c * n + c]).toBeGreaterThan(w[0]);
      expect(w[0]).toBeCloseTo(w[n - 1], 6);
      expect(w[0]).toBeCloseTo(w[(n - 1) * n], 6);
    }
  });
  it('mitchell-smooth coefficient: 1 - 2b/6·… at 0, zero beyond support', () => {
    expect(kernelCoeff('mitchell', 0)).toBeCloseTo((6 - 2 * 0.42) / 6, 9);
    expect(kernelCoeff('mitchell', 2)).toBe(0);
    expect(kernelCoeff('gaussian', 0)).toBeCloseTo(Math.sqrt(2 / Math.PI), 9);
  });
  it('sharpening kernels get a gaussian-0.75 intensity filter; gaussian uses itself; radius 0 → nothing', () => {
    const m = buildSpatialFilters(0.75, 'mitchell');
    expect([m.nc, m.ni]).toEqual([5, 3]);
    expect(m.weights[FILT_INTENSITY_OFFSET + 4]).toBeGreaterThan(0); // 3×3 centre present
    const g = buildSpatialFilters(0.75, 'gaussian');
    expect([g.nc, g.ni]).toEqual([3, 3]);
    expect(buildSpatialFilters(0, 'mitchell')).toMatchObject({ nc: 0, ni: 0 });
  });
});
