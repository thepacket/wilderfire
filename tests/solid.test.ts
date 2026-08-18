import { describe, it, expect } from 'vitest';
import { compileFlame } from '../src/gpu/codegen';
import { solidFilterWeights } from '../src/gpu/filters';
import { defaultSolidRender, flameSignature, normalizeFlame, usesMaterials } from '../src/core/flame';
import { GREY, preset } from './helpers';

describe('solid rendering codegen', () => {
  it('a solid flame compiles to a z-buffer kernel (no histogram writes) and flips the signature', () => {
    const f = preset('Ember');
    const plain = compileFlame(f, 1024);
    expect(plain.solid).toBe(false);
    expect(plain.wgsl).toContain('atomicAdd(&hist');
    expect(plain.wgsl).not.toContain('solidSplat');
    f.solid = defaultSolidRender(true);
    const c = compileFlame(f, 1024);
    expect(c.solid).toBe(true);
    expect(c.usesMat).toBe(false);
    expect(c.wgsl).toContain('solidSplat(');
    expect(c.wgsl).toContain('@binding(7) var<storage, read_write> zkey');
    expect(c.wgsl).not.toContain('atomicAdd(&hist');
    expect(c.wgsl).not.toContain('applyColorMods('); // JWildfire: colour modifiers are skipped in solid mode
    // shadow maps: light-space splat + one bounds sample per walker (bit 31 of the rng word)
    expect(c.wgsl).toContain('shadowSplat(dp)');
    expect(c.wgsl).toContain('var bdone = (rngs[idx].y & 0x80000000u) != 0u');
    expect(c.wgsl).toContain('@binding(10) var<storage, read_write> smaps');
    expect(plain.wgsl).not.toContain('bdone');
    expect(flameSignature(f)).toContain('~solid');
    // a transform with a material index → per-point material state
    f.layers[0].xforms[0].material = 1;
    expect(usesMaterials(f)).toBe(true);
    const cm = compileFlame(f, 1024);
    expect(cm.usesMat).toBe(true);
    expect(cm.wgsl).toContain('@binding(9) var<storage, read_write> mats');
    expect(cm.wgsl).toMatch(/mt = mt \* \(1\.0 \+ xd\[\d+u\]\) \* 0\.5/);
    expect(flameSignature(f)).toContain('~mat');
    // solid off → material state is irrelevant
    f.solid.enabled = false;
    expect(compileFlame(f, 1024).usesMat).toBe(false);
  });

  it('writeData stores material + speed in the block header', () => {
    const f = preset('Ember');
    f.solid = defaultSolidRender(true);
    f.layers[0].xforms[1].material = 0.3;
    f.layers[0].xforms[1].materialSpeed = 0.5;
    const c = compileFlame(f, 1024);
    const out = new Float32Array(c.dataSize);
    c.writeData(f, out);
    // block header slot 64/65 of the second xform: find the pair (0.3, 0.5) somewhere in xd
    let found = false;
    for (let i = 0; i + 1 < out.length; i++) if (Math.abs(out[i] - 0.3) < 1e-6 && Math.abs(out[i + 1] - 0.5) < 1e-6) found = true;
    expect(found).toBe(true);
  });

  it('normalizeFlame keeps a solid block and clamps its lists', () => {
    const f = normalizeFlame({ ...preset('Ember'), solid: { enabled: true, lights: Array.from({ length: 6 }, () => ({ altitude: 10 })), materials: [] } }, GREY);
    expect(f.solid?.enabled).toBe(true);
    expect(f.solid?.lights.length).toBe(4);
    expect(f.solid?.lights[0]).toMatchObject({ altitude: 10, azimuth: -22, intensity: 0.8 });
    expect(f.solid?.materials).toEqual([]);
  });
});

describe('solid spatial filter (JWildfire FilterHolder in raster cells)', () => {
  it('sizes follow int(2·os·support·r)+1 made odd, weights sum to os²', () => {
    expect(solidFilterWeights(0, 'gaussian', 1).n).toBe(0);
    const g = solidFilterWeights(0.5, 'gaussian', 1); // fw = int(1.5) = 1 → 3
    expect(g.n).toBe(3);
    expect(g.w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    const g2 = solidFilterWeights(0.5, 'gaussian', 2); // fw = 3 → 4 → 5
    expect(g2.n).toBe(5);
    expect(g2.w.reduce((a, b) => a + b, 0)).toBeCloseTo(4, 4);
    const m = solidFilterWeights(1, 'mitchell', 2); // fw = 8 → 9
    expect(m.n).toBe(9);
    // symmetric kernel, centre weight is the largest
    const c = m.w[4 * 9 + 4];
    expect(c).toBeGreaterThan(m.w[0]);
    expect(m.w[4 * 9 + 0]).toBeCloseTo(m.w[4 * 9 + 8], 6);
  });
});

describe('AO smoothing kernel', () => {
  it('the 1-D gaussian factor reproduces JWildfire\'s N×N FilterHolder kernel exactly (separable blur)', async () => {
    const { gaussianFilter1D } = await import('../src/gpu/filters');
    for (const r of [0.5, 2.17, 7.3]) {
      const k2 = solidFilterWeights(r, 'gaussian', 1, 45);
      const k1 = gaussianFilter1D(r, 45);
      expect(k1.n).toBe(k2.n);
      for (let i = 0; i < k1.n; i++) for (let j = 0; j < k1.n; j++) expect(k1.w[i] * k1.w[j]).toBeCloseTo(k2.w[i * k1.n + j], 6);
    }
  });
});
