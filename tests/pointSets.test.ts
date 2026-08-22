import { describe, it, expect } from 'vitest';
import { pointSetFor, pointSetKeyFor, Marsaglia, PSET_STRIDE, Turtle } from '../src/core/pointSets';
import { defaultFlame } from '../src/core/flame';
import { defaultParams, VARIATIONS } from '../src/core/variations';
import { compileFlame } from '../src/gpu/codegen';

const pal = Array.from({ length: 256 }, (_, i) => [i / 255, 0.5, 1 - i / 255] as [number, number, number]);
const setOf = (name: string, params: Record<string, number>) => pointSetFor(pointSetKeyFor({ name, params })!);

describe('point sets', () => {
  it('dragon_js: 2^level unit segments from a 90° turtle', () => {
    const s = setOf('dragon_js', { level: 5, line_thickness: 0.5 });
    expect(s.count).toBe(32);
    expect(s.data.length).toBe(32 * PSET_STRIDE);
    expect(s.data[0]).toBe(1); // kind line
    expect(s.data[6]).toBeCloseTo(0.005, 9); // thickness = 0.5 / 100
    const t = new Turtle(); t.goForward(1); t.turnLeft(90); t.goForward(1);
    expect(t.segs.map((v) => +v.toFixed(6))).toEqual([0, 0, 1, 0, 1, 0, 1, 1]);
  });
  it('sunflower: nPoints n-gons on a Fibonacci spiral, colour = radial fraction', () => {
    const s = setOf('sunflower', { ...defaultParams('sunflower'), nPoints: 50 });
    expect(s.count).toBe(50);
    expect(s.data[0]).toBe(3); // ngon
    expect(s.data[4]).toBe(10); // sides
    expect(s.data[1]).toBeGreaterThan(0.8); // first point is near the centre: sc ≈ 1
    expect(s.data[(49) * PSET_STRIDE + 1]).toBeLessThan(0.05);
  });
  it('scrambly: a permutation of the l² cells for a seed > 50, a shift for seed ≤ 50', () => {
    const perm = Array.from(setOf('scrambly', { l: 4, seed: 77, byrows: 0, cellsize: 0.1 }).data.slice(0, 16)).map(Math.round);
    expect([...perm].sort((a, b) => a - b)).toEqual([...Array(16).keys()]);
    expect(perm).not.toEqual([...Array(16).keys()]);
    const shift = Array.from(setOf('scrambly', { l: 3, seed: 2, byrows: 0, cellsize: 0.1 }).data.slice(0, 9)).map(Math.round);
    expect(shift).toEqual([3, 4, 5, 6, 7, 8, 0, 1, 2]);
  });
  it('Marsaglia is deterministic per seed and in 0..1', () => {
    const a = new Marsaglia(666), b = new Marsaglia(666), c = new Marsaglia(667);
    const xs = Array.from({ length: 100 }, () => a.random());
    expect(Array.from({ length: 100 }, () => b.random())).toEqual(xs);
    expect(Array.from({ length: 100 }, () => c.random())).not.toEqual(xs);
    expect(xs.every((x) => x >= 0 && x <= 1)).toBe(true);
  });
  it('a flame with a point-set variation compiles with the pset binding and sampler', () => {
    const f = defaultFlame(pal);
    f.layers[0].xforms[0].variations = [{ name: 'dragon_js', weight: 1, params: defaultParams('dragon_js') }];
    const c = compileFlame(f, 1024);
    expect(c.usesPset).toBe(true);
    expect(c.wgsl).toContain('@binding(13) var<storage, read> pset');
    expect(c.wgsl).toContain('fn psetSample(');
    expect(VARIATIONS.scrambly.extra).toBe(2);
  });
});
