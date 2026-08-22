import { describe, it, expect } from 'vitest';
import { pointSetFor, pointSetKeyFor, Marsaglia, JavaRandom, PSET_STRIDE, Turtle } from '../src/core/pointSets';
import { defaultFlame } from '../src/core/flame';
import { defaultParams, VARIATIONS } from '../src/core/variations';
import { compileFlame } from '../src/gpu/codegen';
import { kleinGenerators } from '../src/core/variations';

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
  it('dla_wf: a connected aggregate around the centre, reproducible per seed', () => {
    const a = setOf('dla_wf', { buffer_size: 200, max_iter: 1500, seed: 666, scale: 10, jitter: 0.01 });
    const b = setOf('dla_wf', { buffer_size: 200, max_iter: 1500, seed: 666, scale: 10, jitter: 0.01 });
    expect(a.count).toBeGreaterThan(1000); expect(a.count).toBeLessThanOrEqual(1501); // deposits onto occupied cells add no point (as in JWildfire)
    expect(a.data).toEqual(b.data);
    expect(a.data[0]).toBe(0); // points
    let maxR = 0; for (let i = 0; i < a.count; i++) maxR = Math.max(maxR, Math.hypot(a.data[i * PSET_STRIDE + 2], a.data[i * PSET_STRIDE + 3]));
    expect(maxR).toBeLessThan(5.5); // inside the scaled buffer
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

describe('turtle family', () => {
  it('koch: 4^level segments of step 0.5; hilbert: 4^level − 1 unit segments; htree: 3·(4^level − 1)/3 H segments; tree: 3^(level+1)… branches', () => {
    expect(setOf('koch_js', { level: 3, show_lines: 1, line_thickness: 0.5, show_points: 0, point_thickness: 3 }).count).toBe(64);
    expect(setOf('hilbert_js', { level: 3, show_lines: 1, line_thickness: 0.5, show_points: 0, point_thickness: 3 }).count).toBe(63);
    expect(setOf('htree_js', { level: 2, size: 2, show_lines: 1, line_thickness: 0.5, show_points: 0, point_thickness: 3 }).count).toBe(3 * 5);
    expect(setOf('tree_js', { level: 2, show_lines: 1, line_thickness: 0.5, bend_angle: 0, branch_angle: 30, branch_ratio: 0.5, show_points: 0, point_thickness: 3 }).count).toBe(1 + 3 + 9);
  });
  it('brownian: 2^level segments, reproducible for a seed, a path from the origin back to the origin', () => {
    const a = setOf('brownian_js', { level: 6, variation: 3, seed: 42, line_thickness: 0.5, show_lines: 1, show_points: 0, point_thickness: 3 });
    const b = setOf('brownian_js', { level: 6, variation: 3, seed: 42, line_thickness: 0.5, show_lines: 1, show_points: 0, point_thickness: 3 });
    expect(a.count).toBe(64); expect(a.data).toEqual(b.data);
    expect([a.data[2], a.data[3]]).toEqual([0, 0]);
    const last = (63) * PSET_STRIDE; expect([a.data[last + 4], a.data[last + 5]]).toEqual([0, 0]);
    expect(a.data.slice(0, 64 * PSET_STRIDE).some((v) => Math.abs(v) > 0.1)).toBe(true);
  });
  it('java.util.Random matches Java: new Random(42).nextDouble() = 0.7275636800328681', () => {
    const r = new JavaRandom(42);
    expect(r.nextDouble()).toBeCloseTo(0.7275636800328681, 15);
    expect(r.nextDouble()).toBeCloseTo(0.6832234717598454, 15);
  });
});

describe('klein_group generators', () => {
  it('Grandma with traces 2, 2 gives finite matrices with det 1 and inverses [d, −b, −c, a]', () => {
    const g = kleinGenerators({ a_re: 2, a_im: 0, b_re: 2, b_im: 0, recipe: 0, avoid_reversal: 0 });
    expect(g.length).toBe(32);
    expect(g.every(Number.isFinite)).toBe(true);
    const det = (m: number[]) => [m[0] * m[6] - m[1] * m[7] - (m[2] * m[4] - m[3] * m[5]), m[0] * m[7] + m[1] * m[6] - (m[2] * m[5] + m[3] * m[4])];
    const a = g.slice(0, 8), A = g.slice(8, 16);
    expect(det(a)[0]).toBeCloseTo(1, 6); expect(det(a)[1]).toBeCloseTo(0, 6);
    expect(A).toEqual([a[6], a[7], -a[2], -a[3], -a[4], -a[5], a[0], a[1]]);
  });
  it('Maskit μ = 1.9+0.05i: a = [−iμ, −i; −i, 0], b = [1, 2; 0, 1]', () => {
    const g = kleinGenerators({ a_re: 1.9, a_im: 0.05, b_re: 2, b_im: 0, recipe: 1, avoid_reversal: 1 });
    expect(g.slice(0, 8).map((v) => +v.toFixed(6))).toEqual([0.05, -1.9, 0, -1, 0, -1, 0, 0]);
    expect(g.slice(16, 24)).toEqual([1, 0, 2, 0, 0, 0, 1, 0]);
  });
  it("matches KleinGroupFunc.init() for every recipe (values probed from JWildfire: mat_a, mat_inv_a, mat_b, mat_inv_b as re, im)", () => {
    const cases: [Record<string, number>, number[]][] = [
    [{ recipe: 0, a_re: 2, a_im: 0, b_re: 2, b_im: 0 }, [1, 0, 0, 0, 0, -2, 1, 0, 1, 0, 0, 0, 0, 2, 1, 0, 1, -1, 1, 0, 1, 0, 1, 1, 1, 1, -1, 0, -1, 0, 1, -1]],
    [{ recipe: 1, a_re: 2, a_im: 1, b_re: 2, b_im: 1 }, [1, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 1, -2, 1, 0, 2, 0, 0, 0, 1, 0, 1, 0, -2, 0, 0, 0, 1, 0]],
    [{ recipe: 2, a_re: 2, a_im: 0, b_re: 1, b_im: 1 }, [2, -1, 6.12323399573676e-17, 1, 2, 0, 0, 1, 0, 1, -6.12323399573676e-17, -1, -2, 0, 2, -1, 0, 0, 0.5, -0.5, -1, -1, 1, 1, 1, 1, -0.5, 0.5, 1, 1, 0, 0]],
    [{ recipe: 3, a_re: 1, a_im: 2.056, b_re: 0.584, b_im: 0 }, [1, 0, 0, 0, 1, 2.056, 1, 0, 1, 0, 0, 0, -1, -2.056, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 1, 0, -2, 0, 0, 0, 1, 0]],
    [{ recipe: 4, a_re: 1, a_im: 2.056, b_re: 0.584, b_im: 0 }, [1, 0, 0, 0, 1, 2.056, 1, 0, 1, 0, 0, 0, -1, -2.056, 1, 0, 1, 0, 0.584, 0, 0, 0, 1, 0, 1, 0, -0.584, 0, 0, 0, 1, 0]],
    [{ recipe: 5, a_re: 2, a_im: 1, b_re: 2, b_im: 1 }, [1, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 1, -2, 1, 0, 2, 1, 0, 0, 1, 0, 1, 0, -2, -1, 0, 0, 1, 0]],
    [{ recipe: 6, a_re: 2, a_im: 1, b_re: 2, b_im: 1 }, [1, -2, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 1, -2, 1, 0, 1.22464679907692e-16, 1, 0, 0, 1, 0, 1, 0, -1.22464679907692e-16, -1, 0, 0, 1, 0]],
    [{ recipe: 0, a_re: 2, a_im: 1, b_re: 2, b_im: 1 }, [1, 0.5, 0.418796525390441, 2.1938972023081, 0.418796525390441, 0.193897202308099, 1, 0.5, 1, 0.5, -0.418796525390441, -2.1938972023081, -0.418796525390441, -0.193897202308099, 1, 0.5, 1, -0.5, 1, 0.5, 1, 0.5, 1, 1.5, 1, 1.5, -1, -0.5, -1, -0.5, 1, -0.5]],
    [{ recipe: 2, a_re: 1.7, a_im: 0.3, b_re: 2.2, b_im: -0.5 }, [1.0154845533202, -0.204691839110991, -0.0775055067039066, 0.232731584196693, 1.7, 0.3, 0.684515446679799, 0.504691839110991, 0.684515446679799, 0.504691839110991, 0.0775055067039066, -0.232731584196693, -1.7, -0.3, 1.0154845533202, -0.204691839110991, 1.86693010284423, -1.05902986775932, -0.0245570211644232, -0.319644195432559, -2.2, 0.5, 0.333069897155773, 0.559029867759323, 0.333069897155773, 0.559029867759323, 0.0245570211644232, 0.319644195432559, 2.2, -0.5, 1.86693010284423, -1.05902986775932]],
    [{ recipe: 6, a_re: 1.5, a_im: 0.7, b_re: 0.4, b_im: 1.1 }, [0.7, -1.5, 0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, 0.7, -1.5, 1, 0, 6.12323399512612e-16, 1.1, 0, 0, 1, 0, 1, 0, -6.12323399512612e-16, -1.1, 0, 0, 1, 0]],
    ];
    for (const [P, want] of cases) {
      const got = kleinGenerators(P);
      expect(got.length).toBe(32);
      got.forEach((v, i) => expect(Math.abs(v - want[i]), `recipe ${P.recipe} slot ${i}`).toBeLessThan(1e-9));
    }
  });
  it('a flame with klein_group compiles with the derived slots and the per-thread memory', () => {
    const f = defaultFlame(pal);
    f.layers[0].xforms[0].variations = [{ name: 'klein_group', weight: 1, params: defaultParams('klein_group') }];
    const c = compileFlame(f, 1024);
    expect(c.wgsl).toContain('var<private> jwx_klein_prev');
    expect(c.wgsl).toContain('cdivk(cmulk(win, A) + Bm, cmulk(win, C) + D)');
    const xd = new Float32Array(c.dataSize); c.writeData(f, xd);
    expect(Array.from(xd).filter((v) => v !== 0).length).toBeGreaterThan(10); // the generators landed in the hidden slots
  });
});
