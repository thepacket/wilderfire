import { describe, it, expect } from 'vitest';
import { compileFormula, FV } from '../src/core/formula';
import { buildSAttractorMesh, sattractorFormulas } from '../src/core/sattractor';
import { SATTRACTOR_PRESETS } from '../src/core/sattractorPresets';
import { meshKeyFor } from '../src/core/meshes';

const vars = (o: Partial<Record<keyof typeof FV, number>>) => { const v = new Float64Array(12); for (const [k, x] of Object.entries(o)) v[FV[k as keyof typeof FV]] = x!; return v; };

describe('formula evaluator', () => {
  it('arithmetic, precedence, unary minus, functions, ternary and comparisons', () => {
    expect(compileFormula('1 + 2 * 3')(vars({}))).toBe(7);
    expect(compileFormula('-(x - 1) / 2')(vars({ x: 5 }))).toBe(-2);
    expect(compileFormula('param_a*x + y - x*z')(vars({ x: 2, y: 3, z: 4, param_a: 1.5 }))).toBe(3 + 3 - 8);
    expect(compileFormula('-param_b*z + param_b*((x>0)?1:0)*x*x')(vars({ x: 2, z: 1, param_b: 0.5 }))).toBe(-0.5 + 0.5 * 4);
    expect(compileFormula('-param_b*z + param_b*((x>0)?1:0)*x*x')(vars({ x: -2, z: 1, param_b: 0.5 }))).toBe(-0.5);
    expect(compileFormula('fabs(-3) + sin(pi/2) + pow(2, 3) + fmod(7, 3)')(vars({}))).toBeCloseTo(3 + 1 + 8 + 1, 12);
    expect(compileFormula('x*(4-y) + param_a*z')(vars({ x: 1, y: 2, z: 3, param_a: 0.3 }))).toBeCloseTo(2 + 0.9, 12);
  });
  it('rejects anything outside the subset', () => {
    expect(() => compileFormula('System.exit(0)')).toThrow();
    expect(() => compileFormula('foo(1)')).toThrow();
    expect(() => compileFormula('x +')).toThrow();
  });
  it('every preset formula compiles', () => {
    expect(SATTRACTOR_PRESETS.length).toBe(21);
    for (const p of SATTRACTOR_PRESETS) for (const f of [p.x, p.y, p.z]) expect(() => compileFormula(f), `${p.name}: ${f}`).not.toThrow();
  });
});

describe('sattractor3D mesh', () => {
  it('builds the tube JWildfire builds: steps·1000 rings of `facets` vertices plus two cap vertices', () => {
    const p = SATTRACTOR_PRESETS[0]; // Aizawa
    const m = buildSAttractorMesh({ x: p.x, y: p.y, z: p.z, steps: 1, radius: p.radius, stepTime: p.stepTime, facets: 3, start: p.start, warmup: 1000, params: p.params });
    const count = 1000, sc = 3;
    expect(m.pos.length / 3).toBe(count * sc + 2);
    expect(m.idx.length / 3).toBe((count - 1) * sc * 2 + sc * 2);
    expect(Array.from(m.pos).every(Number.isFinite)).toBe(true);
    expect(Math.max(...Array.from(m.idx))).toBe(count * sc + 1);
    // the Aizawa attractor lives inside a unit-ish ball
    let maxR = 0; for (let i = 0; i < m.pos.length; i += 3) maxR = Math.max(maxR, Math.hypot(m.pos[i], m.pos[i + 1], m.pos[i + 2]));
    expect(maxR).toBeGreaterThan(0.3); expect(maxR).toBeLessThan(3);
  });
  it('resources override the preset formulas; empty resources fall back to presetId', () => {
    expect(sattractorFormulas(2)).toEqual({ x: SATTRACTOR_PRESETS[2].x, y: SATTRACTOR_PRESETS[2].y, z: SATTRACTOR_PRESETS[2].z });
    expect(sattractorFormulas(2, { xformula: 'y', yformula: '', zformula: '' })).toEqual({ x: 'y', y: '0.0', z: '0.0' });
  });
  it('gets a mesh key that changes with the spec only', () => {
    const vi = (over: Record<string, number>, res?: Record<string, string>) => ({ name: 'sattractor3D', params: { presetId: 0, steps: 5, ...over }, res });
    expect(meshKeyFor(vi({}))).toMatch(/^sattr:[0-9a-f]{8}\w+#0$/);
    expect(meshKeyFor(vi({}))).toBe(meshKeyFor(vi({ scale_x: 2 }))); // scale is applied at sample time, not baked
    expect(meshKeyFor(vi({}))).not.toBe(meshKeyFor(vi({ radius: 0.1 })));
    expect(meshKeyFor(vi({}))).not.toBe(meshKeyFor(vi({}, { xformula: 'y', yformula: '-x', zformula: '0' })));
  });
});
