import { describe, it, expect } from 'vitest';
import { compileFormula, COMPLEX_WGSL } from '../src/core/formula';
import { defaultEscape, normalizeEscape, escapeSignature, escapeFormulaWgsl, ESCAPE_FORMULAS } from '../src/core/escape';
import { buildEscapeWgsl } from '../src/gpu/escapeRenderer';
import { normalizeComposition, escapeLayer, wrapFlame } from '../src/core/composition';
import { PRESETS } from '../src/core/presets';
import { GREY } from './helpers';

describe('custom formula compiler', () => {
  it('compiles the usual expressions to WGSL complex ops', () => {
    expect(compileFormula('z^2 + c')).toBe('(cpow(z, vec2f(2.0, 0.0)) + c)');
    expect(compileFormula('z*z - c/2')).toBe('(cmul(z, z) - cdiv(c, vec2f(2.0, 0.0)))');
    expect(compileFormula('-z')).toBe('(-z)');
    expect(compileFormula('sin(z) + p1*i')).toBe('(csin(z) + cmul(p1, vec2f(0.0, 1.0)))');
    expect(compileFormula('pow(z, p2) + pixel + n')).toBe('((cpow(z, p2) + pixel) + vec2f(n, 0.0))');
    expect(compileFormula('2^3^2')).toBe('cpow(vec2f(2.0, 0.0), cpow(vec2f(3.0, 0.0), vec2f(2.0, 0.0)))'); // right-assoc
    expect(compileFormula('1e-3 + .5')).toBe('(vec2f(0.001, 0.0) + vec2f(0.5, 0.0))');
    expect(compileFormula('1e20')).toBe('vec2f(100000000000000000000.0, 0.0)');
  });
  it('rejects unknown names, bad arity and syntax errors with a message', () => {
    expect(() => compileFormula('z + q')).toThrow(/unknown name "q"/);
    expect(() => compileFormula('foo(z)')).toThrow(/unknown function/);
    expect(() => compileFormula('sin(z, c)')).toThrow(/one argument/);
    expect(() => compileFormula('z +')).toThrow(/unexpected end/);
    expect(() => compileFormula('(z + c')).toThrow(/unexpected end/);
    expect(() => compileFormula('(z + c z')).toThrow(/expected "\)"/);
    expect(() => compileFormula('z $ c')).toThrow(/unexpected character/);
  });
  it('every helper the compiler emits exists in COMPLEX_WGSL', () => {
    for (const fn of ['cmul', 'cdiv', 'cpow', 'csin', 'ccos', 'ctan', 'csinh', 'ccosh', 'ctanh', 'cexp', 'clog', 'csqrt', 'cabs', 'carg', 'cre', 'cim', 'cconj', 'crecip', 'csqr', 'ccube', 'cflip', 'cfloor', 'cround', 'cnorm']) expect(COMPLEX_WGSL).toContain(`fn ${fn}(`);
  });
});

describe('escape-time layer model + shader generator', () => {
  it('defaults normalise to themselves; a broken custom formula falls back to z²+c with an error', () => {
    const d = defaultEscape(GREY);
    expect(normalizeEscape(JSON.parse(JSON.stringify(d)), GREY)).toEqual(d);
    expect(normalizeEscape({ formula: 'nope', maxIter: 1e9, coloring: { outside: 'x' } }, GREY)).toMatchObject({ formula: 'mandelbrot', maxIter: 20000, coloring: { outside: 'smooth' } });
    const bad = { ...d, formula: 'custom' as const, custom: 'z^^2' };
    const w = escapeFormulaWgsl(bad);
    expect(w.error).toBeTruthy();
    expect(w.wgsl).toBe('csqr(z) + c');
    expect(buildEscapeWgsl(bad).error).toBeTruthy();
  });
  it('the signature covers what the shader bakes in, and every built-in formula/colouring generates a shader', () => {
    const d = defaultEscape(GREY);
    expect(escapeSignature({ ...d, zoom: 5, maxIter: 9 })).toBe(escapeSignature(d));
    expect(escapeSignature({ ...d, mode: 'julia' })).not.toBe(escapeSignature(d));
    for (const f of Object.keys(ESCAPE_FORMULAS) as (keyof typeof ESCAPE_FORMULAS)[]) {
      const code = buildEscapeWgsl({ ...d, formula: f }).code;
      expect(code).toContain('@fragment fn fs');
      expect(code).toContain('fn shade(');
    }
    for (const outside of ['smooth', 'iterations', 'exp-smooth', 'orbit-trap', 'distance', 'angle', 'solid'] as const) {
      const code = buildEscapeWgsl({ ...d, coloring: { ...d.coloring, outside } }).code;
      expect(code).toContain('return vec4f(gradient(t)');
    }
    expect(buildEscapeWgsl({ ...d, antialias: 3 }).code).toContain('/ 9.0');
  });
  it('escape layers round-trip inside a composition', () => {
    const c = wrapFlame(PRESETS[0].make());
    const e = defaultEscape(GREY); e.mode = 'julia'; e.custom = 'z^3 + c'; e.formula = 'custom'; e.coloring.trap.shape = 'ring';
    c.layers.push(escapeLayer(e, { blend: 'multiply', ownBackground: false }));
    const back = normalizeComposition(JSON.parse(JSON.stringify(c)), GREY);
    expect(back.layers[1].kind).toBe('escape');
    if (back.layers[1].kind === 'escape') { expect(back.layers[1].escape).toEqual(e); expect(back.layers[1].blend).toBe('multiply'); }
  });
});
