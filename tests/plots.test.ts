import { describe, it, expect } from 'vitest';
import { compileFormula, formulaToWgsl, parseFormula, formulaType, FV } from '../src/core/formula';
import { PLOT_FAMILIES, PLOT_NAMES, plotFormulas, plotPreset } from '../src/core/plots';
import { KNOTS3D_PRESETS } from '../src/core/plotPresets';
import { VARIATIONS } from '../src/core/variations';
import { buildKnotMesh, knotsFormulas } from '../src/core/knots';
import { meshKeyFor } from '../src/core/meshes';
import { importFlameText, flameToXML } from '../src/core/flameXML';
import { flameSignature } from '../src/core/flame';
import { compileFlame } from '../src/gpu/codegen';
import { GREY } from './helpers';

const hex = (s: string) => Array.from(new TextEncoder().encode(s), (b) => b.toString(16).toUpperCase().padStart(2, '0')).join('');
const vars = (o: Partial<Record<keyof typeof FV, number>>) => { const v = new Float64Array(16); for (const [k, x] of Object.entries(o)) v[FV[k as keyof typeof FV]] = x!; return v; };

describe('formula → WGSL', () => {
  const V = { x: 'x_', y: 'y_', z: 'z_', u: 'u_', v: 'v_', t: 't_', param_a: 'pa', param_b: 'pb' };
  it('keeps Java typing: int literals stay ints (truncating division), doubles are f32, comparisons are bools', () => {
    expect(formulaToWgsl('1/2', V)).toBe('f32((1i / 2i))');
    expect(formulaToWgsl('1/2.0', V)).toBe('(f32(1i) / 2.0)');
    expect(formulaToWgsl('x/2', V)).toBe('((x_) / f32(2i))');
    expect(formulaToWgsl('2*pi', V)).toBe('(f32(2i) * 3.141592653589793)');
    expect(formulaToWgsl('(x>0)?1:0', V)).toBe('f32(select(0i, 1i, ((x_) > f32(0i))))');
    expect(formulaToWgsl('x > 0 && y < 1 ? x : -y', V)).toBe('select((-(y_)), (x_), (((x_) > f32(0i)) && ((y_) < f32(1i))))');
    expect(formulaToWgsl('!(x>0)', V)).toBe('select(0.0, 1.0, (!((x_) > f32(0i))))');
    expect(formulaType(parseFormula('1/2', []))).toBe('i');
    expect(formulaType(parseFormula('sign(x)*2', ['x']))).toBe('i');
    expect(compileFormula('1/2')(vars({}))).toBe(0);
    expect(compileFormula('7/2*x')(vars({ x: 1 }))).toBe(3);
    expect(compileFormula('7/2.0*x')(vars({ x: 1 }))).toBe(3.5);
  });
  it('maps MathLib functions and constants; pow and atan2 go through the Java-faithful helpers', () => {
    expect(formulaToWgsl('pow(-2, 3) + atan2(y, x) + sqr(x) + fabs(x) + Math.abs(x)', V)).toBe('((((powc(f32((-2i)), f32(3i)) + atan2j((y_), (x_))) + sqr((x_))) + abs((x_))) + abs((x_)))');
    expect(formulaToWgsl('fmod(x, 2.0) + log10(x) + round(x) + rint(x) + trunc(x) + frac(x)', V)).toContain('((x_) % 2.0)');
    expect(formulaToWgsl('M_PI_2 + M_2PI + EPSILON + TRUE', V)).toBe('(((1.5707963267948966 + 6.283185307179586) + 1.0e-8) + f32(1i))');
    expect(formulaToWgsl('sinh(u)*cosh(v) + tanh(t) + exp(u) + sqrt(v) + floor(t) + min(u, v) + max(u, v)', V)).toContain('sinh((u_))');
  });
  it('rejects unknown names and functions (no eval, no code beyond the subset)', () => {
    expect(() => formulaToWgsl('System.exit(0)', V)).toThrow();
    expect(() => formulaToWgsl('w + 1', V)).toThrow(/unknown name w/);
    expect(() => formulaToWgsl('foo(x)', V)).toThrow(/unknown function/);
    expect(() => formulaToWgsl('x +', V)).toThrow();
    expect(() => formulaToWgsl('(x > 0) + 1', V)).toThrow(/boolean/);
  });
  it('every preset formula of every family (and knots3D) compiles on both backends', () => {
    let n = 0;
    for (const [name, fam] of Object.entries(PLOT_FAMILIES)) {
      const vs: Record<string, string> = {}; for (const v of fam.vars) vs[v] = v + '_';
      for (const c of fam.letters) vs['param_' + c] = 'p' + c;
      for (const pr of fam.presets) for (const k of fam.formulas) {
        expect(() => formulaToWgsl(pr.f[k], vs), `${name} #${pr.id} ${k}: ${pr.f[k]}`).not.toThrow();
        expect(() => compileFormula(pr.f[k]), `${name} #${pr.id} ${k}`).not.toThrow();
        n++;
      }
    }
    expect(KNOTS3D_PRESETS.length).toBe(24);
    expect(n).toBe(14 + 7 + 47 * 3 + 16 + 10 + 42 + 24 * 3);
  });
});

describe('plot family', () => {
  it('resolves formulas like JWildfire: preset_id ≥ 0 wins over the ressource, −1 uses the ressource, unknown ids and empty ressources fall back to the default preset', () => {
    expect(plotFormulas('yplot2d_wf', 3, { formula: 'x' })).toEqual({ formula: 'sin(param_a*x)/cos(x*x)' });
    expect(plotFormulas('yplot2d_wf', -1, { formula: 'x*x' })).toEqual({ formula: 'x*x' });
    expect(plotFormulas('yplot2d_wf', -1)).toEqual({ formula: '0.0' });
    expect(plotFormulas('yplot2d_wf', 999)).toEqual({ formula: '0.0' });
    expect(plotPreset('parplot2d_wf', 4).p.umax).toBeCloseTo(4 * Math.PI, 12); // the "0.0;" preset JEP tolerates
    expect(plotFormulas('parplot2d_wf', -1, { xformula: 'u' })).toEqual({ xformula: 'u', yformula: '0.0', zformula: 'v' });
    expect(PLOT_FAMILIES.polarplot3d_wf.presets.filter((p) => p.p.cylindrical === 1).length).toBeGreaterThan(0);
  });

  it('every plot variation is registered with a formula-dependent signature key and a snippet that inlines the formula', () => {
    for (const name of PLOT_NAMES) {
      const def = VARIATIONS[name];
      expect(def, name).toBeDefined();
      expect(def.sigKey, name).toBeDefined();
      expect(def.res, name).toContain(PLOT_FAMILIES[name].formulas[0]);
      const p = (def.params ?? []).map((_, i) => `xd[${i}u]`);
      const code = def.code('w_', p, () => '0.0', { params: { preset_id: -1 }, res: { formula: 'param_a + 7.25', xformula: 'u + 7.25', yformula: 'v', zformula: '0' } });
      expect(code, name).toContain('7.25');
      expect(def.sigKey!({ params: { preset_id: -1 }, res: { formula: 'x', xformula: 'u', yformula: 'v', zformula: 'u' } })).not.toBe(def.sigKey!({ params: { preset_id: 0 } }));
    }
  });

  it('imports the formula ressource, exports it back, compiles the kernel and changes the signature with the formula', () => {
    const xml = `<flame name="p" size="64 64" scale="10"><xform weight="1" parplot2d_wf="1" parplot2d_wf_preset_id="-1" parplot2d_wf_umin="0" parplot2d_wf_umax="6.2831" parplot2d_wf_vmin="0" parplot2d_wf_vmax="6.2831" parplot2d_wf_color_mode="3" parplot2d_wf_solid="1" parplot2d_wf_param_a="1.5" parplot2d_wf_xformula="${hex('cos(u)*(param_a+cos(v))')}" parplot2d_wf_yformula="${hex('sin(u)*(param_a+cos(v))')}" parplot2d_wf_zformula="${hex('sin(v)')}" coefs="1 0 0 1 0 0"/>` +
      `<xform weight="1" yplot2d_wf="1" yplot2d_wf_preset_id="2" yplot2d_wf_formula="${hex('ignored')}" coefs="1 0 0 1 0 0"/>` +
      `<xform weight="1" isosfplot3d_wf="1" isosfplot3d_wf_preset_id="-1" isosfplot3d_wf_formula="${hex('x*x+y*y+z*z-1')}" isosfplot3d_wf_max_iter="20" coefs="1 0 0 1 0 0"/>` +
      `<xform weight="1" polarplot2d_wf="1" polarplot2d_wf_preset_id="1" coefs="1 0 0 1 0 0"/>` +
      `<xform weight="1" polarplot3d_wf="1" polarplot3d_wf_preset_id="1" coefs="1 0 0 1 0 0"/>` +
      `<xform weight="1" yplot3d_wf="1" yplot3d_wf_preset_id="-1" yplot3d_wf_formula="${hex('sin(x)*cos(z)')}" coefs="1 0 0 1 0 0"/></flame>`;
    const { flame, unknown } = importFlameText(xml, GREY);
    expect(unknown).toEqual([]);
    const xs = flame.layers[0].xforms;
    expect(xs[0].variations[0].res).toEqual({ xformula: 'cos(u)*(param_a+cos(v))', yformula: 'sin(u)*(param_a+cos(v))', zformula: 'sin(v)' });
    expect(xs[0].variations[0].params.param_a).toBe(1.5);
    expect(xs[1].variations[0].res).toEqual({ formula: 'ignored' }); // kept for round-tripping, not rendered (preset 2 is)
    const out = flameToXML(flame);
    expect(out).toContain(`parplot2d_wf_xformula="${hex('cos(u)*(param_a+cos(v))')}"`);
    expect(out).toContain(`isosfplot3d_wf_formula="${hex('x*x+y*y+z*z-1')}"`);
    const c = compileFlame(flame, 1024);
    expect(c.wgsl).toContain('cos((pl_u))'); // the parplot formula, inlined
    expect(c.wgsl).toContain('sin(((f32(2i) * (pl_x)) * (pl_x)))'); // yplot2d preset 2: sin(2*x*x)
    expect(c.wgsl).not.toContain('ignored');
    expect(c.wgsl).toContain('fn powc('); // the formula helpers are in the kernel once
    expect((c.wgsl.match(/fn sqr\(/g) ?? []).length).toBe(1);
    const sig = flameSignature(flame);
    xs[0].variations[0].res!.xformula = 'u';
    expect(flameSignature(flame)).not.toBe(sig);
    xs[0].variations[0].res!.xformula = 'cos(u)*(param_a+cos(v))';
    expect(flameSignature(flame)).toBe(sig);
    xs[1].variations[0].params.preset_id = 5; // another preset = another formula
    expect(flameSignature(flame)).not.toBe(sig);
  });

  it('knots3D: the preset formulas win while presetId ≥ 0, the tube has steps·facets vertices and 2·facets faces per segment, the mesh key follows the spec', () => {
    expect(knotsFormulas(0)).toEqual({ x: KNOTS3D_PRESETS[0].f.xformula, y: KNOTS3D_PRESETS[0].f.yformula, z: KNOTS3D_PRESETS[0].f.zformula });
    expect(knotsFormulas(0, { xformula: 't' })).toEqual(knotsFormulas(0));
    expect(knotsFormulas(-1, { xformula: 't' })).toEqual({ x: 't', y: '0.0', z: '0.0' });
    expect(knotsFormulas(999).x).toBe('100 * cos(t)');
    const pr = KNOTS3D_PRESETS[0];
    const m = buildKnotMesh({ x: pr.f.xformula, y: pr.f.yformula, z: pr.f.zformula, steps: 200, radius: pr.p.radius, facets: 5, params: 'abcdefgh'.split('').map((c) => pr.p['param_' + c]) });
    expect(m.pos.length / 3).toBe(200 * 5);
    expect(m.idx.length / 3).toBe(199 * 5 * 2);
    expect(Array.from(m.pos).every(Number.isFinite)).toBe(true);
    expect(Math.max(...Array.from(m.idx))).toBe(200 * 5 - 1);
    // every ring is a rigid rotation of the first one: its chord lengths equal ring 0's
    const chord = (j: number, i: number, k: number) => { const a = (j * 5 + i) * 3, b = (j * 5 + k) * 3; return Math.hypot(m.pos[a] - m.pos[b], m.pos[a + 1] - m.pos[b + 1], m.pos[a + 2] - m.pos[b + 2]); };
    for (const j of [1, 50, 199]) for (let i = 0; i < 5; i++) expect(chord(j, i, (i + 2) % 5)).toBeCloseTo(chord(0, i, (i + 2) % 5), 3);
    expect(chord(0, 0, 1)).toBeCloseTo(2 * pr.p.radius * Math.sin(Math.PI / 5), 6); // a regular pentagon of radius R
    const vi = (over: Record<string, number>, res?: Record<string, string>) => ({ name: 'knots3D', params: { presetId: 0, steps: 100, ...over }, res });
    expect(meshKeyFor(vi({}))).toMatch(/^knots:[0-9a-f]{8}\w+#0$/);
    expect(meshKeyFor(vi({}))).toBe(meshKeyFor(vi({ scale_x: 2 })));
    expect(meshKeyFor(vi({}))).not.toBe(meshKeyFor(vi({ radius: 0.1 })));
    expect(meshKeyFor(vi({ presetId: -1 }, { xformula: 'cos(t)', yformula: 'sin(t)', zformula: '0' }))).not.toBe(meshKeyFor(vi({ presetId: -1 })));
  });

  it('a formula outside the subset renders the zero curve with a warning instead of breaking the kernel', () => {
    const xml = `<flame name="p" size="64 64" scale="10"><xform weight="1" yplot2d_wf="1" yplot2d_wf_preset_id="-1" yplot2d_wf_formula="${hex('System.exit(0)')}" coefs="1 0 0 1 0 0"/></flame>`;
    const { flame } = importFlameText(xml, GREY);
    const warnings: string[] = [];
    const orig = console.warn; console.warn = (...a: unknown[]) => { warnings.push(a.join(' ')); };
    try { expect(compileFlame(flame, 1024).wgsl).toContain('let pl_y = 0.0;'); } finally { console.warn = orig; }
    expect(warnings.some((w) => /yplot2d_wf: formula/.test(w))).toBe(true);
  });
});

describe('c_var (complex-function code)', () => {
  const bodies = [
    'import js.glsl.vec2;\npublic vec2 f(vec2 z)\n{\n  vec2 a=c_asin(c_inv(z));\nreturn c_sub(a,c_sin(new vec2(0.0, 0.0)));\n}',
    'import js.glsl.vec2;\npublic vec2 f(vec2 z)\n{\n  vec2 a=c_add(c_conj(c_asin(z)),c_cos(c_conj(z)));\n  vec2 b=c_add(c_conj(a),c_asin(new vec2(1.20, 0.0)));\nreturn c_mul(a,b);\n}',
    'import js.glsl.vec2;\npublic vec2 f(vec2 z)\n{\n // vec2 q = broken(;\n vec2 a=c_sub(c_inv(z),c_atan(c_exp(c_acos(z))));\nreturn c_add(a,c_atan(new vec2(-0.0, 0.0)));\n}',
    'import js.glsl.vec2;\npublic vec2 f(vec2 z)\n{\n return c_pow(c_asin(z), c_sqrt(z));\n}',
    'public vec2 f(vec2 w) { vec2 pow=new vec2(-2.0,-.60); return c_pow(w,pow).plus(c_exp(2.0, w)).multiply(0.5); }',
    'public vec2 f(vec2 z) { z = z.plus(new vec2(1, 2)); return c_log(z, 10.0); }',
  ];
  it('compiles the corpus bodies, the default and method chains; rejects everything else', async () => {
    const { cvarToWgsl, CVAR_DEFAULT_CODE } = await import('../src/core/cvar');
    for (const b of bodies) expect(() => cvarToWgsl(b), b).not.toThrow();
    expect(cvarToWgsl(bodies[3])).toBe('cv_ret = cv_powc(cv_asin(cv_z), cv_sqrt(cv_z));');
    expect(cvarToWgsl(bodies[4])).toBe('let cv_l0 = vec2f((-2.0), (-0.6));\n    cv_ret = ((cv_powc(cv_z, cv_l0) + cv_expb(2.0, cv_z)) * 0.5);');
    expect(cvarToWgsl(bodies[5])).toBe('let cv_l0 = (cv_z + vec2f(f32(1i), f32(2i)));\n    cv_ret = cv_log(cv_l0, 10.0);');
    expect(cvarToWgsl(CVAR_DEFAULT_CODE)).toContain('cv_inv(cv_z)');
    expect(() => cvarToWgsl('public vec2 f(vec2 z) { return z.x; }')).toThrow();
    expect(() => cvarToWgsl('public vec2 f(vec2 z) { System.exit(0); return z; }')).toThrow();
    expect(() => cvarToWgsl('public vec2 f(vec2 z) { return foo(z); }')).toThrow(/unknown function/);
    expect(() => cvarToWgsl('public vec2 f(vec2 z) { vec2 a = z; }')).toThrow(/return/);
  });
  it('imports the code ressource, compiles the kernel with the c_* helpers once, and recompiles when the code changes', () => {
    const xml = `<flame name="c" size="64 64" scale="10"><xform weight="1" c_var="1" c_var_mode="0" c_var_zoom="1" c_var_code="${hex(bodies[1])}" coefs="1 0 0 1 0 0"/>` +
      `<xform weight="1" pre_c_var="1" linear="1" post_c_var="0.5" post_c_var_code="${hex(bodies[3])}" coefs="1 0 0 1 0 0"/></flame>`;
    const { flame, unknown } = importFlameText(xml, GREY);
    expect(unknown).toEqual([]);
    const x0 = flame.layers[0].xforms[0];
    expect(x0.variations[0].res?.code).toBe(bodies[1]);
    expect(flame.layers[0].xforms[1].preVariations?.find((v) => v.name === 'pre_c_var')?.res).toBeUndefined(); // the default code is not stored
    expect(flameToXML(flame)).toContain(`pre_c_var_code="${hex('import js.glsl.vec2;')}`); // but exported for JWildfire
    const c = compileFlame(flame, 1024);
    expect(c.wgsl).toContain('cv_conj(cv_asin(cv_z))');
    expect((c.wgsl.match(/fn cv_mul\(/g) ?? []).length).toBe(1);
    expect((c.wgsl.match(/fn powc\(/g) ?? []).length).toBe(1);
    const sig = flameSignature(flame);
    x0.variations[0].res!.code = bodies[0];
    expect(flameSignature(flame)).not.toBe(sig);
  });
});
