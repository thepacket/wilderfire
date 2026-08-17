// Regression tests for java2cu (scripts/jwf-port/java2cu.ts): JWildfire variation classes
// written in Java → the CUDA dialect gen.ts transpiles → WGSL. The fixtures under
// tests/fixtures/java2cu are synthetic classes in JWildfire's style (not JWildfire code),
// each exercising a set of the converter's mechanisms; the CUDA and the WGSL are pinned
// as snapshots so a converter/transpiler change shows up as a reviewable diff.
import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transpileSnippet } from '../scripts/jwf-port/cwgsl';
import { bindingsFor } from '../scripts/jwf-port/bindings';
import type { convertVariation as ConvertVariation, DumpVar } from '../scripts/jwf-port/java2cu';

const kernelLib = readFileSync(resolve(process.cwd(), 'scripts/jwf-port/data/kernel-lib.cu'), 'utf8');
const fixture = (f: string) => readFileSync(resolve(process.cwd(), 'tests/fixtures/java2cu', f), 'latin1');

let convertVariation: typeof ConvertVariation;
beforeAll(async () => {
  process.env.JAVA2CU_LIB = '1'; // import as a library: no JWildfire tree, no main()
  ({ convertVariation } = await import('../scripts/jwf-port/java2cu'));
});

/** Java class → CUDA port → WGSL (both snippets), the way gen.ts does it. */
function port(file: string, d: DumpVar) {
  const p = convertVariation(d, fixture(file), file);
  const libs = [{ name: 'kernel', source: kernelLib }, { name: d.name, source: p.gpuFunctions }];
  const cuda = (s: string) => s.replace(/varpar->/g, '__');
  const env = (code: string) => bindingsFor({ ...d, gpuCode: code, extraParams: p.extraParams });
  const wgsl = transpileSnippet(cuda(p.gpuCode), libs, env(cuda(p.gpuCode)));
  const preWgsl = p.preCode ? transpileSnippet(cuda(p.preCode), libs, env(cuda(p.preCode))) : null;
  return { p, wgsl, preWgsl };
}

describe('java2cu: params, constants, helpers, state', () => {
  const d: DumpVar = { name: 'reg_demo', params: [{ name: 'power', def: 2, int: false }, { name: 'dist', def: 1, int: false }, { name: 'seed', def: 12345, int: true }], priority: 0 };

  it('replays init(), inlines static-final constants, ports long→int, threads params into helpers, keeps written fields as state', () => {
    const { p, wgsl } = port('RegDemoFunc.java', d);
    expect(p.gpuCode).toContain('invPower = 1.0 / __reg_demo_power');           // init() replayed per call
    expect(p.gpuCode).toContain('* (0.75)');                                      // static final SCALE inlined
    expect(p.gpuCode).toContain('jpostinc(&n)');                                  // n++ inside a condition
    expect(p.gpuCode).toContain('__pz += __amount_ * __z');                       // preserve-z branch kept (flag folded)
    expect(p.extraParams).toContain('seed_s:int');                                // long param written by transform → int state (renamed: `seed` is the param)
    expect(p.gpuCode).toContain('varpar->reg_demo_seed_s = __reg_demo_seed;');    // …initialised from the param on the first call
    expect(p.extraParams).toContain('dist_c');                                    // param read by a helper travels as a state copy
    expect(p.gpuFunctions).toContain('__device__ float reg_demo_jitter(float r)');
    expect(p.gpuFunctions).toContain('RANDFLOAT()');                              // pContext.random() in the helper
    expect(wgsl.flags).toContain('state');
    expect(wgsl.functions).toContain('fn reg_demo_jitter(r_: f32, rs: ptr<function, u32>) -> f32');
    expect(p.gpuCode).toMatchSnapshot('cuda');
    expect(wgsl.code).toMatchSnapshot('wgsl');
  });
});

describe('java2cu: plain-data classes, tables, pointer params', () => {
  const d: DumpVar = { name: 'reg_pod', params: [{ name: 'sides', def: 4, int: true }, { name: 'spread', def: 0.5, int: false }], priority: 0 };

  it('inner class → struct + _make, constant array → module table, mutated object param → pointer, chained assignment after else → braces, random field initialiser → per-instance state (initialised once)', () => {
    const { p, wgsl } = port('RegPodFunc.java', d);
    expect(p.gpuFunctions).toContain('struct reg_pod_Pt { float x; float y; };');
    expect(p.gpuFunctions).toContain('reg_pod_Pt_make(float x, float y)');
    expect(p.gpuFunctions).toContain('float reg_pod_OFFSETS[4] = {0.0, 0.25, 0.5, 0.75};');
    expect(p.gpuFunctions).toContain('void reg_pod_rotate(reg_pod_Pt *p, float a)');
    expect(p.gpuCode).toContain('{ p.x = p.x + __reg_pod_spread; d = p.x; }');
    expect(p.gpuCode).toContain('RANDFLOAT() * (float)(__reg_pod_sides)');       // pContext.random(n)
    expect(p.extraParams).toEqual(['inited_', 'phase']);
    // the random initialiser runs once per thread, not on every call
    expect(p.gpuCode.match(/varpar->reg_pod_phase = RANDFLOAT\(\)/g)).toHaveLength(1);
    expect(wgsl.functions).toContain('struct reg_pod_Pt {');
    expect(wgsl.functions).toContain('const reg_pod_OFFSETS: array<f32, 4>');
    expect(wgsl.functions).toContain('fn reg_pod_rotate(p_: ptr<function, reg_pod_Pt>, a: f32)');
    expect(p.gpuCode).toMatchSnapshot('cuda');
    expect(p.gpuFunctions).toMatchSnapshot('cuda functions');
    expect(wgsl.code).toMatchSnapshot('wgsl');
  });
});

describe('java2cu: prepost pairs', () => {
  const d: DumpVar = { name: 'reg_prepost', params: [{ name: 'scale', def: 1.5, int: false }, { name: 'angle', def: 30, int: false }], priority: 2 };

  it('invtransform() becomes preCode (helper taking pAffineTP inlined, precalc refresh appended) and shares the init() preamble with transform()', () => {
    const { p, wgsl, preWgsl } = port('RegPrePostFunc.java', d);
    expect(p.preCode).toBeDefined();
    expect(p.preCode).toContain('__x = x * cosa - y * sina;');                     // moveInput inlined into the pre snippet
    expect(p.preCode).toContain('__r2 = __x*__x+__y*__y;');                        // precalc refresh for the variations that follow
    expect(p.gpuCode).toContain('__px = x * cosa + y * sina;');                    // transform() on the output point
    expect(p.gpuCode).not.toContain('__r2 =');
    // both snippets carry the sina/cosa preamble
    for (const s of [p.gpuCode, p.preCode!]) expect(s).toContain('sina = sinf(__reg_prepost_angle * M_PI / 180.0);');
    expect(preWgsl!.code).toContain('t.x = ((x * cosa) - (y * sina));');
    expect(preWgsl!.code).toContain('r2 = ((t.x * t.x) + (t.y * t.y));');
    expect(wgsl.code).toContain('v.x = ((x * cosa) + (y * sina));');
    expect(p.preCode).toMatchSnapshot('cuda pre');
    expect(preWgsl!.code).toMatchSnapshot('wgsl pre');
    expect(wgsl.code).toMatchSnapshot('wgsl');
  });
});

describe('java2cu: parse errors are reported, not silently ported', () => {
  it('rejects a class without transform()', async () => {
    const src = fixture('RegDemoFunc.java').replace('public void transform(', 'public void transformX(');
    expect(() => convertVariation({ name: 'reg_demo', params: [], priority: 0 }, src, 'x.java')).toThrow();
  });
});
