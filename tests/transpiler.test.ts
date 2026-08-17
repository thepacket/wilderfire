// Regression tests for the CUDA→WGSL transpiler (scripts/jwf-port/cwgsl.ts) that turns
// JWildfire GPU snippets into the registry's WGSL. Each case pins a mechanism with a
// snapshot of the emitted WGSL (tests/__snapshots__) plus a few structural assertions, so a
// transpiler change that alters output shows up as a reviewable diff.
//
// The snippet environment (magic __x/__px/…, weight, params, per-thread state) is the one
// gen.ts uses (scripts/jwf-port/bindings.ts) and the helper library is the real
// data/kernel-lib.cu — the same inputs as the generator.
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { transpileSnippet, TranspileError } from '../scripts/jwf-port/cwgsl';
import { bindingsFor } from '../scripts/jwf-port/bindings';

const kernelLib = readFileSync(resolve(process.cwd(), 'scripts/jwf-port/data/kernel-lib.cu'), 'utf8');

/** Transpile `snippet` as variation `name` (with the given params/extra state) the way gen.ts does. */
function tr(snippet: string, opts: { name?: string; params?: string[]; funcs?: string; extraParams?: string[] } = {}) {
  const v = { name: opts.name ?? 'demo', params: (opts.params ?? []).map((n) => ({ name: n })), gpuCode: snippet, extraParams: opts.extraParams ?? [] };
  const libs = [{ name: 'kernel', source: kernelLib }];
  if (opts.funcs) libs.push({ name: v.name, source: opts.funcs });
  return transpileSnippet(snippet, libs, bindingsFor(v));
}

describe('cwgsl: expressions and magic variables', () => {
  it('maps the point/output magic vars, weight and params; C float math to WGSL', () => {
    const r = tr(`float a = __theta * __demo_power + __demo;
float2 q = make_float2(cosf(a), sinf(a)) * __r;
__px += __demo * q.x;
__py += __demo * fabsf(q.y) + powf(__r2, 0.5f);
__pz += __demo * __z;`, { params: ['power'] });
    expect(r.code).toContain('${w}');
    expect(r.code).toContain('${p[0]}');
    expect(r.flags).toContain('z');
    expect(r.code).toMatchSnapshot();
  });

  it('keeps integer arithmetic integer (int params, casts, %, bit ops)', () => {
    const r = tr(`int n = (int)(__demo_count);
int k = n % 3;
int m = (k << 1) | 1;
float f = (float)k / (float)n;
__px += __demo * (f + (m & 2));`, { params: ['count'] });
    expect(r.code).toContain('i32(');
    expect(r.code).toContain('f32(');
    expect(r.code).toMatchSnapshot();
  });

  it('ternaries, compound assignments and unary minus', () => {
    const r = tr(`float s = __x > 0.f ? 1.f : -1.f;
float t2 = -__y;
t2 *= 2.f; t2 -= s; t2 /= 3.f;
__px += __demo * s * t2;`);
    expect(r.code).toContain('select(');
    expect(r.code).toMatchSnapshot();
  });

  it('gives snippets that assign their own weight/params a mutable local copy', () => {
    const r = tr(`__demo_scale = fabsf(__demo_scale);
__demo = __demo * 0.5f;
__px += __demo * __x * __demo_scale;`, { params: ['scale'] });
    expect(r.code).toContain('var w_: f32 = ${w};');
    expect(r.code).toContain('var p0_: f32 = ${p[0]};');
    expect(r.code).toMatchSnapshot();
  });
});

describe('cwgsl: control flow', () => {
  it('for/while loops, break/continue, switch with block-terminated cases', () => {
    const r = tr(`float acc = 0.f;
for (int i = 0; i < 4; i++) {
  if (i == 2) continue;
  acc += __x * i;
}
int j = 0;
while (j < 10) { j++; if (acc > 3.f) break; acc += 0.1f; }
switch (j & 3) {
  case 0: acc += 1.f; break;
  case 1:
  case 2: acc -= 1.f; break;
  default: acc = 0.f;
}
__px += __demo * acc;`);
    expect(r.code).toContain('for (');
    expect(r.code).toContain('switch');
    expect(r.code).toMatchSnapshot();
  });

  it('a bare `return;` in the snippet becomes a breakable wrapper (no early exit from the xform fn)', () => {
    const r = tr(`if (__r2 < 1.e-6f) return;
__px += __demo * __x / __r2;
__py += __demo * __y / __r2;`);
    expect(r.code).not.toMatch(/^\s*return;/m);
    expect(r.code).toContain('loop');
    expect(r.code).toMatchSnapshot();
  });

  it('`i++` inside an expression goes through the jpostinc builtin', () => {
    const r = tr(`int n = 0;
float a = 0.f;
if (jpostinc(&n) < 1) a += 1.f;
__px += __demo * (a + n);`);
    expect(r.functionNames).toContain('jpostinc');
    expect(r.functions).toContain('fn jpostinc(t: ptr<function, i32>) -> i32');
    expect(r.code).toMatchSnapshot();
  });
});

describe('cwgsl: helpers, structs and state', () => {
  it('emits only the helpers the snippet uses and threads the rng state into helpers that draw randoms', () => {
    const r = tr(`__px += __demo * demo_jit(__x);
__py += __demo * demo_sq(__y);`, {
      funcs: `__device__ float demo_sq(float x) { return x * x; }
__device__ float demo_jit(float x) { return x + (RANDFLOAT() - 0.5f) * 0.01f; }
__device__ float demo_unused(float x) { return x * 3.f; }`,
    });
    expect(r.functionNames).toEqual(expect.arrayContaining(['demo_sq', 'demo_jit']));
    expect(r.functionNames).not.toContain('demo_unused');
    expect(r.functions).toContain('fn demo_jit(x: f32, rs: ptr<function, u32>) -> f32');
    expect(r.functions).toContain('fn demo_sq(x: f32) -> f32');
    expect(r.code).toContain('demo_jit(t.x, rs)');
    expect(r.functions).toMatchSnapshot();
  });

  it('structs, pointer parameters and module-scope constant tables', () => {
    const r = tr(`demo_Pt p = demo_Pt_make(__x, __y);
demo_rot(&p, __demo_angle);
__px += __demo * (p.x + demo_TAB[1]);
__py += __demo * p.y;`, {
      params: ['angle'],
      funcs: `struct demo_Pt { float x; float y; };
float demo_TAB[3] = {0.25f, 0.5f, 0.75f};
__device__ demo_Pt demo_Pt_make(float x, float y) { demo_Pt r_; r_.x = x; r_.y = y; return r_; }
__device__ void demo_rot(demo_Pt *p, float a) { float c = cosf(a), s = sinf(a); float x = p->x * c - p->y * s; p->y = p->x * s + p->y * c; p->x = x; }`,
    });
    expect(r.functions).toContain('struct demo_Pt {');
    expect(r.functions).toContain('ptr<function, demo_Pt>');
    expect(r.functions).toContain('const demo_TAB: array<f32, 3>');
    expect(r.code + r.functions).toMatchSnapshot();
  });

  it('typed per-thread "extra" state becomes jwx_ globals with the state flag', () => {
    const r = tr(`if (__demo_inited_ == 0.0f) { __demo_inited_ = 1.0f; __demo_pos = make_float2(__x, __y); __demo_n = 0; }
__demo_pos = make_float2(__demo_pos.y, -__demo_pos.x);
__demo_n = __demo_n + 1;
__px += __demo * __demo_pos.x;
__py += __demo * __demo_pos.y * __demo_n;`, { extraParams: ['inited_', 'pos:float2', 'n:int'] });
    expect(r.flags).toContain('state');
    expect(r.code).toContain('jwx_demo_pos');
    expect(r.code).toContain('jwx_demo_n');
    expect(r.code).toMatchSnapshot();
  });

  it('direct-colour and hide flags come from the magic vars used', () => {
    const rgb = tr(`__colorR = 1.f; __colorG = __x; __colorB = 0.f; __useRgb = 1.f; __px += __demo * __x;`);
    expect(rgb.flags).toContain('rgb');
    expect(rgb.code).toContain('(*rgb).x');
    const pal = tr(`__pal = fabsf(__x); __px += __demo * __x;`);
    expect(pal.flags).toContain('dc');
    const hide = tr(`if (__r > 1.f) __doHide = true; else __doHide = false; __px += __demo * __x;`);
    expect(hide.flags).toContain('hide');
    expect(hide.code).toContain('(*hd)');
  });
});

describe('cwgsl: errors', () => {
  it('reports unsupported C (++ inside an expression) as a TranspileError instead of emitting bad WGSL', () => {
    expect(() => tr(`int n = 0; if (n++ < 1) __px += __demo * __x;`)).toThrow(TranspileError);
  });
  it('reports unknown identifiers', () => {
    expect(() => tr(`__px += __demo * nosuchthing;`)).toThrow(TranspileError);
  });
});
