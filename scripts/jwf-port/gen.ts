// Generates src/core/variations.jwf.ts from JWildfire's variation catalogue.
//
//   node scripts/jwf-port/gen.ts            # regenerate + print summary
//   node scripts/jwf-port/gen.ts --report   # also write scripts/jwf-port/report.json
//
// Input: data/jwf-variations.jsonl (produced by Dump.java against a JWildfire
// build — see README.md) and data/kernel-lib.cu (shared CUDA helpers).

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { transpileSnippet, TranspileError, type Binding, type Env, type Ty } from './cwgsl.ts';
import { OVERRIDES } from './overrides.ts';
import clampsJson from './data/param-clamps.json' with { type: 'json' };
const PARAM_CLAMPS = clampsJson as Record<string, Record<string, [number, number]>>;
import dcBaseJson from './data/dc-base.json' with { type: 'json' };
import intsJson from './data/param-ints.json' with { type: 'json' };
const PARAM_INTS = intsJson as Record<string, Record<string, 'trunc' | 'round'>>;
const DC_BASE = dcBaseJson as { inherit: string[]; own2: string[]; half: string[]; ownOther: string[] };

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, 'data');
const outFile = join(here, '..', '..', 'src', 'core', 'variations.jwf.ts');
const outFileUnverified = join(here, '..', '..', 'src', 'core', 'variations.jwf.unverified.ts');

interface DumpParam { name: string; def: number | null; int: boolean; stable: boolean }
interface DumpVar {
  name: string; cls: string; priority: number; types: string[]; params: DumpParam[];
  altNames?: string[]; resources: number; defaultsStable: boolean; gpu: boolean;
  gpuCode?: string; preCode?: string; gpuFunctions?: string; stateful?: boolean; extraParams?: string[]; error?: string;
}

const dump: DumpVar[] = readFileSync(join(dataDir, 'jwf-variations.jsonl'), 'utf8')
  .trim().split('\n').map((l) => JSON.parse(l));
// Java→CUDA-dialect ports (java2cu.ts) for variations that have no GPU snippet in JWildfire
const javaPorts = new Map<string, { gpuCode: string; preCode?: string; gpuFunctions: string; extraParams: string[]; note: string }>();
try { for (const l of readFileSync(join(dataDir, 'jwf-java-ports.jsonl'), 'utf8').trim().split('\n')) { const p = JSON.parse(l); javaPorts.set(p.name, p); } } catch { /* none */ }
const kernelLib = readFileSync(join(dataDir, 'kernel-lib.cu'), 'utf8');
// Oracle verdicts (written by the in-browser harness); absent → nothing is verified.
let verified = new Set<string>();
try { verified = new Set<string>(JSON.parse(readFileSync(join(here, 'verified.json'), 'utf8')).jwf); } catch { /* first run */ }
/** Ports the oracle cannot confirm because JWildfire's CPU code is itself broken, but whose GPU intent is right. */
const FORCE_VERIFIED: Record<string, string> = {
  pre_flatten: 'CPU writes pVarTP.z inside a pre-variation (a no-op); the GPU flattens the affine z, which is the intent',
  cut_bricks: 'matches at default seed; the seed param drives java.util.Random on the CPU and nothing on the GPU (same as JWildfire GPU)',
  post_point_crop: 'stateful: crops to the last uncropped point, so the result depends on evaluation order; the snippet is the Java line by line',
  arch: 'same formula as the CPU (v·sin, v·sin²/cos); the 1/cos tail makes per-point mean/std statistics meaningless',
  rays: 'same formula as the CPU (v²·tan(rnd·π·v)/r²); the tan tail makes per-point mean/std statistics meaningless',
  starfractal: 'CPU keeps x,y across calls (a chaos game with its own RNG); the GPU iterates the same IFS 500× per point — same attractor, but the per-point statistics of the heavy-tailed inversions do not compare',
  curliecue2: 'a curlicue walk (x0,y0,phi,theta advance once per call, independent of the input point) kept as per-thread state; a trajectory, so per-point statistics do not compare',
  ...Object.fromEntries(['hopalong', 'macmillan', 'threeply', 'gumowski_mira', 'gingerbread_man'].map((n) => [n, 'chaotic attractor iterated as per-thread state (Java port, line by line); f32 and f64 trajectories decorrelate after a few hundred steps and the spread of these maps keeps evolving, so per-segment statistics do not compare'])),
  circular: 'the jitter is a sin(x·12.9898+y·78.233+seed)·43758 hash of the continuous input point: identical in distribution, but the value at a given point depends on the f64/f32 rounding of that point, so the per-point oracle cannot compare it',
  circular2: 'see circular',
  iconattractor_js: 'JWildfire GPU snippet; the Java re-applies the presetId table (a/b/g/o/l) at init, so only oracle sets whose explicit params contradict their presetId differ — a flame file always carries the preset values',
  recurrenceplot: 'stateful (a rejected sample re-emits the previous accepted point): JWildfire\'s GPU snippet had dropped that state (and never set ldcs); restored as per-thread state — the sequence-dependent per-point means do not compare, the spread and colouring do',
  pre_blur3D: 'gaussian pre-blur whose 6-entry ring buffer is advanced with `& 5` in the Java (entries 2,3 never refresh: per-instance constants); ported as four fresh uniforms plus the constants\' mean — same distribution up to the per-instance offset',
  minkQM: 'Minkowski ?-function: at e near 1 the sum is dominated by late Stern–Brocot branches, which flip when a grid coordinate is (nearly) a small rational — an f32 boundary artefact of the test grid, not a port difference (f32 CPU model matches the Java)',
};
for (const n of Object.keys(FORCE_VERIFIED)) verified.add(n);
// prepost ports carry two snippets; both must have passed (the harness reports the inverse as `name~inv`)
for (const [n, jp] of javaPorts) if (jp.preCode && verified.has(n) && !verified.has(n + '~inv')) verified.delete(n);

/** Deterministic defaults for parameters JWildfire randomizes at construction. */
const DEFAULT_OVERRIDES: Record<string, Record<string, number>> = {
  julian: { power: 3 }, juliascope: { power: 3 }, julia3D: { power: 3 }, julia3Dz: { power: 3 },
  julian2: { power: 3 }, juliaq: { power: 3 }, julia3Dq: { power: 3 }, post_juliaq: { power: 3 }, post_julia3Dq: { power: 3 },
  julian3Dx: { power: 3 }, phoenix_julia: { power: 3 }, jubiQ: { power: 3 }, juliascope3Db: { power: 3 }, juliascopePlus: { power: 3 },
  juliac: { re: 3, im: 0 },
};

/** Variations excluded regardless of transpile success (semantic reasons). */
const EXCLUDE: Record<string, string> = {
  custom_wf: 'user Java code', custom_wf_full: 'user Java code',
};

const UNSUPPORTED_FLAGS: Record<string, string> = {
  wfield: 'weighting fields',
};

interface GenEntry {
  name: string; params: { name: string; def: number }[]; code: string; preCode?: string; funcs: string; funcNames: string[];
  priority: number; types: string[]; flags: string[];
}

const entries: GenEntry[] = [];
const report: { name: string; status: 'ok' | 'skip' | 'error'; reason?: string; flags?: string[] }[] = [];

// Registry of module-scope WGSL items across all variations (name → text) to detect conflicts.
const fnRegistry = new Map<string, string>();

function bindingsFor(v: DumpVar): Env {
  const b = new Map<string, Binding>();
  const F = { k: 'f32' } as const, B = { k: 'bool' } as const, I = { k: 'i32' } as const, U = { k: 'u32' } as const;
  b.set('__x', { code: 't.x', ty: F, lvalue: true });
  b.set('__y', { code: 't.y', ty: F, lvalue: true });
  // z_ / pz_ are declared by the codegen stage (input z, output z accumulator)
  b.set('__z', { code: 'z_', ty: F, lvalue: true, flag: 'z' });
  b.set('__px', { code: 'v.x', ty: F, lvalue: true });
  b.set('__py', { code: 'v.y', ty: F, lvalue: true });
  b.set('__pz', { code: 'pz_', ty: F, lvalue: true, flag: 'z' });
  b.set('__r2', { code: 'r2', ty: F, lvalue: true });
  b.set('__r', { code: 'r', ty: F, lvalue: true });
  b.set('__rinv', { code: 'rinv_', ty: F, lvalue: true, decl: 'var rinv_: f32 = 1.0 / r;' });
  b.set('__phi', { code: 'th', ty: F, lvalue: true });
  b.set('__theta', { code: 'ph', ty: F, lvalue: true });
  b.set('__pal', { code: '(*cp)', ty: F, lvalue: true, flag: 'dc' });
  b.set('__doHide', { code: '(*hd)', ty: B, lvalue: true, flag: 'hide' });
  b.set('__x0', { code: 't0.x', ty: F, lvalue: false, flag: 'x0' });
  b.set('__y0', { code: 't0.y', ty: F, lvalue: false, flag: 'x0' });
  b.set('__z0', { code: 'z_', ty: F, lvalue: false, flag: 'z' });
  // Direct RGB colour: written through the xform function's `rgb` out-pointer
  // (x,y,z = colour, w = 1 when set); the kernel plots it instead of the palette lookup.
  b.set('__colorR', { code: '(*rgb).x', ty: F, lvalue: true, flag: 'rgb' });
  b.set('__colorG', { code: '(*rgb).y', ty: F, lvalue: true, flag: 'rgb' });
  b.set('__colorB', { code: '(*rgb).z', ty: F, lvalue: true, flag: 'rgb' });
  b.set('__colorA', { code: 'colorA_', ty: F, lvalue: true, flag: 'rgb', decl: 'var colorA_: f32 = 1.0;' });
  b.set('__useRgb', { code: '(*rgb).w', ty: F, lvalue: true, flag: 'rgb' });
  b.set('numColors', { code: '256', ty: I, lvalue: false });
  b.set('palette', { code: 'PALB_', ty: U, lvalue: false });
  b.set('__wFieldValue', { code: '0.0', ty: F, flag: 'wfield' });
  b.set('__wFieldAmountScale', { code: '1.0', ty: F, flag: 'wfield' });
  b.set('__was_pre', { code: 'false', ty: B, flag: 'waspre' });
  const pnames = v.params.map((p) => p.name);
  // Snippets occasionally assign to their own weight/params (treating them as
  // scratch); those get a mutable local copy instead of the uniform read.
  const src = (v.gpuCode ?? '').replace(/varpar->/g, '__');
  const assigned = new Set<string>();
  for (const m of src.matchAll(/(__[A-Za-z0-9_]+)\s*(?:=(?!=)|\+=|-=|\*=|\/=|\+\+|--)/g)) assigned.add(m[1]);
  for (const m of src.matchAll(/(?:\+\+|--)\s*(__[A-Za-z0-9_]+)/g)) assigned.add(m[1]);
  // extra params: "name" (f32) or "name:float2|float3|int" (typed per-thread state from java2cu)
  const extraTypes = new Map<string, string>((v.extraParams ?? []).map((e) => { const [n, t] = e.split(':'); return [n, t ?? 'f32']; }));
  const extra = new Set(extraTypes.keys());
  const tyOf = (t: string) => t === 'float2' ? { k: 'vec', n: 2, e: 'f32' } as const : t === 'float3' ? { k: 'vec', n: 3, e: 'f32' } as const : t === 'int' ? I : F;
  const resolveMagic = (name: string): Binding | null => {
    if (name === '__' + v.name || name === '__amount_') {
      if (assigned.has(name) || assigned.has('__amount_') || assigned.has('__' + v.name)) return { code: 'w_', ty: F, lvalue: true, decl: 'var w_: f32 = ${w};' };
      return { code: '${w}', ty: F };
    }
    const pre = '__' + v.name + '_';
    if (name.startsWith(pre)) {
      const pn = name.slice(pre.length);
      // params whose JWildfire name has spaces/punctuation are referenced with `_` (java2cu)
      const i = pnames.indexOf(pn) >= 0 ? pnames.indexOf(pn) : pnames.findIndex((q) => q.replace(/\W/g, '_') === pn);
      if (i >= 0) {
        if (assigned.has(name)) return { code: `p${i}_`, ty: F, lvalue: true, decl: `var p${i}_: f32 = \${p[${i}]};` };
        return { code: `\${p[${i}]}`, ty: F };
      }
      if (extra.has(pn)) {
        // per-thread state (JWildfire "extra GPU parameters" are instance state)
        return { code: `jwx_${v.name}_${pn}`.replace(/\W/g, '_'), ty: tyOf(extraTypes.get(pn)!) as Ty, lvalue: true, flag: 'state' };
      }
    }
    return null;
  };
  return { bindings: b, resolveMagic };
}

for (let v0 of dump) {
  const jp = !v0.gpuCode || v0.gpuCode.startsWith('/*ERROR') ? javaPorts.get(v0.name) : undefined; // JWildfire's getGPUCode threw (prepost_affine) → the Java port
  if (jp) v0 = { ...v0, gpu: true, gpuCode: jp.gpuCode, preCode: jp.preCode, gpuFunctions: jp.gpuFunctions, extraParams: [...(v0.extraParams ?? []), ...jp.extraParams], resources: 0 };
  const ov = OVERRIDES[v0.name];
  // `varpar->x` (JWildfire's per-instance param/state struct) is spelled `__x` in the dialect we transpile
  let ovCode: string | undefined = ov ? (ov.gpuCode ?? v0.gpuCode ?? '') + (ov.append ?? '') : (v0.gpuCode && v0.gpuCode.includes('varpar->') ? v0.gpuCode : undefined);
  if (ovCode) ovCode = ovCode.replace(/varpar->/g, '__');
  if (ov?.retry && ovCode) {
    // wrap in `for (_try < N) { … break-on-success }`: the snippet sets __doHide=true
    // then `if (distance < 0) { __doHide=false; __px=…; }` — add a break inside that block.
    const close = ovCode.lastIndexOf('}');
    if (close < 0) throw new Error(`retry override: no if-block in ${v0.name}`);
    ovCode = `for (int _try = 0; _try < ${ov.retry}; _try++) {\n${ovCode.slice(0, close)} break; }\n}`;
  }
  // Small C-isms the transpiler does not support, rewritten at source level:
  //   `if (++i > n)` → `i++; if (i > n)`; chained `a=b=c` → two statements;
  //   `while ((k++<10) && cond)` → `while (k < 10 && cond) { k++; …` (k++ evaluated first
  //   in C, so the body sees k+1 and k ends one higher — reproduced by incrementing at loop top)
  {
    const fix: Record<string, (c: string, f: string) => [string, string]> = {
      circleRand: (c, f) => [c.replace(/if \(\+\+iter > maxIter\) \{/, 'iter = iter + 1; if (iter > maxIter) {'), f],
      circleTrans1: (c, f) => [c, f.replace(/if \(\+\+iter > maxIter\) break;/g, 'iter = iter + 1; if (iter > maxIter) break;')],
      // `while ((k++<10) && cond) {…}` → k advances on every test (C semantics), so the
      // post-loop `k >= 10` reject fires after 10 tests: rewrite as an increment-first loop
      mandelbrot: (c, f) => [c
        .replace(/while \(\(k\+\+<10\) && \(([\s\S]*?)\)\) \{\n/, 'while (true) { int ci_ = k; k = k + 1; if (!((ci_ < 10) && ($1))) break;\n')
        .replace(/x1=y1=50000\.0-RANDFLOAT\(\)\*100000;/, 'y1 = 50000.0 - RANDFLOAT() * 100000; x1 = y1;'), f],
      // `t++` inside the do-while condition; the 2050-int permutation and 1024×3 gradient tables are
      // function locals in JWildfire's CUDA (20 KB per thread) → module-scope constants; `float *U = grad3[g]` alias inlined
      dc_perlin: (c, f) => {
        const pm = /int p\[2050\] = \{[\s\S]*?\};/.exec(f)!, gm = /float grad3\[1024\]\[3\] = \{[\s\S]*?\}\s*\};/.exec(f)!;
        f = f.replace(pm[0], '').replace(gm[0], '');
        const s0 = f.indexOf('dc_perlin_simplexNoise3D('), s1 = f.indexOf('__device__', s0 + 1);
        f = f.slice(0, s0) + f.slice(s0, s1).replace(/float \*U;/, '').replace(/U = grad3\[gi\[corner\]\];/, '')
          .replace(/U\[_x_\]/g, 'grad3[gi[corner]][_x_]').replace(/U\[_y_\]/g, 'grad3[gi[corner]][_y_]').replace(/U\[_z_\]/g, 'grad3[gi[corner]][_z_]') + f.slice(s1);
        f = pm[0].replace(/^int p/, 'int dc_perlin_p') + '\n' + gm[0].replace(/^float grad3/, 'float dc_perlin_grad3') + '\n' + f.replace(/\bp\[/g, 'dc_perlin_p[').replace(/\bgrad3\[/g, 'dc_perlin_grad3[');
        return [c.replace(/&& t\+\+ < __dc_perlin_select_bailout\)/, '&& jpostinc(&t) < __dc_perlin_select_bailout)'), f];
      },
      // JWildfire's snippet never assigns ldcs (the Java sets it in init) — and its oldx/oldy state is restored by EXTRA_STATE above
      recurrenceplot: (c, f) => [c.replace(/float ldcs;/, 'float ldcs = 1.0 / (__recurrenceplot_scale == 0.0 ? 10E-6 : __recurrenceplot_scale);'), f],
      dc_circuits: (c, f) => [c, f.replace(/float o,ot2,ot=ot2=1000\.0;/, 'float o; float ot2 = 1000.0; float ot = 1000.0;')],
      sym_ng13: (c, f) => [c.replace(/Mathc Tx\[6\]=/, 'Mathc Tx[8]='), f],
      dc_moebiuslog: (c, f) => [c, f.replace(/\(logf\(length\(U=U\+\.5\)\)\)/, '(logf(length(U + .5)))').replace(/if\(Log==1\.0\)(\s*)U =/, 'if(Log==1.0) { U = U + .5; }\n if(Log==1.0)$1U =')],
    };
    // JWildfire GPU snippets that declare per-instance Java state as locals (a JWildfire GPU bug) → per-thread state
    const EXTRA_STATE: Record<string, string[]> = { recurrenceplot: ['oldx', 'oldy'] };
    if (EXTRA_STATE[v0.name]) {
      const c0 = (ovCode ?? v0.gpuCode ?? '');
      const c1 = EXTRA_STATE[v0.name].reduce((c, n) => c.replace(new RegExp(`(?<![\\w.>])${n}\\b`, 'g'), `varpar->${v0.name}_${n}`), c0.replace(new RegExp(`float ${EXTRA_STATE[v0.name].map((n) => `${n} = 0\\.0`).join(', ')};`), ''));
      if (c1 === c0) throw new Error(`EXTRA_STATE ${v0.name}: pattern not found`);
      ovCode = c1;
      v0 = { ...v0, extraParams: [...(v0.extraParams ?? []), ...EXTRA_STATE[v0.name]], stateful: true };
    }
    const fx = fix[v0.name];
    if (fx) {
      const [c, f] = fx((ovCode ?? v0.gpuCode ?? '').replace(/varpar->/g, '__'), v0.gpuFunctions ?? '');
      ovCode = c;
      if (f !== (v0.gpuFunctions ?? '')) v0 = { ...v0, gpuFunctions: f };
    }
    // textual patches from overrides.ts (must match, else the override is stale)
    if (ov?.patch || ov?.patchFuncs) {
      const apply = (t: string, ps: [string | RegExp, string][]) => {
        for (const [from, to] of ps) {
          const t2 = t.replace(from, to);
          if (t2 === t) throw new Error(`override patch for ${v0.name} did not match: ${from}`);
          t = t2;
        }
        return t;
      };
      if (ov.patch) ovCode = apply((ovCode ?? v0.gpuCode ?? '').replace(/varpar->/g, '__'), ov.patch);
      if (ov.patchFuncs) v0 = { ...v0, gpuFunctions: apply(v0.gpuFunctions ?? '', ov.patchFuncs) };
    }
  }
  // fract_* (buddhabrot fractals): helpers carry per-thread state through a
  // `struct VarPar__jwf_<name> *varpar` pointer. Flatten it: drop the pointer
  // parameter/argument and address state fields as `__<name>_<field>` (which
  // resolve to the per-thread `jwx_` globals via extraParams).
  if (v0.name.startsWith('fract_') && (v0.gpuFunctions ?? '').includes('*varpar')) {
    const flat = (t: string) => t
      .replace(/struct VarPar__jwf_\w+ \*varpar\s*,\s*/g, '')
      .replace(/struct VarPar__jwf_\w+ \*varpar\s*\)/g, ')')
      .replace(/\(varpar\s*,\s*/g, '(')
      .replace(/\(varpar\s*\)/g, '()')
      .replace(/\*\(&varpar->(\w+)\)/g, 'varpar->$1')
      .replace(new RegExp(`varpar->jwf_${v0.name}_`, 'g'), `__${v0.name}_`)
      .replace(/varpar->/g, '__')
      // `while ((i++ < n) && cond) {` → explicit increment-first loop (C: i++ evaluated every test)
      .replace(/while \(\((\w+)\+\+ < (\w+)\) && (.*?)\) \{/g, 'while (true) { int ci_ = $1; $1 = $1 + 1; if (!((ci_ < $2) && $3)) break;');
    let code = flat(ovCode ?? v0.gpuCode ?? '');
    let funcs = flat(v0.gpuFunctions ?? '');
    // helpers are shared module text and cannot see `${p[i]}`: route the params they
    // use through per-thread state (`<param>_c`, copied at the top of the snippet)
    const pnames = new Set(v0.params.map((q) => q.name));
    const usedInFuncs = new Set([...funcs.matchAll(new RegExp(`__${v0.name}_(\\w+)`, 'g'))].map((m) => m[1]).filter((n) => pnames.has(n)));
    const extra = [...(v0.extraParams ?? [])];
    let prelude = '';
    for (const pn of usedInFuncs) {
      funcs = funcs.replace(new RegExp(`__${v0.name}_${pn}(?![A-Za-z0-9_])`, 'g'), `__${v0.name}_${pn}_c`);
      extra.push(`${pn}_c`);
      prelude += `__${v0.name}_${pn}_c = __${v0.name}_${pn};\n`;
    }
    ovCode = prelude + code;
    v0 = { ...v0, gpuFunctions: funcs, extraParams: extra };
  }
  // Param-name typos in JWildfire GPU snippets (snippet uses a name the class doesn't declare)
  {
    const aliases: Record<string, Record<string, string>> = {
      post_circlecrop: { scatterarea: 'scatter_area' },
      standing_wave: { freqx: 'freq_x', freqy: 'freq_y' },
    };
    const al = aliases[v0.name];
    if (al) {
      let code = (ovCode ?? v0.gpuCode ?? '').replace(/varpar->/g, '__');
      for (const [bad, good] of Object.entries(al)) code = code.replace(new RegExp(`__${v0.name}_${bad}(?![A-Za-z0-9_])`, 'g'), `__${v0.name}_${good}`);
      ovCode = code;
    }
  }
  // DC_BaseFunc family (dc_* shader-art variations): JWildfire's GPU snippets sample
  // x,y = 2·rnd−1 where the CPU samples rnd−0.5, and leave the outer `z = 0.5` where
  // the CPU sets z = greyscale(colour) (the GPU re-declares z in a nested block).
  {
    let code = ovCode ?? v0.gpuCode ?? '';
    if (/float z\s*=\s*0\.5;/.test(code) && /_getRGBColor\s*\(/.test(code)) {
      // sampling: only classes that inherit DC_BaseFunc.transform (data/dc-base.json);
      // the ones with their own transform() really do use 2·rnd−1 on the CPU too
      if (DC_BASE.inherit.includes(v0.name) || DC_BASE.half.includes(v0.name)) code = code.replace(/2\.0\*RANDFLOAT\(\)-1\.0/g, '(RANDFLOAT()-0.5)');
      code = code.replace(/(color\s*=\s*\w+_getRGBColor\s*\([^;]*\);)/, '$1 { int3 zc_ = dbl2int(color); z = greyscale((float)zc_.x, (float)zc_.y, (float)zc_.z); }');
      ovCode = code;
    }
  }
  // Java setParameter() clamps (data/param-clamps.json, extracted by extract-clamps.py)
  // plus any per-override clamps: wrap every read of the param in the same clamp.
  const clamps = { ...(PARAM_CLAMPS[v0.name] ?? {}), ...(ov?.clampParams ?? {}) };
  if (Object.keys(clamps).length && (ovCode ?? v0.gpuCode)) {
    ovCode = (ovCode ?? v0.gpuCode ?? '').replace(/varpar->/g, '__');
    for (const [pn, [lo, hi]] of Object.entries(clamps)) {
      // reads only: skip assignments to the param (some snippets write to it)
      const re = new RegExp(`(?<![A-Za-z0-9_])__${v0.name}_${pn}(?![A-Za-z0-9_])(?!\\s*(?:[-+*/]?=)(?!=))`, 'g');
      ovCode = ovCode.replace(re, `(fminf(fmaxf(__${v0.name}_${pn}, ${lo.toFixed(4)}f), ${hi.toFixed(4)}f))`);
    }
  }
  // Java setParameter() int casts (data/param-ints.json): the CPU truncates/rounds the
  // value at set time; the GPU snippet reads the raw float — wrap reads the same way.
  const ints = PARAM_INTS[v0.name] ?? {};
  if (Object.keys(ints).length && (ovCode ?? v0.gpuCode)) {
    ovCode = (ovCode ?? v0.gpuCode ?? '').replace(/varpar->/g, '__');
    for (const [pn, how] of Object.entries(ints)) {
      const re = new RegExp(`(?<![A-Za-z0-9_])__${v0.name}_${pn}(?![A-Za-z0-9_])(?!\\s*(?:[-+*/]?=)(?!=))`, 'g');
      ovCode = ovCode.replace(re, how === 'round' ? `((int)lroundf(__${v0.name}_${pn}))` : `((int)(__${v0.name}_${pn}))`);
    }
  }
  // GPU snippets often test a *double* flag param with lroundf(p) > 0 / == 1 where the
  // Java tests the raw double (p > 0). Only for params the dump types as float.
  {
    let code = ovCode ?? v0.gpuCode ?? '';
    const floatParams = v0.params.filter((q) => !q.int).map((q) => q.name);
    let changed = false;
    for (const pn of floatParams) {
      const re = new RegExp(`lroundf\\(\\s*(\\(?\\(?\\(?)__${v0.name}_${pn}(?![A-Za-z0-9_])`, 'g');
      if (re.test(code)) { code = code.replace(new RegExp(`lroundf\\(\\s*__${v0.name}_${pn}\\s*\\)`, 'g'), `(__${v0.name}_${pn})`); changed = true; }
    }
    if (changed) ovCode = code;
  }
  const v: DumpVar = ov || ovCode !== undefined ? { ...v0, gpuCode: ovCode ?? v0.gpuCode, gpuFunctions: ov?.gpuFunctions ?? v0.gpuFunctions } : v0;
  if (v.error) { report.push({ name: v.name, status: 'error', reason: 'instantiation: ' + v.error }); continue; }
  if (EXCLUDE[v.name]) { report.push({ name: v.name, status: 'skip', reason: EXCLUDE[v.name] }); continue; }
  if (!v.gpu || !v.gpuCode) { report.push({ name: v.name, status: 'skip', reason: 'no GPU code in JWildfire' }); continue; }
  if (v.gpuCode.startsWith('/*ERROR')) { report.push({ name: v.name, status: 'skip', reason: 'JWildfire getGPUCode threw' }); continue; }
  if (v.resources > 0) { report.push({ name: v.name, status: 'skip', reason: 'needs resources (images/text)' }); continue; }
  const libs = [{ name: 'kernel', source: kernelLib }];
  if (v.gpuFunctions && v.gpuFunctions.trim() && !v.gpuFunctions.startsWith('/*ERROR')) libs.push({ name: v.name, source: v.gpuFunctions });
  try {
    const r = transpileSnippet(v.gpuCode, libs, bindingsFor(v));
    // prepost variations (java2cu preCode): the inverse runs as a pre step — transpiled as a second snippet sharing the helpers
    const rp = v.preCode ? transpileSnippet(v.preCode, libs, bindingsFor({ ...v, gpuCode: v.preCode })) : null;
    if (rp) {
      for (const f of rp.flags) if (!r.flags.includes(f)) r.flags.push(f);
      for (const n of rp.functionNames) if (!r.functionNames.includes(n)) { r.functions = (r.functions ? r.functions + '\n\n' : '') + extractItem(rp.functions, n); r.functionNames.push(n); }
    }
    const bad = r.flags.find((f) => UNSUPPORTED_FLAGS[f]);
    if (bad) { report.push({ name: v.name, status: 'skip', reason: UNSUPPORTED_FLAGS[bad], flags: r.flags }); continue; }
    if (v.stateful) r.flags.push('stateful');
    if (v.types.includes('VARTYPE_3D')) r.flags.push('3d');
    if (v.types.includes('VARTYPE_DC')) r.flags.push('dc');
    // per-thread state for JWildfire "extra GPU params"
    let funcs = r.functions, code = r.code, preCode = rp?.code;
    const funcNames: string[] = [];
    for (const pe of v.extraParams ?? []) {
      const [pn, pt] = pe.split(':');
      const wty = pt === 'float2' ? 'vec2f' : pt === 'float3' ? 'vec3f' : pt === 'int' ? 'i32' : 'f32';
      const init = pt === 'float2' ? 'vec2f(0.0)' : pt === 'float3' ? 'vec3f(0.0)' : pt === 'int' ? '0' : '0.0';
      const gname = `jwx_${v.name}_${pn}`.replace(/\W/g, '_');
      if (new RegExp(`\\b${gname}\\b`).test(code + (preCode ?? '')) || new RegExp(`\\b${gname}\\b`).test(funcs ?? '')) {
        funcs = (funcs ? funcs + '\n\n' : '') + `var<private> ${gname}: ${wty} = ${init};`;
        funcNames.push(gname);
        fnRegistry.set(gname, `var<private> ${gname}: ${wty} = ${init};`);
      }
    }
    for (const n of r.functionNames) {
      const text = extractItem(funcs, n);
      const prev = fnRegistry.get(n);
      if (prev !== undefined && prev !== text) {
        const nn = `${n}_${v.name.replace(/\W/g, '_')}`;
        const re = new RegExp(`\\b${n}\\b`, 'g');
        funcs = funcs.replace(re, nn);
        code = code.replace(re, nn);
        if (preCode) preCode = preCode.replace(re, nn);
        fnRegistry.set(nn, extractItem(funcs, nn));
        funcNames.push(nn);
      } else {
        if (prev === undefined) fnRegistry.set(n, text);
        funcNames.push(n);
      }
    }
    const params = v.params.map((p) => ({ name: p.name, def: DEFAULT_OVERRIDES[v.name]?.[p.name] ?? (p.def ?? 0) }));
    entries.push({ name: v.name, params, code, preCode, funcs, funcNames, priority: v.priority, types: v.types, flags: [...new Set(r.flags)].sort() });
    report.push({ name: v.name, status: 'ok', flags: r.flags });
  } catch (err) {
    if (!(err instanceof TranspileError)) throw err;
    report.push({ name: v.name, status: 'error', reason: (err as Error).message });
  }
}

/** Pulls the text of one module-scope item (struct/fn/const/var) out of a functions blob. */
function extractItem(funcs: string, name: string): string {
  const re = new RegExp(`(^|\\n)((?:struct|fn|const|var<private>) ${name}\\b[\\s\\S]*?)(?=\\n\\n|$)`);
  const m = re.exec(funcs);
  return m ? m[2] : '';
}

// ---- write TS ----
const esc = (s: string) => s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{(?!w\}|p\[\d+\]\}|A\(\d\)\})/g, '\\${');
let ts = `// AUTO-GENERATED by scripts/jwf-port/gen.ts — do not edit by hand.
// Variations ported from JWildfire (https://github.com/thargor6/JWildfire, LGPL-2.1,
// (c) Andreas Maschke and contributors) by transpiling each variation's CUDA GPU
// snippet to WGSL and verifying it against JWildfire's Java implementation.
// ${entries.filter((e) => verified.has(e.name)).length} verified of ${entries.length} transpiled (${dump.length} in JWildfire).
//
// Snippet scope: t (input point, var), z_ (input z), r2, r, th = atan2(x,y), ph = atan2(y,x),
// v (output accumulator), pz_ (output z), rs (rng), cp (palette coord ptr), hd (hide-flag ptr).
// Flags: z = reads/writes z, hide = uses hide flag, dc = writes color,
// state = keeps per-thread private state, 3d = JWildfire types it 3D, affine = reads xform coefs.

import type { VariationDef } from './variations';

export interface JwfVariationDef extends VariationDef {
  /** True when the oracle harness confirmed this port matches JWildfire's CPU output (scripts/jwf-port/verified.json). */
  verified: boolean;
  /** JWildfire priority: -1 pre (mutates the input point), 0 normal, 1 post (mutates the output). */
  priority: number;
  /** JWildfire "prepost" variations: this snippet runs first in the stage (priority -2), rewriting the input point (the inverse), while \`code\` runs last (priority 2) on the output. */
  preCode?: (w: string, p: string[], A: (i: number) => string) => string;
  /** Module-scope WGSL (helper fns/consts) the snippet needs; codegen dedupes by name. */
  funcs?: string;
  funcNames?: string[];
  flags: string[];
  types: string[];
}

export const JWF_VARIATIONS: Record<string, JwfVariationDef> = {
`;
let tsU = `// AUTO-GENERATED by scripts/jwf-port/gen.ts — do not edit by hand.
// JWildfire ports that did NOT pass the oracle harness (see scripts/jwf-port/verified.json);
// kept out of the app registry, loaded only by the dev harness (src/dev/varTest.ts).

import type { JwfVariationDef } from './variations.jwf';

export const JWF_VARIATIONS_UNVERIFIED: Record<string, JwfVariationDef> = {
`;
for (const e of entries) {
  const codeBody = e.code.replace(/^  /gm, '').trimEnd();
  const usesA = /\$\{A\(/.test(codeBody);
  const usesP = /\$\{p\[/.test(codeBody);
  const usesW = /\$\{w\}/.test(codeBody);
  const args = `(${usesW ? 'w' : '_w'}, ${usesP ? 'p' : '_p'}${usesA ? ', A' : ''})`;
  let item = `  ${JSON.stringify(e.name)}: {\n`;
  item += `    params: [${e.params.map((p) => `{ name: ${JSON.stringify(p.name)}, def: ${p.def} }`).join(', ')}],\n`;
  item += `    verified: ${verified.has(e.name)}, priority: ${e.priority}, flags: ${JSON.stringify(e.flags)}, types: ${JSON.stringify(e.types.filter((t) => !t.startsWith('VARTYPE_SUPPORT')).map((t) => t.replace('VARTYPE_', '')))},\n`;
  if (e.funcs.trim()) {
    item += `    funcNames: ${JSON.stringify(e.funcNames)},\n`;
    item += `    funcs: \`${esc(e.funcs)}\`,\n`;
  }
  item += `    code: ${args} => \`{\n${esc(codeBody).split('\n').map((l) => l.trimEnd()).join('\n')}\n}\`,\n`;
  if (e.preCode) {
    const preBody = e.preCode.replace(/^  /gm, '').trimEnd();
    const pargs = `(${/\$\{w\}/.test(preBody) ? 'w' : '_w'}, ${/\$\{p\[/.test(preBody) ? 'p' : '_p'}${/\$\{A\(/.test(preBody) ? ', A' : ''})`;
    item += `    preCode: ${pargs} => \`{\n${esc(preBody).split('\n').map((l) => l.trimEnd()).join('\n')}\n}\`,\n`;
  }
  item += `  },\n`;
  if (verified.has(e.name)) ts += item; else tsU += item;
}
ts += `};\n`;
tsU += `};\n`;
writeFileSync(outFile, ts);
writeFileSync(outFileUnverified, tsU);
// the deliberately-unported list (data/unportable.json) → src/core/variations.unportable.ts, so the importer can say why a variation is skipped
{
  const up = JSON.parse(readFileSync(join(here, 'data', 'unportable.json'), 'utf8')) as { categories: Record<string, string>; variations: Record<string, string> };
  const SHORT: Record<string, string> = { 'user-code': 'runs user code / a formula at render time', 'external-content': 'renders external content (flame, image, mesh, SVG, text)', 'point-set': 'CPU-built point set', engine: 'JWildfire pre+post inverse pair / nested variation', 'resource-params': 'colour ressources', 'not-yet': 'not ported yet' };
  const bad = Object.keys(up.variations).filter((n) => entries.some((e) => e.name === n));
  if (bad.length) throw new Error(`unportable.json lists ported variations: ${bad.join(' ')}`);
  const missing = dump.filter((d) => !entries.some((e) => e.name === d.name) && !up.variations[d.name]).map((d) => d.name);
  if (missing.length) console.warn(`unportable.json is missing: ${missing.join(' ')}`);
  let tsN = '// AUTO-GENERATED by scripts/jwf-port/gen.ts from data/unportable.json — do not edit by hand.\n// JWildfire variations WilderFire deliberately does not implement, with the reason shown by the importer.\n\n';
  tsN += 'export const UNPORTABLE: Record<string, string> = {\n' + Object.entries(up.variations).map(([n, c]) => `  ${JSON.stringify(n)}: ${JSON.stringify(SHORT[c] ?? c)},`).join('\n') + '\n};\n';
  writeFileSync(join(here, '..', '..', 'src', 'core', 'variations.unportable.ts'), tsN);
}

// ---- summary ----
const ok = report.filter((r) => r.status === 'ok').length;
const skip = report.filter((r) => r.status === 'skip');
const err = report.filter((r) => r.status === 'error');
console.log(`ok ${ok} (verified ${entries.filter((e) => verified.has(e.name)).length})  skip ${skip.length}  error ${err.length}  → ${outFile}`);
const byReason = new Map<string, string[]>();
for (const r of [...skip, ...err]) {
  const key = (r.status === 'error' ? 'ERR ' : 'SKIP ') + (r.reason ?? '').replace(/'[^']*'/g, "'…'").replace(/\(line \d+\)/, '').replace(/at ('.*?')/, '').replace(/unknown identifier \w+/, 'unknown identifier X').replace(/unknown function \w+/, 'unknown function X').trim();
  byReason.set(key, [...(byReason.get(key) ?? []), r.name]);
}
for (const [k, names] of [...byReason.entries()].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`${String(names.length).padStart(4)}  ${k}  [${names.slice(0, 6).join(' ')}${names.length > 6 ? ' …' : ''}]`);
}
const flagCounts = new Map<string, number>();
for (const e of entries) for (const f of e.flags) flagCounts.set(f, (flagCounts.get(f) ?? 0) + 1);
console.log('flags:', [...flagCounts.entries()].map(([k, n]) => `${k}=${n}`).join(' '));
if (process.argv.includes('--report')) writeFileSync(join(here, 'report.json'), JSON.stringify(report, null, 1));
