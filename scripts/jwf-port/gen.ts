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
import { transpileSnippet, TranspileError, type Binding, type Env } from './cwgsl.ts';
import { OVERRIDES } from './overrides.ts';
import clampsJson from './data/param-clamps.json' with { type: 'json' };
const PARAM_CLAMPS = clampsJson as Record<string, Record<string, [number, number]>>;
import dcBaseJson from './data/dc-base.json' with { type: 'json' };
const DC_BASE = dcBaseJson as { inherit: string[]; own2: string[]; ownOther: string[] };

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, 'data');
const outFile = join(here, '..', '..', 'src', 'core', 'variations.jwf.ts');
const outFileUnverified = join(here, '..', '..', 'src', 'core', 'variations.jwf.unverified.ts');

interface DumpParam { name: string; def: number | null; int: boolean; stable: boolean }
interface DumpVar {
  name: string; cls: string; priority: number; types: string[]; params: DumpParam[];
  altNames?: string[]; resources: number; defaultsStable: boolean; gpu: boolean;
  gpuCode?: string; gpuFunctions?: string; stateful?: boolean; extraParams?: string[]; error?: string;
}

const dump: DumpVar[] = readFileSync(join(dataDir, 'jwf-variations.jsonl'), 'utf8')
  .trim().split('\n').map((l) => JSON.parse(l));
const kernelLib = readFileSync(join(dataDir, 'kernel-lib.cu'), 'utf8');
// Oracle verdicts (written by the in-browser harness); absent → nothing is verified.
let verified = new Set<string>();
try { verified = new Set<string>(JSON.parse(readFileSync(join(here, 'verified.json'), 'utf8')).jwf); } catch { /* first run */ }
/** Ports the oracle cannot confirm because JWildfire's CPU code is itself broken, but whose GPU intent is right. */
const FORCE_VERIFIED: Record<string, string> = {
  pre_flatten: 'CPU writes pVarTP.z inside a pre-variation (a no-op); the GPU flattens the affine z, which is the intent',
  cut_bricks: 'matches at default seed; the seed param drives java.util.Random on the CPU and nothing on the GPU (same as JWildfire GPU)',
};
for (const n of Object.keys(FORCE_VERIFIED)) verified.add(n);

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
  name: string; params: { name: string; def: number }[]; code: string; funcs: string; funcNames: string[];
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
  const extra = new Set(v.extraParams ?? []);
  const resolveMagic = (name: string): Binding | null => {
    if (name === '__' + v.name) {
      if (assigned.has(name)) return { code: 'w_', ty: F, lvalue: true, decl: 'var w_: f32 = ${w};' };
      return { code: '${w}', ty: F };
    }
    const pre = '__' + v.name + '_';
    if (name.startsWith(pre)) {
      const pn = name.slice(pre.length);
      const i = pnames.indexOf(pn);
      if (i >= 0) {
        if (assigned.has(name)) return { code: `p${i}_`, ty: F, lvalue: true, decl: `var p${i}_: f32 = \${p[${i}]};` };
        return { code: `\${p[${i}]}`, ty: F };
      }
      if (extra.has(pn)) {
        // per-thread state (JWildfire "extra GPU parameters" are instance state)
        return { code: `jwx_${v.name}_${pn}`.replace(/\W/g, '_'), ty: F, lvalue: true, flag: 'state' };
      }
    }
    return null;
  };
  return { bindings: b, resolveMagic };
}

for (let v0 of dump) {
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
      mandelbrot: (c, f) => [c
        .replace(/while \(\(k\+\+<10\) && \(/, 'while ((k < 10) && (')
        .replace(/x1=y1=50000\.0-RANDFLOAT\(\)\*100000;/, 'y1 = 50000.0 - RANDFLOAT() * 100000; x1 = y1;'), f],
      dc_circuits: (c, f) => [c, f.replace(/float o,ot2,ot=ot2=1000\.0;/, 'float o; float ot2 = 1000.0; float ot = 1000.0;')],
      sym_ng13: (c, f) => [c.replace(/Mathc Tx\[6\]=/, 'Mathc Tx[8]='), f],
      dc_moebiuslog: (c, f) => [c, f.replace(/\(logf\(length\(U=U\+\.5\)\)\)/, '(logf(length(U + .5)))').replace(/if\(Log==1\.0\)(\s*)U =/, 'if(Log==1.0) { U = U + .5; }\n if(Log==1.0)$1U =')],
    };
    const fx = fix[v0.name];
    if (fx) {
      const [c, f] = fx((ovCode ?? v0.gpuCode ?? '').replace(/varpar->/g, '__'), v0.gpuFunctions ?? '');
      ovCode = c;
      if (f !== (v0.gpuFunctions ?? '')) v0 = { ...v0, gpuFunctions: f };
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
    if (/float z\s*=\s*0\.5;/.test(code) && /_getRGBColor\(/.test(code)) {
      // sampling: only classes that inherit DC_BaseFunc.transform (data/dc-base.json);
      // the ones with their own transform() really do use 2·rnd−1 on the CPU too
      if (DC_BASE.inherit.includes(v0.name)) code = code.replace(/2\.0\*RANDFLOAT\(\)-1\.0/g, '(RANDFLOAT()-0.5)');
      code = code.replace(/(color\s*=\s*\w+_getRGBColor\([^;]*\);)/, '$1 { int3 zc_ = dbl2int(color); z = greyscale((float)zc_.x, (float)zc_.y, (float)zc_.z); }');
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
    const bad = r.flags.find((f) => UNSUPPORTED_FLAGS[f]);
    if (bad) { report.push({ name: v.name, status: 'skip', reason: UNSUPPORTED_FLAGS[bad], flags: r.flags }); continue; }
    if (v.stateful) r.flags.push('stateful');
    if (v.types.includes('VARTYPE_3D')) r.flags.push('3d');
    if (v.types.includes('VARTYPE_DC')) r.flags.push('dc');
    // per-thread state for JWildfire "extra GPU params"
    let funcs = r.functions, code = r.code;
    const funcNames: string[] = [];
    for (const pn of v.extraParams ?? []) {
      const gname = `jwx_${v.name}_${pn}`.replace(/\W/g, '_');
      if (new RegExp(`\\b${gname}\\b`).test(code) || new RegExp(`\\b${gname}\\b`).test(funcs ?? '')) {
        funcs = (funcs ? funcs + '\n\n' : '') + `var<private> ${gname}: f32 = 0.0;`;
        funcNames.push(gname);
        fnRegistry.set(gname, `var<private> ${gname}: f32 = 0.0;`);
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
        fnRegistry.set(nn, extractItem(funcs, nn));
        funcNames.push(nn);
      } else {
        if (prev === undefined) fnRegistry.set(n, text);
        funcNames.push(n);
      }
    }
    const params = v.params.map((p) => ({ name: p.name, def: DEFAULT_OVERRIDES[v.name]?.[p.name] ?? (p.def ?? 0) }));
    entries.push({ name: v.name, params, code, funcs, funcNames, priority: v.priority, types: v.types, flags: [...new Set(r.flags)].sort() });
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
  item += `  },\n`;
  if (verified.has(e.name)) ts += item; else tsU += item;
}
ts += `};\n`;
tsU += `};\n`;
writeFileSync(outFile, ts);
writeFileSync(outFileUnverified, tsU);

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
