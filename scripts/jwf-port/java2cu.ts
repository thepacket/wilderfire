// Java → CUDA-dialect pre-processor for JWildfire variations that have no GPU snippet.
//
// JWildfire's variation classes are small numeric Java: `transform()` reads
// pAffineTP.x/y/z, writes pVarTP.x/y/z, uses MathLib statics (sin, sqr, fabs, …),
// param fields set by setParameter(), precalc fields set in init(), and a few
// private helper methods. That subset is close enough to the CUDA dialect of
// JWildfire's own GPU snippets that a textual conversion + the existing
// CUDA→WGSL transpiler (cwgsl.ts) can port most of them mechanically; the oracle
// harness then decides which ones actually match.
//
// Output: data/jwf-java-ports.jsonl — one line per variation with the same shape
// gen.ts reads from the JWildfire dump ({name, gpuCode, gpuFunctions, extraParams,
// note}); gen.ts uses it for dump entries that have no gpuCode.
//
//   node scripts/jwf-port/java2cu.ts [--jwf <path to JWildfire src root>] [names…]
//
// Conventions produced (see gen.ts for how they bind):
//   __x/__y/__z, __px/__py/__pz, __pal, __doHide, __<name> (amount),
//   __<name>_<param>, RANDFLOAT(); per-instance state fields (assigned in
//   transform) become `varpar-><name>_<field>` extra params initialised on first
//   call from their Java initialiser / init() value.

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
let jwfRoot = process.env.JWF ?? '/private/tmp/claude-501/-Users-andrepaquette-Projects-wilderfire/34fc96fe-85dc-4506-a378-926d42cbf78f/scratchpad/jwf';
const only: string[] = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--jwf') jwfRoot = args[++i];
  else only.push(args[i]);
}
const varDir = path.join(jwfRoot, 'src/org/jwildfire/create/tina/variation');

export interface DumpVar { name: string; params: { name: string; def: number; int: boolean }[]; gpuCode?: string; gpu?: boolean; resources?: number; priority?: number }
const dump: DumpVar[] = fs.readFileSync(path.join(here, 'data/jwf-variations.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const dumpByName = new Map(dump.map((d) => [d.name, d]));

// name → java file (by getName())
// The JWildfire source tree is only needed by main() and by MERGE_PARENTS lookups —
// resolved lazily so the module can be imported as a library (tests) without it.
let filesCache: string[] | null = null;
function javaFiles(): string[] {
  if (filesCache) return filesCache;
  if (!fs.existsSync(varDir)) throw new Error(`JWildfire sources not found at ${varDir} (use --jwf or JWF=)`);
  return (filesCache = walk(varDir).filter((f) => f.endsWith('.java')));
}
function fileIndex(): Map<string, string> {
  const fileByName = new Map<string, string>();
  for (const f of javaFiles()) {
    const s = fs.readFileSync(f, 'latin1');
    const m = /public String getName\(\)\s*\{\s*return\s+"([A-Za-z0-9_]+)"\s*;/.exec(s);
    if (m && (!fileByName.has(m[1]) || /public\s+abstract\s+class/.test(fs.readFileSync(fileByName.get(m[1])!, 'latin1')))) fileByName.set(m[1], f); // a concrete class beats an abstract one with the same name
  }
  return fileByName;
}
function walk(d: string): string[] { return fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]); }

const targets = dump.filter((d) => (!d.gpuCode || d.gpuCode.startsWith('/*ERROR')) && (!only.length || only.includes(d.name)));

// ---------------------------------------------------------------- utilities
function stripComments(s: string): string {
  return s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}
/** body between the '{' at/after `from` and its matching '}' (exclusive) */
function braceBody(s: string, from: number): { body: string; end: number } | null {
  const open = s.indexOf('{', from);
  if (open < 0) return null;
  let depth = 0;
  for (let i = open; i < s.length; i++) {
    const c = s[i];
    if (c === '{') depth++;
    else if (c === '}') { depth--; if (depth === 0) return { body: s.slice(open + 1, i), end: i + 1 }; }
    else if (c === '"') { i++; while (i < s.length && s[i] !== '"') { if (s[i] === '\\') i++; i++; } }
    else if (c === "'") { i++; while (i < s.length && s[i] !== "'") { if (s[i] === '\\') i++; i++; } }
  }
  return null;
}
const KNOWN_METHODS = new Set(['transform', 'init', 'initOnce', 'getName', 'getParameterNames', 'getParameterValues', 'setParameter', 'getVariationTypes',
  'dynamicParameterExpansion', 'randomize', 'mutate', 'getGPUCode', 'getGPUFunctions', 'getParameterAlternativeNames', 'getRessourceNames', 'getRessourceValues',
  'setRessource', 'getRessourceType', 'getPriority', 'isStateful', 'getGPUExtraParameterNames', 'validate', 'getInitialParameterValue', 'setSubflame',
  'preprocess', 'getSupportedTransformationTypes', 'getSourceCode', 'getFuncSource', 'getRandomFuncName', 'getPreFuncType', 'getPostFuncType']);

interface Method { name: string; ret: string; params: string; body: string; static: boolean }
interface Field { type: string; name: string; init: string | null; array: number | null }

// ---------------------------------------------------------------- plain-data inner classes → structs
// `private static class Point { double x, y; }` (optionally with a constructor that only assigns
// fields, and setter methods `void setXy(double x, double y)`) → CUDA `struct <Var>_Point { float x; float y; };`
// plus a maker function for the constructor; setters become helper functions taking a pointer.
interface Pod { name: string; fields: { type: string; name: string }[]; ctor: { params: string[]; assigns: [string, string][] } | null; setters: { name: string; params: string; body: string }[]; builtin?: { cuda: string; lib: string; methods: Record<string, string>; ctorFn?: string } }
// library classes used like plain data: struct + functions come from LIB, methods map to `fn(&obj, …)`
const BUILTIN_PODS: Pod[] = [
  { name: 'MarsagliaRandomGenerator', fields: [{ type: 'int', name: 'u' }, { type: 'int', name: 'v' }], ctor: null, setters: [], builtin: { cuda: 'jmrg_', lib: 'jmrg', methods: { randomize: 'jmrg_randomize', random: 'jmrg_random' } } },
  { name: 'XYZPoint', fields: [{ type: 'float', name: 'x' }, { type: 'float', name: 'y' }, { type: 'float', name: 'z' }, { type: 'float', name: 'color' }], ctor: null, setters: [], builtin: { cuda: 'jxyz_', lib: 'jxyz', methods: { assign: 'jxyz_assign' } } },
  { name: 'DoubleWrapperWF', fields: [{ type: 'float', name: 'value' }], ctor: null, setters: [], builtin: { cuda: 'jdw_', lib: 'jdw', methods: {} } },
  // JWildfire's org.jwildfire.base.mathlib.Complex (mutating methods; per_fix/save_* carried): jcx_ (LIB, line-by-line port)
  { name: 'Complex', fields: [{ type: 'float', name: 're' }, { type: 'float', name: 'im' }, { type: 'float', name: 'save_re' }, { type: 'float', name: 'save_im' }, { type: 'float', name: 'per_fix' }], ctor: { params: ['float re', 'float im'], assigns: [] }, setters: [], builtin: { cuda: 'jcx_', lib: 'jcx', ctorFn: 'jcx_make', methods: Object.fromEntries(['One', 'ImOne', 'Zero', 'Copy', 'Flip', 'Conj', 'Neg', 'Sig', 'Sig2', 'Mag2', 'Mag2eps', 'MagInv', 'Save', 'Restore', 'Switch', 'Keep', 'Recall', 'NextPow', 'Sqr', 'Recip', 'Scale', 'Mul', 'Div', 'DivR', 'Add', 'AMean', 'RootMeanS', 'GMean', 'Heronian', 'HMean', 'Sub', 'SubR', 'Inc', 'Dec', 'PerFix', 'Pow', 'Radius', 'Arg', 'ToP', 'UnP', 'Norm', 'Exp', 'SinH', 'Sin', 'CosH', 'Cos', 'Sqrt', 'Log', 'LMean', 'AtanH', 'AsinH', 'AcosH', 'AcotH', 'AsecH', 'AcosecH', 'Atan', 'Asin', 'Acos', 'CPow'].map((m) => [m, 'jcx_' + m])) } },
  { name: 'Random', fields: [{ type: 'int', name: 's0' }, { type: 'int', name: 's1' }, { type: 'int', name: 's2' }], ctor: { params: ['int seed'], assigns: [] }, setters: [], builtin: { cuda: 'jrand_', lib: 'jrand', methods: { nextDouble: 'jrand_nextDouble', nextInt: 'jrand_nextInt', setSeed: 'jrand_setSeed', nextFloat: 'jrand_nextDouble' }, ctorFn: 'jrand_make' } },
];
const POD_TYPE: Record<string, string> = { double: 'float', float: 'float', long: 'int', int: 'int', short: 'int', boolean: 'bool' };
export function parsePod(name: string, body: string): Pod | null {
  const fields: Pod['fields'] = [];
  const setters: Pod['setters'] = [];
  let ctor: Pod['ctor'] = null;
  // methods (constructors, setters, anything else → not plain data)
  const mre = /(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:([\w<>\[\]]+)\s+)?(\w+)\s*\(([^)]*)\)\s*\{/g;
  let m: RegExpExecArray | null;
  const ranges: [number, number][] = [];
  while ((m = mre.exec(body))) {
    const b = braceBody(body, m.index + m[0].length - 1);
    if (!b) { if (process.env.PODDBG) console.log('nobody', m[0]); return null; }
    ranges.push([m.index, b.end]);
    mre.lastIndex = b.end;
    const ret = m[1] && /^(public|private|protected|static|final)$/.test(m[1]) ? undefined : m[1], mn = m[2], params = m[3].trim(), mb = b.body.trim();
    if (mn === name && !ret) {
      // constructor: only `this.f = expr;` / `f = expr;` statements
      const assigns: [string, string][] = [];
      for (const st of mb.split(';').map((x) => x.trim()).filter(Boolean)) {
        const am = /^(?:this\.)?(\w+)\s*=\s*([\s\S]+)$/.exec(st);
        if (!am) { if (process.env.PODDBG) console.log('ctor stmt', st); return null; }
        assigns.push([am[1], am[2]]);
      }
      if (!params && !assigns.length) continue; // empty default constructor
      if (ctor) { if (process.env.PODDBG) console.log('overload'); return null; } // overloaded constructors: not handled
      ctor = { params: params ? params.split(',').map((x) => x.trim()) : [], assigns };
      continue;
    }
    if (ret === 'void' && /^set/.test(mn)) { setters.push({ name: mn, params, body: mb }); continue; }
    if (process.env.PODDBG) console.log('method', ret, mn);
    return null;
  }
  const outside = (pos: number) => !ranges.some(([a, b]) => pos >= a && pos < b);
  for (const fm of body.matchAll(/(?:public|private|protected)?\s*(?:final\s+)?(double|float|int|long|short|boolean)\s+([\w\s,]+?)\s*;/g)) {
    if (!outside(fm.index!)) continue;
    for (const part of fm[2].split(',')) { const nm = part.trim(); if (/^\w+$/.test(nm)) fields.push({ type: POD_TYPE[fm[1]], name: nm }); }
  }
  if (!fields.length) return null;
  // anything else in the body (static final serialVersionUID is fine)
  const stripped = ranges.reduceRight((acc, [a, b]) => acc.slice(0, a) + acc.slice(b), body)
    .replace(/(?:private|public|protected)?\s*static\s+final\s+long\s+serialVersionUID\s*=\s*[^;]+;/g, '');
  if (/\bnew\b|\[\s*\]|String\b|Object\b/.test(stripped)) { if (process.env.PODDBG) console.log('stripped', stripped); return null; }
  return { name, fields, ctor, setters };
}

// ---------------------------------------------------------------- Java class model
export function parseClass(src0: string) {
  const src = stripComments(src0);
  const cls = /public\s+(?:abstract\s+)?class\s+(\w+)\s+extends\s+([\w.]+)/.exec(src);
  if (!cls) throw new Error('no class');
  const body = braceBody(src, cls.index)!.body;
  // strip nested classes/enums (mark ranges)
  const methods: Method[] = [];
  const fields: Field[] = [];
  const consts = new Map<string, string>(); // PARAM_X → "x"
  const numConsts = new Map<string, string>(); // static final numeric constants (raw Java initialiser)
  let i = 0;
  const skipRanges: [number, number][] = [];
  // nested class / enum / interface ranges first: their methods/fields are not the variation's
  const pods: Pod[] = [];
  const nestedRanges: [number, number][] = [];
  for (const cm of body.matchAll(/(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(class|enum|interface)\s+(\w+)([^{]*)\{/g)) {
    const b = braceBody(body, cm.index! + cm[0].length - 1);
    if (!b) continue;
    nestedRanges.push([cm.index!, b.end]);
    if (cm[1] === 'class' && !/\bextends\b/.test(cm[3])) { const pod = parsePod(cm[2], b.body); if (pod) pods.push(pod); }
  }
  skipRanges.push(...nestedRanges);
  const inNested = (pos: number) => nestedRanges.some(([a, b]) => pos >= a && pos < b);
  const mre = /(?:public|private|protected)?\s*(static\s+)?(?:final\s+)?(?:synchronized\s+)?([\w<>\[\],. ]+?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws [\w., ]+)?\s*\{/g;
  let m: RegExpExecArray | null;
  while ((m = mre.exec(body))) {
    if (['if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'else'].includes(m[3])) continue;
    if (inNested(m.index)) continue;
    if (m[3] === cls[1]) { const b0 = braceBody(body, m.index + m[0].length - 1); if (b0) { skipRanges.push([m.index, b0.end]); mre.lastIndex = b0.end; } continue; } // constructor
    const b = braceBody(body, m.index + m[0].length - 1);
    if (!b) break;
    let ret = m[2].trim(); let isStatic = !!m[1];
    for (;;) { const r2 = ret.replace(/^(?:public|private|protected|static|final|synchronized|abstract)\s+/, ''); if (r2 === ret) break; if (/^static\b/.test(ret)) isStatic = true; ret = r2; }
    methods.push({ name: m[3], ret, params: m[4], body: b.body, static: isStatic });
    skipRanges.push([m.index, b.end]);
    mre.lastIndex = b.end;
  }
  const outside = (pos: number) => !skipRanges.some(([a, b]) => pos >= a && pos < b);
  // fields (top-level statements ending with ';' outside methods)
  // one declaration statement may declare several fields: `double a, b = 1, c;`
  const fre = /(?:public|private|protected)?\s*(static\s+)?(final\s+)?(double|int|float|boolean|long|short|String|vec2|vec3|vec4|double\[\]|int\[\]|float\[\]|boolean\[\]|[A-Z]\w*(?:\[\])?)\s+(\w+(?:\s*\[\s*\])?(?:\s*=\s*[^;,]*(?:\([^;]*?\)[^;,]*|\{[^;]*?\}[^;,]*)?)?(?:\s*,\s*\w+(?:\s*\[\s*\])?(?:\s*=\s*[^;,]*(?:\([^;]*?\)[^;,]*|\{[^;]*?\}[^;,]*)?)?)*)\s*;/g;
  const decls: { st: string; fin: string; type0: string; name: string; arrSuffix: string; init: string | undefined; pos: number }[] = [];
  while ((m = fre.exec(body))) {
    if (!outside(m.index)) continue;
    const [, st, fin, type0, list] = m;
    // split the declarator list on top-level commas
    const parts: string[] = []; let depth = 0, cur = '';
    for (const ch of list) { if (ch === '(' || ch === '{' || ch === '[') depth++; else if (ch === ')' || ch === '}' || ch === ']') depth--; if (ch === ',' && depth === 0) { parts.push(cur); cur = ''; } else cur += ch; }
    parts.push(cur);
    for (const p0 of parts) {
      const dm = /^\s*(\w+)(\s*\[\s*\]\s*)?(?:\s*=\s*([\s\S]*?))?\s*$/.exec(p0);
      if (dm) decls.push({ st, fin, type0, name: dm[1], arrSuffix: dm[2] ?? '', init: dm[3], pos: m.index });
    }
  }
  for (const dcl of decls) {
    const { st, fin, type0, name, arrSuffix, init } = dcl;
    if (type0 === 'String' && st && fin && init) { const q = /^"([^"]*)"$/.exec(init.trim()); if (q) consts.set(name, q[1]); continue; }
    if (st && fin && init && !arrSuffix && /^(double|int|float|long|short|boolean)$/.test(type0)) { numConsts.set(name, init.trim()); continue; }
    if (type0.startsWith('String') || /^(MersenneTwister|Object|Layer|XForm|Flame|AbstractRandomGenerator|RandomGeneratorType|Vec\w*)$/.test(type0)) continue;
    if (name === 'serialVersionUID') continue;
    let type = type0; let array: number | null = null;
    if (type.endsWith('[]') || arrSuffix) {
      type = type.replace('[]', '');
      const am = init ? /new\s+\w+\s*\[\s*(\d+)\s*\]/.exec(init) : null;
      const lit = init ? /\{([^}]*)\}/.exec(init) : null;
      array = am ? Number(am[1]) : lit ? lit[1].split(',').filter((x) => x.trim()).length : -1;
      fields.push({ type, name, init: lit ? lit[0] : null, array });
      continue;
    }
    fields.push({ type, name, init: init?.trim() ?? null, array: null });
  }
  for (const bp of BUILTIN_PODS) if (new RegExp(`\\b${bp.name}\\b`).test(body) && !pods.some((x) => x.name === bp.name)) pods.push(bp);
  return { className: cls[1], parent: cls[2], methods, fields, consts, numConsts, pods, body };
}

// ---------------------------------------------------------------- Java expression → CUDA dialect
const MATH_MAP: Record<string, string> = {
  'Math.sin': 'sinf', 'Math.cos': 'cosf', 'Math.tan': 'tanf', 'Math.sqrt': 'sqrtf', 'Math.abs': 'fabsf', 'Math.atan2': 'atan2f', 'Math.atan': 'atanf',
  'Math.asin': 'asinf', 'Math.acos': 'acosf', 'Math.pow': 'powf', 'Math.exp': 'expf', 'Math.log': 'logf', 'Math.log10': 'log10f', 'Math.floor': 'floorf', 'Math.ceil': 'ceilf',
  'Math.round': 'roundf', 'Math.sinh': 'sinhf', 'Math.cosh': 'coshf', 'Math.tanh': 'tanhf', 'Math.max': 'fmaxf', 'Math.min': 'fminf', 'Math.hypot': 'hypotf',
  'Double.isNaN': 'isnan', 'Double.isInfinite': 'isinf', 'Integer.hashCode': 'jhashcode', 'Math.cbrt': 'cbrtf', 'Math.signum': 'sign', 'Math.random': 'RANDFLOAT', 'Math.rint': 'rintf', 'Math.toRadians': 'radians', 'Math.expm1': 'expm1f', 'Math.log1p': 'log1pf',
  'MathLib.sin': 'sinf', 'MathLib.cos': 'cosf', 'MathLib.sqrt': 'sqrtf', 'MathLib.fabs': 'fabsf', 'MathLib.atan2': 'atan2f', 'MathLib.pow': 'powf', 'MathLib.exp': 'expf',
  'MathLib.log': 'logf', 'MathLib.floor': 'floorf', 'MathLib.sqr': 'sqr', 'MathLib.M_PI': 'M_PI', 'MathLib.M_2PI': 'M_2PI',
  'Tools.FTOI': 'lroundf', 'Tools.limitValue': 'limitValue', 'Tools.limitVal': 'limitValue', 'Tools.limitIntVal': 'limitValue',
  'Math.PI': 'M_PI', 'Math.E': 'M_E', 'Math.acosh': 'acoshf', 'Math.asinh': 'asinhf', 'Math.atanh': 'atanhf', 'Math.toDegrees': 'degrees',
  'Integer.signum': 'sign', 'Long.signum': 'sign', 'Integer.MAX_VALUE': '2147483647', 'Integer.MIN_VALUE': '(-2147483647-1)', 'Double.MAX_VALUE': '3.0e38', 'Double.MIN_VALUE': '1.0e-37', 'Float.MAX_VALUE': '3.0e38',
};
// MathLib static imports used unqualified: mostly the same names as C; the exceptions
const BARE_MAP: Record<string, string> = { fabs: 'fabsf', iabs: 'abs', sqrt: 'sqrtf', pow: 'powf', exp: 'expf', log: 'logf', floor: 'floorf', ceil: 'ceilf', round: 'roundf', trunc: 'truncf', rint: 'rintf',
  sin: 'sinf', cos: 'cosf', tan: 'tanf', asin: 'asinf', acos: 'acosf', atan: 'atanf', atan2: 'atan2f', sinh: 'sinhf', cosh: 'coshf', tanh: 'tanhf', hypot: 'hypotf', cbrt: 'cbrtf',
  fmod: 'fmodf', log10: 'log10f', max: 'fmaxf', min: 'fminf', signum: 'sign', lerp: 'lerpf', frac: 'fracf', log2: 'log2f', exp2: 'exp2f', erf: 'erff', erfc: 'erfcf', gamma: 'tgammaf', lgamma: 'lgammaf',
  acosh: 'acoshf', asinh: 'asinhf', atanh: 'atanhf', toRadians: 'radians', toDegrees: 'degrees' };
const RESERVED_LOCALS = new Set(['x', 'y', 'z']);

interface Ctx { name: string; params: Set<string>; fieldMap: Map<string, string>; state: Set<string>; /** state field → varpar slot name (a field that shares its name with a param gets `_s`, or `varpar->x` would read as the param `__x`) */ stateName: (id: string) => string; helperNames: Set<string>; usedHelperParams: Set<string>; inHelper: boolean; consts: Map<string, string>; usedLib: Set<string>; pods: Map<string, string>; podMethods: Map<string, string>; podFields: Map<string, Set<string>>; podValue: boolean }
// small device helpers some conversions need (appended to gpuFunctions when used)
const LIB: Record<string, string> = {
  jcx: `struct jcx_ { float re; float im; float save_re; float save_im; float per_fix; };
__device__ jcx_ jcx__zero() { jcx_ z; z.re = 0.f; z.im = 0.f; z.save_re = 0.f; z.save_im = 0.f; z.per_fix = 0.f; return z; }
__device__ jcx_ jcx_make(float re, float im) { jcx_ z = jcx__zero(); z.re = re; z.im = im; return z; }
__device__ jcx_ jcx_make(float re) { return jcx_make(re, 0.f); }
__device__ jcx_ jcx_make(jcx_ zz) { return jcx_make(zz.re, zz.im); }
__device__ void jcx_One(jcx_ *z) { z->re = 1.f; z->im = 0.f; }
__device__ void jcx_ImOne(jcx_ *z) { z->re = 0.f; z->im = 1.f; }
__device__ void jcx_Zero(jcx_ *z) { z->re = 0.f; z->im = 0.f; }
__device__ void jcx_Copy(jcx_ *z, jcx_ zz) { z->re = zz.re; z->im = zz.im; }
__device__ void jcx_Flip(jcx_ *z) { float r2 = z->im; float i2 = z->re; z->re = r2; z->im = i2; }
__device__ void jcx_Conj(jcx_ *z) { z->im = -z->im; }
__device__ void jcx_Neg(jcx_ *z) { z->re = -z->re; z->im = -z->im; }
__device__ float jcx_Sig(jcx_ *z) { if (z->re == 0.f) return 0.f; if (z->re > 0.f) return 1.f; return -1.f; }
__device__ float jcx_Sig2(jcx_ *z) { if (z->re >= 0.f) return 1.f; return -1.f; }
__device__ float jcx_Mag2(jcx_ *z) { return z->re * z->re + z->im * z->im; }
__device__ float jcx_Mag2eps(jcx_ *z) { return z->re * z->re + z->im * z->im + 1e-20f; }
__device__ float jcx_MagInv(jcx_ *z) { float M2 = jcx_Mag2(z); return (M2 < 1e-100f ? 1.f : 1.f / M2); }
__device__ void jcx_Save(jcx_ *z) { z->save_re = z->re; z->save_im = z->im; }
__device__ void jcx_Restore(jcx_ *z) { z->re = z->save_re; z->im = z->save_im; }
__device__ void jcx_Switch(jcx_ *z) { float r2 = z->save_re; float i2 = z->save_im; z->save_re = z->re; z->save_im = z->im; z->re = r2; z->im = i2; }
__device__ void jcx_Keep(jcx_ *z, jcx_ zz) { z->save_re = zz.re; z->save_im = zz.im; }
__device__ jcx_ jcx_Recall(jcx_ *z) { return jcx_make(z->save_re, z->save_im); }
__device__ void jcx_Sqr(jcx_ *z) { float r2 = z->re * z->re - z->im * z->im; float i2 = 2.f * z->re * z->im; z->re = r2; z->im = i2; }
__device__ void jcx_Recip(jcx_ *z) { float mi = jcx_MagInv(z); z->re = z->re * mi; z->im = -z->im * mi; }
__device__ void jcx_Scale(jcx_ *z, float mul) { z->re = z->re * mul; z->im = z->im * mul; }
__device__ void jcx_Mul(jcx_ *z, jcx_ zz) { if (zz.im == 0.f) { jcx_Scale(z, zz.re); return; } float r2 = z->re * zz.re - z->im * zz.im; float i2 = z->re * zz.im + z->im * zz.re; z->re = r2; z->im = i2; }
__device__ void jcx_NextPow(jcx_ *z) { jcx_Mul(z, jcx_Recall(z)); }
__device__ void jcx_Div(jcx_ *z, jcx_ zz) { float r2 = z->im * zz.im + z->re * zz.re; float i2 = z->im * zz.re - z->re * zz.im; float M2 = jcx_MagInv(&zz); z->re = r2 * M2; z->im = i2 * M2; }
__device__ void jcx_DivR(jcx_ *z, jcx_ zz) { float r2 = zz.im * z->im + zz.re * z->re; float i2 = zz.im * z->re - zz.re * z->im; float M2 = jcx_MagInv(z); z->re = r2 * M2; z->im = i2 * M2; }
__device__ void jcx_Add(jcx_ *z, jcx_ zz) { z->re += zz.re; z->im += zz.im; }
__device__ void jcx_Sub(jcx_ *z, jcx_ zz) { z->re -= zz.re; z->im -= zz.im; }
__device__ void jcx_SubR(jcx_ *z, jcx_ zz) { z->re = zz.re - z->re; z->im = zz.im - z->im; }
__device__ void jcx_Inc(jcx_ *z) { z->re += 1.f; }
__device__ void jcx_Dec(jcx_ *z) { z->re -= 1.f; }
__device__ void jcx_PerFix(jcx_ *z, float v) { z->per_fix = PI * v; }
__device__ float jcx_Radius(jcx_ *z) { return hypotf(z->re, z->im); }
__device__ float jcx_Arg(jcx_ *z) { return z->per_fix + atan2f(z->im, z->re); }
__device__ jcx_ jcx_ToP(jcx_ *z) { return jcx_make(jcx_Radius(z), jcx_Arg(z)); }
__device__ jcx_ jcx_UnP(jcx_ *z) { return jcx_make(z->re * cosf(z->im), z->re * sinf(z->im)); }
__device__ void jcx_Norm(jcx_ *z) { jcx_Scale(z, sqrtf(jcx_MagInv(z))); }
__device__ void jcx_Exp(jcx_ *z) { z->re = expf(z->re); jcx_Copy(z, jcx_UnP(z)); }
__device__ void jcx_Sqrt(jcx_ *z) { float Rad = jcx_Radius(z); float sb = (z->im < 0.f) ? -1.f : 1.f; z->im = sb * sqrtf(0.5f * (Rad - z->re)); z->re = sqrtf(0.5f * (Rad + z->re)); if (z->per_fix < 0.f) jcx_Neg(z); }
__device__ void jcx_Pow(jcx_ *z, float ex0) { if (ex0 == 0.f) { jcx_One(z); return; } float ex = fabsf(ex0); if (ex0 < 0.f) jcx_Recip(z); if (ex == 0.5f) { jcx_Sqrt(z); return; } if (ex == 1.f) return; if (ex == 2.f) { jcx_Sqr(z); return; } jcx_ PF = jcx_ToP(z); PF.re = powf(PF.re, ex); PF.im = PF.im * ex; jcx_Copy(z, jcx_UnP(&PF)); }
__device__ void jcx_AMean(jcx_ *z, jcx_ zz) { jcx_Add(z, zz); jcx_Scale(z, 0.5f); }
__device__ void jcx_RootMeanS(jcx_ *z, jcx_ zz) { jcx_ PF = jcx_make(zz); jcx_Sqr(&PF); jcx_Sqr(z); jcx_Add(z, PF); jcx_Scale(z, 0.5f); jcx_Pow(z, 0.5f); }
__device__ void jcx_GMean(jcx_ *z, jcx_ zz) { jcx_Mul(z, zz); jcx_Pow(z, 0.5f); }
__device__ void jcx_Heronian(jcx_ *z, jcx_ zz) { jcx_ HM = jcx_make(*z); jcx_GMean(&HM, zz); jcx_Add(z, zz); jcx_Add(z, HM); jcx_Scale(z, 0.333333333333333333f); }
__device__ void jcx_HMean(jcx_ *z, jcx_ zz) { float p2 = (zz.re + z->re); float q2 = (zz.im + z->im); float D = 0.5f * (p2 * p2 + q2 * q2); if (D == 0.f) { jcx_Zero(z); return; } D = 1.f / D; p2 = jcx_Mag2(z); q2 = jcx_Mag2(&zz); if (p2 * q2 == 0.f) { jcx_Zero(z); return; } z->re = D * (z->re * q2 + zz.re * p2); z->im = D * (z->im * q2 + zz.im * p2); }
__device__ void jcx_SinH(jcx_ *z) { float er = 1.f; z->re = expf(z->re); er /= z->re; float rr = 0.5f * (z->re - er); float ri = rr + er; z->re = cosf(z->im) * rr; z->im = sinf(z->im) * ri; }
__device__ void jcx_Sin(jcx_ *z) { jcx_Flip(z); jcx_SinH(z); jcx_Flip(z); }
__device__ void jcx_CosH(jcx_ *z) { float er = 1.f; z->re = expf(z->re); er /= z->re; float rr = 0.5f * (z->re - er); float ri = rr + er; z->re = cosf(z->im) * ri; z->im = sinf(z->im) * rr; }
__device__ void jcx_Cos(jcx_ *z) { jcx_Flip(z); jcx_CosH(z); jcx_Flip(z); }
__device__ void jcx_Log(jcx_ *z) { jcx_ L = jcx_make(0.5f * logf(jcx_Mag2eps(z)), jcx_Arg(z)); jcx_Copy(z, L); }
__device__ void jcx_LMean(jcx_ *z, jcx_ zz) { jcx_ dab = jcx_make(*z); jcx_ lab = jcx_make(*z); jcx_Sub(&dab, zz); jcx_Div(&lab, zz); jcx_Log(&lab); jcx_Div(&dab, lab); jcx_Copy(z, dab); }
__device__ void jcx_AtanH(jcx_ *z) { jcx_ D = jcx_make(*z); jcx_Dec(&D); jcx_Neg(&D); jcx_Inc(z); jcx_Div(z, D); jcx_Log(z); jcx_Scale(z, 0.5f); }
__device__ void jcx_AsinH(jcx_ *z) { jcx_ D = jcx_make(*z); jcx_Sqr(&D); jcx_Inc(&D); jcx_Pow(&D, 0.5f); jcx_Add(z, D); jcx_Log(z); }
__device__ void jcx_AcosH(jcx_ *z) { jcx_ D = jcx_make(*z); jcx_Sqr(&D); jcx_Dec(&D); jcx_Pow(&D, 0.5f); jcx_Add(z, D); jcx_Log(z); }
__device__ void jcx_AcotH(jcx_ *z) { jcx_Recip(z); jcx_AtanH(z); }
__device__ void jcx_AsecH(jcx_ *z) { jcx_Recip(z); jcx_AsinH(z); }
__device__ void jcx_AcosecH(jcx_ *z) { jcx_Recip(z); jcx_AcosH(z); }
__device__ void jcx_Atan(jcx_ *z) { jcx_Flip(z); jcx_AtanH(z); jcx_Flip(z); }
__device__ void jcx_Asin(jcx_ *z) { jcx_Flip(z); jcx_AsinH(z); jcx_Flip(z); }
__device__ void jcx_Acos(jcx_ *z) { jcx_Flip(z); jcx_AsinH(z); jcx_Flip(z); z->re = (PI / 2.f) - z->re; z->im = -z->im; }
__device__ void jcx_CPow(jcx_ *z, jcx_ ex) { if (ex.im == 0.f) { jcx_Pow(z, ex.re); return; } jcx_Log(z); jcx_Mul(z, ex); jcx_Exp(z); }`,
  jhashcode: '__device__ int jhashcode(int a) { return a; }', // Integer.hashCode(int) is the int itself
  jgcd: '__device__ int jgcd(int a, int b) { a = abs(a); b = abs(b); while (b != 0) { int t = a % b; a = b; b = t; } return a; }',
  G_Kscope: '__device__ float2 G_Kscope(float2 uv, float k) { float angle = fabsf(mod(atan2f(uv.y, uv.x), 2.0f * k) - k); return make_float2(length(uv) * cosf(angle), length(uv) * sinf(angle)); }',
  G_rot: '__device__ mat3_ G_rot(float3 s) { float sa = sinf(s.x), ca = cosf(s.x), sb = sinf(s.y), cb = cosf(s.y), sc = sinf(s.z), cc = cosf(s.z); return mat3_make(cb*cc, -cb*sc, sb, sa*sb*cc+ca*sc, -sa*sb*sc+ca*cc, -sa*cb, -ca*sb*cc+sa*sc, ca*sb*sc+sa*cc, ca*cb); }',
  G_app: '__device__ float3 G_app(float3 v, float k, mat3_ m) { for (int i = 0; i < 50; i++) { float3 mv = make_float3(m.a00 * v.x + m.a01 * v.y + m.a02 * v.z, m.a10 * v.x + m.a11 * v.y + m.a12 * v.z, m.a20 * v.x + m.a21 * v.y + m.a22 * v.z) * k; v = abs(mv / dot(v, v) * 0.5f - 0.5f) * 2.0f - 1.0f; } return v; }',
  // JWildfire's MarsagliaRandomGenerator (per-cell seeded randoms in de_stijl/greebles/quad): exact 32-bit port
  jmrg: 'struct jmrg_ { int u; int v; };\n__device__ jmrg_ jmrg__zero() { jmrg_ r_; r_.u = 12244355; r_.v = 34384; return r_; }\n__device__ void jmrg_randomize(jmrg_ *g, int seed) { g->u = seed << 16; g->v = (seed << 16) >> 16; }\n__device__ float jmrg_random(jmrg_ *g) { unsigned int v = (unsigned int)g->v; unsigned int u = (unsigned int)g->u; v = 36969u * (v & 65535u) + (unsigned int)(g->v >> 16); u = 18000u * (u & 65535u) + (unsigned int)(g->u >> 16); g->v = (int)v; g->u = (int)u; int rnd = (int)((v << 16) + u); float res = (float)rnd * (1.0f / 2147483647.0f); return res < 0.0f ? -res : res; }',
  // java.util.Random (48-bit LCG) on 16-bit limbs: exact seeding + nextDouble/nextInt sequences (seeded per parameter in cut_*truchet, curliecue2)
  jrand: 'struct jrand_ { int s0; int s1; int s2; };\n__device__ jrand_ jrand__zero() { return jrand_make((int)(RANDINT() >> 1)); }\n__device__ void jrand_setSeed(jrand_ *r, int seed) { jrand_ n = jrand_make(seed); r->s0 = n.s0; r->s1 = n.s1; r->s2 = n.s2; }\n__device__ jrand_ jrand_make(int seed) { jrand_ r; r.s0 = (seed & 0xFFFF) ^ 0xE66D; r.s1 = ((seed >> 16) & 0xFFFF) ^ 0xDEEC; r.s2 = (seed < 0 ? 0xFFFF : 0) ^ 0x5; return r; }\n'
    + '__device__ int jrand_next(jrand_ *r, int bits) { unsigned int a0 = (unsigned int)r->s0, a1 = (unsigned int)r->s1, a2 = (unsigned int)r->s2; unsigned int t0 = a0 * 58989u + 11u; unsigned int r0 = t0 & 65535u; unsigned int c0 = t0 >> 16; unsigned int t1a = a0 * 57068u + c0; unsigned int c1a = t1a >> 16; unsigned int t1b = a1 * 58989u + (t1a & 65535u); unsigned int r1 = t1b & 65535u; unsigned int c1 = c1a + (t1b >> 16); unsigned int r2 = (a0 * 5u + a1 * 57068u + a2 * 58989u + c1) & 65535u; r->s0 = (int)r0; r->s1 = (int)r1; r->s2 = (int)r2; unsigned int hi = (r2 << 16) | r1; return (int)(hi >> (32 - bits)); }\n'
    + '__device__ float jrand_nextDouble(jrand_ *r) { return (float)jrand_next(r, 26) * (1.0f / 67108864.0f) + (float)jrand_next(r, 27) * (1.0f / 9007199254740992.0f); }\n'
    + '__device__ int jrand_nextInt(jrand_ *r, int n) { if (n <= 0) return 0; if ((n & -n) == n) { int k = 0; int m = n; while (m > 1) { m = m >> 1; k = k + 1; } return jrand_next(r, 31) >> (31 - k); } int bits; int val; do { bits = jrand_next(r, 31); val = bits % n; } while (bits - val + (n - 1) < 0); return val; }',
  jxyz: 'struct jxyz_ { float x; float y; float z; float color; };\n__device__ jxyz_ jxyz__zero() { jxyz_ r_; return r_; }\n__device__ void jxyz_assign(jxyz_ *a, jxyz_ b) { a->x = b.x; a->y = b.y; a->z = b.z; a->color = b.color; }\n__device__ jxyz_ jxyz_make(float x, float y, float z, float color) { jxyz_ r_; r_.x = x; r_.y = y; r_.z = z; r_.color = color; return r_; }',
  jdw: 'struct jdw_ { float value; };\n__device__ jdw_ jdw__zero() { jdw_ r_; return r_; }',
  mat3: 'struct mat3_ { float a00; float a10; float a20; float a01; float a11; float a21; float a02; float a12; float a22; };\n__device__ mat3_ mat3_make(float a00, float a10, float a20, float a01, float a11, float a21, float a02, float a12, float a22) { mat3_ m; m.a00 = a00; m.a10 = a10; m.a20 = a20; m.a01 = a01; m.a11 = a11; m.a21 = a21; m.a02 = a02; m.a12 = a12; m.a22 = a22; return m; }\n__device__ mat3_ mat3_scale(mat3_ m, float f) { return mat3_make(m.a00 * f, m.a10 * f, m.a20 * f, m.a01 * f, m.a11 * f, m.a21 * f, m.a02 * f, m.a12 * f, m.a22 * f); }',
};

function convertJava(code: string, ctx: Ctx, extraLocals: Iterable<string> = []): string {
  let s = code;
  s = s.replace(/if\s*\(\s*\w+\s*==\s*null\s*\)\s*\w+\s*=\s*new\s+\w+\s*\([^;]*\)\s*;/g, ''); // lazy creation of a value object (before pod rewriting renames the ctor)
  if (/\b(vec[234]|G\.|mat[23])\b/.test(s) || /\.(multiply|plus|minus|division|add|times|dot|length)\s*\(/.test(s)) s = convertGlsl(s).replace(/\bMAT2_\(/g, 'make_float4(');
  if (/\bmat3_/.test(s)) ctx.usedLib.add('mat3');
  if (/\bG_Kscope\(/.test(s)) ctx.usedLib.add('G_Kscope');
  if (/\bG_rot\(/.test(s)) { ctx.usedLib.add('mat3'); ctx.usedLib.add('G_rot'); }
  if (/\bG_app\(/.test(s)) { ctx.usedLib.add('mat3'); ctx.usedLib.add('G_app'); }
  s = s.replace(/\bfinal\s+/g, '').replace(/\bthis\.(\w)/g, 'THISF_$1');
  // plain-data inner classes: `P v = new P();` → `VAR_P v;`, `new P(a, b)` → `VAR_P_make(a, b)`, `v.setXy(a)` → `VAR_P_setXy(&v, a)`, type names prefixed
  for (const [jn, cn] of ctx.pods) {
    const podVars = new Set<string>([...ctx.podFields.get(jn) ?? []]);
    s = s.replace(new RegExp(`\\b${jn}\\s*\\[\\s*\\]\\s*(\\w+)\\s*=\\s*new\\s+${jn}\\s*\\[([^\\]]+)\\]\\s*;`, 'g'), `${cn} $1[$2];`); // arrays of objects
    s = s.replace(new RegExp(`(?<![\\w.])(\\w+(?:\\[[^\\]]*\\])?)\\.assign\\(\\s*pAffineTP\\s*\\)\\s*;`, 'g'), '$1.x = pAffineTP.x; $1.y = pAffineTP.y; $1.z = pAffineTP.z; $1.color = pAffineTP.color;');
    s = s.replace(new RegExp(`(?<![\\w.])(\\w+(?:\\[[^\\]]*\\])?)\\.assign\\(\\s*pVarTP\\s*\\)\\s*;`, 'g'), '$1.x = pVarTP.x; $1.y = pVarTP.y; $1.z = pVarTP.z; $1.color = pVarTP.color;');
    for (const dm of s.matchAll(new RegExp(`\\b${jn}\\s+(\\w+)`, 'g'))) podVars.add(dm[1]);
    for (const dm of s.matchAll(new RegExp(`\\b${jn}\\s+(\\w+)\\s*[,)]`, 'g'))) podVars.add(dm[1]);
    s = s.replace(new RegExp(`\\b${jn}\\s+(\\w+)\\s*=\\s*new\\s+${jn}\\s*\\(\\s*\\)\\s*;`, 'g'), `${cn} $1 = ${cn}_zero();`);
    s = s.replace(new RegExp(`(?<=[;{}]\\s*)(\\w+(?:\\[[^\\]]*\\])?)\\s*=\\s*new\\s+${jn}\\s*\\(\\s*\\)\\s*;`, 'g'), ''); // re-creating an empty object (statement): already zero-initialised
    s = s.replace(new RegExp(`\\bnew\\s+${jn}\\s*\\(\\s*\\)`, 'g'), `${cn}_zero()`);
    s = s.replace(new RegExp(`\\bnew\\s+${jn}\\s*\\(`, 'g'), `${cls_ctorFn(jn, cn)}(`);
    s = s.replace(new RegExp(`(?<![\\w.])(\\w+)\\.(\\w+)\\s*\\(`, 'g'), (m0, obj: string, fn: string) => (ctx.podMethods.has(`${jn}.${fn}`) && podVars.has(obj) ? `${ctx.podMethods.get(`${jn}.${fn}`)}(&${obj}, ` : m0)).replace(/, \)/g, ')');
    s = s.replace(new RegExp(`\\b${jn}\\b`, 'g'), cn);
  }
  s = s.replace(/\bdouble\b/g, 'float').replace(/\b(?:long|short)\b/g, 'int').replace(/\bboolean\b/g, 'bool'); // long → int: Java longs in variations are seeds/hashes where the low 32 bits carry the meaning
  // locals declared in this body shadow fields of the same name (`double x = …` in dc_cube)
  const locals = new Set<string>(extraLocals);
  for (const dm of s.matchAll(/(?<![\w.])(?:float|int|bool)\s*(?:\[\s*\])?\s+([^;{}()]*?)(?:;|\)\s*\{|=\s*\{)/g)) {
    for (const part of dm[1].split(',')) { const nm = /^\s*(\w+)/.exec(part); if (nm) locals.add(nm[1]); }
  }
  for (const cn of ctx.pods.values()) for (const dm of s.matchAll(new RegExp(`(?<![\\w.])${cn}\\s+(\\w+)`, 'g'))) locals.add(dm[1]);
  for (const dm of s.matchAll(/for\s*\(\s*(?:float|int)\s+(\w+)/g)) locals.add(dm[1]);
  for (const dm of s.matchAll(/(?<![\w.])(?:float|int|bool)\s*(?:\[\s*\])?\s+(\w+)\s*(?:\[[^\]]*\])?\s*(?:=(?!=)|,|;|\))/g)) locals.add(dm[1]);
  // pContext
  s = s.replace(/pContext\.random\(\s*\)/g, 'RANDFLOAT()');
  // random(Integer.MAX_VALUE): a 31-bit uniform int (its low bits are used as coin flips — RANDFLOAT()*2^31 in f32 has none)
  s = s.replace(/pContext\.random\(\s*Integer\.MAX_VALUE\s*\)/g, '((int)(RANDINT() >> 1))');
  s = s.replace(/pContext\.random\(([^()]*(?:\([^()]*\))?[^()]*)\)/g, '((int)(RANDFLOAT() * (float)($1)))');
  s = s.replace(/pContext\.isPreserveZCoordinate\(\)/g, 'false');
  s = s.replace(/randGen\.random\(\)|randomize\.nextDouble\(\)|rand\.nextDouble\(\)|_random\.random\(\)/g, 'RANDFLOAT()');
  if (!ctx.helperNames.has('random')) s = s.replace(/(?<![\w.])random\(\s*\)/g, 'RANDFLOAT()'); // static import of Math.random
  // xform affine coefficients (JWildfire coeffXY: x' = c00 x + c10 y + c20, y' = c01 x + c11 y + c21 → flam3 a b c / d e f)
  s = s.replace(/pXForm\.getXYCoeff00\(\)/g, 'xform->a').replace(/pXForm\.getXYCoeff10\(\)/g, 'xform->b').replace(/pXForm\.getXYCoeff20\(\)/g, 'xform->c')
    .replace(/pXForm\.getXYCoeff01\(\)/g, 'xform->d').replace(/pXForm\.getXYCoeff11\(\)/g, 'xform->e').replace(/pXForm\.getXYCoeff21\(\)/g, 'xform->f');
  // points
  s = s.replace(/pAffineTP\.getPrecalcSumsq\(\)/g, '__r2').replace(/pAffineTP\.getPrecalcSqrt\(\)/g, '__r')
    .replace(/pAffineTP\.getPrecalcAtan\(\)/g, '__phi').replace(/pAffineTP\.getPrecalcAtanYX\(\)/g, '__theta')
    .replace(/pAffineTP\.getPrecalcSinA\(\)/g, '(__x / __r)').replace(/pAffineTP\.getPrecalcCosA\(\)/g, '(__y / __r)');
  // precalcs asked of the output point (a helper called with pVarTP for both points): computed on the spot
  s = s.replace(/pVarTP\.getPrecalcSumsq\(\)/g, '(__px * __px + __py * __py)').replace(/pVarTP\.getPrecalcSqrt\(\)/g, 'sqrtf(__px * __px + __py * __py)')
    .replace(/pVarTP\.getPrecalcAtan\(\)/g, 'atan2f(__px, __py)').replace(/pVarTP\.getPrecalcAtanYX\(\)/g, 'atan2f(__py, __px)');
  s = s.replace(/pAffineTP\.x\b/g, '__x').replace(/pAffineTP\.y\b/g, '__y').replace(/pAffineTP\.z\b/g, '__z').replace(/pAffineTP\.color\b/g, '__pal').replace(/pAffineTP\.doHide\b/g, '__doHide');
  // chained assignment `a = b = c;` → `b = c; a = b;`
  // (a single-statement if/else body keeps its shape by getting braces)
  s = s.replace(/(?<=(?:\)|\belse)\s*)([\w.>-]+)\s*=(?!=)\s*([\w.>-]+)\s*=(?!=)\s*([^;=]+);/g, '{ $2 = $3; $1 = $2; }');
  for (let guard = 0; guard < 4; guard++) { const s2 = s.replace(/(?<![=!<>])\b([\w.>-]+)\s*=(?!=)\s*([\w.>-]+)\s*=(?!=)\s*([^;=]+);/g, '$2 = $3; $1 = $2;'); if (s2 === s) break; s = s2; }
  // helper calls that pass the context along: drop the argument (the helper body uses RANDFLOAT())
  s = s.replace(/\bpContext\s*,\s*/g, '').replace(/\(\s*pContext\s*\)/g, '()');
  s = s.replace(/pVarTP\.x\b/g, '__px').replace(/pVarTP\.y\b/g, '__py').replace(/pVarTP\.z\b/g, '__pz').replace(/pVarTP\.color\b/g, '__pal')
    .replace(/pVarTP\.doHide\b/g, '__doHide').replace(/pVarTP\.rgbColor\b/g, '__useRgb');
  // Java red/green/blueColor are 0..255 ints; the kernel's __colorR/G/B are 0..1
  s = s.replace(/pVarTP\.(red|green|blue)Color\s*=\s*([^;]+);/g, (_m, ch: string, e: string) => `__color${ch[0].toUpperCase()} = (float)(${e}) / 255.0f;`);
  // dbl2int(vec3) → int3 (kernel-lib helper); `int[] t = new int[3]; t = dbl2int(c);` and t[0..2]
  s = s.replace(/int\s*\[\s*\]\s*(\w+)\s*=\s*new\s+int\s*\[\s*3\s*\]\s*;\s*\1\s*=\s*dbl2int\(/g, 'int3 $1 = dbl2int(').replace(/int\s*\[\s*\]\s*(\w+)\s*=\s*dbl2int\(/g, 'int3 $1 = dbl2int(');
  s = s.replace(/\b(tcolor|color[0-9]?)\[0\]/g, '$1.x').replace(/\b(tcolor|color[0-9]?)\[1\]/g, '$1.y').replace(/\b(tcolor|color[0-9]?)\[2\]/g, '$1.z');
  s = s.replace(/\.r\b(?!\w)/g, '.x').replace(/\.g\b(?!\w)/g, '.y').replace(/\.b\b(?!\w)/g, '.z');
  s = s.replace(/\bpAmount\b/g, '__amount_'); // gen.ts binds __amount_ to the weight (`__x` would clash with the input point for the variation named x)
  // sinAndCos(a, sina, cosa) → sina_v = sinf(a); cosa_v = cosf(a)   (declared by the caller prelude)
  s = s.replace(/sinAndCos\(([^;]*?),\s*(\w+),\s*(\w+)\)\s*;/g, (_m, a: string, sn: string, cs: string) => `${sn}.value = sinf(${a}); ${cs}.value = cosf(${a});`);
  if (!ctx.podValue) s = s.replace(/\b(\w+)\.value\b/g, '$1_v');
  s = s.replace(/(?<![\w.])TRUE\b/g, '1').replace(/(?<![\w.])FALSE\b/g, '0'); // MathLib.TRUE/FALSE
  s = s.replace(/\bsuper\.init\s*\([^;]*\)\s*;/g, ''); // VariationFunc.init is a no-op
  s = s.replace(/(?<![\w.])\w+\.invalidate\(\)\s*;/g, ''); // XYZPoint precalc cache: no equivalent
  s = s.replace(/if\s*\(\s*\w+\s*==\s*null\s*\)\s*\w+\s*=\s*new\s+\w+\s*\([^;]*\)\s*;/g, '') // lazy creation of a value object
    .replace(/if\s*\(\s*\w+\s*==\s*null\s*(?:\|\|\s*\w+\s*==\s*null\s*)*\)\s*\{[^{}]*\}\s*(?:else\s*)?/g, '') // a block guarded only by null checks never runs
    .replace(/(?<![\w.])\w+\s*==\s*null\s*\|\|\s*/g, '').replace(/\|\|\s*\w+\s*==\s*null\b/g, ''); // null guards on value objects
  s = s.replace(/\bthrow\s+new\s+\w+\s*\((?:[^()]*|\([^()]*\))*\)\s*;/g, ''); // unreachable-parameter guards
  // the affine/output point handed to a helper as a plain argument (`f(pContext, pAffineTP, s)`): pass a value copy of the point
  s = s.replace(/([(,]\s*)pAffineTP(\s*[,)])/g, '$1jxyz_make(__x, __y, __z, __pal)$2').replace(/([(,]\s*)pVarTP(\s*[,)])/g, '$1jxyz_make(__px, __py, __pz, __pal)$2');
  if (/jxyz_make\(/.test(s)) ctx.usedLib.add('jxyz');
  // Math.* / Tools.* / Integer.*
  s = s.replace(/\bFastMath\./g, 'Math.');
  s = s.replace(/\b(Math|MathLib|Tools|Integer|Double|Float)\.(\w+)/g, (m0) => (Object.hasOwn(MATH_MAP, m0) ? MATH_MAP[m0] : m0));
  if (/\bjhashcode\(/.test(s)) ctx.usedLib.add('jhashcode');
  // qualified statics not mapped → leave (transpiler will report)
  // bare MathLib names
  s = s.replace(/\b([a-z]\w*)\s*\(/g, (m0, fn: string) => (Object.hasOwn(BARE_MAP, fn) && !ctx.helperNames.has(fn) ? BARE_MAP[fn] + '(' : m0));
  // BigInteger gcd idiom (rhodonea): `BigInteger a = BigInteger.valueOf((long) kn); BigInteger b = …; int gcd = a.gcd(b).intValue();`
  s = s.replace(/BigInteger\s+(\w+)\s*=\s*BigInteger\.valueOf\(\s*\((?:int|float)\)\s*(\w+)\s*\)\s*;\s*BigInteger\s+(\w+)\s*=\s*BigInteger\.valueOf\(\s*\((?:int|float)\)\s*(\w+)\s*\)\s*;\s*int\s+(\w+)\s*=\s*\1\.gcd\(\3\)\.intValue\(\)\s*;/g,
    (_m, _a, x: string, _b, y: string, g: string) => { ctx.usedLib.add('jgcd'); return `int ${g} = jgcd((int)(${x}), (int)(${y}));`; });
  // `t++ < n` / `i++ == 0` inside an expression (loop conditions): the transpiler only takes ++ as a statement
  s = s.replace(/(?<![\w.])(\w+)\+\+(?=\s*(?:<=?|>=?|==|!=)\s*[^;])/g, (_m, id: string) => `jpostinc(&${id})`);
  // static final numeric constants of the class → inlined
  s = s.replace(/(?<![\w.])([A-Za-z_]\w*)\b/g, (m0, id: string) => (ctx.consts.has(id) && !locals.has(id) ? `(${ctx.consts.get(id)})` : m0));
  // casts: (int) fine; (float) fine; (long)→(int)
  s = s.replace(/\(\s*(?:long|short)\s*\)/g, '(int)');
  s = s.replace(/\b(\d+)[lL]\b/g, '$1'); // long literals
  // arrays: `float[] a = new float[6];` / `float a[] = new float[6];`
  s = s.replace(/\b(float|int|bool)\s*\[\s*\]\s*(\w+)\s*=\s*new\s+\w+\s*\[([^\]]+)\]\s*;/g, '$1 $2[$3];');
  s = s.replace(/\b(float|int|bool)\s+(\w+)\s*\[\s*\]\s*=\s*new\s+\w+\s*\[([^\]]+)\]\s*;/g, '$1 $2[$3];');
  s = s.replace(/\b(float|int|bool)\s*\[\s*\]\s*(\w+)\s*=\s*new\s+\w+\s*\[\s*\]\s*(\{[^}]*\})\s*;/g, (m0, t: string, n: string, lit: string) => `${t} ${n}[${lit.split(',').filter((x) => x.trim()).length}] = ${lit};`);
  s = s.replace(/\b(float|int|bool)\s*\[\s*\]\s*(\w+)\s*=\s*(\{[^}]*\})\s*;/g, (m0, t: string, n: string, lit: string) => `${t} ${n}[${lit.split(',').filter((x) => x.trim()).length}] = ${lit};`);
  s = s.replace(/\b(float|int|bool)\s*\[\s*\]\s*(\w+)\s*;/g, '$1 $2[64];'); // uninitialised local array: generous default
  s = s.replace(/new\s+(?:float|int|double)\s*\[\s*\]\s*(\{[^}]*\})/g, '$1');
  s = s.replace(/\.length\b/g, '.length');
  // Java `a % b` on floats is fine for the transpiler (typed emitter)
  // param / state / field references
  s = s.replace(/(?<![\w.])([A-Za-z_]\w*)\b(?!\s*\()/g, (m0, id: string, off: number) => {
    // skip after '.' handled by lookbehind; skip declarations handled naturally (types are not fields)
    let forced = false;
    if (id.startsWith('THISF_')) { id = id.slice(6); forced = true; }
    else if (locals.has(id)) return m0;
    if (ctx.params.has(id)) {
      const pn = (ctx.fieldMap.get(id) ?? id).replace(/\W/g, '_');
      if (ctx.inHelper) { ctx.usedHelperParams.add(pn); return `__${ctx.name}_${pn}_c`; }
      return `__${ctx.name}_${pn}`;
    }
    if (ctx.state.has(id)) return `varpar->${ctx.name}_${ctx.stateName(id)}`;
    return forced ? id : m0;
  });
  s = s.replace(/THISF_/g, '');
  return s;
}

// Hand patches on the converted text (Java idioms that do not survive f32 literally).
// Patches on the Java text before conversion (constructs the GLSL parser cannot express)
const PRE_PATCHES: Record<string, [string | RegExp, string][]> = {
  // VecMathLib.VectorD used once, for a normalised direction (normalize() divides by length + 1e-16)
  // the affine point is passed to a helper as an object: copy it into a local first
  boxfold: [['XYZPoint pLocal = rotatePoint(pAffineTP, negatedAnglesRad);', 'XYZPoint pIn = new XYZPoint(); pIn.x = pAffineTP.x; pIn.y = pAffineTP.y; pIn.z = pAffineTP.z;\n    XYZPoint pLocal = rotatePoint(pIn, negatedAnglesRad);']],
  // helpers returning small double[] vectors → vec3/vec4 (GLSL types java2cu already maps to float3/float4)
  cactusGlobe: [
    ['private double[] calculateCactusPoint(', 'private vec4 calculateCactusPoint('],
    ['private double[] applySpikeRealism(', 'private vec3 applySpikeRealism('],
    ['double[] finalCoords = calculateCactusPoint(pContext, pAffineTP, current_size);', 'vec4 finalCoords = calculateCactusPoint(pContext, pAffineTP, current_size);'],
    [/finalCoords\[0\]/g, 'finalCoords.x'], [/finalCoords\[1\]/g, 'finalCoords.y'], [/finalCoords\[2\]/g, 'finalCoords.z'], [/finalCoords\[3\]/g, 'finalCoords.w'],
    [/double\[\] spike_vec = applySpikeRealism\(/g, 'vec3 spike_vec = applySpikeRealism('],
    ['double[] spike_vec = {nx * current_spike_length, ny * current_spike_length, nz * current_spike_length};', 'vec3 spike_vec = new vec3(nx * current_spike_length, ny * current_spike_length, nz * current_spike_length);'],
    [/spike_vec\[0\]/g, 'spike_vec.x'], [/spike_vec\[1\]/g, 'spike_vec.y'], [/spike_vec\[2\]/g, 'spike_vec.z'],
    ['return new double[]{0,0,0,-1.0};', 'return new vec4(0,0,0,-1.0);'],
    ['return new double[]{0,0,0};', 'return new vec3(0,0,0);'],
    ['return new double[]{final_x, final_y, final_z, hasFeature ? 1.0 : -1.0};', 'return new vec4(final_x, final_y, final_z, hasFeature ? 1.0 : -1.0);'],
  ],
  // the `triangle` inner class carries a blur() method → hoist it to the outer class (self param), assign() → struct copy,
  // t1/t2/p become per-call locals, the JWildfire random generator field → pContext.random()
  dc_triTile: [
    [/\btriangle\b/g, 'Triangle'], // field types must be capitalised for the field parser
    [/  private class Triangle implements Serializable \{\n    private static final long serialVersionUID = 1L;\n    int type;\n    double x1, y1, x2, y2, x3, y3;\n    double col;\n\n    public void assign\(Triangle t\) \{[\s\S]*?\n    \}\n\n    private void blur\(XYZPoint p\) \{/,
      '  private static class Triangle implements Serializable {\n    int type;\n    double x1, y1, x2, y2, x3, y3;\n    double col;\n  }\n\n    private void blur(Triangle self, XYZPoint p) {'],
    [/this\.(x[123]|y[123])\b/g, 'self.$1'],
    ['      p.z = 0;\n    }\n  }\n\n  private Triangle orig, t1, t2;', '      p.z = 0;\n    }\n\n  private Triangle orig;'],
    ['  private XYZPoint p;\n', ''],
    ['XYZPoint pVarTP, double pAmount) {\n    /* extension of FiveFold', 'XYZPoint pVarTP, double pAmount) {\n    Triangle t1 = new Triangle(); Triangle t2 = new Triangle(); XYZPoint p = new XYZPoint();\n    /* extension of FiveFold'],
    ['  AbstractRandomGenerator genRand;\n', ''],
    [/genRand\.random\(\)/g, 'pContext.random()'],
    ['    genRand = pContext.getRandGen();\n    t1 = new Triangle();\n    t2 = new Triangle();\n    p = new XYZPoint();\n', ''],
    [/(\w+)\.assign\((\w+)\);/g, '$1 = $2;'],
    ['t2.blur(p);', 'blur(t2, p);'],
  ],
  pre_wave3D_wf: [['VectorD d = new VectorD(pAffineTP.x - centre_x, pAffineTP.y - centre_y, pAffineTP.z - centre_z);\n        d.normalize();',
    'double d_l = sqrt(sqr(pAffineTP.x - centre_x) + sqr(pAffineTP.y - centre_y) + sqr(pAffineTP.z - centre_z)) + 1.0e-16; double d_x = (pAffineTP.x - centre_x) / d_l, d_y = (pAffineTP.y - centre_y) / d_l, d_z = (pAffineTP.z - centre_z) / d_l;'],
    ['pAffineTP.x += d.x * amp;\n        pAffineTP.y += d.y * amp;\n        pAffineTP.z += d.z * amp;', 'pAffineTP.x += d_x * amp;\n        pAffineTP.y += d_y * amp;\n        pAffineTP.z += d_z * amp;'],
    ['fabs(damping) > SMALL_EPSILON', 'fabs(damping) > 1.0e-30']],
  glsl_mandelbox2D: [['double b = (O.a = G.length(I)) < .5 ? 4. : O.a < 1. ? 1. / O.a : 1.;', 'O.a = G.length(I); double b = O.a < .5 ? 4. : O.a < 1. ? 1. / O.a : 1.;']],
};
const JAVA_PATCHES: Record<string, [string | RegExp, string][]> = {
  // random(Integer.MAX_VALUE) * 2π/power: only k mod |power| matters, and k·2π/power in f32 is noise
  // elliptic coordinates: cosh(mu) − 1 cancels near the real axis in f32; compute it stably
  // 4-entry ring buffer of randoms (sliding Irwin-Hall sum): four fresh randoms have the same distribution
  dc_cylinder2: [['float rr = __dc_cylinder2_blur * (r[0] + r[1] + r[2] + r[3] - 2.0);', 'float rr = __dc_cylinder2_blur * (RANDFLOAT() + RANDFLOAT() + RANDFLOAT() + RANDFLOAT() - 2.0);'],
    [/r\[varpar->dc_cylinder2_n\] = RANDFLOAT\(\);\s*varpar->dc_cylinder2_n = varpar->dc_cylinder2_n \+ 1 & 3;/, '']],
  rosoni: [['cerc ^= (r2 <= 0.0);', 'cerc = cerc != (r2 <= 0.0);']],
  // Ken Perlin's doubled permutation table p[512] (filled from `permutation` in a static block) → index the 256 table modulo 256
  camouflage: [[/int p\[512\];\s*/, ''], [/(?<![\w.])p\[([^\[\]]*)\]/g, 'camouflage_permutation[($1) & 255]']],
  // gauss_rnd ring buffer advanced with `& 5` (indices 2,3 never refresh: per-instance constants) → four fresh uniforms + the constants' mean
  pre_blur3D: [['(gauss_rnd[0] + gauss_rnd[1] + gauss_rnd[2] + gauss_rnd[3] + gauss_rnd[4] + gauss_rnd[5] - 3)', '(RANDFLOAT() + RANDFLOAT() + RANDFLOAT() + RANDFLOAT() - 2.0)'],
    [/gauss_rnd\[[^\]]*gauss_N\] = RANDFLOAT\(\);\s*[^;]*gauss_N = \([^;]*gauss_N \+ 1\) & 5;/, '']],
  eSwirl: [
    ['float xmax = (eSwirl_sqrt_safe(tmp + tmp2) + eSwirl_sqrt_safe(tmp - tmp2)) * 0.5;', 'float ea = eSwirl_sqrt_safe(tmp + tmp2); float eb = eSwirl_sqrt_safe(tmp - tmp2); float xmax = (ea + eb) * 0.5;\n    float ed = (__y * __y / (ea + fabsf(__x + 1.0)) + __y * __y / (eb + fabsf(__x - 1.0))) * 0.5 + fmaxf(fabsf(__x) - 1.0, 0.0);'],
    ['float mu = acoshf(xmax);', 'float mu = (ed < 1.0e-4) ? sqrtf(2.0 * ed) * (1.0 - ed / 12.0) : logf(1.0 + ed + sqrtf(ed * (2.0 + ed)));'],
  ],
  ...Object.fromEntries(['post_juliaq', 'post_julia3Dq', 'jubiQ'].map((n) => [n, [[/\(\(int\)\(RANDINT\(\) >> 1\)\)/g, `((int)(RANDFLOAT() * fabsf((float)__${n}_power)))`]] satisfies [string | RegExp, string][]])),
};

// ---------------------------------------------------------------- per-variation conversion
export interface Port { name: string; gpuCode: string; preCode?: string; gpuFunctions: string; extraParams: string[]; note: string; javaFile: string }
const MERGE_PARENTS: Record<string, string> = { GLSLFunc: 'GLSLFunc.java', AbstractFalloff3Func: 'AbstractFalloff3Func.java', AbstractAffine3DFunc: 'AbstractAffine3DFunc.java' };
/** Java variation class source → CUDA-dialect port. `parentSources` lets a caller supply
 *  MERGE_PARENTS sources directly (tests); otherwise they are read from the JWildfire tree. */
export function convertVariation(d: DumpVar, src0: string, javaFile: string, parentSources?: Record<string, string>): Port {
  let src = src0;
  for (const [from, to] of PRE_PATCHES[d.name] ?? []) { const s2 = src.replace(from, to); if (s2 === src) throw new Error(`pre-patch did not match: ${from}`); src = s2; }
  const cls = parseClass(src);
  // known abstract parents whose fields/params/setParameter the subclass relies on (GLSLFunc: resolution, gradient)
  if (MERGE_PARENTS[cls.parent]) {
    const pf = parentSources?.[cls.parent] !== undefined ? null : javaFiles().find((f) => f.endsWith('/' + MERGE_PARENTS[cls.parent]));
    if (!pf && parentSources?.[cls.parent] === undefined) throw new Error(`parent source ${cls.parent} not found`);
    const par = parseClass(parentSources?.[cls.parent] ?? fs.readFileSync(pf!, 'latin1'));
    for (const [k, v] of par.consts) if (!cls.consts.has(k)) cls.consts.set(k, v);
    for (const [k, v] of par.numConsts) if (!cls.numConsts.has(k)) cls.numConsts.set(k, v);
    for (const pd of par.pods) if (!cls.pods.some((x) => x.name === pd.name)) cls.pods.push(pd);
    for (const f of par.fields) if (!cls.fields.some((x) => x.name === f.name)) cls.fields.push(f);
    for (const m of par.methods) {
      if (m.name === 'initOnce' || m.name === 'dbl2int') continue;
      if ((m.name === 'transform' || m.name === 'init') && cls.methods.some((x) => x.name === m.name)) continue;
      const own = cls.methods.find((x) => x.name === m.name);
      if (!own) cls.methods.push(m);
      else if (m.name === 'setParameter') own.body = own.body + '\n' + m.body; // subclass falls through to super.setParameter
    }
    cls.parent = 'VariationFunc';
  }
  const transform = cls.methods.find((m) => m.name === 'transform');
  if (!transform) throw new Error('no transform()');
  // JWildfire "prepost" (func priority 2): invtransform() runs as a pre step on the affine point and transform()
  // as a post step on the variation output → the port carries both snippets (preCode / gpuCode)
  const invtransform = cls.methods.find((m) => m.name === 'invtransform');
  if (!/^(VariationFunc|SimpleVariationFunc)$/.test(cls.parent)) throw new Error(`extends ${cls.parent}`);
  const s = cls.body;
  if (/RessourceType|getRessourceNames|SubFlame|subflame|Flame\b|Layer\s+\w+\s*=|getOwner\(\)|RGBPalette|SimpleImage|WFImage|BufferedImage|String\s+\w+\s*=/.test(transform.body + (cls.methods.find((m) => m.name === 'init')?.body ?? '')))
    throw new Error('uses resources / layer / palette / images');
  const setParam = cls.methods.find((m) => m.name === 'setParameter');
  // param name → field name (from setParameter branches)
  const paramNames = new Set(d.params.map((p) => p.name));
  const fieldMap = new Map<string, string>(); // field → param
  if (setParam) {
    const branches = setParam.body.split(/\belse\s+if\b|\bif\b/);
    for (const br of branches) {
      const cond = /(\w+)\.equalsIgnoreCase\(\s*(\w+)\s*\)|equalsIgnoreCase\(\s*"(\w+)"\s*\)/.exec(br);
      if (!cond) continue;
      const key = cond[3] ?? (cls.consts.get(cond[2]) ?? cls.consts.get(cond[1]));
      if (!key || !paramNames.has(key)) continue;
      // the first assignment to a *field* (a local `double v = max(…, pValue)` in front of it is skipped)
      const fieldNames0 = new Set(cls.fields.map((f) => f.name));
      const rest = br.slice(cond.index + cond[0].length);
      for (const asg of rest.matchAll(/(?:^|[{;\s])(?:this\.)?(\w+)\s*=(?!=)/g)) {
        if (asg[1] === 'pValue' || !fieldNames0.has(asg[1])) continue;
        fieldMap.set(asg[1], key);
        break;
      }
    }
  }
  const fieldNames = new Set(cls.fields.map((f) => f.name));
  // params whose field name is unknown default to the param name
  for (const pn of paramNames) if (![...fieldMap.values()].includes(pn) && fieldNames.has(pn)) fieldMap.set(pn, pn);
  const paramFields = new Set(fieldMap.keys());
  for (const pn of paramNames) if (![...fieldMap.values()].includes(pn)) throw new Error(`param ${pn}: field not found`);
  // helper methods (non-static and static), other than the framework ones
  let helpers = cls.methods.filter((m) => !KNOWN_METHODS.has(m.name));
  // void helpers that take the affine/var points (`fillVIn(pAffineTP, pVarTP, v_in)`) cannot become device
  // functions (the points are snippet-scope bindings) → inline them textually at their call sites
  const inlineHelpers = helpers.filter((h) => h.ret === 'void' && /\bXYZPoint\s+(pAffineTP|pVarTP)\b/.test(h.params));
  if (inlineHelpers.length) {
    helpers = helpers.filter((h) => !inlineHelpers.includes(h));
    const inlineInto = (body: string): string => {
      for (let guard = 0; guard < 8; guard++) {
        let changed = false;
        for (const h of inlineHelpers) {
          const pn = h.params.split(',').map((x) => x.trim().split(/\s+/).pop()!);
          body = body.replace(new RegExp(`(?<![\\w.])${h.name}\\s*\\(([^;]*)\\)\\s*;`, 'g'), (_m, argText: string) => {
            const args = argText.split(',').map((x) => x.trim());
            if (args.length !== pn.length) throw new Error(`inline ${h.name}: argument count`);
            let b = h.body;
            pn.forEach((pname, i) => { b = b.replace(new RegExp(`(?<![\\w.])${pname}\\b`, 'g'), /^\w+$/.test(args[i]) ? args[i] : `(${args[i]})`); });
            changed = true;
            return `{\n${b}\n}`;
          });
        }
        if (!changed) break;
      }
      return body;
    };
    transform.body = inlineInto(transform.body);
    for (const h of helpers) h.body = inlineInto(h.body);
    const invt = cls.methods.find((m) => m.name === 'invtransform');
    if (invt) invt.body = inlineInto(invt.body);
  }
  const init = cls.methods.find((m) => m.name === 'init');
  const initOnce = cls.methods.find((m) => m.name === 'initOnce');
  if (initOnce && /\S/.test(initOnce.body) && !/super\.initOnce/.test(initOnce.body)) throw new Error('initOnce');
  // which non-param fields are written in transform() (state) vs only in init() (precalc)
  const localsOf = (body: string) => {
    const b = body.replace(/\bdouble\b/g, 'float').replace(/\b(?:long|short)\b/g, 'int').replace(/\bboolean\b/g, 'bool');
    const out = new Set<string>();
    for (const dm of b.matchAll(/(?<![\w.])(?:float|int|bool)\s*(?:\[\s*\])?\s+([^;{}()]*?)(?:;|\)\s*\{|=\s*\{)/g)) for (const part of dm[1].split(',')) { const nm = /^\s*(\w+)/.exec(part); if (nm) out.add(nm[1]); }
    for (const dm of b.matchAll(/for\s*\(\s*(?:float|int)\s+(\w+)/g)) out.add(dm[1]);
    for (const dm of b.matchAll(/(?<![\w.])(?:float|int|bool)\s*(?:\[\s*\])?\s+(\w+)\s*(?:\[[^\]]*\])?\s*(?:=(?!=)|,|;|\))/g)) out.add(dm[1]); // initialisers with calls
    return out;
  };
  const assignedIn = (body: string) => {
    const loc = localsOf(body);
    const all = [...body.matchAll(/(?<![\w.])(\w+)\s*(?:=(?!=)|\+=|-=|\*=|\/=|\+\+|--)/g)].map((m) => m[1]).concat([...body.matchAll(/(?:\+\+|--)\s*(\w+)/g)].map((m) => m[1]));
    const forced = [...body.matchAll(/this\.(\w+)\s*(?:=(?!=)|\+=|-=|\*=|\/=|\+\+|--)/g)].map((m) => m[1]);
    return new Set(all.filter((n) => !loc.has(n)).concat(forced));
  };
  const tAssigned = assignedIn(transform.body);
  const hAssigned = new Set(helpers.flatMap((h) => [...assignedIn(h.body)]));
  const nonParamFields = cls.fields.filter((f) => !paramFields.has(f.name));
  const state = new Set<string>();
  for (const f of nonParamFields) {
    if (f.array !== null) continue;
    if (tAssigned.has(f.name) || hAssigned.has(f.name)) state.add(f.name);
  }
  // fields read inside helper methods cannot be snippet locals → make them state
  const helperParamNames = (h: Method) => new Set(h.params.split(',').map((x) => x.trim().split(/\s+/).pop()!.replace(/\[\]/g, '')).filter(Boolean));
  const podScratch = new Map<string, Set<string>>(); // helper name → pod-typed fields it uses as scratch objects
  for (const h of helpers) {
    const hp = helperParamNames(h);
    for (const f of nonParamFields) {
      if (f.array !== null || hp.has(f.name) || !new RegExp(`(?<![\\w.])${f.name}\\b`).test(h.body.replace(/\bthis\./g, ''))) continue;
      // a plain-data object only touched inside helpers is a scratch object: give each helper its own local copy
      if (cls.pods.some((pd) => pd.name === f.type) && !new RegExp(`(?<![\\w.])${f.name}\\b`).test(transform.body + (init?.body ?? ''))) { podScratch.set(h.name, (podScratch.get(h.name) ?? new Set()).add(f.name)); continue; }
      state.add(f.name);
    }
  }
  // fields initialised from a random draw (`int k = rand.nextInt(n)`: fixed per JWildfire instance) → per-thread state, drawn once
  for (const f of nonParamFields) if (f.array === null && f.init && /\bnextInt\(|\bnextDouble\(|\brandom\(\)|Math\.random/.test(f.init)) state.add(f.name);
  // params written in transform → also state (initialised from the param)
  const paramState: string[] = [];
  for (const f of paramFields) if (tAssigned.has(f)) { state.add(f); paramState.push(f); }
  // init() that writes state (attractor start values) runs once → inside the first-call block; the
  // other fields it writes must then persist too
  const initAssigned = init ? assignedIn(init.body) : new Set<string>();
  // only real per-point state (written by transform/helpers) makes init() a first-call block; fields that are
  // state merely because a helper reads them are derived values and keep the plain (per-call) init replay
  const trueState = new Set([...state].filter((f) => tAssigned.has(f) || hAssigned.has(f)));
  const initTouchesState = [...initAssigned].some((f) => trueState.has(f));
  const usedOutsideInit = (n: string) => new RegExp(`(?<![\\w.])${n}\\b`).test(transform.body.replace(/\bthis\./g, '') + helpers.map((h) => h.body.replace(/\bthis\./g, '')).join(''));
  if (initTouchesState) for (const f of nonParamFields) if (f.array === null && initAssigned.has(f.name) && !(cls.pods.some((pd) => pd.name === f.type) && !usedOutsideInit(f.name))) state.add(f.name);
  const arrayFields = nonParamFields.filter((f) => f.array !== null);
  for (const f of arrayFields) if (f.array === -1) throw new Error(`array field ${f.name} of unknown size`);
  for (const f of nonParamFields) if (state.has(f.name) && cls.pods.some((pd) => pd.name === f.type)) {
    // an object re-created inside transform() (`rand = new Random(seed)`) is a per-call scratch object: keep it a local
    if (new RegExp(`(?<![\\w.])${f.name}\\s*=\\s*new\\s+${f.type}\\b`).test(transform.body)) { state.delete(f.name); continue; }
    throw new Error(`object field ${f.name} reassigned per point`);
  }
  if (state.size > 40) throw new Error(`too much state (${[...state].join(',')})`);
  const helperNames = new Set(helpers.map((h) => h.name));
  const stateName = (id: string) => (paramNames.has(id) ? `${id}_s` : id);
  const ctx: Ctx = { name: d.name, params: new Set([...paramFields].filter((f) => !state.has(f))), fieldMap, state, stateName, helperNames, usedHelperParams: new Set(), inHelper: false, consts: new Map(), usedLib: new Set(), podFields: new Map(), pods: new Map(cls.pods.map((pd) => [pd.name, pd.builtin ? pd.builtin.cuda : `${d.name}_${pd.name}`])), podMethods: new Map(cls.pods.flatMap((pd) => pd.builtin ? Object.entries(pd.builtin.methods).map(([m, fn]) => [`${pd.name}.${m}`, fn] as [string, string]) : pd.setters.map((st) => [`${pd.name}.${st.name}`, `${d.name}_${pd.name}_${st.name}`] as [string, string]))), podValue: cls.pods.some((pd) => pd.fields.some((f) => f.name === 'value')) };
  for (const pd of cls.pods) if (pd.builtin) ctx.usedLib.add(pd.builtin.lib);
  ctx.podFields = new Map(cls.pods.map((pd) => [pd.name, new Set(cls.fields.filter((f) => f.type === pd.name).map((f) => f.name))]));
  for (const [k, v] of cls.numConsts) ctx.consts.set(k, convertJava(v, ctx));
  glslKinds.clear();
  for (const m of cls.methods) if (/^(mat2|mat3|vec[234])$/.test(m.ret)) glslKinds.set(m.name, m.ret === 'mat2' ? 'mat2' : m.ret === 'mat3' ? 'mat3' : 'vec');
  for (const f of cls.fields) if (/^(mat2|mat3|vec[234])$/.test(f.type)) glslKinds.set(f.name, f.type === 'mat2' ? 'mat2' : f.type === 'mat3' ? 'mat3' : 'vec');
  const others = new Set(['pContext', 'pXForm', 'pLayer', 'pAffineTP', 'pVarTP', 'pAmount']);
  // --- plain-data inner classes → structs (+ maker / setter functions)
  let funcs = '';
  for (const pd of cls.pods) {
    if (pd.builtin) continue;
    const cn = ctx.pods.get(pd.name)!;
    funcs += `struct ${cn} { ${pd.fields.map((f) => `${f.type} ${f.name};`).join(' ')} };\n__device__ ${cn} ${cn}_zero() { ${cn} r_; return r_; }\n`;
    if (pd.ctor) {
      const ps = pd.ctor.params.map((pp) => convertJava(pp, ctx));
      const pn = pd.ctor.params.map((pp) => pp.split(/\s+/).pop()!);
      funcs += `__device__ ${cn} ${cn}_make(${ps.join(', ')}) { ${cn} r_; ${pd.ctor.assigns.map(([f, e]) => `r_.${f} = ${convertJava(e, ctx, pn)};`).join(' ')} return r_; }\n`;
    }
    for (const st of pd.setters) {
      const ps = st.params ? st.params.split(',').map((pp) => convertJava(pp.trim(), ctx)) : [];
      const pn = st.params ? st.params.split(',').map((pp) => pp.trim().split(/\s+/).pop()!) : [];
      const bodyStmts = st.body.split(';').map((x) => x.trim()).filter(Boolean).map((x) => {
        const am = /^(?:this\.)?(\w+)\s*=\s*([\s\S]+)$/.exec(x);
        if (!am || !pd.fields.some((f) => f.name === am[1])) throw new Error(`setter ${pd.name}.${st.name}: ${x}`);
        return `self_->${am[1]} = ${convertJava(am[2], ctx, pn)};`;
      });
      funcs += `__device__ void ${cn}_${st.name}(${cn} *self_${ps.length ? ', ' + ps.join(', ') : ''}) { ${bodyStmts.join(' ')} }\n`;
    }
  }
  // --- helpers
  // Which object params are references in effect (mutated → pointer): a helper that assigns `p.x = …` (or takes `&p`)
  // mutates the caller's object; and a helper that merely hands its param on to such a pointer parameter of another
  // helper must pass a pointer too — propagated to a fixed point (dc_triTile: devis(t1, t2) → fiveFoldDevis(in, *out)).
  const objParams = (h: Method) => (h.params.trim() ? h.params.split(',').map((x) => x.trim()) : []).filter((x) => !/^(FlameTransformationContext|XForm|Layer)\b/.test(x)).map((x) => x.split(/\s+/).pop()!.replace(/\[\]/g, ''));
  const ptrParams = new Map<string, Set<string>>();
  for (const h of helpers) {
    const set = new Set<string>();
    for (const pn of objParams(h)) if (new RegExp(`(?<![\\w.])${pn}\\.\\w+\\s*(?:=(?!=)|[-+*\\/]=)`).test(h.body) || new RegExp(`\\(&${pn}\\b`).test(h.body)) set.add(pn);
    ptrParams.set(h.name, set);
  }
  for (let changed = true; changed;) {
    changed = false;
    for (const h of helpers) {
      for (const g of helpers) {
        const gp = objParams(g), gptr = ptrParams.get(g.name)!;
        if (!gptr.size) continue;
        for (const cm of h.body.matchAll(new RegExp(`(?<![\\w.])${g.name}\\s*\\(([^;]*)\\)\\s*;`, 'g'))) {
          const args = splitTop(cm[1]).filter((a) => !/^pContext\b|^pXForm\b|^pLayer\b/.test(a));
          args.forEach((a, i) => { if (/^\w+$/.test(a) && gptr.has(gp[i]) && objParams(h).includes(a) && !ptrParams.get(h.name)!.has(a)) { ptrParams.get(h.name)!.add(a); changed = true; } });
        }
      }
    }
  }
  for (const h of helpers) {
    ctx.inHelper = true;
    // a non-void helper taking `XYZPoint pAffineTP/pVarTP` (a value copy at the call site): rename the parameter so its
    // fields are not rewritten as the snippet's magic point
    let hbody = h.body, hps = h.params;
    for (const pn of ['pAffineTP', 'pVarTP']) if (new RegExp(`\\bXYZPoint\\s+${pn}\\b`).test(hps)) { const re = new RegExp(`\\b${pn}\\b`, 'g'); hps = hps.replace(re, pn + '_'); hbody = hbody.replace(re, pn + '_'); }
    const hparams = hps.trim() ? hps.split(',').map((p) => p.trim()).filter((p) => !/^(FlameTransformationContext|XForm|Layer)\b/.test(p)) : [];
    const pnames = hparams.map((p) => p.split(/\s+/).pop()!.replace(/\[\]/g, ''));
    for (const p of hparams) { const pm = /^(mat2|mat3|vec[234])\s+(\w+)$/.exec(p); if (pm) glslKinds.set(pm[2], pm[1] === 'mat2' ? 'mat2' : pm[1] === 'mat3' ? 'mat3' : 'vec'); }
    let body = convertJava(hbody, ctx, pnames);
    for (const sn of podScratch.get(h.name) ?? []) { const sf = cls.fields.find((x) => x.name === sn)!; body = `${ctx.pods.get(sf.type)} ${sn};\n` + body; }
    const params = hparams.map((p, i) => {
      let q = convertJava(p, ctx, pnames).replace(/\bfloat\s+/, 'float ');
      // Java arrays are references: `float[] a` → `float a[]` (pointer param, pointee inferred from the argument)
      const am = /^(\w+)\s*\[\s*\]\s*(\w+)$/.exec(q) ?? /^(\w+)\s+(\w+)\s*\[\s*\]$/.exec(q);
      if (am) return `${am[1]} ${am[2]}[]`;
      // vec/struct objects are references too: a helper that assigns `p.x = …` mutates the caller's object → pointer param
      const vm = /^(float[234]|\w+)\s+(\w+)$/.exec(q);
      if (vm && !/^(float[234])$/.test(vm[1]) && ![...ctx.pods.values()].includes(vm[1])) return q;
      if (vm && (new RegExp(`(?<![\\w.])${vm[2]}\\.\\w+\\s*(?:=(?!=)|[-+*\\/]=)`).test(body) || new RegExp(`\\(&${vm[2]}\\b`).test(body) || ptrParams.get(h.name)?.has(vm[2]))) {
        body = body.replace(new RegExp(`(?<![\\w.])${vm[2]}\\.`, 'g'), `${vm[2]}->`);
        return `${vm[1]} *${vm[2]}`;
      }
      return q;
    }).join(', ');
    const ret = h.ret === 'double' ? 'float' : h.ret === 'long' || h.ret === 'short' ? 'int' : h.ret === 'boolean' ? 'bool' : (VEC_TYPES[h.ret] ?? ctx.pods.get(h.ret) ?? h.ret);
    if (!/^(float|int|bool|void|float2|float3|float4|mat3_)$/.test(ret) && ![...ctx.pods.values()].includes(ret)) throw new Error(`helper ${h.name} returns ${h.ret}`);
    if (ret === 'mat3_') ctx.usedLib.add('mat3');
    if (/\b(pXForm|pLayer|pAffineTP|pVarTP)\b/.test(body)) throw new Error(`helper ${h.name} uses context`);
    funcs += `__device__ ${ret} ${d.name}_${h.name}(${params}) {${body}}\n`;
    ctx.inHelper = false;
  }
  // helper names in code → prefixed
  const prefixHelpers = (t: string) => t.replace(/(?<![\w.])(\w+)\s*\(/g, (m0, fn: string) => helperNames.has(fn) ? `${d.name}_${fn}(` : m0);
  funcs = prefixHelpers(funcs);
  // --- snippet: field locals, state init, init(), transform()
  let code = '';
  // params used inside helpers travel as state copies
  for (const pn of ctx.usedHelperParams) code += `varpar->${d.name}_${pn}_c = __${d.name}_${pn};\n`;
  const extra = [...ctx.usedHelperParams].map((pn) => `${pn}_c`);
  // non-param, non-state, non-array fields → locals with their initialisers
  for (const f of nonParamFields) {
    if (state.has(f.name) || f.array !== null) continue;
    if (ctx.pods.has(f.type)) {
      if ([...podScratch.values()].some((st) => st.has(f.name))) continue;
      const cn = ctx.pods.get(f.type)!;
      const im = f.init ? new RegExp(`^new\\s+${f.type}\\s*\\(([\\s\\S]*)\\)$`).exec(f.init.trim()) : null;
      code += im && im[1].trim() ? `${cn} ${f.name} = ${cls_ctorFn(f.type, cn)}(${convertJava(im[1], ctx)});\n` : `${cn} ${f.name};\n`;
      continue;
    }
    const t = f.type === 'double' ? 'float' : f.type === 'long' || f.type === 'short' ? 'int' : f.type === 'boolean' ? 'bool' : (VEC_TYPES[f.type] ?? f.type);
    if (!/^(float|int|bool|float2|float3|float4|mat3_)$/.test(t)) throw new Error(`field ${f.name}: ${f.type}`);
    code += `${t} ${f.name}${f.init !== null ? ' = ' + convertJava(f.init, ctx) : ' = 0'};\n`;
  }
  // constant tables (array fields with a literal initialiser that nothing writes) → module-scope constants,
  // shared by helpers and free per thread; other arrays stay per-call locals
  let globalsCode = '';
  const tableRename = new Map<string, string>();
  for (const f of arrayFields) {
    const t = f.type === 'double' ? 'float' : f.type === 'boolean' ? 'bool' : f.type;
    const written = tAssigned.has(f.name) || hAssigned.has(f.name) || initAssigned.has(f.name) || new RegExp(`(?<![\\w.])${f.name}\\s*\\[[^\\]]*\\]\\s*(?:=(?!=)|[-+*\\/]=|\\+\\+|--)`).test(transform.body + helpers.map((h) => h.body).join('') + (init?.body ?? ''));
    if (f.init && !written) { globalsCode += `${t} ${d.name}_${f.name}[${f.array}] = ${f.init};\n`; tableRename.set(f.name, `${d.name}_${f.name}`); continue; }
    code += `${t} ${f.name}[${f.array}]${f.init ? ' = ' + f.init : ''};\n`;
  }
  const renameTables = (t: string) => { for (const [from, to] of tableRename) t = t.replace(new RegExp(`(?<![\\w.])${from}\\b`, 'g'), to); return t; };
  funcs = renameTables(funcs);
  // state: first-call init from Java initialiser (or param value)
  const stateType = (n: string) => { const f = cls.fields.find((x) => x.name === n); const t = f?.type ?? 'double'; return t in VEC_TYPES ? ':' + VEC_TYPES[t] : t === 'int' || t === 'long' || t === 'short' ? ':int' : ''; };
  if (state.size) {
    extra.push('inited_', ...[...state].map((n) => stateName(n) + stateType(n)));
    code += `if (varpar->${d.name}_inited_ == 0.0f) {\n  varpar->${d.name}_inited_ = 1.0f;\n`;
    for (const f of cls.fields) {
      if (!state.has(f.name)) continue;
      const initExpr = paramState.includes(f.name) ? `__${d.name}_${fieldMap.get(f.name)}` : f.init !== null ? convertJava(f.init, ctx) : '0';
      code += `  varpar->${d.name}_${stateName(f.name)} = ${initExpr};\n`;
    }
    if (initTouchesState && init) code += '  {\n' + convertJava(init.body, ctx) + '\n  }\n';
    // trajectory state (attractor coordinates: float state written by transform()): Java runs a
    // handful of long identical trajectories; our 65k walkers would all trace the same short
    // one, so each thread starts a hair off — chaotic maps decorrelate within the fuse
    for (const f of cls.fields) {
      if (!state.has(f.name) || !tAssigned.has(f.name) || f.array !== null || f.type === 'int' || f.type === 'long' || f.type === 'short' || f.type === 'boolean' || f.type in VEC_TYPES) continue;
      code += `  varpar->${d.name}_${stateName(f.name)} += (RANDFLOAT() - 0.5f) * 1.0e-4f;\n`;
    }
    code += '}\n';
  }
  // state fields that are only read by helpers (never written per point) are plain derived values:
  // reset them from their initialiser every call before the setParameter/init replay
  // (a random initialiser is a per-instance constant: the first-call init above is all it needs)
  for (const f of nonParamFields) {
    if (!state.has(f.name) || f.array !== null || tAssigned.has(f.name) || hAssigned.has(f.name) || initAssigned.has(f.name)) continue;
    const initExpr = f.init !== null ? convertJava(f.init, ctx) : '0';
    if (/RANDFLOAT|RANDINT|jrand_|jmrg_/.test(initExpr)) continue;
    code += `varpar->${d.name}_${stateName(f.name)} = ${initExpr};\n`;
  }
  // derived fields computed in setParameter() (`pwx = pValue * M_2PI;`): replay those branches
  if (setParam) {
    const derived = new Set(nonParamFields.filter((f) => !state.has(f.name) && f.array === null).map((f) => f.name));
    let extraCode = '';
    const bre = /(?:(\w+)\.equalsIgnoreCase\(\s*(\w+)\s*\)|equalsIgnoreCase\(\s*"(\w+)"\s*\))\s*\)\s*(\{)?/g;
    let bm: RegExpExecArray | null;
    while ((bm = bre.exec(setParam.body))) {
      const key = bm[3] ?? (cls.consts.get(bm[2]) ?? cls.consts.get(bm[1]));
      if (!key || !paramNames.has(key)) continue;
      let stmt: string;
      if (bm[4]) { const bb = braceBody(setParam.body, bm.index + bm[0].length - 1); if (!bb) continue; stmt = bb.body; bre.lastIndex = bb.end; }
      else { const rest = setParam.body.slice(bm.index + bm[0].length); stmt = (rest.match(/^[^;]*;/) ?? [''])[0]; bre.lastIndex = bm.index + bm[0].length + stmt.length; }
      const drop = (x: string) => /\bnew\b|System\.|nextDouble|nextInt|randomize|elapsed_time|last_time|throw\b|return\b/.test(x);
      const touches = [...derived].some((dn) => new RegExp(`(?<![\\w.])${dn}\\s*(?:=(?!=)|\\+=|-=|\\*=|\\/=)`).test(stmt));
      if (!touches) continue;
      let text = stmt.replace(/\bpValue\b/g, `(__${d.name}_${key})`);
      // drop whole simple statements we cannot express (Random re-seeding etc.)
      text = text.replace(/[^;{}]*;/g, (st) => drop(st) ? '' : st);
      extraCode += '{\n' + convertJava(text, ctx) + '\n}\n';
    }
    code += extraCode;
  }
  if (init && /\S/.test(init.body) && !initTouchesState) {
    code += '{\n' + convertJava(init.body, ctx) + '\n}\n';
  }
  const PRECALC = '\n__r2 = __x*__x+__y*__y; __r = sqrtf(__r2); __rinv = 1.0f/__r; __phi = atan2f(__x,__y); __theta = 0.5f*M_PI-__phi; if (__theta > M_PI) __theta -= 2.0f*M_PI;\n';
  const preamble = code; // field locals, state init, setParameter/init replay — needed by both snippets
  code += convertJava(transform.body, ctx);
  if ((d.priority ?? 0) < 0) code += PRECALC;
  code = prefixHelpers(renameTables(code));
  let preCode: string | undefined;
  if (invtransform) preCode = prefixHelpers(renameTables(preamble + convertJava(invtransform.body, ctx) + PRECALC));
  for (const [from, to] of JAVA_PATCHES[d.name] ?? []) { const c2 = code.replace(from, to), f2 = funcs.replace(from, to); if (c2 === code && f2 === funcs) throw new Error(`patch did not match: ${from}`); code = c2; funcs = f2; if (preCode) preCode = preCode.replace(from, to); }
  // sinAndCos locals
  const scv = new Set([...(code + funcs).matchAll(/\b(\w+)_v\b/g)].map((m) => m[1]));
  const declV = [...scv].filter((n) => /^(sina?|cosa?|sin\w*|cos\w*|s|c)$/.test(n) || /_v = (sinf|cosf)\(/.test(code)).map((n) => `float ${n}_v = 0.0f;`).join(' ');
  if (declV) code = declV + '\n' + code;
  // sanity: leftovers the transpiler cannot know
  const bad = /\b(pContext|pXForm|pLayer|new\s+\w+|super\.|Random\b|String\b|\.length\b|Object\b|throw\b|try\b|catch\b|instanceof|\bnull\b|Complex\b|vec[234]\b|mat[234]\b|MAT2\()\b/.exec(code + funcs);
  if (bad) throw new Error(`unsupported construct: ${bad[0]}`);
  const knownTypes = new Set(['M_PI', 'M_2PI', 'M_E', 'M_PI_2', 'M_PI_4', 'M_1_PI', 'M_2_PI', 'M_SQRT2', 'M_SQRT1_2', 'RANDFLOAT', 'RANDINT', 'EPSILON', 'HSIN2', 'mat3_', ...ctx.pods.values()]);
  const tm = [...(code + funcs).matchAll(/(?<![\w.])([A-Z]\w*)\s+[a-z_]\w*\s*(?:=|;|,|\))/g)].find((x) => !knownTypes.has(x[1]) && !/^[A-Z0-9_]+$/.test(x[1]));
  if (tm) throw new Error(`type ${tm[1]}`);
  for (const o of others) if (new RegExp(`\\b${o}\\b`).test(code + funcs + (preCode ?? ''))) { if (process.env.JAVA2CU_DEBUG) console.error(code + funcs); throw new Error(`leftover ${o}`); }
  const lib = [...ctx.usedLib].map((k) => LIB[k]).join('\n');
  return { name: d.name, gpuCode: code, preCode, gpuFunctions: (lib ? lib + '\n' : '') + globalsCode + funcs, extraParams: extra, note: `java port of ${javaFile}`, javaFile };
}

// ---------------------------------------------------------------- GLSL-style Java (js.glsl vec2/vec3/G) → CUDA dialect
// JWildfire's shader-art variations are written with vec2/vec3/vec4 objects and the static
// class G (abs, mod, mix, …). Method chains `p.multiply(2.0).minus(0.5)` become plain vector
// arithmetic for the transpiler; `new vec2(a,b)` → make_float2, `G.f(…)` → f(…). This is a
// small expression parser (precedence climbing) over a token stream; statements are located
// by delimiters and only their expressions are re-emitted.
type Tok = { t: 'id' | 'num' | 'op' | 'str'; v: string };
function glslTokenize(s: string): Tok[] {
  const out: Tok[] = [];
  const re = /\s+|\/\/[^\n]*|\/\*[\s\S]*?\*\/|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[A-Za-z_]\w*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFdDlL]?|>>>=|<<=|>>=|>>>|\+\+|--|&&|\|\||==|!=|<=|>=|\+=|-=|\*=|\/=|%=|&=|\|=|\^=|<<|>>|->|[-+*\/%=<>!&|^~?:;,.(){}\[\]]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(s))) {
    const v = m[0];
    if (/^\s/.test(v) || v.startsWith('//') || v.startsWith('/*')) continue;
    if (/^[A-Za-z_]/.test(v)) out.push({ t: 'id', v });
    else if (/^(\d|\.\d)/.test(v)) out.push({ t: 'num', v: v.replace(/[fFdDlL]$/, '') });
    else if (/^["']/.test(v)) out.push({ t: 'str', v });
    else out.push({ t: 'op', v });
  }
  return out;
}
const VEC_TYPES: Record<string, string> = { vec2: 'float2', vec3: 'float3', vec4: 'float4', mat2: 'float4', mat3: 'mat3_' };
const G_MAP: Record<string, string> = { abs: 'abs', mod: 'mod', mix: 'mix', smoothstep: 'smoothstep', floor: 'floor', fract: 'fract', clamp: 'clamp', sin: 'sin', cos: 'cos', tan: 'tan',
  step: 'step', normalize: 'normalize', min: 'min', max: 'max', pow: 'pow', exp: 'exp', exp2: 'exp2', log2: 'log2', sign: 'sign', cross: 'cross', sqrt: 'sqrt', length: 'length', dot: 'dot', distance: 'distance',
  trunc: 'trunc', round: 'round', ceil: 'ceil', reflect: 'reflect', tanh: 'tanh', sinh: 'sinh', cosh: 'cosh', invSqrt: 'rsqrtf', atan2: 'atan2f' };
const VEC_BINOPS: Record<string, string> = { plus: '+', add: '+', minus: '-', multiply: '*', division: '/' };
// js.glsl matrices: mat2 travels as float4 (a00, a10, a01, a11 — the constructor order), mat3 as the
// struct mat3_ (LIB.mat3). `.times()` needs to know which operand is the matrix, so declared variable /
// field / helper-return kinds are tracked here (set per variation by convertVariation, per body by convertGlsl).
const glslKinds = new Map<string, 'mat2' | 'mat3' | 'vec'>();
const cls_ctorFn = (jn: string, cn: string) => BUILTIN_PODS.find((b) => b.name === jn)?.builtin?.ctorFn ?? `${cn}_make`;
function glslKindOf(e: string): 'mat2' | 'mat3' | 'vec' | 'scalar' {
  let x = e.trim();
  while (/^\(.*\)$/.test(x) && splitTop(x.slice(1, -1)).length === 1 && balanced(x.slice(1, -1))) x = x.slice(1, -1).trim();
  if (/^MAT2_\(/.test(x)) return 'mat2';
  if (/^mat3_(make|scale)\(/.test(x) || /^G_rot\(/.test(x)) return 'mat3';
  if (/^make_float[234]\(/.test(x)) return 'vec';
  const id = /^([A-Za-z_]\w*)\s*(\(|$)/.exec(x);
  if (id) { const k = glslKinds.get(id[1]); if (k) return k; if (id[2]) return 'vec'; return 'scalar'; }
  if (/^[\d.]/.test(x)) return 'scalar';
  return 'vec';
}
function balanced(x: string): boolean { let d = 0; for (const ch of x) { if (ch === '(') d++; else if (ch === ')') { d--; if (d < 0) return false; } } return d === 0; }
const MAT2_MUL_VEC = (m: string, v: string) => `make_float2((${m}).x * (${v}).x + (${m}).z * (${v}).y, (${m}).y * (${v}).x + (${m}).w * (${v}).y)`;
const VEC_MUL_MAT2 = (v: string, m: string) => `make_float2((${m}).x * (${v}).x + (${m}).y * (${v}).y, (${m}).z * (${v}).x + (${m}).w * (${v}).y)`;
const MAT3_MUL_VEC = (m: string, v: string) => `make_float3((${m}).a00 * (${v}).x + (${m}).a01 * (${v}).y + (${m}).a02 * (${v}).z, (${m}).a10 * (${v}).x + (${m}).a11 * (${v}).y + (${m}).a12 * (${v}).z, (${m}).a20 * (${v}).x + (${m}).a21 * (${v}).y + (${m}).a22 * (${v}).z)`;
const VEC_MUL_MAT3 = (v: string, m: string) => `make_float3((${v}).x * (${m}).a00 + (${v}).y * (${m}).a10 + (${v}).z * (${m}).a20, (${v}).x * (${m}).a01 + (${v}).y * (${m}).a11 + (${v}).z * (${m}).a21, (${v}).x * (${m}).a02 + (${v}).y * (${m}).a12 + (${v}).z * (${m}).a22)`;
const PREC: Record<string, number> = { '||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6, '<': 7, '>': 7, '<=': 7, '>=': 7, '<<': 8, '>>': 8, '>>>': 8, '+': 9, '-': 9, '*': 10, '/': 10, '%': 10 };
class GlslExpr {
  i = 0;
  toks: Tok[];
  constructor(toks: Tok[]) { this.toks = toks; }
  peek(k = 0) { return this.toks[this.i + k]; }
  eat(v?: string): Tok { const t = this.toks[this.i++]; if (!t || (v !== undefined && t.v !== v)) throw new Error(`glsl: expected ${v} got ${t?.v}`); return t; }
  at(v: string) { const t = this.peek(); return !!t && t.t === 'op' && t.v === v; }
  expr(): string { return this.ternary(); }
  ternary(): string {
    const c = this.binary(0);
    if (this.at('?')) { this.eat('?'); const a = this.ternary(); this.eat(':'); const b = this.ternary(); return `(${c} ? ${a} : ${b})`; }
    return c;
  }
  binary(minPrec: number): string {
    let lhs = this.unary();
    for (;;) {
      const t = this.peek();
      if (!t || t.t !== 'op' || !(t.v in PREC) || PREC[t.v] < minPrec) return lhs;
      this.eat();
      const rhs = this.binary(PREC[t.v] + 1);
      lhs = `(${lhs} ${t.v} ${rhs})`;
    }
  }
  unary(): string {
    const t = this.peek();
    if (t && t.t === 'op' && (t.v === '-' || t.v === '!' || t.v === '~' || t.v === '+')) { this.eat(); return `${t.v}${this.unary()}`; }
    if (t && t.t === 'op' && (t.v === '++' || t.v === '--')) { this.eat(); return `${t.v}${this.unary()}`; }
    // cast: ( type ) unary
    if (t && t.v === '(' && this.peek(1)?.t === 'id' && this.peek(2)?.v === ')' && /^(int|double|float|long|short|byte|boolean)$/.test(this.peek(1)!.v)) {
      this.eat('('); const ty = this.eat().v; this.eat(')');
      const tt = ty === 'double' ? 'float' : ty === 'long' || ty === 'short' ? 'int' : ty === 'boolean' ? 'bool' : ty;
      return `((${tt})${this.unary()})`;
    }
    return this.postfix(this.primary());
  }
  args(): string[] { this.eat('('); const out: string[] = []; if (!this.at(')')) { out.push(this.expr()); while (this.at(',')) { this.eat(','); out.push(this.expr()); } } this.eat(')'); return out; }
  primary(): string {
    const t = this.eat();
    if (t.t === 'num') return t.v;
    if (t.t === 'str') throw new Error('glsl: string');
    if (t.t === 'op') {
      if (t.v === '(') { const e = this.expr(); this.eat(')'); return `(${e})`; }
      throw new Error(`glsl: unexpected ${t.v}`);
    }
    if (t.v === 'new') {
      const ty = this.eat().v;
      if (ty === 'mat2') { // float4 after conversion (see convertJava); mat2(vec4) is the same layout, mat2(vec2, vec2) two columns
        const a = this.args();
        if (a.length === 4) return `MAT2_(${a.join(', ')})`;
        if (a.length === 1) return `MAT2_((${a[0]}).x, (${a[0]}).y, (${a[0]}).z, (${a[0]}).w)`;
        if (a.length === 2) return `MAT2_((${a[0]}).x, (${a[0]}).y, (${a[1]}).x, (${a[1]}).y)`;
        throw new Error('glsl: mat2 args');
      }
      if (ty === 'mat3') {
        const a = this.args();
        if (a.length === 9) return `mat3_make(${a.join(', ')})`;
        if (a.length === 3) return `mat3_make(${a.map((c) => `(${c}).x, (${c}).y, (${c}).z`).join(', ')})`;
        throw new Error('glsl: mat3 args');
      }
      if (ty in VEC_TYPES) {
        const a = this.args();
        const n = Number(ty[3]);
        if (a.length === 1) return `make_${VEC_TYPES[ty]}(${Array(n).fill(a[0]).join(', ')})`;
        if (ty === 'vec3' && a.length === 2) return `make_float3((${a[0]}).x, (${a[0]}).y, ${a[1]})`;
        return `make_${VEC_TYPES[ty]}(${a.join(', ')})`;
      }
      if (/^(int|double|float|boolean|long)$/.test(ty) && this.at('[')) { // new int[3] / new double[]{…}: left for the array regexes
        this.eat('['); let n = ''; if (!this.at(']')) n = this.expr(); this.eat(']');
        if (this.at('{')) { let d = 0, lit = ''; for (;;) { const t2 = this.eat(); lit += t2.v; if (t2.v === '{') d++; if (t2.v === '}') { d--; if (d === 0) break; } if (t2.t === 'op' && t2.v === ',') lit += ' '; } return `new ${ty}[]${lit}`; }
        return `new ${ty}[${n}]`;
      }
      if (/^[A-Z]\w*$/.test(ty) && this.at('(')) { const a = this.args(); return `new ${ty}(${a.join(', ')})`; } // objects: handled by the pod pass afterwards
      throw new Error(`glsl: new ${ty}`);
    }
    if (t.v === 'G' && this.at('.')) {
      this.eat('.'); const fn = this.eat().v; const a = this.args();
      if (fn === 'atan') return a.length === 2 ? `atan2f(${a[0]}, ${a[1]})` : `atanf(${a[0]})`;
      if (fn === 'Kscope') return `G_Kscope(${a.join(', ')})`;
      if (fn === 'rot') return `G_rot(${a[0]})`;
      if (fn === 'app') return `G_app(${a.join(', ')})`;
      const m = G_MAP[fn]; if (!m) throw new Error(`glsl: G.${fn}`);
      return `${m}(${a.join(', ')})`;
    }
    if (t.v === 'Math' && this.at('.')) { this.eat('.'); const fn = this.eat().v; if (this.at('(')) { const a = this.args(); return `Math.${fn}(${a.join(', ')})`; } return `Math.${fn}`; }
    // identifier / call
    if (this.at('(')) { const a = this.args(); return `${t.v}(${a.join(', ')})`; }
    return t.v;
  }
  postfix(recv: string): string {
    for (;;) {
      if (this.at('.')) {
        this.eat('.'); const name = this.eat().v;
        if (this.at('(')) {
          const a = this.args();
          if (name in VEC_BINOPS) { recv = `(${recv} ${VEC_BINOPS[name]} ${a[0]})`; continue; }
          if (name === 'times' || ((name === 'add' || name === 'minus' || name === 'division') && glslKindOf(recv) === 'mat2')) {
            const rk = glslKindOf(recv), ak = glslKindOf(a[0]);
            if (name !== 'times') { recv = `(${recv} ${VEC_BINOPS[name]} ${a[0]})`; continue; } // mat2 ± scalar, componentwise on the float4
            if (rk === 'vec' && ak === 'mat2') { recv = VEC_MUL_MAT2(recv, a[0]); continue; }
            if (rk === 'mat2' && ak === 'vec') { recv = MAT2_MUL_VEC(recv, a[0]); continue; }
            if (rk === 'vec' && ak === 'mat3') { recv = VEC_MUL_MAT3(recv, a[0]); continue; }
            if (rk === 'mat3' && ak === 'vec') { recv = MAT3_MUL_VEC(recv, a[0]); continue; }
            if (rk === 'mat2' && ak === 'scalar') { recv = `(${recv} * ${a[0]})`; continue; }
            if (rk === 'mat3' && ak === 'scalar') { recv = `mat3_scale(${recv}, ${a[0]})`; continue; }
            if (rk === 'mat3' || ak === 'mat3') throw new Error('glsl: mat3 product');
            recv = `(${recv} * ${a[0]})`; continue; // vec.times(scalar)
          }
          if (name === 'dot') { recv = `dot(${recv}, ${a[0]})`; continue; }
          if (name === 'length') { recv = `length(${recv})`; continue; }
          recv = `${recv}.${name}(${a.join(', ')})`; continue; // pContext.random() etc. — handled by the later regexes
        }
        recv = `${recv}.${name}`;
        continue;
      }
      if (this.at('[')) { this.eat('['); const e = this.expr(); this.eat(']'); recv = `${recv}[${e}]`; continue; }
      if (this.at('++') || this.at('--')) { recv = `${recv}${this.eat().v}`; continue; }
      return recv;
    }
  }
}
function splitTop(s: string): string[] { const out: string[] = []; let d = 0, cur = ''; for (const ch of s) { if ('([{'.includes(ch)) d++; else if (')]}'.includes(ch)) d--; if (ch === ',' && d === 0) { out.push(cur.trim()); cur = ''; } else cur += ch; } out.push(cur.trim()); return out; }
function glslExpr(toks: Tok[]): string { const p = new GlslExpr(toks); const e = p.expr(); if (p.i !== toks.length) throw new Error(`glsl: trailing ${toks[p.i]?.v}`); return e; }

/** Re-emit a Java body: statement structure kept, every expression parsed and re-emitted. */
let pendingArray: { ty: string; name: string } | null = null;
export function convertGlsl(code: string): string {
  const toks = glslTokenize(code);
  let out = '';
  let seg: Tok[] = [];
  let depth = 0;
  pendingArray = null;
  const flush = (term: string) => { out += convertSegment(seg) + term + '\n'; seg = []; };
  for (let ti = 0; ti < toks.length; ti++) {
    const t = toks[ti];
    if (t.t === 'op' && t.v === '{' && depth === 0 && seg.length >= 5 && seg[seg.length - 1].v === '=' && seg[seg.length - 3].v === ']' && seg[seg.length - 4].v === '[') {
      const tyTok = seg[seg.length - 5].v;
      pendingArray = { ty: VEC_TYPES[tyTok] ?? (tyTok === 'double' ? 'float' : tyTok === 'long' ? 'int' : tyTok === 'boolean' ? 'bool' : tyTok), name: seg[seg.length - 2].v };
      glslKinds.set(pendingArray.name, tyTok === 'mat2' ? 'mat2' : tyTok === 'mat3' ? 'mat3' : 'vec');
      seg = seg.slice(0, seg.length - 5);
      if (seg.length && seg[seg.length - 1].v !== 'final') { out += convertSegment(seg) + '\n'; }
      seg = [];
    }
    if (pendingArray && t.t === 'op' && t.v === '{' && depth === 0) {
      // collect the literal up to the matching '}' and emit the whole array declaration
      let d = 0, j = ti; const items: Tok[][] = [[]];
      for (; j < toks.length; j++) { const u = toks[j]; if (u.v === '{' || u.v === '(' || u.v === '[') d++; else if (u.v === '}' || u.v === ')' || u.v === ']') { d--; if (d === 0) break; } if (u.v === ',' && d === 1) items.push([]); else if (!(u.v === '{' && d === 1)) items[items.length - 1].push(u); }
      const its = items.filter((x) => x.length).map((x) => glslExpr(x));
      out += `${pendingArray.ty} ${pendingArray.name}[${its.length}] = { ${its.join(', ')} };\n`;
      pendingArray = null; ti = j; if (toks[ti + 1]?.v === ';') ti++; seg = []; continue;
    }
    if (t.t === 'op' && (t.v === '(' || t.v === '[')) depth++;
    if (t.t === 'op' && (t.v === ')' || t.v === ']')) depth--;
    if (t.t === 'op' && depth === 0 && (t.v === ';' || t.v === '{' || t.v === '}')) { flush(t.v); continue; }
    if (t.t === 'op' && depth === 0 && t.v === ':' && seg.length && (seg[0].v === 'case' || seg[0].v === 'default')) { flush(':'); continue; }
    seg.push(t);
  }
  if (seg.length) flush('');
  return out;
}
function convertSegment(seg: Tok[]): string {
  if (!seg.length) return '';
  const raw = () => seg.map((t) => t.v).join(' ');
  const first = seg[0].v;
  const exprOf = (ts: Tok[]) => (ts.length ? glslExpr(ts) : '');
  const closeParen = (from: number) => { let d = 0; for (let k = from; k < seg.length; k++) { if (seg[k].v === '(') d++; else if (seg[k].v === ')') { d--; if (d === 0) return k; } } return -1; };
  if (first === 'else' && seg.length === 1) return 'else';
  if (first === 'else') { return 'else ' + convertSegment(seg.slice(1)); }
  if (first === 'if' || first === 'while' || first === 'switch') {
    const c = closeParen(1); if (c < 0) return raw();
    const head = `${first} (${exprOf(seg.slice(2, c))})`;
    return c === seg.length - 1 ? head : head + ' ' + convertSegment(seg.slice(c + 1)); // single-statement body on the same segment
  }
  if (first === 'for') {
    const c = closeParen(1); if (c < 0) return raw();
    const inner = seg.slice(2, c);
    const parts: Tok[][] = [[]]; let d = 0;
    for (const t of inner) { if (t.v === '(') d++; if (t.v === ')') d--; if (t.v === ';' && d === 0) parts.push([]); else parts[parts.length - 1].push(t); }
    const head = `for (${parts.map((p) => convertSegment(p)).join('; ')})`;
    return c === seg.length - 1 ? head : head + ' ' + convertSegment(seg.slice(c + 1));
  }
  if (first === 'return') {
    const ra = seg.findIndex((t, i) => i > 1 && t.t === 'op' && t.v === '=');
    if (ra > 1 && seg.slice(1, ra).every((t) => t.t === 'id' || t.v === '.')) { const lhs = seg.slice(1, ra).map((t) => t.v).join(''); return `${lhs} = ${exprOf(seg.slice(ra + 1))};\nreturn ${lhs}`; } // return v = expr;
    return 'return' + (seg.length > 1 ? ' ' + exprOf(seg.slice(1)) : '');
  }
  if (first === 'case') return 'case ' + exprOf(seg.slice(1));
  if (first === 'default' || first === 'break' || first === 'continue') return raw();
  // declaration: [final] type [ [] ] name [= expr] {, name [= expr]}
  let k = 0;
  if (seg[k]?.v === 'final') k++;
  const typeTok = seg[k];
  const isType = typeTok?.t === 'id' && (typeTok.v in VEC_TYPES || /^(double|int|float|boolean|long|short|mat2|mat3)$/.test(typeTok.v) || (/^[A-Z]\w*$/.test(typeTok.v) && !/^(Math|G|Integer|Double|Float|Tools|MathLib|M_\w+|PI)$/.test(typeTok.v))) && seg[k + 1] && (seg[k + 1].t === 'id' || seg[k + 1].v === '[');
  if (isType) {
    let ty = typeTok.v; k++;
    let arr = '';
    if (seg[k]?.v === '[' && seg[k + 1]?.v === ']') { arr = '[]'; k += 2; }
    const jty = ty;
    ty = VEC_TYPES[ty] ?? (ty === 'double' ? 'float' : ty === 'long' || ty === 'short' ? 'int' : ty === 'boolean' ? 'bool' : ty === 'mat2' ? 'float4' : ty === 'mat3' ? 'mat3_' : ty);
    const rest = seg.slice(k);
    // `vec2[] a = { new vec2(…), … };` → `float2 a[N] = { … }` (the '{' ends the segment in convertGlsl, so the
    // literal arrives as its own segments; handled by convertGlsl's arrayLiteral state instead)
    // split declarators on top-level commas
    const decls: Tok[][] = [[]]; let d = 0;
    for (const t of rest) { if ('([{'.includes(t.v)) d++; if (')]}'.includes(t.v)) d--; if (t.v === ',' && d === 0) decls.push([]); else decls[decls.length - 1].push(t); }
    for (const dc of decls) if (dc[0]?.t === 'id') { if (jty === 'mat2' || jty === 'mat3') glslKinds.set(dc[0].v, jty); else if (jty in VEC_TYPES) glslKinds.set(dc[0].v, 'vec'); else glslKinds.delete(dc[0].v); }
    const after: string[] = []; // `double o, ot2, ot = ot2 = 1000.0;` → declare ot = 1000.0, then ot2 = ot
    const decl = `${ty}${arr} ` + decls.map((dc) => {
      const eq = dc.findIndex((t) => t.v === '='); if (eq < 0) return dc.map((t) => t.v).join('');
      let rhs = dc.slice(eq + 1); const nm = dc.slice(0, eq).map((t) => t.v).join('');
      for (;;) { const eq2 = rhs.findIndex((t) => t.t === 'op' && t.v === '='); if (eq2 <= 0 || !rhs.slice(0, eq2).every((t) => t.t === 'id' || t.v === '.')) break; after.push(`${rhs.slice(0, eq2).map((t) => t.v).join('')} = ${nm}`); rhs = rhs.slice(eq2 + 1); }
      return `${nm} = ${exprOf(rhs)}`;
    }).join(', ');
    return after.length ? decl + ';\n' + after.join(';\n') : decl;
  }
  // assignment: lvalue op= expr
  const asg = seg.findIndex((t) => t.t === 'op' && /^(=|\+=|-=|\*=|\/=|%=|&=|\|=|\^=)$/.test(t.v));
  if (asg > 0) {
    const rhs = seg.slice(asg + 1);
    const asg2 = rhs.findIndex((t) => t.t === 'op' && t.v === '=');
    if (asg2 > 0 && seg[asg].v === '=') { // chained a = b = c → b = c; a = b
      const inner = convertSegment(rhs);
      return `${inner};\n${exprOf(seg.slice(0, asg))} = ${exprOf(rhs.slice(0, asg2))}`;
    }
    return `${exprOf(seg.slice(0, asg))} ${seg[asg].v} ${exprOf(rhs)}`;
  }
  return exprOf(seg);
}

// ---------------------------------------------------------------- main
function main() {
const ports: Port[] = [];
const failures: [string, string][] = [];
const fileByName = fileIndex();

for (const d of targets) {
  const file = fileByName.get(d.name);
  if (!file) { failures.push([d.name, 'no java file']); continue; }
  const src = fs.readFileSync(file, 'latin1');
  try {
    const port = convertVariation(d, src, path.relative(jwfRoot, file));
    ports.push(port);
  } catch (e) {
    failures.push([d.name, String((e as Error).message ?? e)]);
  }
}

// ---------------------------------------------------------------- output
const out = path.join(here, 'data/jwf-java-ports.jsonl');
fs.writeFileSync(out, ports.map((p) => JSON.stringify(p)).join('\n') + '\n');
console.log(`java2cu: ${ports.length} converted, ${failures.length} skipped → ${path.relative(process.cwd(), out)}`);
const byReason = new Map<string, string[]>();
for (const [n, r] of failures) { const k = r.replace(/\b\w+:/, '').slice(0, 40); byReason.set(k, [...(byReason.get(k) ?? []), n]); }
for (const [r, ns] of [...byReason.entries()].sort((a, b) => b[1].length - a[1].length)) console.log(`  ${ns.length.toString().padStart(3)}  ${r}  [${ns.slice(0, 8).join(' ')}${ns.length > 8 ? ' …' : ''}]`);
if (process.env.JAVA2CU_FAILS) fs.writeFileSync(process.env.JAVA2CU_FAILS, failures.map(([n, r]) => `${n}\t${r}`).join('\n') + '\n');

}
if (!process.env.JAVA2CU_LIB) main();
