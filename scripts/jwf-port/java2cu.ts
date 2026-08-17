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
if (!fs.existsSync(varDir)) { console.error(`JWildfire sources not found at ${varDir} (use --jwf)`); process.exit(1); }

interface DumpVar { name: string; params: { name: string; def: number; int: boolean }[]; gpuCode?: string; gpu?: boolean; resources?: number }
const dump: DumpVar[] = fs.readFileSync(path.join(here, 'data/jwf-variations.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const dumpByName = new Map(dump.map((d) => [d.name, d]));

// name → java file (by getName())
const files = walk(varDir).filter((f) => f.endsWith('.java'));
const fileByName = new Map<string, string>();
for (const f of files) {
  const s = fs.readFileSync(f, 'latin1');
  const m = /public String getName\(\)\s*\{\s*return\s+"([A-Za-z0-9_]+)"\s*;/.exec(s);
  if (m && !fileByName.has(m[1])) fileByName.set(m[1], f);
}
function walk(d: string): string[] { return fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]); }

const targets = dump.filter((d) => !d.gpuCode && (!only.length || only.includes(d.name)));

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
  'preprocess', 'getSupportedTransformationTypes', 'getSourceCode', 'getFuncSource', 'getRandomFuncName']);

interface Method { name: string; ret: string; params: string; body: string; static: boolean }
interface Field { type: string; name: string; init: string | null; array: number | null }

// ---------------------------------------------------------------- Java class model
function parseClass(src0: string) {
  const src = stripComments(src0);
  const cls = /public\s+(?:abstract\s+)?class\s+(\w+)\s+extends\s+([\w.]+)/.exec(src);
  if (!cls) throw new Error('no class');
  const body = braceBody(src, cls.index)!.body;
  // strip nested classes/enums (mark ranges)
  const methods: Method[] = [];
  const fields: Field[] = [];
  const consts = new Map<string, string>(); // PARAM_X → "x"
  let i = 0;
  const skipRanges: [number, number][] = [];
  const mre = /(?:public|private|protected)?\s*(static\s+)?(?:final\s+)?(?:synchronized\s+)?([\w<>\[\],. ]+?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws [\w., ]+)?\s*\{/g;
  let m: RegExpExecArray | null;
  while ((m = mre.exec(body))) {
    if (['if', 'for', 'while', 'switch', 'catch', 'return', 'new', 'else'].includes(m[3])) continue;
    const b = braceBody(body, m.index + m[0].length - 1);
    if (!b) break;
    let ret = m[2].trim(); let isStatic = !!m[1];
    for (;;) { const r2 = ret.replace(/^(?:public|private|protected|static|final|synchronized|abstract)\s+/, ''); if (r2 === ret) break; if (/^static\b/.test(ret)) isStatic = true; ret = r2; }
    methods.push({ name: m[3], ret, params: m[4], body: b.body, static: isStatic });
    skipRanges.push([m.index, b.end]);
    mre.lastIndex = b.end;
  }
  // nested class / enum / static blocks
  for (const cm of body.matchAll(/(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:class|enum|interface)\s+\w+[^{]*\{/g)) {
    const b = braceBody(body, cm.index! + cm[0].length - 1);
    if (b) skipRanges.push([cm.index!, b.end]);
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
    if (type0.startsWith('String') || /^(Random|MersenneTwister|Object|Layer|XForm|Flame|AbstractRandomGenerator|RandomGeneratorType|Vec\w*)$/.test(type0)) continue;
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
  return { className: cls[1], parent: cls[2], methods, fields, consts, body };
}

// ---------------------------------------------------------------- Java expression → CUDA dialect
const MATH_MAP: Record<string, string> = {
  'Math.sin': 'sinf', 'Math.cos': 'cosf', 'Math.tan': 'tanf', 'Math.sqrt': 'sqrtf', 'Math.abs': 'fabsf', 'Math.atan2': 'atan2f', 'Math.atan': 'atanf',
  'Math.asin': 'asinf', 'Math.acos': 'acosf', 'Math.pow': 'powf', 'Math.exp': 'expf', 'Math.log': 'logf', 'Math.log10': 'log10f', 'Math.floor': 'floorf', 'Math.ceil': 'ceilf',
  'Math.round': 'roundf', 'Math.sinh': 'sinhf', 'Math.cosh': 'coshf', 'Math.tanh': 'tanhf', 'Math.max': 'fmaxf', 'Math.min': 'fminf', 'Math.hypot': 'hypotf',
  'Math.cbrt': 'cbrtf', 'Math.signum': 'sign', 'Math.random': 'RANDFLOAT', 'Math.rint': 'rintf', 'Math.toRadians': 'radians', 'Math.expm1': 'expm1f', 'Math.log1p': 'log1pf',
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

interface Ctx { name: string; params: Set<string>; fieldMap: Map<string, string>; state: Set<string>; helperNames: Set<string>; usedHelperParams: Set<string>; inHelper: boolean }

function convertJava(code: string, ctx: Ctx, extraLocals: Iterable<string> = []): string {
  let s = code;
  if (/\b(vec[234]|G\.|mat2)\b/.test(s)) s = convertGlsl(s);
  s = s.replace(/\bfinal\s+/g, '').replace(/\bthis\.(\w)/g, 'THISF_$1');
  s = s.replace(/\b(?:double|long|short)\b/g, 'float').replace(/\bboolean\b/g, 'bool');
  // locals declared in this body shadow fields of the same name (`double x = …` in dc_cube)
  const locals = new Set<string>(extraLocals);
  for (const dm of s.matchAll(/(?<![\w.])(?:float|int|bool)\s*(?:\[\s*\])?\s+([^;{}()]*?)(?:;|\)\s*\{|=\s*\{)/g)) {
    for (const part of dm[1].split(',')) { const nm = /^\s*(\w+)/.exec(part); if (nm) locals.add(nm[1]); }
  }
  for (const dm of s.matchAll(/for\s*\(\s*(?:float|int)\s+(\w+)/g)) locals.add(dm[1]);
  // pContext
  s = s.replace(/pContext\.random\(\s*\)/g, 'RANDFLOAT()');
  // random(Integer.MAX_VALUE): a 31-bit uniform int (its low bits are used as coin flips — RANDFLOAT()*2^31 in f32 has none)
  s = s.replace(/pContext\.random\(\s*Integer\.MAX_VALUE\s*\)/g, '((int)(RANDINT() >> 1))');
  s = s.replace(/pContext\.random\(([^()]*(?:\([^()]*\))?[^()]*)\)/g, '((int)(RANDFLOAT() * (float)($1)))');
  s = s.replace(/pContext\.isPreserveZCoordinate\(\)/g, 'false');
  s = s.replace(/randGen\.random\(\)|randomize\.nextDouble\(\)|rand\.nextDouble\(\)|_random\.random\(\)/g, 'RANDFLOAT()');
  if (!ctx.helperNames.has('random')) s = s.replace(/(?<![\w.])random\(\s*\)/g, 'RANDFLOAT()'); // static import of Math.random
  // points
  s = s.replace(/pAffineTP\.getPrecalcSumsq\(\)/g, '__r2').replace(/pAffineTP\.getPrecalcSqrt\(\)/g, '__r')
    .replace(/pAffineTP\.getPrecalcAtan\(\)/g, '__phi').replace(/pAffineTP\.getPrecalcAtanYX\(\)/g, '__theta')
    .replace(/pAffineTP\.getPrecalcSinA\(\)/g, '(__x / __r)').replace(/pAffineTP\.getPrecalcCosA\(\)/g, '(__y / __r)');
  s = s.replace(/pAffineTP\.x\b/g, '__x').replace(/pAffineTP\.y\b/g, '__y').replace(/pAffineTP\.z\b/g, '__z').replace(/pAffineTP\.color\b/g, '__pal').replace(/pAffineTP\.doHide\b/g, '__doHide');
  // chained assignment `a = b = c;` → `b = c; a = b;`
  s = s.replace(/(?<![=!<>])\b([\w.>-]+)\s*=(?!=)\s*([\w.>-]+)\s*=(?!=)\s*([^;=]+);/g, '$2 = $3; $1 = $2;');
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
  s = s.replace(/sinAndCos\(([^;]*?),\s*(\w+),\s*(\w+)\)\s*;/g, (_m, a: string, sn: string, cs: string) => `${sn}_v = sinf(${a}); ${cs}_v = cosf(${a});`);
  s = s.replace(/\b(\w+)\.value\b/g, '$1_v');
  // Math.* / Tools.* / Integer.*
  s = s.replace(/\bFastMath\./g, 'Math.');
  s = s.replace(/\b(Math|MathLib|Tools|Integer|Double|Float)\.(\w+)/g, (m0) => MATH_MAP[m0] ?? m0);
  // qualified statics not mapped → leave (transpiler will report)
  // bare MathLib names
  s = s.replace(/\b([a-z]\w*)\s*\(/g, (m0, fn: string) => (BARE_MAP[fn] && !ctx.helperNames.has(fn) ? BARE_MAP[fn] + '(' : m0));
  // casts: (int) fine; (float) fine; (long)→(int)
  s = s.replace(/\(\s*(?:long|short)\s*\)/g, '(int)');
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
    if (ctx.state.has(id)) return `varpar->${ctx.name}_${id}`;
    return forced ? id : m0;
  });
  s = s.replace(/THISF_/g, '');
  return s;
}

// Hand patches on the converted text (Java idioms that do not survive f32 literally).
const JAVA_PATCHES: Record<string, [string | RegExp, string][]> = {
  // random(Integer.MAX_VALUE) * 2π/power: only k mod |power| matters, and k·2π/power in f32 is noise
  // elliptic coordinates: cosh(mu) − 1 cancels near the real axis in f32; compute it stably
  // 4-entry ring buffer of randoms (sliding Irwin-Hall sum): four fresh randoms have the same distribution
  dc_cylinder2: [['float rr = __dc_cylinder2_blur * (r[0] + r[1] + r[2] + r[3] - 2.0);', 'float rr = __dc_cylinder2_blur * (RANDFLOAT() + RANDFLOAT() + RANDFLOAT() + RANDFLOAT() - 2.0);'],
    [/r\[varpar->dc_cylinder2_n\] = RANDFLOAT\(\);\s*varpar->dc_cylinder2_n = varpar->dc_cylinder2_n \+ 1 & 3;/, '']],
  rosoni: [['cerc ^= (r2 <= 0.0);', 'cerc = cerc != (r2 <= 0.0);']],
  eSwirl: [
    ['float xmax = (eSwirl_sqrt_safe(tmp + tmp2) + eSwirl_sqrt_safe(tmp - tmp2)) * 0.5;', 'float ea = eSwirl_sqrt_safe(tmp + tmp2); float eb = eSwirl_sqrt_safe(tmp - tmp2); float xmax = (ea + eb) * 0.5;\n    float ed = (__y * __y / (ea + fabsf(__x + 1.0)) + __y * __y / (eb + fabsf(__x - 1.0))) * 0.5 + fmaxf(fabsf(__x) - 1.0, 0.0);'],
    ['float mu = acoshf(xmax);', 'float mu = (ed < 1.0e-4) ? sqrtf(2.0 * ed) * (1.0 - ed / 12.0) : logf(1.0 + ed + sqrtf(ed * (2.0 + ed)));'],
  ],
  ...Object.fromEntries(['post_juliaq', 'post_julia3Dq', 'jubiQ'].map((n) => [n, [[/\(\(int\)\(RANDINT\(\) >> 1\)\)/g, `((int)(RANDFLOAT() * fabsf((float)__${n}_power)))`]] satisfies [string | RegExp, string][]])),
};

// ---------------------------------------------------------------- per-variation conversion
interface Port { name: string; gpuCode: string; gpuFunctions: string; extraParams: string[]; note: string; javaFile: string }
const MERGE_PARENTS: Record<string, string> = { GLSLFunc: 'GLSLFunc.java' };
function convertVariation(d: DumpVar, src: string, javaFile: string): Port {
  const cls = parseClass(src);
  // known abstract parents whose fields/params/setParameter the subclass relies on (GLSLFunc: resolution, gradient)
  if (MERGE_PARENTS[cls.parent]) {
    const pf = files.find((f) => f.endsWith('/' + MERGE_PARENTS[cls.parent]));
    if (!pf) throw new Error(`parent source ${cls.parent} not found`);
    const par = parseClass(fs.readFileSync(pf, 'latin1'));
    for (const [k, v] of par.consts) if (!cls.consts.has(k)) cls.consts.set(k, v);
    for (const f of par.fields) if (!cls.fields.some((x) => x.name === f.name)) cls.fields.push(f);
    for (const m of par.methods) {
      if (m.name === 'transform' || m.name === 'init' || m.name === 'initOnce' || m.name === 'dbl2int') continue;
      const own = cls.methods.find((x) => x.name === m.name);
      if (!own) cls.methods.push(m);
      else if (m.name === 'setParameter') own.body = own.body + '\n' + m.body; // subclass falls through to super.setParameter
    }
    cls.parent = 'VariationFunc';
  }
  const transform = cls.methods.find((m) => m.name === 'transform');
  if (!transform) throw new Error('no transform()');
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
      const asg = /\{?\s*(?:this\.)?(\w+)\s*=/.exec(br.slice(cond.index + cond[0].length));
      if (asg && asg[1] !== 'pValue') fieldMap.set(asg[1], key);
    }
  }
  const fieldNames = new Set(cls.fields.map((f) => f.name));
  // params whose field name is unknown default to the param name
  for (const pn of paramNames) if (![...fieldMap.values()].includes(pn) && fieldNames.has(pn)) fieldMap.set(pn, pn);
  const paramFields = new Set(fieldMap.keys());
  for (const pn of paramNames) if (![...fieldMap.values()].includes(pn)) throw new Error(`param ${pn}: field not found`);
  // helper methods (non-static and static), other than the framework ones
  const helpers = cls.methods.filter((m) => !KNOWN_METHODS.has(m.name));
  const init = cls.methods.find((m) => m.name === 'init');
  const initOnce = cls.methods.find((m) => m.name === 'initOnce');
  if (initOnce && /\S/.test(initOnce.body) && !/super\.initOnce/.test(initOnce.body)) throw new Error('initOnce');
  // which non-param fields are written in transform() (state) vs only in init() (precalc)
  const localsOf = (body: string) => {
    const b = body.replace(/\b(?:double|long|short)\b/g, 'float').replace(/\bboolean\b/g, 'bool');
    const out = new Set<string>();
    for (const dm of b.matchAll(/(?<![\w.])(?:float|int|bool)\s*(?:\[\s*\])?\s+([^;{}()]*?)(?:;|\)\s*\{|=\s*\{)/g)) for (const part of dm[1].split(',')) { const nm = /^\s*(\w+)/.exec(part); if (nm) out.add(nm[1]); }
    for (const dm of b.matchAll(/for\s*\(\s*(?:float|int)\s+(\w+)/g)) out.add(dm[1]);
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
  for (const h of helpers) for (const f of nonParamFields) if (f.array === null && new RegExp(`(?<![\\w.])${f.name}\\b`).test(h.body)) state.add(f.name);
  // params written in transform → also state (initialised from the param)
  const paramState: string[] = [];
  for (const f of paramFields) if (tAssigned.has(f)) { state.add(f); paramState.push(f); }
  // init() that writes state (attractor start values) runs once → inside the first-call block; the
  // other fields it writes must then persist too
  const initAssigned = init ? assignedIn(init.body) : new Set<string>();
  const initTouchesState = [...initAssigned].some((f) => state.has(f));
  if (initTouchesState) for (const f of nonParamFields) if (f.array === null && initAssigned.has(f.name)) state.add(f.name);
  const arrayFields = nonParamFields.filter((f) => f.array !== null);
  for (const f of arrayFields) if (f.array === -1) throw new Error(`array field ${f.name} of unknown size`);
  if (state.size > 8) throw new Error(`too much state (${[...state].join(',')})`);
  const helperNames = new Set(helpers.map((h) => h.name));
  const ctx: Ctx = { name: d.name, params: new Set([...paramFields].filter((f) => !state.has(f))), fieldMap, state, helperNames, usedHelperParams: new Set(), inHelper: false };
  const others = new Set(['pContext', 'pXForm', 'pLayer', 'pAffineTP', 'pVarTP', 'pAmount']);
  // --- helpers
  let funcs = '';
  for (const h of helpers) {
    ctx.inHelper = true;
    const hparams = h.params.trim() ? h.params.split(',').map((p) => p.trim()).filter((p) => !/^(FlameTransformationContext|XForm|Layer)\b/.test(p)) : [];
    const pnames = hparams.map((p) => p.split(/\s+/).pop()!.replace(/\[\]/g, ''));
    const params = hparams.map((p) => convertJava(p, ctx, pnames).replace(/\bfloat\s+/, 'float ')).join(', ');
    const ret = h.ret === 'double' || h.ret === 'long' ? 'float' : h.ret === 'boolean' ? 'bool' : (VEC_TYPES[h.ret] ?? h.ret);
    if (!/^(float|int|bool|void|float2|float3|float4)$/.test(ret)) throw new Error(`helper ${h.name} returns ${h.ret}`);
    const body = convertJava(h.body, ctx, pnames);
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
    const t = f.type === 'double' || f.type === 'long' ? 'float' : f.type === 'boolean' ? 'bool' : (VEC_TYPES[f.type] ?? f.type);
    if (!/^(float|int|bool|float2|float3|float4)$/.test(t)) throw new Error(`field ${f.name}: ${f.type}`);
    code += `${t} ${f.name}${f.init !== null ? ' = ' + convertJava(f.init, ctx) : ' = 0'};\n`;
  }
  for (const f of arrayFields) {
    const t = f.type === 'double' ? 'float' : f.type === 'boolean' ? 'bool' : f.type;
    code += `${t} ${f.name}[${f.array}]${f.init ? ' = ' + f.init : ''};\n`;
  }
  // state: first-call init from Java initialiser (or param value)
  const stateType = (n: string) => { const f = cls.fields.find((x) => x.name === n); const t = f?.type ?? 'double'; return t in VEC_TYPES ? ':' + VEC_TYPES[t] : t === 'int' ? ':int' : ''; };
  if (state.size) {
    extra.push('inited_', ...[...state].map((n) => n + stateType(n)));
    code += `if (varpar->${d.name}_inited_ == 0.0f) {\n  varpar->${d.name}_inited_ = 1.0f;\n`;
    for (const f of cls.fields) {
      if (!state.has(f.name)) continue;
      const initExpr = paramState.includes(f.name) ? `__${d.name}_${fieldMap.get(f.name)}` : f.init !== null ? convertJava(f.init, ctx) : '0';
      code += `  varpar->${d.name}_${f.name} = ${initExpr};\n`;
    }
    if (initTouchesState && init) code += '  {\n' + convertJava(init.body, ctx) + '\n  }\n';
    // trajectory state (attractor coordinates: float state written by transform()): Java runs a
    // handful of long identical trajectories; our 65k walkers would all trace the same short
    // one, so each thread starts a hair off — chaotic maps decorrelate within the fuse
    for (const f of cls.fields) {
      if (!state.has(f.name) || !tAssigned.has(f.name) || f.array !== null || f.type === 'int' || f.type === 'boolean' || f.type in VEC_TYPES) continue;
      code += `  varpar->${d.name}_${f.name} += (RANDFLOAT() - 0.5f) * 1.0e-4f;\n`;
    }
    code += '}\n';
  }
  // state fields that are only read by helpers (never written per point) are plain derived values:
  // reset them from their initialiser every call before the setParameter/init replay
  for (const f of nonParamFields) {
    if (!state.has(f.name) || f.array !== null || tAssigned.has(f.name) || hAssigned.has(f.name) || initAssigned.has(f.name)) continue;
    code += `varpar->${d.name}_${f.name} = ${f.init !== null ? convertJava(f.init, ctx) : '0'};\n`;
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
    if (/super\.init/.test(init.body)) throw new Error('init calls super');
    code += '{\n' + convertJava(init.body, ctx) + '\n}\n';
  }
  code += convertJava(transform.body, ctx);
  code = prefixHelpers(code);
  for (const [from, to] of JAVA_PATCHES[d.name] ?? []) { const c2 = code.replace(from, to); if (c2 === code) throw new Error(`patch did not match: ${from}`); code = c2; }
  // sinAndCos locals
  const scv = new Set([...(code + funcs).matchAll(/\b(\w+)_v\b/g)].map((m) => m[1]));
  const declV = [...scv].filter((n) => /^(sina?|cosa?|sin\w*|cos\w*|s|c)$/.test(n) || /_v = (sinf|cosf)\(/.test(code)).map((n) => `float ${n}_v = 0.0f;`).join(' ');
  if (declV) code = declV + '\n' + code;
  // sanity: leftovers the transpiler cannot know
  const bad = /\b(pContext|pXForm|pLayer|new\s+\w+|super\.|Random\b|String\b|\.length\b|Object\b|throw\b|try\b|catch\b|instanceof|\bnull\b|Complex\b|vec[234]\b|mat[234]\b|MAT2\()\b/.exec(code + funcs);
  if (bad) throw new Error(`unsupported construct: ${bad[0]}`);
  for (const o of others) if (new RegExp(`\\b${o}\\b`).test(code + funcs)) throw new Error(`leftover ${o}`);
  return { name: d.name, gpuCode: code, gpuFunctions: funcs, extraParams: extra, note: `java port of ${javaFile}`, javaFile };
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
const VEC_TYPES: Record<string, string> = { vec2: 'float2', vec3: 'float3', vec4: 'float4' };
const G_MAP: Record<string, string> = { abs: 'abs', mod: 'mod', mix: 'mix', smoothstep: 'smoothstep', floor: 'floor', fract: 'fract', clamp: 'clamp', sin: 'sin', cos: 'cos', tan: 'tan',
  step: 'step', normalize: 'normalize', min: 'min', max: 'max', pow: 'pow', exp: 'exp', exp2: 'exp2', log2: 'log2', sign: 'sign', cross: 'cross', sqrt: 'sqrt', length: 'length', dot: 'dot', distance: 'distance',
  trunc: 'trunc', round: 'round', ceil: 'ceil', reflect: 'reflect', tanh: 'tanh', sinh: 'sinh', cosh: 'cosh', invSqrt: 'rsqrtf', atan2: 'atan2f' };
const VEC_BINOPS: Record<string, string> = { plus: '+', add: '+', minus: '-', multiply: '*', division: '/' };
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
      const tt = ty === 'double' || ty === 'long' ? 'float' : ty === 'boolean' ? 'bool' : ty;
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
      if (ty in VEC_TYPES) {
        const a = this.args();
        const n = Number(ty[3]);
        if (a.length === 1) return `make_${VEC_TYPES[ty]}(${Array(n).fill(a[0]).join(', ')})`;
        if (ty === 'vec3' && a.length === 2) return `make_float3((${a[0]}).x, (${a[0]}).y, ${a[1]})`;
        return `make_${VEC_TYPES[ty]}(${a.join(', ')})`;
      }
      if (ty === 'mat2') { const a = this.args(); return `MAT2(${a.join(', ')})`; } // consumed by .times() below
      if (/^(int|double|float|boolean|long)$/.test(ty) && this.at('[')) { // new int[3] / new double[]{…}: left for the array regexes
        this.eat('['); let n = ''; if (!this.at(']')) n = this.expr(); this.eat(']');
        if (this.at('{')) { let d = 0, lit = ''; for (;;) { const t2 = this.eat(); lit += t2.v; if (t2.v === '{') d++; if (t2.v === '}') { d--; if (d === 0) break; } if (t2.t === 'op' && t2.v === ',') lit += ' '; } return `new ${ty}[]${lit}`; }
        return `new ${ty}[${n}]`;
      }
      throw new Error(`glsl: new ${ty}`);
    }
    if (t.v === 'G' && this.at('.')) {
      this.eat('.'); const fn = this.eat().v; const a = this.args();
      if (fn === 'atan') return a.length === 2 ? `atan2f(${a[0]}, ${a[1]})` : `atanf(${a[0]})`;
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
          if (name === 'times') {
            const mm = /^MAT2\((.*)\)$/.exec(a[0]);
            if (mm) { // vec2 · mat2(a00,a10,a01,a11) = (a00 x + a10 y, a01 x + a11 y)
              const p = splitTop(mm[1]); if (p.length !== 4) throw new Error('glsl: mat2 args');
              recv = `make_float2((${p[0]}) * (${recv}).x + (${p[1]}) * (${recv}).y, (${p[2]}) * (${recv}).x + (${p[3]}) * (${recv}).y)`; continue;
            }
            throw new Error('glsl: times() with a matrix variable');
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
export function convertGlsl(code: string): string {
  const toks = glslTokenize(code);
  let out = '';
  let seg: Tok[] = [];
  let depth = 0;
  const flush = (term: string) => { out += convertSegment(seg) + term + '\n'; seg = []; };
  for (const t of toks) {
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
  if (first === 'return') return 'return' + (seg.length > 1 ? ' ' + exprOf(seg.slice(1)) : '');
  if (first === 'case') return 'case ' + exprOf(seg.slice(1));
  if (first === 'default' || first === 'break' || first === 'continue') return raw();
  // declaration: [final] type [ [] ] name [= expr] {, name [= expr]}
  let k = 0;
  if (seg[k]?.v === 'final') k++;
  const typeTok = seg[k];
  const isType = typeTok?.t === 'id' && (typeTok.v in VEC_TYPES || /^(double|int|float|boolean|long|short|mat2)$/.test(typeTok.v)) && seg[k + 1] && (seg[k + 1].t === 'id' || seg[k + 1].v === '[');
  if (isType) {
    let ty = typeTok.v; k++;
    let arr = '';
    if (seg[k]?.v === '[' && seg[k + 1]?.v === ']') { arr = '[]'; k += 2; }
    ty = VEC_TYPES[ty] ?? (ty === 'double' || ty === 'long' ? 'float' : ty === 'boolean' ? 'bool' : ty);
    if (ty === 'mat2') throw new Error('glsl: mat2 variable');
    const rest = seg.slice(k);
    // split declarators on top-level commas
    const decls: Tok[][] = [[]]; let d = 0;
    for (const t of rest) { if ('([{'.includes(t.v)) d++; if (')]}'.includes(t.v)) d--; if (t.v === ',' && d === 0) decls.push([]); else decls[decls.length - 1].push(t); }
    return `${ty}${arr} ` + decls.map((dc) => { const eq = dc.findIndex((t) => t.v === '='); if (eq < 0) return dc.map((t) => t.v).join(''); return `${dc.slice(0, eq).map((t) => t.v).join('')} = ${exprOf(dc.slice(eq + 1))}`; }).join(', ');
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

}
if (!process.env.JAVA2CU_LIB) main();
