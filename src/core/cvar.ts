// c_var / pre_c_var / post_c_var — JWildfire's ZVarFunc family (Jesus Sosa; LGPL 2.1+, see NOTICE.md): the point
// (or a random one, mode 1), scaled by `zoom` and shifted, goes through a complex function the user writes as a Java
// method body over js.glsl's vec2 complex helpers:
//
//     import js.glsl.vec2;
//     public vec2 f(vec2 z) {
//       vec2 a = c_add(c_inv(z), c_exp(c_inv(z)));
//       return c_add(a, c_exp(new vec2(0.0, 0.0)));
//     }
//
// JWildfire compiles that with Janino; WilderFire parses the body — `vec2 <name> = <expr>;` declarations and a
// `return <expr>;`, where an expression is `z`, a declared name, `new vec2(<scalar>, <scalar>)`, one of the c_*
// helpers of js.glsl.G applied to complex/scalar arguments, or a `vec2` method call (`.plus/.minus/.multiply/.division`
// and friends) — and emits it as WGSL over a port of those helpers (src/core/formula.ts compiles the scalar
// sub-expressions). Comments are stripped first. Anything else is rejected and the instance maps to 0 (JWildfire's
// runner stays null and the transform writes nothing). The helpers are glslFuncRunner's own copies (js.glsl.G has
// another set using atan(y/x) — the runner's use its rational `atan2` approximation, up to 0.07 rad off: `G_atan2`,
// the kernel helper the transpiled cut_*/dc_* ports share), so c_to_polar — hence c_ln, c_sqrt, c_pow, … — keep it.
import { formulaToWgsl } from './formula';

/** ZVarFunc's default code (the uncommented lines of its code_func) */
export const CVAR_DEFAULT_CODE = 'import js.glsl.vec2;\npublic vec2 f(vec2 z)\n{\n  vec2 a=c_add(c_inv(z),c_exp(c_inv(z)));\nreturn c_add(a,c_exp(new vec2(0.0, 0.0)));\n}';

type Tok = { t: 'id' | 'num' | 'op'; v: string; a: number; b: number };
function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; continue; }
    if ((c >= '0' && c <= '9') || (c === '.' && src[i + 1] >= '0' && src[i + 1] <= '9')) {
      const m = /^(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?[dDfFlL]?/.exec(src.slice(i))!;
      out.push({ t: 'num', v: m[0], a: i, b: i + m[0].length }); i += m[0].length; continue;
    }
    if (/[A-Za-z_]/.test(c)) { const m = /^[A-Za-z_]\w*/.exec(src.slice(i))!; out.push({ t: 'id', v: m[0], a: i, b: i + m[0].length }); i += m[0].length; continue; }
    const two = src.slice(i, i + 2);
    if (['<=', '>=', '==', '!=', '&&', '||'].includes(two)) { out.push({ t: 'op', v: two, a: i, b: i + 2 }); i += 2; continue; }
    if ('+-*/%()?:,<>!=;{}.'.includes(c)) { out.push({ t: 'op', v: c, a: i, b: i + 1 }); i++; continue; }
    throw new Error(`unexpected "${c}"`);
  }
  return out;
}

/** complex helpers: name → [argument kinds, WGSL function]; 'c' = complex, 's' = scalar */
const CFN: Record<string, { args: string; fn: string }> = {
  c_one: { args: '', fn: 'cv_one' }, c_i: { args: '', fn: 'cv_i' }, c_ni: { args: '', fn: 'cv_ni' },
  c_conj: { args: 'c', fn: 'cv_conj' }, c_from_polar: { args: 'ss', fn: 'cv_from_polar' }, c_to_polar: { args: 'c', fn: 'cv_to_polar' },
  c_exp: { args: 'c|sc', fn: 'cv_exp' }, c_ln: { args: 'c', fn: 'cv_ln' }, c_log: { args: 'cs', fn: 'cv_log' }, c_sqrt: { args: 'c', fn: 'cv_sqrt' },
  c_pow: { args: 'cs|cc', fn: 'cv_pow' }, c_add: { args: 'cc', fn: 'cv_add' }, c_sub: { args: 'cc', fn: 'cv_sub' }, c_mul: { args: 'cc', fn: 'cv_mul' }, c_div: { args: 'cc', fn: 'cv_div' },
  c_sin: { args: 'c', fn: 'cv_sin' }, c_cos: { args: 'c', fn: 'cv_cos' }, c_tan: { args: 'c', fn: 'cv_tan' }, c_atan: { args: 'c', fn: 'cv_atan' }, c_asin: { args: 'c', fn: 'cv_asin' }, c_acos: { args: 'c', fn: 'cv_acos' },
  c_sinh: { args: 'c', fn: 'cv_sinh' }, c_cosh: { args: 'c', fn: 'cv_cosh' }, c_tanh: { args: 'c', fn: 'cv_tanh' }, c_asinh: { args: 'c', fn: 'cv_asinh' }, c_acosh: { args: 'c', fn: 'cv_acosh' }, c_atanh: { args: 'c', fn: 'cv_atanh' },
  c_rem: { args: 'cc', fn: 'cv_rem' }, c_inv: { args: 'c', fn: 'cv_inv' },
  iabs: { args: 'c', fn: 'cv_iabs' }, c_iabs: { args: 'c', fn: 'cv_iabs' }, c_abs: { args: 'c', fn: 'cv_abs' },
  // G's vec2 overloads of the elementary functions (exp/sinh/cosh/tanh of a vec2 act per component or as above)
  exp: { args: 'c', fn: 'cv_gexp' }, sinh: { args: 'c', fn: 'cv_gsinh' }, cosh: { args: 'c', fn: 'cv_gcosh' }, tanh: { args: 'c', fn: 'cv_gtanh' },
};
/** vec2 methods: name → [argument kind, WGSL] ('c' complex, 's' scalar, '' none) */
const CMETHOD: Record<string, { args: string; fn: (self: string, a: string[]) => string }> = {
  plus: { args: 'c|s', fn: (s, a) => `(${s} + ${a[0]})` }, add: { args: 'c|s', fn: (s, a) => `(${s} + ${a[0]})` },
  minus: { args: 'c|s', fn: (s, a) => `(${s} - ${a[0]})` }, sub: { args: 'c|s', fn: (s, a) => `(${s} - ${a[0]})` },
  multiply: { args: 'c|s', fn: (s, a) => `(${s} * ${a[0]})` }, times: { args: 'c|s', fn: (s, a) => `(${s} * ${a[0]})` },
  division: { args: 'c|s', fn: (s, a) => `(${s} / ${a[0]})` }, div: { args: 'c|s', fn: (s, a) => `(${s} / ${a[0]})` },
  negate: { args: '', fn: (s) => `(-${s})` }, neg: { args: '', fn: (s) => `(-${s})` },
};

/** Parse the method body and return the WGSL statements computing `cv_ret` (vec2f) from `cv_z` (vec2f). */
export function cvarToWgsl(code: string): string {
  const src = code.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
  const open = src.indexOf('{', src.search(/\bf\s*\(\s*vec2\s+(\w+)\s*\)/));
  const m = /\bf\s*\(\s*vec2\s+(\w+)\s*\)/.exec(src);
  if (!m || open < 0) throw new Error('no "public vec2 f(vec2 z) { … }" method');
  const zName = m[1];
  const close = src.lastIndexOf('}');
  if (close <= open) throw new Error('unbalanced braces');
  const body = src.slice(open + 1, close);
  const toks = tokenize(body);
  let p = 0;
  const locals = new Map<string, string>([[zName, 'cv_z']]);
  const out: string[] = [];
  const isOp = (v: string) => toks[p]?.t === 'op' && toks[p].v === v;
  const expect = (v: string) => { if (!isOp(v)) throw new Error(`expected "${v}"`); p++; };
  // a scalar sub-expression: the token run up to the next ',' or ')' at depth 0, handed to the formula compiler
  const scalar = (): string => {
    let depth = 0; const start = p;
    while (p < toks.length) {
      const t = toks[p];
      if (t.t === 'op' && (t.v === '(')) depth++;
      else if (t.t === 'op' && t.v === ')') { if (depth === 0) break; depth--; }
      else if (t.t === 'op' && t.v === ',' && depth === 0) break;
      p++;
    }
    if (p === start) throw new Error('empty scalar argument');
    return formulaToWgsl(body.slice(toks[start].a, toks[p - 1].b), {});
  };
  const args = (kinds: string): { vals: string[]; alt: string } => {
    // kinds: alternatives separated by '|', each a string of 'c'/'s' per argument (the first that parses wins)
    const alts = kinds.split('|');
    const saved = p;
    for (const alt of alts) {
      p = saved;
      try {
        const vals: string[] = [];
        expect('(');
        for (let i = 0; i < alt.length; i++) {
          if (i) expect(',');
          vals.push(alt[i] === 'c' ? complex() : scalar());
        }
        expect(')');
        return { vals, alt };
      } catch (e) { if (alt === alts[alts.length - 1]) throw e; }
    }
    throw new Error('bad arguments');
  };
  const postfix = (base: string): string => {
    let cur = base;
    while (isOp('.')) {
      p++;
      const t = toks[p++];
      if (!t || t.t !== 'id') throw new Error('method name expected');
      if (t.v === 'x' || t.v === 'y') throw new Error('scalar component access is not supported');
      const mth = CMETHOD[t.v];
      if (!mth) throw new Error(`unknown vec2 method ${t.v}`);
      const a = mth.args ? args(mth.args).vals : (expect('('), expect(')'), []);
      cur = mth.fn(cur, a);
    }
    return cur;
  };
  const complex = (): string => {
    const t = toks[p++];
    if (!t) throw new Error('unexpected end of code');
    if (t.t === 'op' && t.v === '(') { const e = complex(); expect(')'); return postfix(e); }
    if (t.t === 'op' && t.v === '-') { const e = complex(); return `(-${e})`; }
    if (t.t === 'id') {
      if (t.v === 'new') {
        const n = toks[p++];
        if (!n || n.t !== 'id' || n.v !== 'vec2') throw new Error('only new vec2(…) is supported');
        expect('('); const re = scalar(); expect(','); const im = scalar(); expect(')');
        return postfix(`vec2f(${re}, ${im})`);
      }
      if (t.v === 'G' && isOp('.')) { p++; const f = toks[p++]; if (!f || f.t !== 'id') throw new Error('G.?'); return postfix(call(f.v)); }
      if (isOp('(')) return postfix(call(t.v));
      const local = locals.get(t.v);
      if (local === undefined) throw new Error(`unknown name ${t.v}`);
      return postfix(local);
    }
    throw new Error(`unexpected "${t.v}"`);
  };
  const call = (name: string): string => {
    const f = CFN[name];
    if (!f) throw new Error(`unknown function ${name}`);
    const { vals: a, alt } = f.args ? args(f.args) : (expect('('), expect(')'), { vals: [], alt: '' });
    // overloads by argument kind: c_exp(base, c) / c_pow(c, vec2)
    if (name === 'c_exp' && alt === 'sc') return `cv_expb(${a.join(', ')})`;
    if (name === 'c_pow' && alt === 'cc') return `cv_powc(${a.join(', ')})`;
    return `${f.fn}(${a.join(', ')})`;
  };
  let returned = false;
  let n = 0;
  while (p < toks.length && !returned) {
    const t = toks[p];
    if (t.t === 'op' && t.v === ';') { p++; continue; }
    if (t.t === 'id' && t.v === 'vec2') {
      p++;
      const name = toks[p++];
      if (!name || name.t !== 'id') throw new Error('variable name expected');
      expect('=');
      const e = complex();
      expect(';');
      const w = `cv_l${n++}`;
      out.push(`let ${w} = ${e};`);
      locals.set(name.v, w);
      continue;
    }
    if (t.t === 'id' && t.v === 'return') {
      p++;
      const e = complex();
      if (isOp(';')) p++;
      out.push(`cv_ret = ${e};`);
      returned = true;
      continue;
    }
    if (t.t === 'id' && locals.has(t.v) && toks[p + 1]?.t === 'op' && toks[p + 1].v === '=') {
      // reassignment of a local (`z = z.plus(…);`)
      p += 2;
      const e = complex();
      expect(';');
      const w = `cv_l${n++}`;
      out.push(`let ${w} = ${e};`);
      locals.set(t.v, w);
      continue;
    }
    throw new Error(`unexpected "${t.v}" (only vec2 declarations and return are supported)`);
  }
  if (!returned) throw new Error('no return statement');
  return out.join('\n    ');
}

export const CVAR_WGSL_FUNC_NAMES = ['G_atan2', 'cv_iabs', 'cv_abs', 'cv_gsinh1', 'cv_gcosh1', 'cv_one', 'cv_i', 'cv_ni', 'cv_arg', 'cv_conj', 'cv_from_polar', 'cv_to_polar', 'cv_exp', 'cv_expb', 'cv_ln', 'cv_log', 'cv_sqrt', 'cv_pow', 'cv_powc', 'cv_add', 'cv_sub', 'cv_mul', 'cv_div', 'cv_sin', 'cv_cos', 'cv_tan', 'cv_atan', 'cv_asin', 'cv_acos', 'cv_sinh', 'cv_cosh', 'cv_tanh', 'cv_asinh', 'cv_acosh', 'cv_atanh', 'cv_rem', 'cv_inv', 'cv_gexp', 'cv_gsinh', 'cv_gcosh', 'cv_gtanh', 'powc'];
/** js.glsl.G's complex helpers, line for line (G.sinh/cosh/tanh of a double go through exp like G's) */
export const CVAR_WGSL_FUNCS = `fn G_atan2(y: f32, x: f32) -> f32 { let c1 = PI / 4.0; let ay = abs(y); var ang: f32;
  if (x >= 0.0) { let r = (x - ay) / (x + ay); ang = c1 - c1 * r; } else { let r = (x + ay) / (ay - x); ang = 3.0 * c1 - c1 * r; }
  return select(ang, -ang, y < 0.0); }

fn cv_iabs(c: vec2f) -> vec2f { return vec2f(c.x, abs(c.y)); }

fn cv_abs(c: vec2f) -> vec2f { return abs(c); }

fn cv_one() -> vec2f { return vec2f(1.0, 0.0); }

fn cv_i() -> vec2f { return vec2f(0.0, 1.0); }

fn cv_ni() -> vec2f { return vec2f(0.0, -1.0); }

fn cv_arg(c: vec2f) -> f32 { return G_atan2(c.y, c.x); }

fn cv_conj(c: vec2f) -> vec2f { return vec2f(c.x, -c.y); }

fn cv_from_polar(r: f32, theta: f32) -> vec2f { return vec2f(r * cos(theta), r * sin(theta)); }

fn cv_to_polar(c: vec2f) -> vec2f { return vec2f(length(c), G_atan2(c.y, c.x)); }

fn cv_exp(c: vec2f) -> vec2f { return cv_from_polar(exp(c.x), c.y); }

fn cv_expb(base: f32, c: vec2f) -> vec2f { return cv_from_polar(powc(base, c.x), c.y * log(base)); }

fn cv_ln(c: vec2f) -> vec2f { let polar = cv_to_polar(c); return vec2f(log(polar.x), polar.y); }

fn cv_log(c: vec2f, base: f32) -> vec2f { let polar = cv_to_polar(c); return vec2f(log(polar.x) / log(base), polar.y / log(base)); }

fn cv_sqrt(c: vec2f) -> vec2f { let p = cv_to_polar(c); return cv_from_polar(sqrt(p.x), p.y / 2.0); }

fn cv_pow(c: vec2f, e: f32) -> vec2f { let p = cv_to_polar(c); return cv_from_polar(powc(p.x, e), p.y * e); }

fn cv_powc(c: vec2f, e: vec2f) -> vec2f { let polar = cv_to_polar(c); return cv_from_polar(powc(polar.x, e.x) * exp(-e.y * polar.y), e.x * polar.y + e.y * log(polar.x)); }

fn cv_add(a: vec2f, b: vec2f) -> vec2f { return a + b; }

fn cv_sub(a: vec2f, b: vec2f) -> vec2f { return a - b; }

fn cv_mul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

fn cv_div(a: vec2f, b: vec2f) -> vec2f { let norm = length(b); return vec2f((a.x * b.x + a.y * b.y) / (norm * norm), (a.y * b.x - a.x * b.y) / (norm * norm)); }

fn cv_gsinh1(v: f32) -> f32 { let tmp = exp(v); return (tmp - 1.0 / tmp) / 2.0; }

fn cv_gcosh1(v: f32) -> f32 { let tmp = exp(v); return (tmp + 1.0 / tmp) / 2.0; }

fn cv_sin(c: vec2f) -> vec2f { return vec2f(sin(c.x) * cv_gcosh1(c.y), cos(c.x) * cv_gsinh1(c.y)); }

fn cv_cos(c: vec2f) -> vec2f { return vec2f(cos(c.x) * cv_gcosh1(c.y), -sin(c.x) * cv_gsinh1(c.y)); }

fn cv_tan(c: vec2f) -> vec2f { let c2 = c * 2.0; return vec2f(sin(c2.x), cv_gsinh1(c2.y)) / (cos(c2.x) + cv_gcosh1(c2.y)); }

fn cv_atan(c: vec2f) -> vec2f {
  if (c.x == 0.0 && c.y == 1.0) { return vec2f(0.0, 3.0e38); }
  if (c.x == 0.0 && c.y == -1.0) { return vec2f(0.0, -3.0e38); }
  let i = cv_i(); let one = cv_one(); let two = one + one;
  return cv_div(cv_sub(cv_ln(cv_add(one, cv_mul(i, c))), cv_ln(cv_sub(one, cv_mul(i, c)))), cv_mul(two, i));
}

fn cv_asin(c: vec2f) -> vec2f { let i = cv_i(); return cv_mul(cv_ni(), cv_ln(cv_add(cv_sqrt(cv_sub(cv_one(), cv_mul(c, c))), cv_mul(i, c)))); }

fn cv_acos(c: vec2f) -> vec2f { let i = cv_i(); return cv_mul(cv_ni(), cv_ln(cv_add(cv_mul(i, cv_sqrt(cv_sub(cv_one(), cv_mul(c, c)))), c))); }

fn cv_sinh(c: vec2f) -> vec2f { return vec2f(cv_gsinh1(c.x) * cos(c.y), cv_gcosh1(c.x) * sin(c.y)); }

fn cv_cosh(c: vec2f) -> vec2f { return vec2f(cv_gcosh1(c.x) * cos(c.y), cv_gsinh1(c.x) * sin(c.y)); }

fn cv_tanh(c: vec2f) -> vec2f { let c2 = vec2f(2.0 * c.x, 2.0 * c.y); return vec2f(cv_gsinh1(c2.x) / (cv_gcosh1(c2.x) + cos(c2.y)), sin(c2.y) / (cv_gcosh1(c2.x) + cos(c2.y))); }

fn cv_asinh(c: vec2f) -> vec2f { return cv_ln(cv_add(c, cv_sqrt(cv_add(cv_one(), cv_mul(c, c))))); }

fn cv_acosh(c: vec2f) -> vec2f { let one = cv_one(); let two = one + one; return cv_mul(two, cv_ln(cv_add(cv_sqrt(cv_div(c + one, two)), cv_sqrt(cv_div(c - one, two))))); }

fn cv_atanh(c: vec2f) -> vec2f {
  if (c.x == 1.0 && c.y == 0.0) { return vec2f(3.0e38, 0.0); }
  if (c.x == -1.0 && c.y == 0.0) { return vec2f(-3.0e38, 0.0); }
  let one = cv_one(); let two = one + one;
  return cv_div(cv_sub(cv_ln(one + c), cv_ln(one - c)), two);
}

fn cv_rem(c: vec2f, modulus: vec2f) -> vec2f { let c0 = cv_div(c, modulus); let c1 = vec2f(c0.x - (c0.x - floor(c0.x)), c0.y - (c0.y - floor(c0.y))); return c - cv_mul(modulus, c1); }

fn cv_inv(c: vec2f) -> vec2f { let norm = length(c); return vec2f(c.x / (norm * norm), -c.y / (norm * norm)); }

fn cv_gexp(c: vec2f) -> vec2f { return vec2f(exp(c.x), exp(c.y)); }

fn cv_gsinh(c: vec2f) -> vec2f { let ep = cv_gexp(c); let t1 = cv_sub(ep, cv_div(cv_one(), ep)); return vec2f(t1.x / 2.0, t1.y / 2.0); }

fn cv_gcosh(c: vec2f) -> vec2f { let ep = cv_gexp(c); let t1 = cv_add(ep, cv_div(cv_one(), ep)); return vec2f(t1.x / 2.0, t1.y / 2.0); }

fn cv_gtanh(c: vec2f) -> vec2f { let ep = cv_gexp(c); return cv_div(cv_sub(ep, cv_div(cv_one(), ep)), cv_add(ep, cv_div(cv_one(), ep))); }

fn powc(x: f32, y: f32) -> f32 {
  if (x >= 0.0) { return pow(x, y); }
  let yi = round(y);
  if (abs(y - yi) > 1e-6) { return pow(x, y); }
  let m = pow(-x, y);
  return select(m, -m, (i32(yi) & 1) != 0);
}`;

type Inst = { params: Record<string, number>; res?: Record<string, string> };
/** The statements of an instance's function, or "cv_ret = vec2f(0.0)" with a warning when the code is outside the subset */
function statements(name: string, inst?: Inst): string {
  const code = inst?.res?.code?.trim() || CVAR_DEFAULT_CODE;
  try { return cvarToWgsl(code); }
  catch (e) { console.warn(`${name}: code not understood (${(e as Error).message}); the function returns 0`); return 'cv_ret = vec2f(0.0);'; }
}
const PARAMS = [{ name: 'shiftX', def: 0 }, { name: 'shiftY', def: 0 }, { name: 'mode', def: 0, int: true }, { name: 'zoom', def: 1 }];
const common = (name: string) => ({
  params: PARAMS, res: ['code'], resDef: { code: CVAR_DEFAULT_CODE },
  types: ['2D'], funcNames: CVAR_WGSL_FUNC_NAMES, funcs: CVAR_WGSL_FUNCS,
  sigKey: (inst: Inst) => inst.res?.code?.trim() || '',
  /** the function value at the instance's input point `inp` (vec2f): mode 1 draws a random point instead */
  body: (inst: Inst | undefined, p: string[], inp: string) => `var cv_in = ${inp}; if (i32(${p[2]}) != 0) { cv_in = vec2f(2.0 * rnd(rs) - 1.0, 2.0 * rnd(rs) - 1.0); }
    let cv_z = cv_in * ${p[3]} + vec2f(${p[0]}, ${p[1]});
    var cv_ret = vec2f(0.0);
    ${statements(name, inst)}`,
});
export const CVAR_VARIATIONS = {
  // the output becomes amount·f(z) (assigned, not added — ZVarFunc writes pVarTP); the preserve-z clause is the standard one
  c_var: { ...(({ body, ...rest }) => ({ ...rest, flags: ['formula'], code: (w: string, p: string[], _A: unknown, inst?: Inst) => `{ ${body(inst, p, 't')}\n    v = ${w} * cv_ret; }` }))(common('c_var')) },
  // pre: the affine point becomes amount·f(point); with preserve_z JWildfire doubles its z (pAffineTP.z += pAmount·pAffineTP.z)
  pre_c_var: { ...(({ body, ...rest }) => ({ ...rest, priority: -1, flags: ['formula', 'z'], types: ['2D', 'PRE'], code: (w: string, p: string[], _A: unknown, inst?: Inst) => `{ ${body(inst, p, 't')}\n    t = ${w} * cv_ret; if ((P.flags & 1u) != 0u) { z_ += ${w} * z_; } }` }))(common('pre_c_var')) },
  // post: the output point becomes amount·f(output); z gains amount·(affine z)
  post_c_var: { ...(({ body, ...rest }) => ({ ...rest, priority: 1, flags: ['formula', 'z'], types: ['2D', 'POST'], code: (w: string, p: string[], _A: unknown, inst?: Inst) => `{ ${body(inst, p, 'v')}\n    v = ${w} * cv_ret; if ((P.flags & 1u) != 0u) { pz_ += ${w} * z_; } }` }))(common('post_c_var')) },
};
