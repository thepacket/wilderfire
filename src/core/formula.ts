// A small, safe evaluator for JWildfire formula text (the `sattractor3D` x/y/z formulas, written in the Java
// expression subset JWildfire compiles with Janino at run time): numbers, the variables x y z param_a…param_h
// pi, arithmetic + - * / %, unary -, comparisons, && || !, the ternary ?:, and MathLib's functions. No eval —
// the text is parsed into closures once and evaluated per point.
export type Formula = (v: Float64Array) => number;

/** variable slots in the Float64Array handed to a compiled formula */
export const FV = { x: 0, y: 1, z: 2, param_a: 3, param_b: 4, param_c: 5, param_d: 6, param_e: 7, param_f: 8, param_g: 9, param_h: 10, delta_t: 11 } as const;

const CONSTS: Record<string, number> = { pi: Math.PI, M_PI: Math.PI, M_E: Math.E, e: Math.E };
const F1: Record<string, (a: number) => number> = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh, asinh: Math.asinh, acosh: Math.acosh, atanh: Math.atanh,
  exp: Math.exp, log: Math.log, log10: Math.log10, sqrt: Math.sqrt, cbrt: Math.cbrt,
  fabs: Math.abs, abs: Math.abs, floor: Math.floor, ceil: Math.ceil, round: Math.round, rint: Math.round, trunc: Math.trunc,
  sqr: (a) => a * a, frac: (a) => a - Math.floor(a), sign: Math.sign, erf: erf,
};
const F2: Record<string, (a: number, b: number) => number> = { pow: Math.pow, atan2: Math.atan2, fmod: (a, b) => a % b, min: Math.min, max: Math.max, hypot: Math.hypot };

function erf(x: number): number { // Abramowitz–Stegun 7.1.26 (JWildfire's MathLib.erf is the same order of accuracy)
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

type Tok = { t: 'num'; v: number } | { t: 'id'; v: string } | { t: 'op'; v: string };
function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    if ((c >= '0' && c <= '9') || (c === '.' && src[i + 1] >= '0' && src[i + 1] <= '9')) {
      const m = /^(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?[dDfF]?/.exec(src.slice(i))!;
      out.push({ t: 'num', v: parseFloat(m[0]) }); i += m[0].length; continue;
    }
    if (/[A-Za-z_]/.test(c)) { const m = /^[A-Za-z_]\w*/.exec(src.slice(i))!; out.push({ t: 'id', v: m[0] }); i += m[0].length; continue; }
    const two = src.slice(i, i + 2);
    if (['<=', '>=', '==', '!=', '&&', '||'].includes(two)) { out.push({ t: 'op', v: two }); i += 2; continue; }
    if ('+-*/%()?:,<>!'.includes(c)) { out.push({ t: 'op', v: c }); i++; continue; }
    throw new Error(`unexpected "${c}" at ${i}`);
  }
  return out;
}

/** Parse a formula; throws with a message on anything outside the subset. */
export function compileFormula(src: string): Formula {
  const toks = tokenize(src);
  let p = 0;
  const peek = () => toks[p];
  const isOp = (v: string) => { const t = toks[p]; return t !== undefined && t.t === 'op' && t.v === v; };
  const expect = (v: string) => { if (!isOp(v)) throw new Error(`expected "${v}"`); p++; };
  const ternary = (): Formula => {
    const c = or();
    if (!isOp('?')) return c;
    p++; const a = ternary(); expect(':'); const b = ternary();
    return (v) => (c(v) !== 0 ? a(v) : b(v));
  };
  const or = (): Formula => { let l = and(); while (isOp('||')) { p++; const r = and(); const ll = l; l = (v) => (ll(v) !== 0 || r(v) !== 0 ? 1 : 0); } return l; };
  const and = (): Formula => { let l = eq(); while (isOp('&&')) { p++; const r = eq(); const ll = l; l = (v) => (ll(v) !== 0 && r(v) !== 0 ? 1 : 0); } return l; };
  const eq = (): Formula => {
    let l = rel();
    for (;;) {
      if (isOp('==')) { p++; const r = rel(); const ll = l; l = (v) => (ll(v) === r(v) ? 1 : 0); }
      else if (isOp('!=')) { p++; const r = rel(); const ll = l; l = (v) => (ll(v) !== r(v) ? 1 : 0); }
      else return l;
    }
  };
  const rel = (): Formula => {
    let l = add();
    for (;;) {
      const t = peek();
      if (!t || t.t !== 'op' || !['<', '>', '<=', '>='].includes(t.v)) return l;
      p++; const r = add(); const ll = l; const o = t.v;
      l = o === '<' ? (v) => (ll(v) < r(v) ? 1 : 0) : o === '>' ? (v) => (ll(v) > r(v) ? 1 : 0) : o === '<=' ? (v) => (ll(v) <= r(v) ? 1 : 0) : (v) => (ll(v) >= r(v) ? 1 : 0);
    }
  };
  const add = (): Formula => {
    let l = mul();
    for (;;) {
      if (isOp('+')) { p++; const r = mul(); const ll = l; l = (v) => ll(v) + r(v); }
      else if (isOp('-')) { p++; const r = mul(); const ll = l; l = (v) => ll(v) - r(v); }
      else return l;
    }
  };
  const mul = (): Formula => {
    let l = unary();
    for (;;) {
      if (isOp('*')) { p++; const r = unary(); const ll = l; l = (v) => ll(v) * r(v); }
      else if (isOp('/')) { p++; const r = unary(); const ll = l; l = (v) => ll(v) / r(v); }
      else if (isOp('%')) { p++; const r = unary(); const ll = l; l = (v) => ll(v) % r(v); }
      else return l;
    }
  };
  const unary = (): Formula => {
    if (isOp('-')) { p++; const a = unary(); return (v) => -a(v); }
    if (isOp('+')) { p++; return unary(); }
    if (isOp('!')) { p++; const a = unary(); return (v) => (a(v) === 0 ? 1 : 0); }
    return primary();
  };
  const primary = (): Formula => {
    const t = toks[p++];
    if (!t) throw new Error('unexpected end of formula');
    if (t.t === 'num') { const c = t.v; return () => c; }
    if (t.t === 'op' && t.v === '(') { const e = ternary(); expect(')'); return e; }
    if (t.t === 'id') {
      if (isOp('(')) {
        p++;
        const args: Formula[] = [];
        if (!isOp(')')) { args.push(ternary()); while (isOp(',')) { p++; args.push(ternary()); } }
        expect(')');
        const f1 = F1[t.v], f2 = F2[t.v];
        if (f1 && args.length === 1) { const a = args[0]; return (v) => f1(a(v)); }
        if (f2 && args.length === 2) { const [a, b] = args; return (v) => f2(a(v), b(v)); }
        throw new Error(`unknown function ${t.v}/${args.length}`);
      }
      if (t.v in FV) { const i = FV[t.v as keyof typeof FV]; return (v) => v[i]; }
      if (t.v in CONSTS) { const c = CONSTS[t.v]; return () => c; }
      throw new Error(`unknown name ${t.v}`);
    }
    throw new Error(`unexpected "${t.v}"`);
  };
  const f = ternary();
  if (p !== toks.length) throw new Error(`unexpected "${(toks[p] as { v: unknown }).v}"`);
  return f;
}
