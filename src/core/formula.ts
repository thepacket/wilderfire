// JWildfire formula text (the `sattractor3D` x/y/z formulas, the plot family's `formula`/`xformula`…, `knots3D`):
// the Java expression subset JWildfire compiles with Janino at run time under `import static MathLib.*` —
// numbers, variables, arithmetic + - * / %, unary -, comparisons, && || !, the ternary ?:, and MathLib's
// functions/constants. The text is parsed once into a small AST with Java's typing (an int literal stays an int:
// `1/2` is 0, `1/2.0` is 0.5; comparisons are booleans) and then either evaluated on the CPU through closures
// (`compileFormula`, no eval) or emitted as a WGSL expression for the GPU kernel (`formulaToWgsl`).
export type Formula = (v: Float64Array) => number;

/** variable slots in the Float64Array handed to a compiled formula */
export const FV = { x: 0, y: 1, z: 2, param_a: 3, param_b: 4, param_c: 5, param_d: 6, param_e: 7, param_f: 8, param_g: 9, param_h: 10, delta_t: 11, u: 12, v: 13, t: 14 } as const;

// MathLib's constants (and `pi`, which every JWildfire formula wrapper declares as M_PI)
const CONSTS: Record<string, number> = {
  pi: Math.PI, M_PI: Math.PI, M_PI_2: Math.PI / 2, M_PI_4: Math.PI / 4, M_1_PI: 1 / Math.PI, M_2_PI: 2 / Math.PI, M_2PI: 2 * Math.PI,
  M_1_2PI: 1 / (2 * Math.PI), M_SQRT2: Math.SQRT2, M_E: Math.E, e: Math.E, M_LOG2E: Math.LOG2E, EPSILON: 1e-8, SMALL_EPSILON: 1e-300, C_255: 255,
};
const INT_CONSTS: Record<string, number> = { TRUE: 1, FALSE: 0 };
const F1: Record<string, (a: number) => number> = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh, asinh: Math.asinh, acosh: (d) => Math.log(Math.sqrt(d * d - 1) + d), atanh: Math.atanh,
  exp: Math.exp, log: Math.log, log10: Math.log10, sqrt: Math.sqrt, cbrt: Math.cbrt,
  fabs: Math.abs, abs: Math.abs, floor: Math.floor, ceil: Math.ceil, round: (a) => Math.floor(a + 0.5), rint: roundEven, trunc: Math.trunc,
  sqr: (a) => a * a, frac: (a) => a - Math.trunc(a), sign: (a) => (a > 0 ? 1 : a < 0 ? -1 : 0), signum: Math.sign, erf: erf, lgamma: lgamma,
  iabs: Math.abs,
};
const F2: Record<string, (a: number, b: number) => number> = {
  pow: Math.pow, atan2: Math.atan2, fmod: (a, b) => a % b, min: Math.min, max: Math.max, hypot: Math.hypot, iMin: Math.min, iMax: Math.max,
};
/** functions whose Java result is an int (the rest return double) */
const INT_RESULT = new Set(['sign', 'iabs', 'iMin', 'iMax']);

function roundEven(x: number): number { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; }
function erf(z: number): number { // MathLib.erf (Sedgewick & Wayne's Chebyshev fit, |error| < 1.2e-7)
  const t = 1 / (1 + 0.5 * Math.abs(z));
  const ans = 1 - t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return z >= 0 ? ans : -ans;
}
function lgamma(x: number): number { // MathLib.lgamma
  const tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
  const ser = 1 + 76.18009173 / x - 86.50532033 / (x + 1) + 24.01409822 / (x + 2) - 1.231739516 / (x + 3) + 0.00120858003 / (x + 4) - 0.00000536382 / (x + 5);
  return tmp + Math.log(ser * Math.sqrt(2 * Math.PI));
}

// ---- AST ----
export type FNode =
  | { k: 'num'; v: number; int: boolean }
  | { k: 'var'; name: string }
  | { k: 'const'; name: string }
  | { k: 'call'; fn: string; args: FNode[] }
  | { k: 'un'; op: '-' | '+' | '!'; a: FNode }
  | { k: 'bin'; op: string; a: FNode; b: FNode }
  | { k: 'tern'; c: FNode; a: FNode; b: FNode };

type Tok = { t: 'num'; v: number; int: boolean } | { t: 'id'; v: string } | { t: 'op'; v: string };
function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    if ((c >= '0' && c <= '9') || (c === '.' && src[i + 1] >= '0' && src[i + 1] <= '9')) {
      const m = /^(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?([dDfFlL])?/.exec(src.slice(i))!;
      // Java typing: a literal without '.', exponent or d/f suffix is an int (long: l suffix, also integral)
      out.push({ t: 'num', v: parseFloat(m[0]), int: !/[.eEdDfF]/.test(m[0]) }); i += m[0].length; continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      const m = /^[A-Za-z_]\w*/.exec(src.slice(i))!;
      let id = m[0]; i += m[0].length;
      // `Math.sin`, `MathLib.sqr`: qualified statics → the bare name
      if ((id === 'Math' || id === 'MathLib') && src[i] === '.') { const m2 = /^\.([A-Za-z_]\w*)/.exec(src.slice(i))!; id = m2[1]; i += m2[0].length; }
      out.push({ t: 'id', v: id }); continue;
    }
    const two = src.slice(i, i + 2);
    if (['<=', '>=', '==', '!=', '&&', '||'].includes(two)) { out.push({ t: 'op', v: two }); i += 2; continue; }
    if ('+-*/%()?:,<>!'.includes(c)) { out.push({ t: 'op', v: c }); i++; continue; }
    throw new Error(`unexpected "${c}" at ${i}`);
  }
  return out;
}

/** Parse a formula into its AST; throws with a message on anything outside the subset. Unknown identifiers are
 *  accepted as variables (`vars` decides which exist: FV's names for the CPU evaluator, the caller's map for WGSL). */
export function parseFormula(src: string, vars: Iterable<string>): FNode {
  const known = new Set(vars);
  const toks = tokenize(src);
  let p = 0;
  const peek = () => toks[p];
  const isOp = (v: string) => { const t = toks[p]; return t !== undefined && t.t === 'op' && t.v === v; };
  const expect = (v: string) => { if (!isOp(v)) throw new Error(`expected "${v}"`); p++; };
  const ternary = (): FNode => {
    const c = or();
    if (!isOp('?')) return c;
    p++; const a = ternary(); expect(':'); const b = ternary();
    return { k: 'tern', c, a, b };
  };
  const or = (): FNode => { let l = and(); while (isOp('||')) { p++; l = { k: 'bin', op: '||', a: l, b: and() }; } return l; };
  const and = (): FNode => { let l = eq(); while (isOp('&&')) { p++; l = { k: 'bin', op: '&&', a: l, b: eq() }; } return l; };
  const eq = (): FNode => {
    let l = rel();
    for (;;) {
      if (isOp('==') || isOp('!=')) { const op = (toks[p] as { v: string }).v; p++; l = { k: 'bin', op, a: l, b: rel() }; }
      else return l;
    }
  };
  const rel = (): FNode => {
    let l = add();
    for (;;) {
      const t = peek();
      if (!t || t.t !== 'op' || !['<', '>', '<=', '>='].includes(t.v)) return l;
      p++; l = { k: 'bin', op: t.v, a: l, b: add() };
    }
  };
  const add = (): FNode => {
    let l = mul();
    for (;;) {
      if (isOp('+') || isOp('-')) { const op = (toks[p] as { v: string }).v; p++; l = { k: 'bin', op, a: l, b: mul() }; }
      else return l;
    }
  };
  const mul = (): FNode => {
    let l = unary();
    for (;;) {
      if (isOp('*') || isOp('/') || isOp('%')) { const op = (toks[p] as { v: string }).v; p++; l = { k: 'bin', op, a: l, b: unary() }; }
      else return l;
    }
  };
  const unary = (): FNode => {
    if (isOp('-')) { p++; return { k: 'un', op: '-', a: unary() }; }
    if (isOp('+')) { p++; return { k: 'un', op: '+', a: unary() }; }
    if (isOp('!')) { p++; return { k: 'un', op: '!', a: unary() }; }
    return primary();
  };
  const primary = (): FNode => {
    const t = toks[p++];
    if (!t) throw new Error('unexpected end of formula');
    if (t.t === 'num') return { k: 'num', v: t.v, int: t.int };
    if (t.t === 'op' && t.v === '(') { const e = ternary(); expect(')'); return e; }
    if (t.t === 'id') {
      if (isOp('(')) {
        p++;
        const args: FNode[] = [];
        if (!isOp(')')) { args.push(ternary()); while (isOp(',')) { p++; args.push(ternary()); } }
        expect(')');
        const n = args.length;
        if (!((n === 1 && t.v in F1) || (n === 2 && t.v in F2))) throw new Error(`unknown function ${t.v}/${n}`);
        return { k: 'call', fn: t.v, args };
      }
      if (known.has(t.v)) return { k: 'var', name: t.v };
      if (t.v in CONSTS || t.v in INT_CONSTS) return { k: 'const', name: t.v };
      throw new Error(`unknown name ${t.v}`);
    }
    throw new Error(`unexpected "${t.v}"`);
  };
  const f = ternary();
  if (p !== toks.length) throw new Error(`unexpected "${(toks[p] as { v: unknown }).v}"`);
  return f;
}

/** Java's static type of a node: int, double or boolean (int·int stays int — integer division; anything with a
 *  double is a double; comparisons and logic are booleans). */
export function formulaType(n: FNode): 'i' | 'f' | 'b' {
  switch (n.k) {
    case 'num': return n.int ? 'i' : 'f';
    case 'var': return 'f';
    case 'const': return n.name in INT_CONSTS ? 'i' : 'f';
    case 'call': return INT_RESULT.has(n.fn) ? 'i' : 'f';
    case 'un': return n.op === '!' ? 'b' : formulaType(n.a);
    case 'bin': {
      if (['<', '>', '<=', '>=', '==', '!=', '&&', '||'].includes(n.op)) return 'b';
      const ta = formulaType(n.a), tb = formulaType(n.b);
      if (ta === 'b' || tb === 'b') throw new Error(`boolean operand of ${n.op}`);
      return ta === 'i' && tb === 'i' ? 'i' : 'f';
    }
    case 'tern': { const ta = formulaType(n.a), tb = formulaType(n.b); return ta === tb ? ta : 'f'; }
  }
}

/** Compile a formula to a closure over the FV slots (booleans are 1/0; int arithmetic truncates like Java). */
export function compileFormula(src: string): Formula {
  const ast = parseFormula(src, Object.keys(FV));
  const build = (n: FNode): Formula => {
    switch (n.k) {
      case 'num': { const c = n.v; return () => c; }
      case 'var': { const i = FV[n.name as keyof typeof FV]; return (v) => v[i]; }
      case 'const': { const c = n.name in INT_CONSTS ? INT_CONSTS[n.name] : CONSTS[n.name]; return () => c; }
      case 'call': {
        if (n.args.length === 1) { const f1 = F1[n.fn], a = build(n.args[0]); return (v) => f1(a(v)); }
        const f2 = F2[n.fn], a = build(n.args[0]), b = build(n.args[1]); return (v) => f2(a(v), b(v));
      }
      case 'un': { const a = build(n.a); return n.op === '-' ? (v) => -a(v) : n.op === '!' ? (v) => (a(v) === 0 ? 1 : 0) : a; }
      case 'bin': {
        const a = build(n.a), b = build(n.b);
        const intOp = formulaType(n) === 'i';
        switch (n.op) {
          case '+': return (v) => a(v) + b(v);
          case '-': return (v) => a(v) - b(v);
          case '*': return (v) => a(v) * b(v);
          case '/': return intOp ? (v) => Math.trunc(a(v) / b(v)) : (v) => a(v) / b(v);
          case '%': return (v) => a(v) % b(v);
          case '<': return (v) => (a(v) < b(v) ? 1 : 0);
          case '>': return (v) => (a(v) > b(v) ? 1 : 0);
          case '<=': return (v) => (a(v) <= b(v) ? 1 : 0);
          case '>=': return (v) => (a(v) >= b(v) ? 1 : 0);
          case '==': return (v) => (a(v) === b(v) ? 1 : 0);
          case '!=': return (v) => (a(v) !== b(v) ? 1 : 0);
          case '&&': return (v) => (a(v) !== 0 && b(v) !== 0 ? 1 : 0);
          case '||': return (v) => (a(v) !== 0 || b(v) !== 0 ? 1 : 0);
        }
        throw new Error(`operator ${n.op}`);
      }
      case 'tern': { const c = build(n.c), a = build(n.a), b = build(n.b); return (v) => (c(v) !== 0 ? a(v) : b(v)); }
    }
  };
  return build(ast);
}

// ---- WGSL backend ----
const lit = (v: number): string => { if (!isFinite(v)) return v > 0 ? '3.0e38' : v < 0 ? '-3.0e38' : '0.0'; let s = String(v); if (!/[.e]/.test(s)) s += '.0'; if (/^-?\d+e/.test(s)) s = s.replace(/^(-?\d+)e/, '$1.0e'); return s; };
/** MathLib function → WGSL expression of the f32 arguments; helpers named here must be in FORMULA_WGSL_FUNCS. */
const WGSL_FN: Record<string, (a: string[]) => string> = {
  sin: (a) => `sin(${a[0]})`, cos: (a) => `cos(${a[0]})`, tan: (a) => `tan(${a[0]})`, asin: (a) => `asin(${a[0]})`, acos: (a) => `acos(${a[0]})`, atan: (a) => `atan(${a[0]})`,
  sinh: (a) => `sinh(${a[0]})`, cosh: (a) => `cosh(${a[0]})`, tanh: (a) => `tanh(${a[0]})`, asinh: (a) => `asinh(${a[0]})`, acosh: (a) => `log(sqrt(${a[0]} * ${a[0]} - 1.0) + ${a[0]})`, atanh: (a) => `atanh(${a[0]})`,
  exp: (a) => `exp(${a[0]})`, log: (a) => `log(${a[0]})`, log10: (a) => `(log(${a[0]}) * 0.4342944819032518)`, sqrt: (a) => `sqrt(${a[0]})`, cbrt: (a) => `cbrt(${a[0]})`,
  fabs: (a) => `abs(${a[0]})`, abs: (a) => `abs(${a[0]})`, iabs: (a) => `abs(${a[0]})`, floor: (a) => `floor(${a[0]})`, ceil: (a) => `ceil(${a[0]})`,
  round: (a) => `floor(${a[0]} + 0.5)`, rint: (a) => `round(${a[0]})`, trunc: (a) => `trunc(${a[0]})`,
  sqr: (a) => `sqr(${a[0]})`, frac: (a) => `(${a[0]} - trunc(${a[0]}))`, sign: (a) => `i32(sign(${a[0]}))`, signum: (a) => `sign(${a[0]})`, erf: (a) => `ferf_(${a[0]})`, lgamma: (a) => `flgamma_(${a[0]})`,
  pow: (a) => `powc(${a[0]}, ${a[1]})`, atan2: (a) => `atan2j(${a[0]}, ${a[1]})`, fmod: (a) => `(${a[0]} % ${a[1]})`, min: (a) => `min(${a[0]}, ${a[1]})`, max: (a) => `max(${a[0]}, ${a[1]})`,
  hypot: (a) => `length(vec2f(${a[0]}, ${a[1]}))`, iMin: (a) => `min(${a[0]}, ${a[1]})`, iMax: (a) => `max(${a[0]}, ${a[1]})`,
};
const INT_ARGS = new Set(['iabs', 'iMin', 'iMax']);

/** WGSL helpers the emitted expressions may call (names shared with the transpiled ports — the kernel dedupes by name). */
export const FORMULA_WGSL_FUNC_NAMES = ['sqr', 'powc', 'atan2j', 'cbrt', 'ferf_', 'flgamma_'];
export const FORMULA_WGSL_FUNCS = `fn sqr(x: f32) -> f32 { return x * x; }

fn powc(x: f32, y: f32) -> f32 {
  if (x >= 0.0) { return pow(x, y); }
  let yi = round(y);
  if (abs(y - yi) > 1e-6) { return pow(x, y); }
  let m = pow(-x, y);
  return select(m, -m, (i32(yi) & 1) != 0);
}

fn atan2j(y: f32, x: f32) -> f32 { if (x == 0.0 && y == 0.0) { return select(0.0, PI, (bitcast<u32>(x) >> 31u) == 1u) * select(1.0, -1.0, (bitcast<u32>(y) >> 31u) == 1u); } return atan2(y, x); }

fn cbrt(x: f32) -> f32 { return sign(x) * pow(abs(x), 1.0 / 3.0); }

fn ferf_(z: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.5 * abs(z));
  let ans = 1.0 - t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(-ans, ans, z >= 0.0);
}

fn flgamma_(x: f32) -> f32 {
  let tmp = (x - 0.5) * log(x + 4.5) - (x + 4.5);
  let ser = 1.0 + 76.18009173 / x - 86.50532033 / (x + 1.0) + 24.01409822 / (x + 2.0) - 1.231739516 / (x + 3.0) + 0.00120858003 / (x + 4.0) - 0.00000536382 / (x + 5.0);
  return tmp + log(ser * sqrt(6.283185307179586));
}`;

/** Emit a formula as one WGSL f32 expression. `vars` maps each variable name the formula may use to the WGSL
 *  expression (f32) that holds it. Throws on anything outside the subset, like `compileFormula`. */
export function formulaToWgsl(src: string, vars: Record<string, string>): string {
  const ast = parseFormula(src, Object.keys(vars));
  type R = { c: string; t: 'i' | 'f' | 'b' };
  const toF = (r: R): string => (r.t === 'f' ? r.c : r.t === 'i' ? `f32(${r.c})` : `select(0.0, 1.0, ${r.c})`);
  const toI = (r: R): string => (r.t === 'i' ? r.c : r.t === 'f' ? `i32(${r.c})` : `select(0, 1, ${r.c})`);
  const toB = (r: R): string => (r.t === 'b' ? r.c : `(${r.c} != ${r.t === 'i' ? '0' : '0.0'})`);
  const emit = (n: FNode): R => {
    switch (n.k) {
      case 'num': return n.int ? { c: `${Math.trunc(n.v)}i`, t: 'i' } : { c: lit(n.v), t: 'f' };
      case 'var': return { c: `(${vars[n.name]})`, t: 'f' };
      case 'const': return n.name in INT_CONSTS ? { c: `${INT_CONSTS[n.name]}i`, t: 'i' } : { c: lit(CONSTS[n.name] < 1.18e-38 && CONSTS[n.name] > 0 ? 1.1754944e-38 : CONSTS[n.name]), t: 'f' };
      case 'call': {
        const args = n.args.map(emit).map((r) => (INT_ARGS.has(n.fn) ? toI(r) : toF(r)));
        return { c: WGSL_FN[n.fn](args), t: INT_RESULT.has(n.fn) ? 'i' : 'f' };
      }
      case 'un': {
        const a = emit(n.a);
        if (n.op === '!') return { c: `(!${toB(a)})`, t: 'b' };
        if (n.op === '+') return a;
        if (a.t === 'b') throw new Error('unary - on a boolean');
        return { c: `(-${a.c})`, t: a.t };
      }
      case 'bin': {
        const a = emit(n.a), b = emit(n.b);
        if (n.op === '&&' || n.op === '||') return { c: `(${toB(a)} ${n.op} ${toB(b)})`, t: 'b' };
        if (['<', '>', '<=', '>=', '==', '!='].includes(n.op)) {
          if (a.t === 'b' || b.t === 'b') { if (n.op !== '==' && n.op !== '!=') throw new Error(`boolean operand of ${n.op}`); return { c: `(${toB(a)} ${n.op} ${toB(b)})`, t: 'b' }; }
          return a.t === 'i' && b.t === 'i' ? { c: `(${a.c} ${n.op} ${b.c})`, t: 'b' } : { c: `(${toF(a)} ${n.op} ${toF(b)})`, t: 'b' };
        }
        if (a.t === 'b' || b.t === 'b') throw new Error(`boolean operand of ${n.op}`);
        if (a.t === 'i' && b.t === 'i') return { c: `(${a.c} ${n.op} ${b.c})`, t: 'i' }; // Java int arithmetic (truncating / and %)
        return { c: `(${toF(a)} ${n.op} ${toF(b)})`, t: 'f' };
      }
      case 'tern': {
        const c = toB(emit(n.c)), a = emit(n.a), b = emit(n.b);
        if (a.t === b.t) return { c: `select(${b.c}, ${a.c}, ${c})`, t: a.t };
        if (a.t === 'b' || b.t === 'b') throw new Error('mixed boolean/number branches of ?:');
        return { c: `select(${toF(b)}, ${toF(a)}, ${c})`, t: 'f' };
      }
    }
  };
  return toF(emit(ast));
}
