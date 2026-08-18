// A tiny complex-expression language for custom escape-time formulas (Ultra-Fractal style "z = …" lines),
// compiled to WGSL. Everything is a complex number (vec2f); real functions of a complex argument return
// (value, 0). Grammar (usual precedence, right-assoc ^):
//   expr := term (('+'|'-') term)* ; term := unary (('*'|'/') unary)* ; unary := ('-'|'+') unary | pow
//   pow  := atom ('^' unary)? ; atom := number | ident | ident '(' args ')' | '(' expr ')' | 'i'
// Identifiers: z (current value), c (the parameter: pixel in Mandelbrot mode, the constant in Julia mode),
// pixel (the pixel's complex coordinate), n (iteration number), p1..p4 (user parameters), i, pi, e.
// Functions: sin cos tan sinh cosh tanh exp log sqrt abs arg re im conj recip sqr cube pow(z, w) flip
// (swaps re/im), floor round, norm (|z|²).

export type WgslExpr = string;

const FUNCS_1: Record<string, string> = {
  sin: 'csin', cos: 'ccos', tan: 'ctan', sinh: 'csinh', cosh: 'ccosh', tanh: 'ctanh', exp: 'cexp', log: 'clog', sqrt: 'csqrt',
  abs: 'cabs', arg: 'carg', re: 'cre', im: 'cim', conj: 'cconj', recip: 'crecip', sqr: 'csqr', cube: 'ccube', flip: 'cflip',
  floor: 'cfloor', round: 'cround', norm: 'cnorm',
};
const FUNCS_2: Record<string, string> = { pow: 'cpow' };
const CONSTS: Record<string, string> = { i: 'vec2f(0.0, 1.0)', pi: 'vec2f(3.14159265358979, 0.0)', e: 'vec2f(2.71828182845905, 0.0)' };
/** identifiers the shader provides (as vec2f) */
export const FORMULA_VARS = ['z', 'c', 'pixel', 'n', 'p1', 'p2', 'p3', 'p4'] as const;

type Tok = { t: 'num'; v: number } | { t: 'id'; v: string } | { t: 'op'; v: string };

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (/[0-9.]/.test(ch)) {
      const m = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i))!;
      out.push({ t: 'num', v: parseFloat(m[0]) }); i += m[0].length; continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      const m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i))!;
      out.push({ t: 'id', v: m[0] }); i += m[0].length; continue;
    }
    if ('+-*/^(),'.includes(ch)) { out.push({ t: 'op', v: ch }); i++; continue; }
    throw new Error(`unexpected character "${ch}" at ${i + 1}`);
  }
  return out;
}

const f = (v: number) => { const s = String(v); return /[.eE]/.test(s) ? s : s + '.0'; };

/** Compile an expression to a WGSL vec2f expression. Throws a message on syntax errors / unknown names. */
export function compileFormula(src: string): WgslExpr {
  const toks = tokenize(src);
  let p = 0;
  const peek = () => toks[p];
  const eat = (v?: string): Tok => { const t = toks[p++]; if (!t) throw new Error('unexpected end of formula'); if (v !== undefined && !(t.t === 'op' && t.v === v)) throw new Error(`expected "${v}"`); return t; };
  const expr = (): string => {
    let a = term();
    while (peek()?.t === 'op' && (peek()!.v === '+' || peek()!.v === '-')) { const op = (eat() as any).v; const b = term(); a = `(${a} ${op} ${b})`; }
    return a;
  };
  const term = (): string => {
    let a = unary();
    while (peek()?.t === 'op' && (peek()!.v === '*' || peek()!.v === '/')) { const op = (eat() as any).v; const b = unary(); a = op === '*' ? `cmul(${a}, ${b})` : `cdiv(${a}, ${b})`; }
    return a;
  };
  const unary = (): string => {
    if (peek()?.t === 'op' && peek()!.v === '-') { eat(); return `(-${unary()})`; }
    if (peek()?.t === 'op' && peek()!.v === '+') { eat(); return unary(); }
    return powr();
  };
  const powr = (): string => {
    const a = atom();
    if (peek()?.t === 'op' && peek()!.v === '^') { eat(); const b = unary(); return `cpow(${a}, ${b})`; }
    return a;
  };
  const atom = (): string => {
    const t = eat();
    if (t.t === 'num') return `vec2f(${f(t.v)}, 0.0)`;
    if (t.t === 'op' && t.v === '(') { const e = expr(); eat(')'); return e; }
    if (t.t === 'id') {
      if (peek()?.t === 'op' && peek()!.v === '(') {
        eat('(');
        const args: string[] = [];
        if (!(peek()?.t === 'op' && peek()!.v === ')')) { args.push(expr()); while (peek()?.t === 'op' && peek()!.v === ',') { eat(); args.push(expr()); } }
        eat(')');
        if (FUNCS_1[t.v]) { if (args.length !== 1) throw new Error(`${t.v}() takes one argument`); return `${FUNCS_1[t.v]}(${args[0]})`; }
        if (FUNCS_2[t.v]) { if (args.length !== 2) throw new Error(`${t.v}() takes two arguments`); return `${FUNCS_2[t.v]}(${args[0]}, ${args[1]})`; }
        throw new Error(`unknown function "${t.v}"`);
      }
      if (CONSTS[t.v]) return CONSTS[t.v];
      if ((FORMULA_VARS as readonly string[]).includes(t.v)) return t.v === 'n' ? 'vec2f(n, 0.0)' : t.v;
      throw new Error(`unknown name "${t.v}"`);
    }
    throw new Error(`unexpected "${(t as any).v}"`);
  };
  const out = expr();
  if (p < toks.length) throw new Error(`unexpected "${(toks[p] as any).v}"`);
  return out;
}

/** The complex helpers the compiled expressions (and the built-in formulas) use. */
export const COMPLEX_WGSL = `
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
fn cdiv(a: vec2f, b: vec2f) -> vec2f { let d = max(dot(b, b), 1e-30); return vec2f(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d; }
fn csqr(a: vec2f) -> vec2f { return vec2f(a.x * a.x - a.y * a.y, 2.0 * a.x * a.y); }
fn ccube(a: vec2f) -> vec2f { return cmul(csqr(a), a); }
fn cconj(a: vec2f) -> vec2f { return vec2f(a.x, -a.y); }
fn cflip(a: vec2f) -> vec2f { return vec2f(a.y, a.x); }
fn crecip(a: vec2f) -> vec2f { return cconj(a) / max(dot(a, a), 1e-30); }
fn cabs(a: vec2f) -> vec2f { return vec2f(length(a), 0.0); }
fn cnorm(a: vec2f) -> vec2f { return vec2f(dot(a, a), 0.0); }
fn carg(a: vec2f) -> vec2f { return vec2f(atan2(a.y, a.x), 0.0); }
fn cre(a: vec2f) -> vec2f { return vec2f(a.x, 0.0); }
fn cim(a: vec2f) -> vec2f { return vec2f(a.y, 0.0); }
fn cfloor(a: vec2f) -> vec2f { return floor(a); }
fn cround(a: vec2f) -> vec2f { return round(a); }
fn cexp(a: vec2f) -> vec2f { let e = exp(clamp(a.x, -80.0, 80.0)); return vec2f(e * cos(a.y), e * sin(a.y)); }
fn clog(a: vec2f) -> vec2f { return vec2f(0.5 * log(max(dot(a, a), 1e-30)), atan2(a.y, a.x)); }
fn csqrt(a: vec2f) -> vec2f { let r = length(a); let re = sqrt(max(0.5 * (r + a.x), 0.0)); let im = sqrt(max(0.5 * (r - a.x), 0.0)); return vec2f(re, select(-im, im, a.y >= 0.0)); }
fn cpow(a: vec2f, b: vec2f) -> vec2f {
  if (dot(a, a) < 1e-30) { return vec2f(0.0); }
  // integer real exponents by repeated squaring keep the branch cut of z^n exact (and are what z^2, z^3… mean)
  if (b.y == 0.0 && abs(b.x - round(b.x)) < 1e-6 && abs(b.x) <= 16.0) {
    var n = i32(abs(round(b.x))); var r = vec2f(1.0, 0.0); var base = a;
    loop { if (n == 0) { break; } if ((n & 1) != 0) { r = cmul(r, base); } base = csqr(base); n = n >> 1; }
    return select(r, crecip(r), b.x < 0.0);
  }
  return cexp(cmul(b, clog(a)));
}
fn csin(a: vec2f) -> vec2f { return vec2f(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y)); }
fn ccos(a: vec2f) -> vec2f { return vec2f(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y)); }
fn ctan(a: vec2f) -> vec2f { return cdiv(csin(a), ccos(a)); }
fn csinh(a: vec2f) -> vec2f { return vec2f(sinh(a.x) * cos(a.y), cosh(a.x) * sin(a.y)); }
fn ccosh(a: vec2f) -> vec2f { return vec2f(cosh(a.x) * cos(a.y), sinh(a.x) * sin(a.y)); }
fn ctanh(a: vec2f) -> vec2f { return cdiv(csinh(a), ccosh(a)); }
`;
