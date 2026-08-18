// A tiny complex-expression language for custom escape-time formulas ("z = …" lines),
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

// ---- double-single (hi, lo f32 pairs) complex arithmetic — the deep-zoom "ds" tier ----
// A real is a vec2f (hi, lo); a complex is DC { x, y }. TwoSum/TwoProduct error terms are routed through
// `op_` (an integer add of the runtime zero `df_zero`) so the shader compiler cannot reassociate them away —
// the same trick as the flame kernel's double-float helpers. Functions with no cheap DS form (sin, exp, log,
// sqrt, non-integer powers) evaluate in f32 and come back as DS (their result is only f32-accurate).
export const DS_WGSL = `
struct DC { x: vec2f, y: vec2f }
fn op_(v: f32) -> f32 { return bitcast<f32>(bitcast<u32>(v) + df_zero); }
fn ds_qts(a0: f32, b0: f32) -> vec2f { let a = op_(a0); let b = op_(b0); let s = op_(a + b); return vec2f(s, b - op_(s - a)); }
fn ds_ts(a0: f32, b0: f32) -> vec2f { let a = op_(a0); let b = op_(b0); let s = op_(a + b); let bb = op_(s - a); return vec2f(s, op_(a - op_(s - bb)) + op_(b - bb)); }
fn ds_add(x: vec2f, y: vec2f) -> vec2f { var s = ds_ts(x.x, y.x); let t = ds_ts(x.y, y.y); s.y = op_(s.y + t.x); s = ds_qts(s.x, s.y); s.y = op_(s.y + t.y); return ds_qts(s.x, s.y); }
fn ds_neg(x: vec2f) -> vec2f { return vec2f(-x.x, -x.y); }
fn ds_sub(x: vec2f, y: vec2f) -> vec2f { return ds_add(x, ds_neg(y)); }
// TwoProduct by Veltkamp/Dekker splitting rather than fma: WGSL's fma is not fused on every backend (HLSL mad), and an
// unfused fma silently turns the error term into 0 — which would make double-single no better than f32 exactly where it
// is needed. Every intermediate goes through op_ so the compiler can neither reassociate nor contract them.
fn ds_split(a: f32) -> vec2f { let t = op_(4097.0 * a); let hi = op_(t - op_(t - a)); return vec2f(hi, op_(a - hi)); }
fn ds_twoprod(a: f32, b: f32) -> vec2f {
  let p = op_(a * b);
  let as_ = ds_split(a); let bs = ds_split(b);
  let e = op_(op_(op_(op_(as_.x * bs.x) - p) + op_(as_.x * bs.y)) + op_(as_.y * bs.x)) + op_(as_.y * bs.y);
  return vec2f(p, e);
}
fn ds_mulf(x: vec2f, b: f32) -> vec2f { let pe = ds_twoprod(x.x, b); let e = op_(pe.y + op_(x.y * b)); return ds_qts(pe.x, e); }
fn ds_mul(x: vec2f, y: vec2f) -> vec2f { let pe = ds_twoprod(x.x, y.x); let e = op_(pe.y + op_(op_(x.x * y.y) + op_(x.y * y.x))); return ds_qts(pe.x, e); }
fn ds_div(x: vec2f, y: vec2f) -> vec2f {
  let q1 = x.x / y.x;
  let r = ds_sub(x, ds_mulf(y, q1));
  let q2 = r.x / y.x;
  let r2 = ds_sub(r, ds_mulf(y, q2));
  let q3 = r2.x / y.x;
  return ds_add(ds_qts(q1, q2), vec2f(q3, 0.0));
}
fn ds_f(v: f32) -> vec2f { return vec2f(v, 0.0); }
fn ds_hi(x: vec2f) -> f32 { return x.x + x.y; }
fn ds_abs(x: vec2f) -> vec2f { return select(x, ds_neg(x), x.x < 0.0 || (x.x == 0.0 && x.y < 0.0)); }
fn ds_lt(x: vec2f, y: vec2f) -> bool { return x.x < y.x || (x.x == y.x && x.y < y.y); }
fn dc_f(re: f32, im: f32) -> DC { return DC(vec2f(re, 0.0), vec2f(im, 0.0)); }
fn dc_c(a: vec2f) -> DC { return DC(vec2f(a.x, 0.0), vec2f(a.y, 0.0)); }
fn dc_to(a: DC) -> vec2f { return vec2f(ds_hi(a.x), ds_hi(a.y)); }
fn dc_add(a: DC, b: DC) -> DC { return DC(ds_add(a.x, b.x), ds_add(a.y, b.y)); }
fn dc_sub(a: DC, b: DC) -> DC { return DC(ds_sub(a.x, b.x), ds_sub(a.y, b.y)); }
fn dc_neg(a: DC) -> DC { return DC(ds_neg(a.x), ds_neg(a.y)); }
fn dc_mul(a: DC, b: DC) -> DC { return DC(ds_sub(ds_mul(a.x, b.x), ds_mul(a.y, b.y)), ds_add(ds_mul(a.x, b.y), ds_mul(a.y, b.x))); }
fn dc_mulf(a: DC, s: f32) -> DC { return DC(ds_mulf(a.x, s), ds_mulf(a.y, s)); }
fn dc_sqr(a: DC) -> DC { return DC(ds_sub(ds_mul(a.x, a.x), ds_mul(a.y, a.y)), ds_mulf(ds_mul(a.x, a.y), 2.0)); }
fn dc_cube(a: DC) -> DC { return dc_mul(dc_sqr(a), a); }
fn dc_norm(a: DC) -> vec2f { return ds_add(ds_mul(a.x, a.x), ds_mul(a.y, a.y)); }
fn dc_conj(a: DC) -> DC { return DC(a.x, ds_neg(a.y)); }
fn dc_flip(a: DC) -> DC { return DC(a.y, a.x); }
fn dc_abs2(a: DC) -> DC { return DC(ds_abs(a.x), ds_abs(a.y)); }   // componentwise |re|, |im| (the burning-ship abs)
fn dc_re(a: DC) -> DC { return DC(a.x, vec2f(0.0)); }
fn dc_im(a: DC) -> DC { return DC(a.y, vec2f(0.0)); }
fn dc_div(a: DC, b: DC) -> DC { let d = dc_norm(b); let n = dc_mul(a, dc_conj(b)); return DC(ds_div(n.x, d), ds_div(n.y, d)); }
fn dc_recip(a: DC) -> DC { return dc_div(dc_f(1.0, 0.0), a); }
fn dc_powi(a: DC, n0: i32) -> DC {
  var n = abs(n0); var r = dc_f(1.0, 0.0); var base = a;
  loop { if (n == 0) { break; } if ((n & 1) != 0) { r = dc_mul(r, base); } base = dc_sqr(base); n = n >> 1; }
  if (n0 < 0) { return dc_recip(r); }
  return r;
}
// exponent as a complex: integer real exponents stay in DS, anything else through f32
fn dc_pow(a: DC, b: DC) -> DC {
  let br = ds_hi(b.x); let bi = ds_hi(b.y);
  if (bi == 0.0 && abs(br - round(br)) < 1e-6 && abs(br) <= 16.0) { return dc_powi(a, i32(round(br))); }
  return dc_c(cpow(dc_to(a), vec2f(br, bi)));
}
fn dc_absc(a: DC) -> DC { return DC(vec2f(sqrt(ds_hi(dc_norm(a))), 0.0), vec2f(0.0)); } // |z| as a (f32-accurate) real
`;

/** Compile an expression to a WGSL DC (double-single complex) expression. Same language as compileFormula. */
export function compileFormulaDS(src: string): WgslExpr {
  const F1: Record<string, (a: string) => string> = {
    sin: (a) => `dc_c(csin(dc_to(${a})))`, cos: (a) => `dc_c(ccos(dc_to(${a})))`, tan: (a) => `dc_c(ctan(dc_to(${a})))`,
    sinh: (a) => `dc_c(csinh(dc_to(${a})))`, cosh: (a) => `dc_c(ccosh(dc_to(${a})))`, tanh: (a) => `dc_c(ctanh(dc_to(${a})))`,
    exp: (a) => `dc_c(cexp(dc_to(${a})))`, log: (a) => `dc_c(clog(dc_to(${a})))`, sqrt: (a) => `dc_c(csqrt(dc_to(${a})))`,
    abs: (a) => `dc_absc(${a})`, arg: (a) => `dc_c(carg(dc_to(${a})))`, re: (a) => `dc_re(${a})`, im: (a) => `dc_im(${a})`,
    conj: (a) => `dc_conj(${a})`, recip: (a) => `dc_recip(${a})`, sqr: (a) => `dc_sqr(${a})`, cube: (a) => `dc_cube(${a})`,
    flip: (a) => `dc_flip(${a})`, floor: (a) => `dc_c(floor(dc_to(${a})))`, round: (a) => `dc_c(round(dc_to(${a})))`, norm: (a) => `DC(dc_norm(${a}), vec2f(0.0))`,
  };
  const F2: Record<string, (a: string, b: string) => string> = { pow: (a, b) => `dc_pow(${a}, ${b})` };
  const C: Record<string, string> = { i: 'dc_f(0.0, 1.0)', pi: 'dc_f(3.14159265358979, 0.0)', e: 'dc_f(2.71828182845905, 0.0)' };
  const toks = tokenizeExport(src);
  let p = 0;
  const peek = () => toks[p];
  const eat = (v?: string): Tok => { const t = toks[p++]; if (!t) throw new Error('unexpected end of formula'); if (v !== undefined && !(t.t === 'op' && t.v === v)) throw new Error(`expected "${v}"`); return t; };
  const expr = (): string => { let a = term(); while (peek()?.t === 'op' && (peek()!.v === '+' || peek()!.v === '-')) { const op = (eat() as any).v; const b = term(); a = op === '+' ? `dc_add(${a}, ${b})` : `dc_sub(${a}, ${b})`; } return a; };
  const term = (): string => { let a = unary(); while (peek()?.t === 'op' && (peek()!.v === '*' || peek()!.v === '/')) { const op = (eat() as any).v; const b = unary(); a = op === '*' ? `dc_mul(${a}, ${b})` : `dc_div(${a}, ${b})`; } return a; };
  const unary = (): string => { if (peek()?.t === 'op' && peek()!.v === '-') { eat(); return `dc_neg(${unary()})`; } if (peek()?.t === 'op' && peek()!.v === '+') { eat(); return unary(); } return powr(); };
  const powr = (): string => { const a = atom(); if (peek()?.t === 'op' && peek()!.v === '^') { eat(); const b = unary(); return `dc_pow(${a}, ${b})`; } return a; };
  const atom = (): string => {
    const t = eat();
    if (t.t === 'num') return `dc_f(${f(t.v)}, 0.0)`;
    if (t.t === 'op' && t.v === '(') { const e = expr(); eat(')'); return e; }
    if (t.t === 'id') {
      if (peek()?.t === 'op' && peek()!.v === '(') {
        eat('(');
        const args: string[] = [];
        if (!(peek()?.t === 'op' && peek()!.v === ')')) { args.push(expr()); while (peek()?.t === 'op' && peek()!.v === ',') { eat(); args.push(expr()); } }
        eat(')');
        if (F1[t.v]) { if (args.length !== 1) throw new Error(`${t.v}() takes one argument`); return F1[t.v](args[0]); }
        if (F2[t.v]) { if (args.length !== 2) throw new Error(`${t.v}() takes two arguments`); return F2[t.v](args[0], args[1]); }
        throw new Error(`unknown function "${t.v}"`);
      }
      if (C[t.v]) return C[t.v];
      if ((FORMULA_VARS as readonly string[]).includes(t.v)) return t.v === 'n' ? 'dc_f(n, 0.0)' : t.v;
      throw new Error(`unknown name "${t.v}"`);
    }
    throw new Error(`unexpected "${(t as any).v}"`);
  };
  const out = expr();
  if (p < toks.length) throw new Error(`unexpected "${(toks[p] as any).v}"`);
  return out;
}
function tokenizeExport(src: string): Tok[] { return tokenize(src); }
