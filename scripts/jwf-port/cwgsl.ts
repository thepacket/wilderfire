// CUDA-C (JWildfire GPU-snippet dialect) → WGSL transpiler.
//
// JWildfire variations that implement SupportsGPU carry a CUDA-C snippet
// (getGPUCode) plus optional helper functions (getGPUFunctions). This module
// parses that C subset into an AST, type-checks it (C's implicit int/float
// promotions have to become explicit WGSL conversions) and emits WGSL.
//
// Snippet conventions (see JWildfire's Flam4_3dKernal_TemplateJWF.cu):
//   __x __y __z        affine-transformed input point       → t.x t.y (z ≡ 0)
//   __px __py __pz     accumulated output                    → v.x v.y (pz dummy)
//   __r2 __r __rinv    r² r 1/r of the input                 → r2 r (1/r)
//   __phi              atan2(x, y)                          → th
//   __theta            atan2(y, x)                          → ph
//   __pal              palette coordinate (direct color)     → *cp
//   __doHide           hide flag                             → *hd
//   __<name>           variation weight                      → w
//   __<name>_<param>   variation parameter                   → p[i]
//   varpar-><...>      same as __<...>
//   RANDFLOAT()        uniform [0,1)                         → rnd(rs)
//   xform->a..f        the xform's affine coefficients       → A(0..5)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Ty =
  | { k: 'f32' } | { k: 'i32' } | { k: 'u32' } | { k: 'bool' } | { k: 'void' }
  | { k: 'aint' } | { k: 'afloat' }              // abstract literals (no conversion needed)
  | { k: 'vec'; n: 2 | 3 | 4; e: 'f32' | 'i32' | 'u32' }
  | { k: 'arr'; e: Ty; n: number }
  | { k: 'ptr'; e: Ty | null }                    // e null = not yet inferred
  | { k: 'struct'; name: string };

const F32: Ty = { k: 'f32' }, I32: Ty = { k: 'i32' }, U32: Ty = { k: 'u32' }, BOOL: Ty = { k: 'bool' };
const VOID: Ty = { k: 'void' }, AINT: Ty = { k: 'aint' }, AFLOAT: Ty = { k: 'afloat' };
const vec = (n: 2 | 3 | 4, e: 'f32' | 'i32' | 'u32' = 'f32'): Ty => ({ k: 'vec', n, e });

export function tyStr(t: Ty): string {
  switch (t.k) {
    case 'f32': case 'afloat': return 'f32';
    case 'i32': case 'aint': return 'i32';
    case 'u32': return 'u32';
    case 'bool': return 'bool';
    case 'void': return 'void';
    case 'vec': return `vec${t.n}${t.e === 'f32' ? 'f' : t.e === 'i32' ? 'i' : 'u'}`;
    case 'arr': return `array<${tyStr(t.e)}, ${t.n}>`;
    case 'ptr': return `ptr<function, ${t.e ? tyStr(t.e) : '?'}>`;
    case 'struct': return t.name;
  }
}
const isNum = (t: Ty) => t.k === 'f32' || t.k === 'i32' || t.k === 'u32' || t.k === 'aint' || t.k === 'afloat';
const isFloatish = (t: Ty) => t.k === 'f32' || t.k === 'afloat';
const isIntish = (t: Ty) => t.k === 'i32' || t.k === 'u32' || t.k === 'aint';
const isAbstract = (t: Ty) => t.k === 'aint' || t.k === 'afloat';
const isVec = (t: Ty): t is { k: 'vec'; n: 2 | 3 | 4; e: 'f32' | 'i32' | 'u32' } => t.k === 'vec';
const concrete = (t: Ty): Ty => t.k === 'aint' ? I32 : t.k === 'afloat' ? F32 : t;
function tyEq(a: Ty, b: Ty): boolean { return tyStr(a) === tyStr(b); }

export class TranspileError extends Error {}
const fail = (msg: string): never => { throw new TranspileError(msg); };

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

interface Tok { k: 'id' | 'num' | 'op' | 'str' | 'eof'; s: string; line: number }

const OPS3 = ['<<=', '>>=', '...'];
const OPS2 = ['->', '++', '--', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '##'];

function lex(src: string): Tok[] {
  const toks: Tok[] = [];
  let i = 0, line = 1;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === '\n') { line++; i++; continue; }
    if (c === ' ' || c === '\t' || c === '\r' || c === '\f') { i++; continue; }
    if (c === '\\' && src[i + 1] === '\n') { i += 2; line++; continue; }
    if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') {
      i += 2;
      while (i < n && !(src[i] === '*' && src[i + 1] === '/')) { if (src[i] === '\n') line++; i++; }
      i += 2; continue;
    }
    if (c === '#') {
      // preprocessor line: keep as a single token (up to end of line, honoring \ continuation)
      let j = i;
      let s = '';
      while (j < n && src[j] !== '\n') {
        if (src[j] === '\\' && src[j + 1] === '\n') { j += 2; line++; continue; }
        s += src[j]; j++;
      }
      toks.push({ k: 'op', s: s.trim(), line });
      i = j; continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i + 1;
      while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
      toks.push({ k: 'id', s: src.slice(i, j), line });
      i = j; continue;
    }
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      let j = i;
      if (c === '0' && (src[i + 1] === 'x' || src[i + 1] === 'X')) {
        j = i + 2;
        while (j < n && /[0-9a-fA-F]/.test(src[j])) j++;
      } else {
        while (j < n && /[0-9]/.test(src[j])) j++;
        if (src[j] === '.') { j++; while (j < n && /[0-9]/.test(src[j])) j++; }
        if (src[j] === 'e' || src[j] === 'E') {
          let k = j + 1;
          if (src[k] === '+' || src[k] === '-') k++;
          if (/[0-9]/.test(src[k] ?? '')) { j = k; while (j < n && /[0-9]/.test(src[j])) j++; }
        }
      }
      while (j < n && /[fFuUlL]/.test(src[j])) j++;
      toks.push({ k: 'num', s: src.slice(i, j), line });
      i = j; continue;
    }
    if (c === '"' || c === "'") {
      let j = i + 1;
      while (j < n && src[j] !== c) { if (src[j] === '\\') j++; j++; }
      toks.push({ k: 'str', s: src.slice(i, j + 1), line });
      i = j + 1; continue;
    }
    const three = src.slice(i, i + 3), two = src.slice(i, i + 2);
    if (OPS3.includes(three)) { toks.push({ k: 'op', s: three, line }); i += 3; continue; }
    if (OPS2.includes(two)) { toks.push({ k: 'op', s: two, line }); i += 2; continue; }
    toks.push({ k: 'op', s: c, line });
    i++;
  }
  toks.push({ k: 'eof', s: '', line });
  return toks;
}

// ---------------------------------------------------------------------------
// Preprocessor (object-like and function-like #define, no conditionals)
// ---------------------------------------------------------------------------

interface Macro { params: string[] | null; body: Tok[] }

function preprocess(toks: Tok[], macros: Map<string, Macro>): Tok[] {
  // Collect #define lines
  const out: Tok[] = [];
  for (const t of toks) {
    if (t.k === 'op' && t.s.startsWith('#')) {
      const m = /^#\s*define\s+([A-Za-z_]\w*)(\(([^)]*)\))?\s*(.*)$/s.exec(t.s);
      if (m) {
        const params = m[2] ? m[3].split(',').map((s) => s.trim()).filter(Boolean) : null;
        const body = lex(m[4] ?? '');
        body.pop(); // eof
        macros.set(m[1], { params, body });
        continue;
      }
      if (/^#\s*(include|pragma|ifdef|ifndef|if|else|endif|undef)/.test(t.s)) {
        if (/^#\s*(ifdef|ifndef|if|else|endif)/.test(t.s)) fail(`preprocessor conditional not supported: ${t.s}`);
        continue;
      }
      fail(`unsupported preprocessor line: ${t.s}`);
    }
    out.push(t);
  }
  return expandMacros(out, macros, 0);
}

function expandMacros(toks: Tok[], macros: Map<string, Macro>, depth: number): Tok[] {
  if (depth > 8) fail('macro recursion too deep');
  const out: Tok[] = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    const mac = t.k === 'id' ? macros.get(t.s) : undefined;
    if (!mac) { out.push(t); continue; }
    if (mac.params === null) {
      const body = expandMacros(mac.body, macros, depth + 1);
      // wrap complete expressions in parens; operator-led bodies (e.g. "+epsilon") splice in as-is
      const wrap = body.length > 1 && !(body[0].k === 'op' && ['+', '-', '*', '/'].includes(body[0].s));
      if (wrap) out.push({ k: 'op', s: '(', line: t.line });
      for (const b of body) out.push({ ...b, line: t.line });
      if (wrap) out.push({ k: 'op', s: ')', line: t.line });
      continue;
    }
    // function-like: gather args
    if (toks[i + 1]?.s !== '(') { out.push(t); continue; }
    let j = i + 2, depthP = 0;
    const args: Tok[][] = [[]];
    for (; j < toks.length; j++) {
      const u = toks[j];
      if (u.s === '(' || u.s === '[' || u.s === '{') depthP++;
      if (u.s === ')' || u.s === ']' || u.s === '}') { if (depthP === 0) break; depthP--; }
      if (u.s === ',' && depthP === 0) { args.push([]); continue; }
      args[args.length - 1].push(u);
    }
    if (args.length === 1 && args[0].length === 0) args.pop();
    if (args.length !== mac.params.length) fail(`macro ${t.s} expects ${mac.params.length} args, got ${args.length}`);
    const body: Tok[] = [];
    body.push({ k: 'op', s: '(', line: t.line });
    for (const b of mac.body) {
      const pi = mac.params.indexOf(b.s);
      if (b.k === 'id' && pi >= 0) {
        body.push({ k: 'op', s: '(', line: t.line });
        for (const a of args[pi]) body.push(a);
        body.push({ k: 'op', s: ')', line: t.line });
      } else body.push({ ...b, line: t.line });
    }
    body.push({ k: 'op', s: ')', line: t.line });
    for (const b of expandMacros(body, macros, depth + 1)) out.push(b);
    i = j;
  }
  return out;
}

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

type Expr =
  | { t: 'num'; s: string; isFloat: boolean; v: number }
  | { t: 'id'; name: string }
  | { t: 'str'; s: string }
  | { t: 'call'; name: string; args: Expr[] }
  | { t: 'index'; obj: Expr; idx: Expr }
  | { t: 'member'; obj: Expr; name: string; arrow: boolean }
  | { t: 'unary'; op: string; e: Expr }
  | { t: 'post'; op: string; e: Expr }
  | { t: 'binary'; op: string; l: Expr; r: Expr }
  | { t: 'assign'; op: string; l: Expr; r: Expr }
  | { t: 'cond'; c: Expr; a: Expr; b: Expr }
  | { t: 'cast'; ty: Ty; e: Expr }
  | { t: 'sizeof'; e: Expr | null; ty: Ty | null }
  | { t: 'init'; items: Expr[] }
  | { t: 'comma'; l: Expr; r: Expr };

interface Declarator { name: string; dims: number[]; ptr: boolean; init: Expr | null }
type Stmt =
  | { t: 'decl'; ty: Ty; decls: Declarator[] }
  | { t: 'expr'; e: Expr }
  | { t: 'if'; c: Expr; a: Stmt; b: Stmt | null }
  | { t: 'for'; init: Stmt | null; c: Expr | null; u: Expr | null; body: Stmt }
  | { t: 'while'; c: Expr; body: Stmt }
  | { t: 'do'; body: Stmt; c: Expr }
  | { t: 'block'; stmts: Stmt[] }
  | { t: 'return'; e: Expr | null }
  | { t: 'break' } | { t: 'continue' } | { t: 'empty' }
  | { t: 'switch'; e: Expr; cases: { vals: (Expr | null)[]; body: Stmt[] }[] };

interface Param { ty: Ty; name: string; ptr: boolean; dims: number[] | null }
interface FuncDef { t: 'func'; ret: Ty; name: string; params: Param[]; body: Stmt & { t: 'block' } }
interface GlobalDecl { t: 'gdecl'; ty: Ty; decls: Declarator[]; constant: boolean }
interface StructDef { t: 'struct'; name: string; fields: { ty: Ty; name: string; dims: number[] }[] }
type TopLevel = FuncDef | GlobalDecl | StructDef;

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

const BASE_TYPES: Record<string, Ty> = {
  float: F32, double: F32, int: I32, short: I32, char: I32, long: I32, uint: U32, bool: BOOL, void: VOID,
  float2: vec(2), float3: vec(3), float4: vec(4),
  int2: vec(2, 'i32'), int3: vec(3, 'i32'), int4: vec(4, 'i32'),
  uint2: vec(2, 'u32'), uint3: vec(3, 'u32'), uint4: vec(4, 'u32'),
  vec2: vec(2), vec3: vec(3), vec4: vec(4),
};
const QUALIFIERS = new Set(['const', 'static', '__device__', '__constant__', '__host__', '__forceinline__', 'inline', 'signed', 'volatile', 'register', '__shared__']);

class Parser {
  toks: Tok[]; i = 0;
  structs: Set<string>;
  constructor(toks: Tok[], structs: Set<string>) { this.toks = toks; this.structs = structs; }
  peek(o = 0) { return this.toks[Math.min(this.i + o, this.toks.length - 1)]; }
  next() { return this.toks[this.i++]; }
  at(s: string, o = 0) { const t = this.peek(o); return t.k !== 'str' && t.k !== 'num' && t.s === s; }
  accept(s: string) { if (this.at(s)) { this.i++; return true; } return false; }
  expect(s: string) { if (!this.accept(s)) fail(`expected '${s}' but got '${this.peek().s}' (line ${this.peek().line})`); }
  err(msg: string): never { return fail(`${msg} at '${this.peek().s}' (line ${this.peek().line})`); }

  // ---- types ----
  isTypeStart(o = 0): boolean {
    let k = o;
    while (QUALIFIERS.has(this.peek(k).s) || this.peek(k).s === 'unsigned') k++;
    const t = this.peek(k);
    if (t.k !== 'id') return false;
    if (t.s === 'struct') return true;
    if (BASE_TYPES[t.s]) return true;
    if (this.structs.has(t.s)) return true;
    if (t.s === '__align__') return true;
    return false;
  }
  parseType(): Ty {
    let unsigned = false;
    let ty: Ty | null = null;
    for (;;) {
      const t = this.peek();
      if (t.k === 'id' && QUALIFIERS.has(t.s)) { this.next(); continue; }
      if (t.k === 'id' && t.s === '__align__') { this.next(); this.expect('('); this.next(); this.expect(')'); continue; }
      if (t.k === 'id' && t.s === 'unsigned') { this.next(); unsigned = true; continue; }
      if (t.k === 'id' && t.s === 'struct') {
        this.next();
        while (this.at('__align__')) { this.next(); this.expect('('); this.next(); this.expect(')'); }
        const nm = this.next();
        ty = { k: 'struct', name: nm.s };
        continue;
      }
      if (t.k === 'id' && BASE_TYPES[t.s]) {
        this.next();
        const b = BASE_TYPES[t.s];
        if (ty === null || (ty.k === 'i32' && b.k === 'i32')) ty = b; // "long int", "short int"
        continue;
      }
      if (t.k === 'id' && this.structs.has(t.s) && ty === null) { this.next(); ty = { k: 'struct', name: t.s }; continue; }
      break;
    }
    if (ty === null) { if (unsigned) ty = U32; else this.err('expected type'); }
    if (unsigned && ty!.k === 'i32') ty = U32;
    while (this.at('const')) this.next();
    return ty!;
  }
  parseDeclarator(): { name: string; ptr: boolean; dims: number[] } {
    let ptr = false;
    while (this.accept('*')) ptr = true;
    while (this.at('const')) this.next();
    const nm = this.next();
    if (nm.k !== 'id') this.err('expected identifier');
    const dims: number[] = [];
    while (this.accept('[')) {
      if (this.accept(']')) { dims.push(-1); continue; }
      const e = this.parseExpr();
      this.expect(']');
      const v = constEval(e);
      if (v === null) this.err('array dimension must be constant');
      dims.push(v);
    }
    return { name: nm.s, ptr, dims };
  }

  // ---- top level ----
  parseTop(): TopLevel[] {
    const out: TopLevel[] = [];
    while (this.peek().k !== 'eof') {
      if (this.accept(';')) continue;
      if (this.at('typedef')) {
        this.next();
        // typedef struct {...} Name;  or typedef struct Name {...} Name;
        const st = this.parseStructDef(true);
        out.push(st);
        continue;
      }
      if (this.at('struct') && (this.peek(1).s === '__align__' ? this.peek(5).s === '{' || this.peek(6).s === '{' : this.peek(2).s === '{')) {
        out.push(this.parseStructDef(false));
        continue;
      }
      const ty = this.parseType();
      const d = this.parseDeclarator();
      if (this.at('(')) {
        // function
        this.next();
        const params: Param[] = [];
        if (!this.at(')')) {
          do {
            if (this.at('void') && this.peek(1).s === ')') { this.next(); break; }
            const pty = this.parseType();
            let ptr = false;
            while (this.accept('*')) ptr = true;
            let name = '';
            let dims: number[] | null = null;
            if (this.peek().k === 'id') {
              name = this.next().s;
              while (this.accept('[')) {
                dims ??= [];
                if (this.accept(']')) { dims.push(-1); continue; }
                const e = this.parseExpr(); this.expect(']');
                dims.push(constEval(e) ?? -1);
              }
            }
            params.push({ ty: pty, name, ptr, dims });
          } while (this.accept(','));
        }
        this.expect(')');
        if (this.accept(';')) continue; // prototype
        const body = this.parseBlock();
        out.push({ t: 'func', ret: d.ptr ? { k: 'ptr', e: ty } : ty, name: d.name, params, body });
        continue;
      }
      // global declaration(s)
      const decls: Declarator[] = [];
      let cur = d;
      for (;;) {
        let init: Expr | null = null;
        if (this.accept('=')) init = this.parseInitializer();
        decls.push({ ...cur, init });
        if (this.accept(',')) { cur = this.parseDeclarator(); continue; }
        break;
      }
      this.expect(';');
      out.push({ t: 'gdecl', ty, decls, constant: true });
    }
    return out;
  }
  parseStructDef(typedefd: boolean): StructDef {
    this.expect('struct');
    while (this.at('__align__')) { this.next(); this.expect('('); this.next(); this.expect(')'); }
    let name = '';
    if (this.peek().k === 'id' && !this.at('{')) name = this.next().s;
    this.expect('{');
    const fields: StructDef['fields'] = [];
    while (!this.accept('}')) {
      const ty = this.parseType();
      do {
        const d = this.parseDeclarator();
        fields.push({ ty, name: d.name, dims: d.dims });
      } while (this.accept(','));
      this.expect(';');
    }
    if (typedefd || this.peek().k === 'id') {
      const nm = this.next();
      if (nm.k === 'id') name = name || nm.s;
    }
    this.expect(';');
    this.structs.add(name);
    return { t: 'struct', name, fields };
  }
  parseInitializer(): Expr {
    if (this.accept('{')) {
      const items: Expr[] = [];
      while (!this.at('}')) {
        items.push(this.parseInitializer());
        if (!this.accept(',')) break;
      }
      this.expect('}');
      return { t: 'init', items };
    }
    return this.parseAssign();
  }

  // ---- statements ----
  parseBlock(): Stmt & { t: 'block' } {
    this.expect('{');
    const stmts: Stmt[] = [];
    while (!this.accept('}')) {
      if (this.peek().k === 'eof') this.err('unexpected end of input in block');
      stmts.push(this.parseStmt());
    }
    return { t: 'block', stmts };
  }
  parseStmt(): Stmt {
    const t = this.peek();
    if (this.at('{')) return this.parseBlock();
    if (this.accept(';')) return { t: 'empty' };
    if (t.k === 'id') {
      switch (t.s) {
        case 'if': {
          this.next(); this.expect('(');
          const c = this.parseExpr(); this.expect(')');
          const a = this.parseStmt();
          const b = this.accept('else') ? this.parseStmt() : null;
          return { t: 'if', c, a, b };
        }
        case 'for': {
          this.next(); this.expect('(');
          let init: Stmt | null = null;
          if (!this.at(';')) {
            if (this.isTypeStart()) init = this.parseDeclStmt(false);
            else init = { t: 'expr', e: this.parseExpr() };
          }
          this.expect(';');
          const c = this.at(';') ? null : this.parseExpr();
          this.expect(';');
          const u = this.at(')') ? null : this.parseExpr();
          this.expect(')');
          const body = this.parseStmt();
          return { t: 'for', init, c, u, body };
        }
        case 'while': {
          this.next(); this.expect('(');
          const c = this.parseExpr(); this.expect(')');
          return { t: 'while', c, body: this.parseStmt() };
        }
        case 'do': {
          this.next();
          const body = this.parseStmt();
          if (!this.accept('while')) this.err("expected 'while'");
          this.expect('(');
          const c = this.parseExpr(); this.expect(')'); this.accept(';');
          return { t: 'do', body, c };
        }
        case 'return': {
          this.next();
          const e = this.at(';') ? null : this.parseExpr();
          this.expect(';');
          return { t: 'return', e };
        }
        case 'break': this.next(); this.expect(';'); return { t: 'break' };
        case 'continue': this.next(); this.expect(';'); return { t: 'continue' };
        case 'switch': {
          this.next(); this.expect('(');
          const e = this.parseExpr(); this.expect(')'); this.expect('{');
          const cases: { vals: (Expr | null)[]; body: Stmt[] }[] = [];
          while (!this.accept('}')) {
            const vals: (Expr | null)[] = [];
            while (this.at('case') || this.at('default')) {
              if (this.accept('default')) { vals.push(null); }
              else { this.next(); vals.push(this.parseCond()); }
              this.expect(':');
            }
            if (!vals.length) this.err('expected case');
            const body: Stmt[] = [];
            while (!this.at('case') && !this.at('default') && !this.at('}')) body.push(this.parseStmt());
            cases.push({ vals, body });
          }
          return { t: 'switch', e, cases };
        }
        case 'goto': this.err('goto not supported');
      }
    }
    if (this.isTypeStart()) {
      const d = this.parseDeclStmt(true);
      return d;
    }
    const e = this.parseExpr();
    this.expect(';');
    return { t: 'expr', e };
  }
  parseDeclStmt(semi: boolean): Stmt {
    const ty = this.parseType();
    const decls: Declarator[] = [];
    do {
      const d = this.parseDeclarator();
      let init: Expr | null = null;
      if (this.accept('=')) init = this.parseInitializer();
      decls.push({ ...d, init });
    } while (this.accept(','));
    if (semi) this.expect(';');
    return { t: 'decl', ty, decls };
  }

  // ---- expressions ----
  parseExpr(): Expr {
    let e = this.parseAssign();
    while (this.at(',')) { this.next(); e = { t: 'comma', l: e, r: this.parseAssign() }; }
    return e;
  }
  parseAssign(): Expr {
    const l = this.parseCond();
    const t = this.peek();
    if (t.k === 'op' && ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='].includes(t.s)) {
      this.next();
      const r = this.parseAssign();
      return { t: 'assign', op: t.s, l, r };
    }
    return l;
  }
  parseCond(): Expr {
    const c = this.parseBin(0);
    if (this.accept('?')) {
      const a = this.parseExpr();
      this.expect(':');
      const b = this.parseCond();
      return { t: 'cond', c, a, b };
    }
    return c;
  }
  static PREC: string[][] = [['||'], ['&&'], ['|'], ['^'], ['&'], ['==', '!='], ['<', '>', '<=', '>='], ['<<', '>>'], ['+', '-'], ['*', '/', '%']];
  parseBin(level: number): Expr {
    if (level >= Parser.PREC.length) return this.parseUnary();
    let l = this.parseBin(level + 1);
    for (;;) {
      const t = this.peek();
      if (t.k === 'op' && Parser.PREC[level].includes(t.s)) {
        this.next();
        const r = this.parseBin(level + 1);
        l = { t: 'binary', op: t.s, l, r };
      } else return l;
    }
  }
  parseUnary(): Expr {
    const t = this.peek();
    if (t.k === 'op') {
      if (['-', '+', '!', '~', '*', '&'].includes(t.s)) { this.next(); return { t: 'unary', op: t.s, e: this.parseUnary() }; }
      if (t.s === '++' || t.s === '--') { this.next(); return { t: 'unary', op: t.s, e: this.parseUnary() }; }
      if (t.s === '(' && this.isTypeStart(1)) {
        // cast (but not a compound literal / function-style: e.g. (float)x, (int)(x))
        this.next();
        const ty = this.parseType();
        let ptr = false;
        while (this.accept('*')) ptr = true;
        this.expect(')');
        return { t: 'cast', ty: ptr ? { k: 'ptr', e: ty } : ty, e: this.parseUnary() };
      }
    }
    if (t.k === 'id' && t.s === 'sizeof') {
      this.next();
      if (this.at('(') && this.isTypeStart(1)) {
        this.next(); const ty = this.parseType(); this.expect(')');
        return { t: 'sizeof', e: null, ty };
      }
      return { t: 'sizeof', e: this.parseUnary(), ty: null };
    }
    return this.parsePostfix();
  }
  parsePostfix(): Expr {
    let e = this.parsePrimary();
    for (;;) {
      if (this.at('(') && e.t === 'id') {
        this.next();
        const args: Expr[] = [];
        if (!this.at(')')) { do { args.push(this.parseAssign()); } while (this.accept(',')); }
        this.expect(')');
        e = { t: 'call', name: e.name, args };
      } else if (this.accept('[')) {
        const idx = this.parseExpr(); this.expect(']');
        e = { t: 'index', obj: e, idx };
      } else if (this.at('.') && this.peek(1).k === 'id') {
        this.next(); e = { t: 'member', obj: e, name: this.next().s, arrow: false };
      } else if (this.at('->')) {
        this.next(); e = { t: 'member', obj: e, name: this.next().s, arrow: true };
      } else if (this.at('++') || this.at('--')) {
        e = { t: 'post', op: this.next().s, e };
      } else return e;
    }
  }
  parsePrimary(): Expr {
    const t = this.next();
    if (t.k === 'num') return parseNum(t.s);
    if (t.k === 'str') return { t: 'str', s: t.s };
    if (t.k === 'id') {
      if (BASE_TYPES[t.s] && this.at('(')) {
        // functional cast e.g. float(x) / float2(a,b)
        this.next();
        const args: Expr[] = [];
        if (!this.at(')')) { do { args.push(this.parseAssign()); } while (this.accept(',')); }
        this.expect(')');
        return { t: 'call', name: t.s, args };
      }
      return { t: 'id', name: t.s };
    }
    if (t.s === '(') {
      const e = this.parseExpr();
      this.expect(')');
      return e;
    }
    return this.err(`unexpected token '${t.s}'`);
  }
}

function parseNum(s: string): Expr {
  if (/^0[xX]/.test(s)) {
    const clean = s.replace(/[uUlL]+$/, '');
    let v = parseInt(clean, 16);
    if (v > 0x7fffffff) v = v - 0x100000000; // C's unsigned literal reinterpreted as i32 bit pattern
    return { t: 'num', s: String(v), isFloat: false, v };
  }
  const clean = s.replace(/[fFuUlL]+$/, '');
  const isFloat = /[.eE]/.test(clean);
  const v = Number(clean);
  return { t: 'num', s: clean, isFloat: isFloat || /[fF]$/.test(s), v };
}

function constEval(e: Expr): number | null {
  switch (e.t) {
    case 'num': return e.v;
    case 'unary': { const v = constEval(e.e); return v === null ? null : e.op === '-' ? -v : e.op === '+' ? v : null; }
    case 'binary': {
      const a = constEval(e.l), b = constEval(e.r);
      if (a === null || b === null) return null;
      switch (e.op) {
        case '+': return a + b; case '-': return a - b; case '*': return a * b;
        case '/': return Number.isInteger(a) && Number.isInteger(b) ? Math.trunc(a / b) : a / b;
        case '%': return a % b; case '<<': return a << b; case '>>': return a >> b;
      }
      return null;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

export interface Binding { code: string; ty: Ty; lvalue?: boolean; flag?: string; decl?: string }
export interface Env {
  /** identifier bindings visible in the snippet (magic vars, weight, params). */
  bindings: Map<string, Binding>;
  /** unresolved __name → binding hook (e.g. for `__varname_param`) */
  resolveMagic?: (name: string) => Binding | null;
}

interface Sym { ty: Ty; wname: string; isParam?: boolean }
interface FnSig { name: string; wname: string; ret: Ty; params: { ty: Ty; name: string; ptrInferred: boolean }[]; def: FuncDef | null; used: boolean; body?: string; deps: Set<string>; assignsParams: Set<string> }

// WGSL reserved words + builtin names + our prelude names — C identifiers matching these get a suffix.
const RESERVED = new Set(`alias break case const const_assert continue continuing default diagnostic discard else enable false fn for if let loop override requires return struct switch true var while
NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become binding_array cast catch class co_await co_return co_yield coherent column_major common compile compile_fragment concept const_cast consteval constexpr constinit crate debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit export extends extern external fallthrough filter final finally friend from fxgroup get goto groupshared highp impl implements import inline instanceof interface layout lowp macro macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept noinline nointerpolation non_coherent noncoherent noperspective null nullptr of operator package packoffset partition pass patch pixelfragment precise precision premerge priv protected pub public readonly ref regardless register reinterpret_cast require resource restrict self set shared sizeof smooth snorm static static_assert static_cast std subroutine super target template this thread_local throw trait try type typedef typeid typename typeof union unless unorm unsafe unsized use using varying virtual volatile wgsl where with writeonly yield
f32 f16 i32 u32 bool vec2 vec3 vec4 vec2f vec3f vec4f vec2i vec3i vec4i vec2u vec3u vec4u mat2x2 mat3x3 mat4x4 array ptr atomic sampler texture function private workgroup uniform storage read write read_write
abs acos acosh all any asin asinh atan atan2 atanh bitcast ceil clamp cos cosh countLeadingZeros countOneBits countTrailingZeros cross degrees determinant distance dot dot4I8Packed dot4U8Packed exp exp2 extractBits faceForward firstLeadingBit firstTrailingBit floor fma fract frexp insertBits inverseSqrt ldexp length log log2 max min mix modf normalize pow quantizeToF16 radians reflect refract reverseBits round saturate select sign sin sinh smoothstep sqrt step tan tanh transpose trunc arrayLength
t v r r2 th ph rs cp hd w p A PI xd pts hist pal P rnd rndi mmod pin np op main iters
in out`.split(/\s+/));

const LN10_INV = '0.4342944819032518';

// builtin function table: C name → emitter. Returns code+type.
type EmitRes = { c: string; ty: Ty; lv?: boolean };
type BuiltinFn = (args: EmitRes[], em: Emitter) => EmitRes;

function unaryF(wname: string): BuiltinFn {
  return (a, em) => {
    if (a.length !== 1) fail(`${wname} expects 1 arg`);
    const x = a[0];
    if (isVec(x.ty)) return { c: `${wname}(${x.c})`, ty: x.ty };
    return { c: `${wname}(${em.toF32(x)})`, ty: F32 };
  };
}
function binaryF(wname: string): BuiltinFn {
  return (a, em) => {
    if (a.length !== 2) fail(`${wname} expects 2 args`);
    if (isVec(a[0].ty) || isVec(a[1].ty)) {
      const [x, y] = em.unifyVec(a[0], a[1]);
      return { c: `${wname}(${x.c}, ${y.c})`, ty: x.ty };
    }
    return { c: `${wname}(${em.toF32(a[0])}, ${em.toF32(a[1])})`, ty: F32 };
  };
}
function minmax(wname: string): BuiltinFn {
  return (a, em) => {
    if (a.length !== 2) fail(`${wname} expects 2 args`);
    const [x, y] = em.unify(a[0], a[1]);
    return { c: `${wname}(${x.c}, ${y.c})`, ty: concrete(x.ty) };
  };
}
const BUILTINS: Record<string, BuiltinFn> = {};
for (const [cn, wn] of Object.entries({
  sinf: 'sin', sin: 'sin', __sinf: 'sin', cosf: 'cos', cos: 'cos', __cosf: 'cos', tanf: 'tan', tan: 'tan',
  asinf: 'asin', asin: 'asin', acosf: 'acos', acos: 'acos', atanf: 'atan', sinhf: 'sinh', sinh: 'sinh', coshf: 'cosh', cosh: 'cosh',
  tanhf: 'tanh', tanh: 'tanh', asinhf: 'asinh', acoshf: 'acosh', atanhf: 'atanh',
  expf: 'exp', exp: 'exp', __expf: 'exp', exp2f: 'exp2', exp2: 'exp2', logf: 'log', log: 'log', log2f: 'log2', log2: 'log2',
  sqrtf: 'sqrt', sqrt: 'sqrt', rsqrtf: 'inverseSqrt', fabsf: 'abs', fabs: 'abs',
  floorf: 'floor', floor: 'floor', ceilf: 'ceil', ceil: 'ceil', truncf: 'trunc', trunc: 'trunc',
  rintf: 'round', rint: 'round', nearbyintf: 'round',
  fract: 'fract', fracf: 'fract', normalize: 'normalize', saturate: 'saturate',
})) BUILTINS[cn] = unaryF(wn);
for (const [cn, wn] of Object.entries({ atan2f: 'atan2', atan2: 'atan2', step: 'step', distance: 'distance', reflect: 'reflect' }))
  BUILTINS[cn] = binaryF(wn);
// C powf: negative bases with integer-valued exponents are defined (WGSL pow is NaN there)
BUILTINS.powf = (a, em) => {
  if (a.length !== 2) fail('powf expects 2 args');
  if (isVec(a[0].ty) || isVec(a[1].ty)) return binaryF('pow')(a, em);
  return { c: `powc(${em.toF32(a[0])}, ${em.toF32(a[1])})`, ty: F32 };
};
BUILTINS.pow = BUILTINS.powf; BUILTINS.__powf = BUILTINS.powf;
BUILTINS.atan = (a, em) => a.length === 2 ? binaryF('atan2')(a, em) : unaryF('atan')(a, em);
for (const [cn, wn] of Object.entries({ fmaxf: 'max', fmax: 'max', max: 'max', fminf: 'min', fmin: 'min', min: 'min' })) BUILTINS[cn] = minmax(wn);
BUILTINS.abs = (a, em) => {
  const x = a[0];
  if (isVec(x.ty)) return { c: `abs(${x.c})`, ty: x.ty };
  if (isIntish(x.ty)) return { c: `abs(${em.toI32(x)})`, ty: I32 };
  return { c: `abs(${em.toF32(x)})`, ty: F32 };
};
BUILTINS.iabs = BUILTINS.abs;
BUILTINS.sign = (a, em) => isVec(a[0].ty) ? { c: `sign(${a[0].c})`, ty: a[0].ty } : { c: `sign(${em.toF32(a[0])})`, ty: F32 };
BUILTINS.sgn = BUILTINS.sign;
BUILTINS.roundf = (a, em) => isVec(a[0].ty) ? { c: `roundc${a[0].ty.n}(${a[0].c})`, ty: a[0].ty } : { c: `roundc(${em.toF32(a[0])})`, ty: F32 };
BUILTINS.round = BUILTINS.roundf;
BUILTINS.lroundf = (a, em) => ({ c: `i32(roundc(${em.toF32(a[0])}))`, ty: I32 });
BUILTINS.lround = BUILTINS.lroundf;
BUILTINS.lrintf = (a, em) => ({ c: `i32(round(${em.toF32(a[0])}))`, ty: I32 });
BUILTINS.fmodf = (a, em) => ({ c: `(${em.toF32(a[0])} % ${em.toF32(a[1])})`, ty: F32 });
BUILTINS.fmod = BUILTINS.fmodf;
BUILTINS.mod = (a, em) => {
  if (isVec(a[0].ty)) {
    const t = a[0].ty as { k: 'vec'; n: number };
    const y = isVec(a[1].ty) ? a[1].c : `${tyStr(a[0].ty)}(${em.toF32(a[1])})`;
    return { c: `mmod${t.n}(${a[0].c}, ${y})`, ty: a[0].ty };
  }
  return { c: `mmod(${em.toF32(a[0])}, ${em.toF32(a[1])})`, ty: F32 };
};
BUILTINS.log10f = (a, em) => ({ c: `(log(${em.toF32(a[0])}) * ${LN10_INV})`, ty: F32 });
BUILTINS.log10 = BUILTINS.log10f;
BUILTINS.hypotf = (a, em) => ({ c: `length(vec2f(${em.toF32(a[0])}, ${em.toF32(a[1])}))`, ty: F32 });
BUILTINS.hypot = BUILTINS.hypotf;
BUILTINS.sqrf = (a, em) => { const x = em.toF32(a[0]); return { c: `sqr(${x})`, ty: F32 }; };
BUILTINS.sqr = BUILTINS.sqrf;
BUILTINS.sqrtf_safe = (a, em) => ({ c: `sqrt(max(${em.toF32(a[0])}, 0.0))`, ty: F32 });
BUILTINS.cbrtf = (a, em) => ({ c: `cbrt(${em.toF32(a[0])})`, ty: F32 });
BUILTINS.cbrt = BUILTINS.cbrtf;
BUILTINS.expm1f = (a, em) => ({ c: `(exp(${em.toF32(a[0])}) - 1.0)`, ty: F32 });
BUILTINS.log1pf = (a, em) => ({ c: `log(1.0 + ${em.toF32(a[0])})`, ty: F32 });
BUILTINS.copysignf = (a, em) => ({ c: `copysign(${em.toF32(a[0])}, ${em.toF32(a[1])})`, ty: F32 });
BUILTINS.__fdividef = (a, em) => ({ c: `(${em.toF32(a[0])} / ${em.toF32(a[1])})`, ty: F32 });
BUILTINS.fdividef = BUILTINS.__fdividef;
BUILTINS.__int_as_float = (a, em) => ({ c: `bitcast<f32>(${em.toI32(a[0])})`, ty: F32 });
BUILTINS.__float_as_int = (a, em) => ({ c: `bitcast<i32>(${em.toF32(a[0])})`, ty: I32 });
BUILTINS.isnan = (a, em) => { const x = em.toF32(a[0]); return { c: `(${x} != ${x})`, ty: BOOL }; };
BUILTINS.isinf = (a, em) => ({ c: `(abs(${em.toF32(a[0])}) > 3.0e38)`, ty: BOOL });
BUILTINS.lerpf = (a, em) => ({ c: `mix(${em.toF32(a[0])}, ${em.toF32(a[1])}, ${em.toF32(a[2])})`, ty: F32 });
BUILTINS.lerp = BUILTINS.lerpf;
BUILTINS.mix = (a, em) => {
  if (isVec(a[0].ty)) {
    const [x, y] = em.unifyVec(a[0], a[1]);
    const s = isVec(a[2].ty) ? a[2].c : em.toF32(a[2]);
    return { c: `mix(${x.c}, ${y.c}, ${s})`, ty: x.ty };
  }
  return BUILTINS.lerpf(a, em);
};
BUILTINS.clamp = (a, em) => {
  if (isVec(a[0].ty)) {
    const t = a[0].ty;
    const lo = isVec(a[1].ty) ? a[1].c : `${tyStr(t)}(${em.toF32(a[1])})`;
    const hi = isVec(a[2].ty) ? a[2].c : `${tyStr(t)}(${em.toF32(a[2])})`;
    return { c: `clamp(${a[0].c}, ${lo}, ${hi})`, ty: t };
  }
  const [x, y] = em.unify(a[0], a[1]);
  const [x2, z] = em.unify(x, a[2]);
  const [y2] = em.unify(y, z);
  return { c: `clamp(${x2.c}, ${y2.c}, ${z.c})`, ty: concrete(x2.ty) };
};
BUILTINS.smoothstep = (a, em) => {
  if (isVec(a[2].ty)) {
    const t = a[2].ty;
    const lo = isVec(a[0].ty) ? a[0].c : `${tyStr(t)}(${em.toF32(a[0])})`;
    const hi = isVec(a[1].ty) ? a[1].c : `${tyStr(t)}(${em.toF32(a[1])})`;
    return { c: `smoothstep(${lo}, ${hi}, ${a[2].c})`, ty: t };
  }
  // scalar: helper that tolerates equal edges (WGSL rejects smoothstep(1.0, 1.0, x) at compile time)
  return { c: `smoothstepc(${em.toF32(a[0])}, ${em.toF32(a[1])}, ${em.toF32(a[2])})`, ty: F32 };
};
BUILTINS.distance = (a, em) => {
  if (isVec(a[0].ty)) { const [x, y] = em.unifyVec(a[0], a[1]); return { c: `distance(${x.c}, ${y.c})`, ty: F32 }; }
  return { c: `abs(${em.toF32(a[0])} - ${em.toF32(a[1])})`, ty: F32 };
};
BUILTINS.length = (a) => { if (!isVec(a[0].ty)) fail('length() of non-vector'); return { c: `length(${a[0].c})`, ty: F32 }; };
BUILTINS.dot = (a, em) => {
  if (isVec(a[0].ty)) { const [x, y] = em.unifyVec(a[0], a[1]); return { c: `dot(${x.c}, ${y.c})`, ty: F32 }; }
  return { c: `(${em.toF32(a[0])} * ${em.toF32(a[1])})`, ty: F32 };
};
BUILTINS.cross = (a) => ({ c: `cross(${a[0].c}, ${a[1].c})`, ty: vec(3) });
for (const n of [2, 3, 4] as const) {
  const mk = (e: 'f32' | 'i32' | 'u32'): BuiltinFn => (a, em) => {
    const conv = e === 'f32' ? (x: EmitRes) => em.toF32(x) : e === 'i32' ? (x: EmitRes) => em.toI32(x) : (x: EmitRes) => em.toU32(x);
    return { c: `${tyStr(vec(n, e))}(${a.map((x) => isVec(x.ty) ? x.c : conv(x)).join(', ')})`, ty: vec(n, e) };
  };
  BUILTINS[`make_float${n}`] = mk('f32'); BUILTINS[`float${n}`] = mk('f32'); BUILTINS[`vec${n}`] = mk('f32');
  BUILTINS[`make_int${n}`] = mk('i32'); BUILTINS[`int${n}`] = mk('i32');
  BUILTINS[`make_uint${n}`] = mk('u32'); BUILTINS[`uint${n}`] = mk('u32');
}
BUILTINS.float = (a, em) => ({ c: em.toF32(a[0], true), ty: F32 });
BUILTINS.double = BUILTINS.float;
BUILTINS.int = (a, em) => ({ c: em.toI32(a[0], true), ty: I32 });
BUILTINS.uint = (a, em) => ({ c: em.toU32(a[0], true), ty: U32 });
BUILTINS.bool = (a, em) => ({ c: em.toBool(a[0]), ty: BOOL });
BUILTINS.RANDFLOAT = () => ({ c: 'rnd(rs)', ty: F32 });
// JWildfire GPU palette read (dc_* direct-colour variations): `palette` binds to the
// layer's palette base index, `numColors` to 256; helper defined in HELPER_FUNCS.
BUILTINS.read_imageStepMode = (a, em) => ({ c: `read_imageStepMode(${em.toU32(a[0])}, ${em.toI32(a[1])}, ${em.toF32(a[2])})`, ty: vec(4, 'f32') });
BUILTINS.RANDINT = () => ({ c: 'rndi(rs)', ty: U32 });
BUILTINS.erff = (a, em) => ({ c: `erf(${em.toF32(a[0])})`, ty: F32 });
BUILTINS.erf = BUILTINS.erff;
BUILTINS.j1f = (a, em) => ({ c: `besselJ1(${em.toF32(a[0])})`, ty: F32 });
BUILTINS.j0f = (a, em) => ({ c: `besselJ0(${em.toF32(a[0])})`, ty: F32 });
BUILTINS.tgammaf = (a, em) => ({ c: `tgamma(${em.toF32(a[0])})`, ty: F32 });
BUILTINS.lgammaf = (a, em) => ({ c: `lgamma(${em.toF32(a[0])})`, ty: F32 });
BUILTINS.fmaf = (a, em) => ({ c: `fma(${em.toF32(a[0])}, ${em.toF32(a[1])}, ${em.toF32(a[2])})`, ty: F32 });
BUILTINS.fdimf = (a, em) => ({ c: `max(${em.toF32(a[0])} - ${em.toF32(a[1])}, 0.0)`, ty: F32 });
BUILTINS.remainderf = (a, em) => { const x = em.toF32(a[0]), y = em.toF32(a[1]); return { c: `(${x} - ${y} * round(${x} / ${y}))`, ty: F32 }; };
BUILTINS.sinpif = (a, em) => ({ c: `sin(PI * ${em.toF32(a[0])})`, ty: F32 });
BUILTINS.cospif = (a, em) => ({ c: `cos(PI * ${em.toF32(a[0])})`, ty: F32 });
BUILTINS.any = (a) => ({ c: `any(${a[0].c})`, ty: BOOL });
BUILTINS.all = (a) => ({ c: `all(${a[0].c})`, ty: BOOL });
BUILTINS.rand = () => ({ c: 'rnd(rs)', ty: F32 });

// Extra WGSL helper functions referenced by builtin mappings (included on demand).
export const HELPER_FUNCS: Record<string, string> = {
  sqr: `fn sqr(x: f32) -> f32 { return x * x; }`,
  smoothstepc: `fn smoothstepc(a: f32, b: f32, x: f32) -> f32 { if (a == b) { return step(a, x); } let t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }`,
  read_imageStepMode: `fn read_imageStepMode(base: u32, n: i32, t: f32) -> vec4f { return pal[base + u32(clamp(t, 0.0, 0.99999) * f32(max(n, 1)))]; }`,
  // C powf semantics: a negative base with an integer-valued exponent is defined
  powc: `fn powc(x: f32, y: f32) -> f32 {
  if (x >= 0.0) { return pow(x, y); }
  let yi = round(y);
  if (abs(y - yi) > 1e-6) { return pow(x, y); }
  let m = pow(-x, y);
  return select(m, -m, (i32(yi) & 1) != 0);
}`,
  roundc: `fn roundc(x: f32) -> f32 { return sign(x) * floor(abs(x) + 0.5); }`,
  roundc2: `fn roundc2(x: vec2f) -> vec2f { return sign(x) * floor(abs(x) + 0.5); }`,
  roundc3: `fn roundc3(x: vec3f) -> vec3f { return sign(x) * floor(abs(x) + 0.5); }`,
  roundc4: `fn roundc4(x: vec4f) -> vec4f { return sign(x) * floor(abs(x) + 0.5); }`,
  mmod2: `fn mmod2(a: vec2f, b: vec2f) -> vec2f { return a - b * floor(a / b); }`,
  mmod3: `fn mmod3(a: vec3f, b: vec3f) -> vec3f { return a - b * floor(a / b); }`,
  mmod4: `fn mmod4(a: vec4f, b: vec4f) -> vec4f { return a - b * floor(a / b); }`,
  cbrt: `fn cbrt(x: f32) -> f32 { return sign(x) * pow(abs(x), 1.0 / 3.0); }`,
  copysign: `fn copysign(a: f32, b: f32) -> f32 { return select(-abs(a), abs(a), b >= 0.0); }`,
  rndi: `fn rndi(state: ptr<function, u32>) -> u32 { var x = *state; x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u; *state = x; return x; }`,
  erf: `fn erf(x: f32) -> f32 { let s = sign(x); let a = abs(x); let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return s * y; }`,
  besselJ0: `fn besselJ0(x: f32) -> f32 { let ax = abs(x);
  if (ax < 8.0) { let y = x * x;
    let a1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
    let a2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
    return a1 / a2; }
  let z = 8.0 / ax; let y = z * z; let xx = ax - 0.785398164;
  let a1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  let a2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
  return sqrt(0.636619772 / ax) * (cos(xx) * a1 - z * sin(xx) * a2); }`,
  besselJ1: `fn besselJ1(x: f32) -> f32 { let ax = abs(x);
  if (ax < 8.0) { let y = x * x;
    let a1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
    let a2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
    return a1 / a2; }
  let z = 8.0 / ax; let y = z * z; let xx = ax - 2.356194491;
  let a1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
  let a2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
  let ans = sqrt(0.636619772 / ax) * (cos(xx) * a1 - z * sin(xx) * a2);
  return select(ans, -ans, x < 0.0); }`,
  lgamma: `fn lgamma(xx: f32) -> f32 { var x = xx; var y = x; var tmp = x + 5.5; tmp -= (x + 0.5) * log(tmp);
  var ser = 1.000000000190015;
  let cof = array<f32, 6>(76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5);
  for (var j = 0; j < 6; j++) { y += 1.0; ser += cof[j] / y; }
  return -tmp + log(2.5066282746310005 * ser / x); }`,
  tgamma: `fn tgamma(x: f32) -> f32 { return exp(lgamma(x)); }`,
};
const HELPER_DEPS: Record<string, string[]> = { tgamma: ['lgamma'] };

function isLvalueExpr(e: Expr): boolean {
  return e.t === 'id' || e.t === 'index' || e.t === 'member' || (e.t === 'unary' && e.op === '*');
}

export interface Program {
  structs: Map<string, StructDef>;
  funcs: Map<string, FnSig[]>;      // by C name (overloads)
  globals: Map<string, { wname: string; ty: Ty; code: string; used: boolean; deps: Set<string> }>;
  helpersUsed: Set<string>;
  flags: Set<string>;
  macros: Map<string, Macro>;
}

class Scope {
  syms = new Map<string, Sym>();
  parent: Scope | null;
  fn: FnSig | null;
  constructor(parent: Scope | null, fn: FnSig | null) { this.parent = parent; this.fn = fn; }
  lookup(n: string): Sym | null { return this.syms.get(n) ?? this.parent?.lookup(n) ?? null; }
}

class Emitter {
  prog: Program;
  env: Env;
  usedNames = new Set<string>();
  declsUsed = new Set<string>();       // binding declarations to prepend to the snippet
  fnDeps: Set<string> | null = null;   // deps of the function currently being emitted
  flags: Set<string>;
  curFn: FnSig | null = null;
  constructor(prog: Program, env: Env, flags: Set<string>) { this.prog = prog; this.env = env; this.flags = flags; }

  // ---- naming ----
  wname(n: string): string {
    let m = n.replace(/^__+/, 'jw_');
    if (RESERVED.has(m) || /^_$/.test(m)) m = m + '_';
    return m;
  }

  // ---- conversions ----
  toF32(x: EmitRes, explicit = false): string {
    if (x.ty.k === 'f32' || x.ty.k === 'afloat') return x.c;
    if (x.ty.k === 'aint') return explicit ? `f32(${x.c})` : x.c;
    if (x.ty.k === 'i32' || x.ty.k === 'u32') return `f32(${x.c})`;
    if (x.ty.k === 'bool') return `select(0.0, 1.0, ${x.c})`;
    if (isVec(x.ty)) return x.c; // caller handles vectors
    return fail(`cannot convert ${tyStr(x.ty)} to f32`);
  }
  toI32(x: EmitRes, explicit = false): string {
    if (x.ty.k === 'i32') return x.c;
    if (x.ty.k === 'aint') return explicit ? `i32(${x.c})` : x.c;
    if (x.ty.k === 'f32' || x.ty.k === 'afloat' || x.ty.k === 'u32') return `i32(${x.c})`;
    if (x.ty.k === 'bool') return `select(0, 1, ${x.c})`;
    return fail(`cannot convert ${tyStr(x.ty)} to i32`);
  }
  toU32(x: EmitRes, explicit = false): string {
    if (x.ty.k === 'u32') return x.c;
    if (x.ty.k === 'aint') return explicit ? `u32(${x.c})` : x.c;
    if (x.ty.k === 'f32' || x.ty.k === 'afloat' || x.ty.k === 'i32') return `u32(${x.c})`;
    if (x.ty.k === 'bool') return `select(0u, 1u, ${x.c})`;
    return fail(`cannot convert ${tyStr(x.ty)} to u32`);
  }
  toBool(x: EmitRes): string {
    if (x.ty.k === 'bool') return x.c;
    if (x.ty.k === 'f32' || x.ty.k === 'afloat') return `(${x.c} != 0.0)`;
    if (x.ty.k === 'i32' || x.ty.k === 'aint') return `(${x.c} != 0)`;
    if (x.ty.k === 'u32') return `(${x.c} != 0u)`;
    return fail(`cannot convert ${tyStr(x.ty)} to bool`);
  }
  convTo(x: EmitRes, ty: Ty): string {
    switch (ty.k) {
      case 'f32': case 'afloat': return this.toF32(x);
      case 'i32': case 'aint': return this.toI32(x);
      case 'u32': return this.toU32(x);
      case 'bool': return this.toBool(x);
      case 'vec': {
        if (isVec(x.ty)) {
          if (x.ty.n !== ty.n) fail(`vector size mismatch ${tyStr(x.ty)} → ${tyStr(ty)}`);
          if (x.ty.e === ty.e) return x.c;
          return `${tyStr(ty)}(${x.c})`;
        }
        if (isNum(x.ty)) return `${tyStr(ty)}(${ty.e === 'f32' ? this.toF32(x) : ty.e === 'i32' ? this.toI32(x) : this.toU32(x)})`;
        return fail(`cannot convert ${tyStr(x.ty)} to ${tyStr(ty)}`);
      }
      case 'struct': case 'arr': case 'ptr':
        if (tyEq(x.ty, ty) || (ty.k === 'ptr' && x.ty.k === 'ptr')) return x.c;
        return fail(`cannot convert ${tyStr(x.ty)} to ${tyStr(ty)}`);
      case 'void': return x.c;
    }
  }
  /** unify two scalar operands for arithmetic/comparison (C promotion rules) */
  unify(a: EmitRes, b: EmitRes): [EmitRes, EmitRes] {
    if (isVec(a.ty) || isVec(b.ty)) return this.unifyVec(a, b);
    const A = a.ty.k, B = b.ty.k;
    if (A === 'bool') a = { c: this.toI32(a), ty: I32 };
    if (B === 'bool') b = { c: this.toI32(b), ty: I32 };
    if (a.ty.k === 'ptr' || b.ty.k === 'ptr' || a.ty.k === 'struct' || b.ty.k === 'struct' || a.ty.k === 'arr' || b.ty.k === 'arr')
      fail(`arithmetic on ${tyStr(a.ty)} and ${tyStr(b.ty)}`);
    if (isFloatish(a.ty) || isFloatish(b.ty)) {
      const ta: Ty = (a.ty.k === 'f32' || b.ty.k === 'f32') ? F32 : AFLOAT;
      return [{ c: this.toF32(a), ty: ta }, { c: this.toF32(b), ty: ta }];
    }
    if (a.ty.k === 'u32' || b.ty.k === 'u32') {
      if (a.ty.k === 'i32' || b.ty.k === 'i32') { // C promotes int to unsigned; keep i32 arithmetic (safer for our uses)
        return [{ c: this.toI32(a), ty: I32 }, { c: this.toI32(b), ty: I32 }];
      }
      return [{ c: this.toU32(a), ty: U32 }, { c: this.toU32(b), ty: U32 }];
    }
    if (a.ty.k === 'i32' || b.ty.k === 'i32') return [{ c: this.toI32(a), ty: I32 }, { c: this.toI32(b), ty: I32 }];
    return [a, b]; // both abstract int
  }
  unifyVec(a: EmitRes, b: EmitRes): [EmitRes, EmitRes] {
    if (isVec(a.ty) && isVec(b.ty)) {
      if (a.ty.n !== b.ty.n) fail(`vector size mismatch ${tyStr(a.ty)} vs ${tyStr(b.ty)}`);
      if (a.ty.e === b.ty.e) return [a, b];
      const t = vec(a.ty.n, 'f32');
      return [{ c: this.convTo(a, t), ty: t }, { c: this.convTo(b, t), ty: t }];
    }
    if (isVec(a.ty)) return [a, { c: this.convTo(b, a.ty), ty: a.ty }];
    if (isVec(b.ty)) return [{ c: this.convTo(a, b.ty), ty: b.ty }, b];
    return [a, b];
  }

  // ---- expressions ----
  expr(e: Expr, sc: Scope): EmitRes {
    switch (e.t) {
      case 'num': return this.num(e);
      case 'str': return fail('string literal not supported');
      case 'id': return this.ident(e.name, sc);
      case 'call': return this.call(e, sc);
      case 'index': {
        const o = this.expr(e.obj, sc);
        const i = this.expr(e.idx, sc);
        let ot = o.ty, oc = o.c;
        if (ot.k === 'ptr') { if (!ot.e) fail('index of untyped pointer'); ot = ot.e; oc = `(*${o.c})`; }
        if (ot.k === 'arr') return { c: `${oc}[${this.toI32(i)}]`, ty: ot.e, lv: true };
        if (isVec(ot)) return { c: `${oc}[${this.toI32(i)}]`, ty: ot.e === 'f32' ? F32 : ot.e === 'i32' ? I32 : U32, lv: true };
        return fail(`index of non-array ${tyStr(o.ty)}`);
      }
      case 'member': {
        if (e.obj.t === 'id' && e.obj.name === 'varpar') return this.magic('__' + e.name);
        if (e.obj.t === 'id' && e.obj.name === 'xform') {
          const i = 'abcdef'.indexOf(e.name);
          if (i < 0 || e.name.length !== 1) fail(`unsupported xform field: ${e.name}`);
          this.flags.add('affine');
          return { c: `\${A(${i})}`, ty: F32 };
        }
        const o = this.expr(e.obj, sc);
        let ot = o.ty, oc = o.c;
        if (ot.k === 'ptr') { if (!ot.e) fail('member of untyped pointer'); ot = ot.e; oc = `(*${o.c})`; }
        if (isVec(ot)) {
          if (!/^[xyzw]{1,4}$/.test(e.name) && !/^[rgba]{1,4}$/.test(e.name)) fail(`bad vector member .${e.name}`);
          const et: Ty = ot.e === 'f32' ? F32 : ot.e === 'i32' ? I32 : U32;
          const ty: Ty = e.name.length === 1 ? et : vec(e.name.length as 2 | 3 | 4, ot.e);
          return { c: `${oc}.${e.name}`, ty, lv: e.name.length === 1 };
        }
        if (ot.k === 'struct') {
          const sd = this.prog.structs.get(ot.name);
          if (!sd) fail(`unknown struct ${ot.name}`);
          const f = sd.fields.find((x) => x.name === e.name);
          if (!f) fail(`struct ${ot.name} has no field ${e.name}`);
          const fty = f.dims.length ? f.dims.reduceRight<Ty>((acc, n) => ({ k: 'arr', e: acc, n }), f.ty) : f.ty;
          return { c: `${oc}.${this.wname(e.name)}`, ty: fty, lv: true };
        }
        return fail(`member access on ${tyStr(o.ty)}`);
      }
      case 'unary': {
        if (e.op === '&') {
          const x = this.expr(e.e, sc);
          if (x.ty.k === 'ptr') return x;
          return { c: `&(${x.c})`, ty: { k: 'ptr', e: x.ty } };
        }
        if (e.op === '*') {
          const x = this.expr(e.e, sc);
          if (x.ty.k !== 'ptr' || !x.ty.e) fail('deref of non-pointer');
          return { c: `(*${x.c})`, ty: x.ty.e, lv: true };
        }
        if (e.op === '++' || e.op === '--') fail('++/-- inside expression not supported');
        const x = this.expr(e.e, sc);
        if (e.op === '!') return { c: `!${this.paren(this.toBool(x))}`, ty: BOOL };
        if (e.op === '~') return { c: `~${this.paren(this.toI32(x))}`, ty: I32 };
        if (e.op === '+') return x;
        // '-'
        if (x.ty.k === 'bool') return { c: `-${this.toI32(x)}`, ty: I32 };
        if (x.ty.k === 'u32') return { c: `-i32(${x.c})`, ty: I32 };
        return { c: `-${this.paren(x.c)}`, ty: x.ty };
      }
      case 'post': return fail('++/-- inside expression not supported');
      case 'binary': return this.binary(e, sc);
      case 'assign': return fail('assignment inside expression not supported');
      case 'cond': {
        const c = this.expr(e.c, sc);
        const a = this.expr(e.a, sc), b = this.expr(e.b, sc);
        const [x, y] = this.unify(a, b);
        return { c: `select(${y.c}, ${x.c}, ${this.toBool(c)})`, ty: concrete(x.ty) };
      }
      case 'cast': {
        const x = this.expr(e.e, sc);
        if (e.ty.k === 'ptr') return { c: x.c, ty: e.ty };
        return { c: this.convTo(x, e.ty), ty: concrete(e.ty) };
      }
      case 'sizeof': {
        const sizeOf = (t: Ty): number => t.k === 'arr' ? t.n * sizeOf(t.e) : isVec(t) ? t.n * 4 : 4;
        const t = e.ty ?? this.expr(e.e!, sc).ty;
        return { c: String(sizeOf(t)), ty: AINT };
      }
      case 'init': return fail('brace initializer only allowed in declarations');
      case 'comma': return fail('comma expression not supported');
    }
  }
  paren(c: string): string { return /^[A-Za-z_][\w.]*$|^\(.*\)$|^[0-9.]+(e[-+]?\d+)?$/.test(c) && balanced(c) ? c : `(${c})`; }
  num(e: Expr & { t: 'num' }): EmitRes {
    if (!e.isFloat) return { c: e.s, ty: AINT };
    let v = e.v;
    if (v !== 0 && Math.abs(v) < 1e-37) v = Math.sign(v) * 1e-37;
    if (Math.abs(v) > 3.0e38) v = Math.sign(v) * 3.0e38;
    let s = String(v);
    if (!/[.e]/.test(s)) s += '.0';
    if (/^-?\d+e/.test(s)) s = s.replace(/^(-?\d+)e/, '$1.0e');
    return { c: s, ty: AFLOAT };
  }
  magic(name: string): EmitRes {
    const b = this.env.bindings.get(name) ?? this.env.resolveMagic?.(name) ?? null;
    if (!b) fail(`unknown identifier ${name}`);
    if (b.flag) this.flags.add(b.flag);
    if (b.decl) this.declsUsed.add(b.decl);
    return { c: b.code, ty: b.ty, lv: b.lvalue };
  }
  ident(name: string, sc: Scope): EmitRes {
    const s = sc.lookup(name);
    if (s) {
      if (s.isParam && this.curFn?.assignsParams.has(name)) return { c: s.wname, ty: s.ty, lv: true };
      return { c: s.wname, ty: s.ty, lv: !s.isParam || s.ty.k === 'ptr' };
    }
    const g = this.prog.globals.get(name);
    if (g) { (this.fnDeps ?? this.prog.helpersUsed).add('g:' + name); return { c: g.wname, ty: g.ty, lv: true }; }
    const b = this.env.bindings.get(name);
    if (b) { if (b.flag) this.flags.add(b.flag); if (b.decl) this.declsUsed.add(b.decl); return { c: b.code, ty: b.ty, lv: b.lvalue }; }
    if (name.startsWith('__') || name === 'varpar') return this.magic(name);
    switch (name) {
      case 'PI': case 'M_PI': case 'M_PI_F': return { c: 'PI', ty: F32 };
      case 'M_PI_2': case 'M_PI_2_F': return { c: '(PI * 0.5)', ty: F32 };
      case 'M_PI_4': case 'M_PI_4_F': return { c: '(PI * 0.25)', ty: F32 };
      case 'M_1_PI': case 'M_1_PI_F': return { c: '(1.0 / PI)', ty: F32 };
      case 'M_2_PI': case 'M_2_PI_F': return { c: '(2.0 / PI)', ty: F32 };
      case 'M_2PI': return { c: '(2.0 * PI)', ty: F32 };
      case 'M_1_2PI': return { c: '(0.5 / PI)', ty: F32 };
      case 'M_SQRT2': return { c: '1.4142135623730951', ty: AFLOAT };
      case 'M_SQRT1_2': return { c: '0.7071067811865476', ty: AFLOAT };
      case 'M_E': return { c: '2.718281828459045', ty: AFLOAT };
      case 'M_LN2': return { c: '0.6931471805599453', ty: AFLOAT };
      case 'M_LN10': return { c: '2.302585092994046', ty: AFLOAT };
      case 'EPSILON': return { c: '1.0e-9', ty: AFLOAT };
      case 'SMALL_EPSILON': return { c: '1.0e-30', ty: AFLOAT };
      case 'epsilon': return { c: '1.0e-8', ty: AFLOAT };
      case 'FLT_MAX': return { c: '3.0e38', ty: AFLOAT };
      case 'FLT_MIN': return { c: '1.0e-37', ty: AFLOAT };
      case 'FLT_EPSILON': return { c: '1.1920929e-7', ty: AFLOAT };
      case 'INT_MAX': return { c: '2147483647', ty: AINT };
      case 'true': return { c: 'true', ty: BOOL };
      case 'false': return { c: 'false', ty: BOOL };
      case 'NULL': return fail('NULL not supported');
    }
    return fail(`unknown identifier ${name}`);
  }
  binary(e: Expr & { t: 'binary' }, sc: Scope): EmitRes {
    const op = e.op;
    if (op === '&&' || op === '||') {
      const a = this.expr(e.l, sc), b = this.expr(e.r, sc);
      return { c: `(${this.toBool(a)} ${op} ${this.toBool(b)})`, ty: BOOL };
    }
    const a = this.expr(e.l, sc), b = this.expr(e.r, sc);
    if (op === '<<' || op === '>>') {
      const l = isIntish(a.ty) ? a : { c: this.toI32(a), ty: I32 };
      const shift = b.ty.k === 'aint' ? b.c : `u32(${b.c})`;
      return { c: `(${l.c} ${op} ${shift})`, ty: concrete(l.ty) };
    }
    if (op === '&' || op === '|' || op === '^') {
      if (a.ty.k === 'bool' && b.ty.k === 'bool') {
        const wop = op === '^' ? '!=' : op === '&' ? '&&' : '||';
        return { c: `(${a.c} ${wop} ${b.c})`, ty: BOOL };
      }
      const [x, y] = this.unify(isFloatish(a.ty) ? { c: this.toI32(a), ty: I32 } : a, isFloatish(b.ty) ? { c: this.toI32(b), ty: I32 } : b);
      return { c: `(${x.c} ${op} ${y.c})`, ty: concrete(x.ty) };
    }
    if (op === '%') {
      const [x, y] = this.unify(a, b);
      return { c: `(${x.c} % ${y.c})`, ty: concrete(x.ty) };
    }
    if (['==', '!=', '<', '>', '<=', '>='].includes(op)) {
      if (a.ty.k === 'bool' || b.ty.k === 'bool') return { c: `(${this.toBool(a)} ${op} ${this.toBool(b)})`, ty: BOOL };
      const [x, y] = this.unify(a, b);
      if (isVec(x.ty)) return { c: `all(${x.c} ${op} ${y.c})`, ty: BOOL };
      return { c: `(${x.c} ${op} ${y.c})`, ty: BOOL };
    }
    // + - * /
    const [x, y] = this.unify(a, b);
    return { c: `(${x.c} ${op} ${y.c})`, ty: concrete(isAbstract(x.ty) ? x.ty : x.ty) };
  }
  call(e: Expr & { t: 'call' }, sc: Scope): EmitRes {
    const name = e.name;
    // user function?
    const overloads = this.prog.funcs.get(name);
    if (overloads) {
      const args = e.args.map((a) => this.argExpr(a, sc));
      const fn = this.pickOverload(overloads, args);
      (this.fnDeps ?? this.prog.helpersUsed).add('f:' + fn.wname);
      fn.used = true;
      const parts: string[] = [];
      fn.params.forEach((p, i) => {
        const a = args[i];
        if (p.ty.k === 'ptr') {
          // infer pointee from argument
          let pointee: Ty | null = null;
          let code: string;
          if (a.ty.k === 'ptr') { pointee = a.ty.e; code = a.c; if (!pointee && !p.ty.e) fail(`pointer type of ${name} param ${p.name} not yet known`); }
          else if (a.lv) { pointee = a.ty; code = `&(${a.c})`; }
          else fail(`argument ${i} of ${name} must be an lvalue (pointer param)`);
          if (pointee) {
            if (!p.ty.e) { p.ty = { k: 'ptr', e: pointee }; p.ptrInferred = true; this.prog.flags.add('reinfer'); }
            else if (!tyEq(p.ty.e, pointee)) {
              // allow array-of-T where param expects T? (C decays) — mismatch otherwise
              fail(`pointer type mismatch for param ${p.name} of ${name}: ${tyStr(p.ty.e)} vs ${tyStr(pointee)}`);
            }
          }
          parts.push(code!);
        } else {
          parts.push(this.convTo(a, p.ty));
        }
      });
      return { c: `${fn.wname}(${parts.join(', ')})`, ty: fn.ret };
    }
    const b = BUILTINS[name];
    if (b) {
      const args = e.args.map((a) => this.expr(a, sc));
      const r = b(args, this);
      // helper usage
      for (const h of Object.keys(HELPER_FUNCS)) {
        if (new RegExp(`\\b${h}\\(`).test(r.c) && !args.some((a) => new RegExp(`\\b${h}\\(`).test(a.c))) {
          this.prog.helpersUsed.add('h:' + h);
          for (const d of HELPER_DEPS[h] ?? []) this.prog.helpersUsed.add('h:' + d);
        }
      }
      return r;
    }
    return fail(`unknown function ${name}`);
  }
  argExpr(a: Expr, sc: Scope): EmitRes {
    // Arrays decay to pointers; &x yields pointer
    if (a.t === 'unary' && a.op === '&') {
      const x = this.expr(a.e, sc);
      if (x.ty.k === 'ptr') return x;
      return { c: x.c, ty: x.ty, lv: true };
    }
    return this.expr(a, sc);
  }
  pickOverload(ov: FnSig[], args: EmitRes[]): FnSig {
    const cands = ov.filter((f) => f.params.length === args.length);
    if (!cands.length) fail(`no overload of ${ov[0].name} takes ${args.length} args`);
    if (cands.length === 1) return cands[0];
    // score by exact type matches
    let best: FnSig | null = null, bestScore = -1;
    for (const f of cands) {
      let score = 0, ok = true;
      f.params.forEach((p, i) => {
        const a = args[i];
        if (p.ty.k === 'ptr') { if (a.ty.k === 'ptr' || a.lv) score += 1; else ok = false; return; }
        if (isVec(p.ty)) { if (isVec(a.ty) && a.ty.n === p.ty.n) score += 3; else ok = false; return; }
        if (isVec(a.ty)) { ok = false; return; }
        if (tyEq(concrete(a.ty), concrete(p.ty))) score += 2; else if (isNum(a.ty) && isNum(p.ty)) score += 1; else if (a.ty.k === 'struct' && tyEq(a.ty, p.ty)) score += 3; else ok = false;
      });
      if (ok && score > bestScore) { best = f; bestScore = score; }
    }
    if (!best) fail(`no matching overload of ${ov[0].name}`);
    return best;
  }

  // ---- statements ----
  stmts(list: Stmt[], sc: Scope, ind: string): string {
    return list.map((s) => this.stmt(s, sc, ind)).filter((s) => s.length).join('');
  }
  declTy(base: Ty, d: { ptr: boolean; dims: number[]; init?: Expr | null }): Ty {
    let ty: Ty = base;
    for (let i = d.dims.length - 1; i >= 0; i--) {
      let n = d.dims[i];
      if (n < 0) {
        // unsized outer dimension: take it from a brace initializer
        if (i === 0 && d.init && d.init.t === 'init') n = d.init.items.length;
        else fail('array with unknown size');
      }
      ty = { k: 'arr', e: ty, n };
    }
    if (d.ptr) ty = { k: 'ptr', e: ty };
    return ty;
  }
  initExpr(init: Expr, ty: Ty, sc: Scope): string {
    if (init.t === 'init') {
      if (ty.k === 'arr') {
        const items = init.items.map((it) => this.initExpr(it, (ty as { k: 'arr'; e: Ty }).e, sc));
        while (items.length < ty.n) items.push(this.zeroOf(ty.e));
        return `${tyStr(ty)}(${items.join(', ')})`;
      }
      if (isVec(ty)) {
        const items = init.items.map((it) => this.convTo(this.expr(it, sc), ty.e === 'f32' ? F32 : ty.e === 'i32' ? I32 : U32));
        return `${tyStr(ty)}(${items.join(', ')})`;
      }
      if (ty.k === 'struct') {
        const sd = this.prog.structs.get(ty.name)!;
        const items = sd.fields.map((f, i) => init.items[i] ? this.initExpr(init.items[i], f.ty, sc) : this.zeroOf(f.ty));
        return `${ty.name}(${items.join(', ')})`;
      }
      if (init.items.length === 1) return this.initExpr(init.items[0], ty, sc);
      return fail('bad initializer');
    }
    // arrays cannot be assigned from another array-typed value except copy
    return this.convTo(this.expr(init, sc), ty);
  }
  zeroOf(ty: Ty): string {
    switch (ty.k) {
      case 'f32': case 'afloat': return '0.0';
      case 'i32': case 'aint': return '0';
      case 'u32': return '0u';
      case 'bool': return 'false';
      case 'vec': return `${tyStr(ty)}()`;
      case 'arr': return `${tyStr(ty)}()`;
      case 'struct': return `${ty.name}()`;
      default: return fail(`no zero for ${tyStr(ty)}`);
    }
  }
  declare(name: string, ty: Ty, sc: Scope, isParam = false): string {
    let w = this.wname(name);
    // avoid clashing with an outer name that would break WGSL (shadowing is fine, redeclare in same scope is not)
    if (sc.syms.has(name)) fail(`redeclaration of ${name}`);
    sc.syms.set(name, { ty, wname: w, isParam });
    return w;
  }
  stmt(s: Stmt, sc: Scope, ind: string): string {
    switch (s.t) {
      case 'empty': return '';
      case 'block': {
        const inner = new Scope(sc, sc.fn);
        return `${ind}{\n${this.stmts(s.stmts, inner, ind + '  ')}${ind}}\n`;
      }
      case 'decl': {
        let out = '';
        for (const d of s.decls) {
          const ty = this.declTy(s.ty, d);
          if (ty.k === 'ptr') {
            // pointer local: `float *p = &x;` → alias; support only with initializer to an lvalue
            if (!d.init) fail('uninitialized pointer local');
            const x = this.argExpr(d.init, sc);
            const w = this.declare(d.name, { k: 'ptr', e: x.ty.k === 'ptr' ? x.ty.e : x.ty }, sc);
            out += `${ind}let ${w} = ${x.ty.k === 'ptr' ? x.c : `&(${x.c})`};\n`;
            continue;
          }
          const initCode = d.init ? this.initExpr(d.init, ty, sc) : null;
          const w = this.declare(d.name, ty, sc);
          out += initCode !== null ? `${ind}var ${w}: ${tyStr(ty)} = ${initCode};\n` : `${ind}var ${w}: ${tyStr(ty)};\n`;
        }
        return out;
      }
      case 'expr': return this.exprStmt(s.e, sc, ind);
      case 'if': {
        const c = this.expr(s.c, sc);
        let out = `${ind}if (${this.toBool(c)}) ${this.blockOf(s.a, sc, ind)}`;
        if (s.b) {
          if (s.b.t === 'if') out = out.trimEnd() + ` else ${this.stmt(s.b, sc, ind).trimStart()}`;
          else out = out.trimEnd() + ` else ${this.blockOf(s.b, sc, ind)}`;
        }
        return out;
      }
      case 'while': {
        const c = this.expr(s.c, sc);
        return `${ind}while (${this.toBool(c)}) ${this.blockOf(s.body, sc, ind)}`;
      }
      case 'do': {
        const inner = new Scope(sc, sc.fn);
        const body = this.stmts(s.body.t === 'block' ? s.body.stmts : [s.body], inner, ind + '  ');
        const c = this.expr(s.c, inner);
        return `${ind}loop {\n${body}${ind}  if (!${this.paren(this.toBool(c))}) { break; }\n${ind}}\n`;
      }
      case 'for': {
        const inner = new Scope(sc, sc.fn);
        let init = '';
        if (s.init) {
          if (s.init.t === 'decl') {
            if (s.init.decls.length !== 1) {
              // hoist multi-decl into an enclosing block
              const pre = this.stmt(s.init, inner, ind + '  ');
              const rest = this.stmt({ ...s, init: null }, inner, ind + '  ');
              return `${ind}{\n${pre}${rest}${ind}}\n`;
            }
            init = this.stmt(s.init, inner, '').trim().replace(/;$/, '');
          } else {
            init = this.exprStmt(s.init.e, inner, '').trim().replace(/;$/, '');
            if (init.includes('\n')) fail('complex for-init');
          }
        }
        const c = s.c ? this.toBool(this.expr(s.c, inner)) : '';
        let u = '';
        if (s.u) { u = this.exprStmt(s.u, inner, '').trim().replace(/;$/, ''); if (u.includes('\n')) fail('complex for-update'); }
        return `${ind}for (${init}; ${c}; ${u}) ${this.blockOf(s.body, inner, ind)}`;
      }
      case 'return': {
        if (!s.e) return `${ind}return;\n`;
        const fn = sc.fn;
        if (!fn) fail('return outside function');
        const x = this.expr(s.e, sc);
        return `${ind}return ${this.convTo(x, fn.ret)};\n`;
      }
      case 'break': return `${ind}break;\n`;
      case 'continue': return `${ind}continue;\n`;
      case 'switch': {
        const sel = this.expr(s.e, sc);
        let out = `${ind}switch ${this.toI32(sel)} {\n`;
        let hasDefault = false;
        // a case body that ends in break/return/continue — possibly inside a trailing block
        const jumpEnd = (st: Stmt | undefined): 'break' | 'other' | null => {
          if (!st) return null;
          if (st.t === 'break') return 'break';
          if (st.t === 'return' || st.t === 'continue') return 'other';
          if (st.t === 'block') return jumpEnd(st.stmts[st.stmts.length - 1]);
          return null;
        };
        s.cases.forEach((cs, idx) => {
          const last = cs.body[cs.body.length - 1];
          const je = jumpEnd(last);
          const terminated = je !== null;
          if (!terminated && idx !== s.cases.length - 1 && cs.body.length) fail('switch fallthrough not supported');
          const labels = cs.vals.map((v) => v === null ? (hasDefault = true, 'default') : this.toI32(this.expr(v, sc)));
          // drop a direct trailing break (WGSL cases don't fall through); a break nested in a block is harmless
          const body = je === 'break' && last.t === 'break' ? cs.body.slice(0, -1) : cs.body;
          const inner = new Scope(sc, sc.fn);
          out += `${ind}  case ${labels.join(', ')}: {\n${this.stmts(body, inner, ind + '    ')}${ind}  }\n`;
        });
        if (!hasDefault) out += `${ind}  default: {}\n`;
        return out + `${ind}}\n`;
      }
    }
  }
  blockOf(s: Stmt, sc: Scope, ind: string): string {
    if (s.t === 'block') return this.stmt(s, sc, ind).trimStart();
    const inner = new Scope(sc, sc.fn);
    return `{\n${this.stmt(s, inner, ind + '  ')}${ind}}\n`;
  }
  exprStmt(e: Expr, sc: Scope, ind: string): string {
    if (e.t === 'comma') return this.exprStmt(e.l, sc, ind) + this.exprStmt(e.r, sc, ind);
    if (e.t === 'assign') {
      // chained assignment a = b = c
      if (e.r.t === 'assign') {
        const first = this.exprStmt(e.r, sc, ind);
        return first + this.exprStmt({ t: 'assign', op: e.op, l: e.l, r: e.r.l }, sc, ind);
      }
      const l = this.expr(e.l, sc);
      if (!l.lv) fail(`assignment to non-lvalue: ${l.c}`);
      const r = this.expr(e.r, sc);
      if (e.op === '=') {
        if (l.ty.k === 'arr') fail('array assignment not supported');
        return `${ind}${l.c} = ${this.convTo(r, l.ty)};\n`;
      }
      const bop = e.op.slice(0, -1);
      if (bop === '<<' || bop === '>>') return `${ind}${l.c} ${e.op} ${r.ty.k === 'aint' ? r.c : `u32(${r.c})`};\n`;
      if (isVec(l.ty)) return `${ind}${l.c} ${e.op} ${this.convTo(r, l.ty)};\n`;
      if (l.ty.k === 'bool') fail('compound assign on bool');
      // compute as binary then assign (handles int/float promotion like C: l = (T)(l op r))
      const [x, y] = this.unify(l, r);
      if (tyEq(concrete(x.ty), l.ty)) return `${ind}${l.c} ${e.op} ${y.c};\n`;
      return `${ind}${l.c} = ${this.convTo({ c: `(${x.c} ${bop} ${y.c})`, ty: concrete(x.ty) }, l.ty)};\n`;
    }
    if ((e.t === 'post' || e.t === 'unary') && (e.op === '++' || e.op === '--')) {
      const l = this.expr(e.e, sc);
      if (!l.lv) fail('++/-- on non-lvalue');
      if (l.ty.k === 'f32') return `${ind}${l.c} ${e.op[0]}= 1.0;\n`;
      return `${ind}${l.c}${e.op};\n`;
    }
    if (e.t === 'call' && (e.name === 'sincosf' || e.name === 'sincos')) {
      const a = this.expr(e.args[0], sc);
      const s = this.expr(e.args[1].t === 'unary' && e.args[1].op === '&' ? e.args[1].e : e.args[1], sc);
      const c = this.expr(e.args[2].t === 'unary' && e.args[2].op === '&' ? e.args[2].e : e.args[2], sc);
      const ac = this.toF32(a);
      const sl = s.ty.k === 'ptr' ? `(*${s.c})` : s.c, cl = c.ty.k === 'ptr' ? `(*${c.c})` : c.c;
      return `${ind}${sl} = sin(${ac});\n${ind}${cl} = cos(${ac});\n`;
    }
    if (e.t === 'call') {
      const r = this.expr(e, sc);
      if (r.ty.k === 'void') return `${ind}${r.c};\n`;
      return `${ind}_ = ${r.c};\n`;
    }
    const r = this.expr(e, sc);
    return `${ind}_ = ${r.c};\n`;
  }
}

function balanced(s: string): boolean {
  let d = 0;
  for (let i = 0; i < s.length; i++) { if (s[i] === '(') d++; else if (s[i] === ')') { d--; if (d === 0 && i !== s.length - 1) return false; } }
  return d === 0;
}

// analysis: which params are assigned in a function body
function collectAssignedParams(fn: FuncDef): Set<string> {
  const names = new Set(fn.params.filter((p) => !p.ptr && !p.dims).map((p) => p.name));
  const out = new Set<string>();
  const root = (e: Expr): string | null => e.t === 'id' ? e.name : e.t === 'index' || e.t === 'member' ? root(e.obj) : null;
  const visitE = (e: Expr | null): void => {
    if (!e) return;
    switch (e.t) {
      case 'assign': { const r = root(e.l); if (r && names.has(r)) out.add(r); visitE(e.l); visitE(e.r); break; }
      case 'post': { const r = root(e.e); if (r && names.has(r)) out.add(r); break; }
      case 'unary': { if (e.op === '++' || e.op === '--' || e.op === '&') { const r = root(e.e); if (r && names.has(r)) out.add(r); } visitE(e.e); break; }
      case 'binary': visitE(e.l); visitE(e.r); break;
      case 'call': e.args.forEach(visitE); break;
      case 'index': visitE(e.obj); visitE(e.idx); break;
      case 'member': visitE(e.obj); break;
      case 'cond': visitE(e.c); visitE(e.a); visitE(e.b); break;
      case 'cast': visitE(e.e); break;
      case 'comma': visitE(e.l); visitE(e.r); break;
      case 'init': e.items.forEach(visitE); break;
    }
  };
  const visitS = (s: Stmt | null): void => {
    if (!s) return;
    switch (s.t) {
      case 'decl': s.decls.forEach((d) => visitE(d.init)); break;
      case 'expr': visitE(s.e); break;
      case 'if': visitE(s.c); visitS(s.a); visitS(s.b); break;
      case 'for': visitS(s.init); visitE(s.c); visitE(s.u); visitS(s.body); break;
      case 'while': visitE(s.c); visitS(s.body); break;
      case 'do': visitS(s.body); visitE(s.c); break;
      case 'block': s.stmts.forEach(visitS); break;
      case 'return': visitE(s.e); break;
      case 'switch': visitE(s.e); s.cases.forEach((c) => c.body.forEach(visitS)); break;
    }
  };
  visitS(fn.body);
  return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface LibraryInput { name: string; source: string }

export interface TranspileResult {
  /** WGSL statements for the snippet body (no wrapping braces). */
  code: string;
  /** Module-scope WGSL (structs, consts, functions) the snippet depends on, in dependency order. */
  functions: string;
  /** Names of module-scope items (for dedupe across variations). */
  functionNames: string[];
  flags: string[];
}

/**
 * Transpile a JWildfire GPU snippet. `libraries` are additional C sources
 * (the variation's getGPUFunctions plus the shared kernel helper library);
 * only the items the snippet transitively uses are emitted.
 */
export function transpileSnippet(snippet: string, libraries: LibraryInput[], env: Env): TranspileResult {
  const macros = new Map<string, Macro>();
  // Kernel-provided macros
  macros.set('ADD_EPSILON', { params: null, body: lex('+ 1.0e-20').slice(0, -1) });
  const structs = new Set<string>();
  const prog: Program = { structs: new Map(), funcs: new Map(), globals: new Map(), helpersUsed: new Set(), flags: new Set(), macros };
  const flags = new Set<string>();
  const em = new Emitter(prog, env, flags);

  // Parse libraries
  const topLevels: TopLevel[] = [];
  for (const lib of libraries) {
    const src = lib.source.replace(/%d/g, '0');
    const toks = preprocess(lex(src), macros);
    const p = new Parser(toks, structs);
    for (const tl of p.parseTop()) topLevels.push(tl);
  }
  // Register structs, functions, globals
  const usedFnNames = new Set<string>();
  for (const tl of topLevels) {
    if (tl.t === 'struct') prog.structs.set(tl.name, tl);
    else if (tl.t === 'func') {
      const list = prog.funcs.get(tl.name) ?? [];
      let wname = em.wname(tl.name);
      if (list.length) wname = `${wname}_ov${list.length}`;
      if (usedFnNames.has(wname)) wname = `${wname}_${usedFnNames.size}`;
      usedFnNames.add(wname);
      const params = tl.params.map((p) => {
        let ty: Ty = p.ty;
        if (p.dims) {
          // array param: known dims → ptr to array; unknown leading dim → infer
          const dims = p.dims;
          if (dims.every((d) => d > 0)) { for (let i = dims.length - 1; i >= 0; i--) ty = { k: 'arr', e: ty, n: dims[i] }; ty = { k: 'ptr', e: ty }; }
          else if (dims.length === 1) ty = { k: 'ptr', e: null };
          else { for (let i = dims.length - 1; i >= 1; i--) ty = { k: 'arr', e: ty, n: dims[i] }; ty = { k: 'ptr', e: null }; }
        } else if (p.ptr) ty = { k: 'ptr', e: null };
        return { ty, name: p.name, ptrInferred: false };
      });
      list.push({ name: tl.name, wname, ret: tl.ret, params, def: tl, used: false, deps: new Set(), assignsParams: collectAssignedParams(tl) });
      prog.funcs.set(tl.name, list);
    } else {
      // globals (constants / private state)
      for (const d of tl.decls) {
        const ty = em.declTy(tl.ty, d);
        prog.globals.set(d.name, { wname: em.wname(d.name), ty, code: '', used: false, deps: new Set() });
      }
    }
  }
  // Emit globals now (they may reference other globals/macros only)
  const gscope = new Scope(null, null);
  for (const tl of topLevels) {
    if (tl.t !== 'gdecl') continue;
    for (const d of tl.decls) {
      const g = prog.globals.get(d.name)!;
      em.fnDeps = g.deps;
      const isConst = d.init !== null && !hasMutableGlobalUse(topLevels, d.name);
      if (d.init) {
        const init = em.initExpr(d.init, g.ty, gscope);
        g.code = isConst ? `const ${g.wname}: ${tyStr(g.ty)} = ${init};` : `var<private> ${g.wname}: ${tyStr(g.ty)} = ${init};`;
      } else {
        g.code = `var<private> ${g.wname}: ${tyStr(g.ty)};`;
        flags.add('state');
      }
      if (!isConst && d.init) flags.add('state');
      em.fnDeps = null;
    }
  }

  // Parse the snippet as a function body
  const snipToks = preprocess(lex('{' + snippet.replace(/%d/g, '0') + '}'), macros);
  const sp = new Parser(snipToks, structs);
  const body = sp.parseBlock();
  if (sp.peek().k !== 'eof') sp.err('trailing tokens after snippet');

  // Emit: iterate to a fixpoint for pointer-param inference
  let code = '';
  const fnBodies = new Map<string, string>();
  for (let pass = 0; pass < 16; pass++) {
    prog.flags.delete('reinfer');
    prog.helpersUsed.clear();
    flags.clear();
    em.declsUsed.clear();
    for (const list of prog.funcs.values()) for (const f of list) { f.used = false; f.deps.clear(); }
    // snippet
    const sc = new Scope(null, null);
    em.curFn = null;
    em.fnDeps = null;
    code = em.stmts(body.stmts, sc, '  ');
    if (em.declsUsed.size) code = [...em.declsUsed].map((d) => `  ${d}\n`).join('') + code;
    // functions (all, so signatures/pointers get inferred even for transitively used ones)
    fnBodies.clear();
    for (const list of prog.funcs.values()) {
      for (const f of list) {
        if (!f.def) continue;
        em.curFn = f;
        em.fnDeps = f.deps;
        try {
          fnBodies.set(f.wname, emitFunction(em, f));
        } catch (err) {
          if (!(err instanceof TranspileError)) throw err;
          fnBodies.set(f.wname, `/*ERROR:${(err as Error).message}*/`);
        }
        em.fnDeps = null;
        em.curFn = null;
      }
    }
    if (!prog.flags.has('reinfer')) break;
  }

  // Collect transitive deps from snippet
  const need = new Set<string>();
  const order: string[] = [];
  const visit = (key: string) => {
    if (need.has(key)) return;
    need.add(key);
    if (key.startsWith('f:')) {
      const f = [...prog.funcs.values()].flat().find((x) => x.wname === key.slice(2))!;
      for (const d of f.deps) visit(d);
      const b = fnBodies.get(f.wname) ?? '';
      if (b.startsWith('/*ERROR:')) fail(`in function ${f.name}: ${b.slice(8, -2)}`);
      order.push(key);
    } else if (key.startsWith('g:')) {
      const g = prog.globals.get(key.slice(2))!;
      for (const d of g.deps) visit(d);
      order.push(key);
    } else order.push(key);
  };
  for (const k of prog.helpersUsed) visit(k);
  // helper functions used inside emitted functions
  for (const key of [...need]) {
    if (!key.startsWith('f:')) continue;
    const b = fnBodies.get(key.slice(2)) ?? '';
    for (const h of Object.keys(HELPER_FUNCS)) if (new RegExp(`\\b${h}\\(`).test(b)) { visit('h:' + h); for (const d of HELPER_DEPS[h] ?? []) visit('h:' + d); }
  }
  // structs used by any emitted function/global (emit all referenced structs first)
  const structsOut: string[] = [];
  const emittedText = order.map((k) => k.startsWith('f:') ? fnBodies.get(k.slice(2)) ?? '' : k.startsWith('g:') ? prog.globals.get(k.slice(2))!.code : '').join('\n') + code;
  for (const [name, sd] of prog.structs) {
    if (new RegExp(`\\b${name}\\b`).test(emittedText)) {
      structsOut.push(`struct ${name} {\n${sd.fields.map((f) => `  ${em.wname(f.name)}: ${tyStr(f.dims.length ? f.dims.reduceRight<Ty>((acc, n) => ({ k: 'arr', e: acc, n }), f.ty) : f.ty)},`).join('\n')}\n}`);
    }
  }
  const parts: string[] = [...structsOut];
  const names: string[] = structsOut.map((s) => /struct (\w+)/.exec(s)![1]);
  // helpers first (they have no deps except each other), then globals/functions in dep order
  for (const k of order) {
    if (k.startsWith('h:')) { parts.push(HELPER_FUNCS[k.slice(2)]); names.push(k.slice(2)); }
  }
  for (const k of order) {
    if (k.startsWith('g:')) { const g = prog.globals.get(k.slice(2))!; parts.push(g.code); names.push(g.wname); }
    else if (k.startsWith('f:')) { parts.push(fnBodies.get(k.slice(2))!); names.push(k.slice(2)); }
  }
  return { code, functions: parts.join('\n\n'), functionNames: names, flags: [...flags] };
}

function hasMutableGlobalUse(tops: TopLevel[], name: string): boolean {
  // crude: if any function body assigns to the global, it's mutable
  const root = (e: Expr): string | null => e.t === 'id' ? e.name : e.t === 'index' || e.t === 'member' ? root(e.obj) : null;
  let found = false;
  const visitE = (e: Expr | null): void => {
    if (!e || found) return;
    switch (e.t) {
      case 'assign': if (root(e.l) === name) found = true; visitE(e.l); visitE(e.r); break;
      case 'post': if (root(e.e) === name) found = true; break;
      case 'unary': if ((e.op === '++' || e.op === '--' || e.op === '&') && root(e.e) === name) found = true; visitE(e.e); break;
      case 'binary': visitE(e.l); visitE(e.r); break;
      case 'call': e.args.forEach(visitE); break;
      case 'index': visitE(e.obj); visitE(e.idx); break;
      case 'member': visitE(e.obj); break;
      case 'cond': visitE(e.c); visitE(e.a); visitE(e.b); break;
      case 'cast': visitE(e.e); break;
      case 'comma': visitE(e.l); visitE(e.r); break;
    }
  };
  const visitS = (s: Stmt | null): void => {
    if (!s || found) return;
    switch (s.t) {
      case 'decl': s.decls.forEach((d) => visitE(d.init)); break;
      case 'expr': visitE(s.e); break;
      case 'if': visitE(s.c); visitS(s.a); visitS(s.b); break;
      case 'for': visitS(s.init); visitE(s.c); visitE(s.u); visitS(s.body); break;
      case 'while': visitE(s.c); visitS(s.body); break;
      case 'do': visitS(s.body); visitE(s.c); break;
      case 'block': s.stmts.forEach(visitS); break;
      case 'return': visitE(s.e); break;
      case 'switch': visitE(s.e); s.cases.forEach((c) => c.body.forEach(visitS)); break;
    }
  };
  for (const t of tops) if (t.t === 'func') visitS(t.body);
  return found;
}

function emitFunction(em: Emitter, f: FnSig): string {
  const def = f.def!;
  const sc = new Scope(null, f);
  const params: string[] = [];
  let prologue = '';
  def.params.forEach((p, i) => {
    const sig = f.params[i];
    if (sig.ty.k === 'ptr' && !sig.ty.e) {
      // pointee not inferred (yet) — uses of it will fail this pass; a later pass may resolve it
      params.push(`${em.wname(p.name)}: ptr<function, ?>`);
      sc.syms.set(p.name, { ty: { k: 'ptr', e: null }, wname: em.wname(p.name), isParam: true });
      return;
    }
    if (f.assignsParams.has(p.name)) {
      const w = em.wname(p.name);
      params.push(`${w}_in: ${tyStr(sig.ty)}`);
      prologue += `  var ${w}: ${tyStr(sig.ty)} = ${w}_in;\n`;
      sc.syms.set(p.name, { ty: sig.ty, wname: w, isParam: true });
    } else {
      const w = em.wname(p.name);
      params.push(`${w}: ${tyStr(sig.ty)}`);
      sc.syms.set(p.name, { ty: sig.ty, wname: w, isParam: true });
    }
  });
  const body = em.stmts(def.body.stmts, sc, '  ');
  const ret = f.ret.k === 'void' ? '' : ` -> ${tyStr(f.ret)}`;
  return `fn ${f.wname}(${params.join(', ')})${ret} {\n${prologue}${body}}`;
}
