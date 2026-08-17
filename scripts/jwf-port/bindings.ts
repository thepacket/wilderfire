// The snippet environment shared by gen.ts and the transpiler regression tests:
// which identifiers a JWildfire GPU snippet may use (magic __x/__y/…, its own
// weight/params, per-thread "extra" state) and what WGSL they map to.
import type { Binding, Env, Ty } from './cwgsl.ts';

export interface SnippetVar {
  name: string;
  params: { name: string }[];
  gpuCode?: string;
  /** "name" (f32) or "name:float2|float3|int" (typed per-thread state from java2cu) */
  extraParams?: string[];
}

export function bindingsFor(v: SnippetVar): Env {
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

