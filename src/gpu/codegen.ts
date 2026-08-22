// WGSL code generation: each flame compiles to a dedicated compute shader with
// its variation set inlined. Numeric parameters (affines, weights, variation
// params, layer densities) live in a storage buffer (`xd`) so slider tweaks
// never recompile — only structural edits (adding/removing layers, xforms or
// variations, toggling layer visibility) do.
//
// Layers: walker threads are partitioned equally across visible layers (JWildfire iterates every
// layer equally often); the layer weight multiplies the plotted colour (xd header slots 8+li, so
// weight tweaks stay numeric).
// Each layer gets its own iteration function, CDF rows (xaos), xform blocks,
// and a 256-entry slice of the palette buffer.
//
// xd layout:
//   [0..7]              cumulative per-layer thread cutoffs (f32 thread index)
//   [8..15]             reserved
//   per visible layer:  CDF table ((n+1) rows × 16) then xform blocks
//   per-xform block:    6 affine, 6 post, color, colorSpeed, opacity, pad,
//                       6 yz, 6 zx, 6 yzPost, 6 zxPost (JWildfire 3D affines),
//                       8 colour modifiers, 16 weighting field, material + materialSpeed (solid rendering),
//                       6 spare, then per variation: weight + params

import type { Flame, XForm, VarInstance, Layer, PostSymmetry } from '../core/flame';
import { visibleLayers, usesMaterials } from '../core/flame';
import { VARIATIONS, DFLT_SUBFLAME_XML } from '../core/variations';
import { importFlameText } from '../core/flameXML';
import { WFIELD_WGSL } from './wfield.wgsl';
import { SOLID_KERNEL_WGSL, SOLID_PAY_WORDS } from './solid.wgsl';
import { meshKeyFor, meshLayout } from '../core/meshes';

/** JWildfire pre/post-priority variation classes whose Java transform() ends with the preserve-z clause
 *  (`pVarTP.z += pAmount * pAffineTP.z`) like a 2D normal-priority one — grep of isPreserveZCoordinate over classes with a
 *  non-zero getPriority(): the post crop family, post_trig, pre_recip, pre/post_c_symmetry, pre/post_c_var, ringtile. */
const PREPOST_PRESERVE_Z = new Set(['post_circlecrop', 'post_crop_box', 'post_crop_polygon', 'post_crop_cross', 'post_crop_stars', 'post_crop_triangle',
  'post_crop_trapezoid', 'post_crop_rhombus', 'post_crop_x', 'post_crosscrop', 'post_crop_vesica', 'post_point_crop', 'post_trig', 'post_c_symmetry',
  'post_c_var', 'pre_recip', 'pre_c_symmetry', 'pre_c_var', 'ringtile']);
/** 3D variations (flag `z`) whose Java transform() nevertheless carries the preserve-z clause at normal
 *  priority — the port compiled `isPreserveZCoordinate()` to false, so the engine adds the line for them.
 *  Found by comparing a real flame (`_pdofR`, dc_carpet3D under a 60° pitch) against JWildfire: with
 *  preserve_z on it diverged (corr 0.09), with it off it matched (1.00). */
const Z_PRESERVE_TOO = new Set(['dc_carpet3D', 'whirligig']);
const CDF_ROW = 16;
const HEADER = 72;    // floats per xform block header (6 affine, 6 post, color, colorSpeed, opacity, pad, 24 3D affines, 8 colour modifiers, 16 weighting field, material, materialSpeed, 6 spare)
const XD_HEADER = 16; // floats reserved at the front of xd

export interface CompiledFlame {
  wgsl: string;
  dataSize: number; // float count for the xd buffer
  writeData(flame: Flame, out: Float32Array): void;
  /** The kernel carries per-point colour modifiers (JWildfire mod_gamma/…): binding 6 `mods` must be bound. */
  usesMods: boolean;
  /** JWildfire solid rendering: the kernel splats into a z-buffer (bindings 7 `zkey` + 8 `zpay`) instead of the histogram. */
  solid: boolean;
  /** The kernel carries a per-point material index (binding 9 `mats`). */
  usesMat: boolean;
  /** A variation samples the shared mesh buffer (binding 12 `mesh`; obj_mesh_primitive_wf). */
  usesMesh: boolean;
}

/** All variation lists of an xform in serialized order: pre, main, post. */
function varLists(x: XForm): [VarInstance[], VarInstance[], VarInstance[]] {
  return [x.preVariations ?? [], x.variations, x.postVariations ?? []];
}

function blockSize(x: XForm): number {
  let n = HEADER;
  for (const list of varLists(x)) {
    for (const v of list) {
      const def = VARIATIONS[v.name];
      if (def) n += 1 + (def.params?.length ?? 0) + (def.extra ?? 0);
    }
  }
  return n;
}

/** Module-scope WGSL (helper fns/consts) required by the variations of a flame, deduped by name. */
function collectFuncs(layersAll: Layer[]): string {
  const seen = new Set<string>();
  const parts: string[] = [];
  for (const ly of layersAll) {
    const xs = [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals];
    for (const x of xs) {
      for (const list of varLists(x)) {
        for (const vi of list) {
          const def = VARIATIONS[vi.name];
          if (!def?.funcs || !def.funcNames) continue;
          const key = def.funcNames.join('|');
          if (seen.has(key)) continue;
          // Item-level dedupe: two variations may share a helper (e.g. sqr, Complex struct)
          const items = splitItems(def.funcs);
          for (const [nm, text] of items) {
            if (seen.has('item:' + nm)) continue;
            seen.add('item:' + nm);
            parts.push(text);
          }
          seen.add(key);
        }
      }
    }
  }
  // JWildfire weighting fields: FastNoise + wfieldValue (deduped item by item against the variation helpers)
  if (layersAll.some((ly) => [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals].some((x) => x.wfield && WFIELD_TYPE_ORD[x.wfield.type]))) {
    for (const [nm, text] of splitItems(WFIELD_WGSL)) {
      if (seen.has('item:' + nm)) continue;
      seen.add('item:' + nm);
      parts.push(text);
    }
  }
  return parts.join('\n\n');
}

/** Splits a funcs blob into [name, text] items (struct/fn/const/var<private>). */
function splitItems(blob: string): [string, string][] {
  const out: [string, string][] = [];
  const chunks = blob.split(/\n\n(?=(?:struct|fn|const|var<private>) )/);
  for (const c of chunks) {
    const m = /^(?:struct|fn|const|var<private>) ([A-Za-z_]\w*)/.exec(c.trim());
    if (m) out.push([m[1], c.trim()]);
  }
  return out;
}

function genXformFn(name: string, x: XForm, B: number, palBase: number, pzCond = '(P.flags & 1u) != 0u'): string {
  const A = (i: number) => `xd[${B + i}u]`;
  const [pre, main, post] = varLists(x);
  let off = B + HEADER;
  // JWildfire weighting field (slots 48..63): the noise value wfv scales amounts/params/colour and jitters the output
  const wf = x.wfield && WFIELD_TYPE_ORD[x.wfield.type] ? x.wfield : null;
  const wfParamScale = (vname: string, pname: string): string => {
    if (!wf) return '';
    const k = wf.params.findIndex((pp) => pp.varName === vname && pp.paramName === pname);
    return k >= 0 ? ` * (1.0 + wfv * ${A(61 + k)})` : '';
  };

  // Emits one stage. The point (t), its polar precalcs and the accumulator (v)
  // are all mutable so JWildfire-style pre-priority variations can rewrite the
  // input and post-priority ones can rewrite the output; snippets run in
  // priority order (pre → normal → post), list order within a priority.
  // `z_` is the stage's input z (mutable, JWildfire's __z), `pz_` its z output
  // accumulator (__pz). Only the main stage carries z out; the WilderFire pre/post
  // sum-stages pass z through untouched. With preserve_z, 2D variations at normal
  // priority add w·z (JWildfire's isPreserveZCoordinate behaviour).
  const genStage = (list: VarInstance[], input: string, output: string, initV: string, zIn: string, zOut: string | null): string => {
    // xd layout follows list order (writeData mirrors it), so bind offsets first…
    const bound = list.map((vi) => {
      const def = VARIATIONS[vi.name];
      if (!def) return null;
      const w = wf ? `(xd[${off}u] * wfAmp${wfParamScale(vi.name, 'amount')})` : `xd[${off}u]`;
      // a modulated int param is rounded like JWildfire's Tools.FTOI (the snippet's own (int) cast would truncate)
      const p = (def.params ?? []).map((pd, k) => (wf && wfParamScale(vi.name, pd.name) ? (pd.int ? `round(xd[${off + 1 + k}u]${wfParamScale(vi.name, pd.name)})` : `(xd[${off + 1 + k}u]${wfParamScale(vi.name, pd.name)})`) : `xd[${off + 1 + k}u]`));
      const nP = def.params?.length ?? 0;
      for (let k = 0; k < (def.extra ?? 0); k++) p.push(`xd[${off + 1 + nP + k}u]`); // hidden slots (data hook)
      off += 1 + nP + (def.extra ?? 0);
      const dprio = def.priority ?? 0;
      // JWildfire per-instance priority override (`<var>_fx_priority`):
      //  · a normal variation forced to pre/post — EnforcedPre/PostVariationTransformationStep: the input/output point becomes
      //    p + w·f(p) (its preserve-z clause included, so z grows by w·z too);
      //  · a pre/post-definition variation forced to 0 — EnforcedVariationTransformationStep: transform(tmp = copy of the
      //    output, output): a pre-priority function (writes its "affine" argument = the copy) has NO effect and is dropped,
      //    a post-priority one rewrites the output at its place in the normal list (`forcedZero`);
      //  · anything else keeps the definition.
      const ovr = vi.priority !== undefined && vi.priority !== dprio ? vi.priority : undefined;
      if (ovr === 0 && dprio === -1) return null;
      const forced = ovr !== undefined && dprio === 0 && (ovr === -1 || ovr === 1) ? ovr : 0;
      const forcedZero = ovr === 0 && dprio === 1;
      return { def, name: vi.name, w, p, prio: forced !== 0 ? forced : forcedZero ? 0 : dprio, forced, forcedZero };
    }).filter((b): b is NonNullable<typeof b> => b !== null);
    // …then emit in priority order.
    let snips = '';
    const precalc = '    r2 = max(dot(t, t), 1e-12); r = sqrt(r2); th = atan2(t.x, t.y); ph = atan2(t.y, t.x);\n';
    for (const prio of [-2, -1, 0, 1, 2]) {
      for (const b of bound) {
        // prepost variations: their inverse snippet rewrites the input first (prio -2), the forward one runs last (prio 2)
        if (prio === -2 && b.def.preCode) { snips += '    ' + b.def.preCode(b.w, b.p, A) + '\n'; continue; }
        if (b.prio !== prio) continue;
        // a 2D variation's preserve-z clause (JWildfire adds w·z of its input point to its output z)
        const snippet = b.def.code(b.w, b.p, A);
        const pzLine = zOut && (!(b.def.flags ?? []).includes('z') || Z_PRESERVE_TOO.has(b.name)) ? `    if ${pzCond} { pz_ += ${b.w} * z_; }\n` : '';
        // the few pre/post-priority JWildfire functions that carry the clause too (their input is the affine point, the
        // output the accumulator, so it is the same line) — unless the port already writes pz_ itself
        const pzPrePost = zOut && prio !== 0 && PREPOST_PRESERVE_Z.has(b.name) && !/\bpz_\b/.test(snippet) ? `    if ${pzCond} { pz_ += ${b.w} * z_; }\n` : '';
        if (b.forced === -1) {
          // enforced pre: input ← input + w·f(input) (the snippet adds into v; v is borrowed and restored)
          snips += `    {\n      let v_keep = v; let pz_keep = pz_; v = t; pz_ = z_;\n    ${b.def.code(b.w, b.p, A)}\n${pzLine}      t = v; z_ = pz_; v = v_keep; pz_ = pz_keep;\n    }\n` + precalc;
          continue;
        }
        if (b.forced === 1) {
          // enforced post: output ← output + w·f(output) (the snippet reads t; t is borrowed and restored)
          snips += `    {\n      let t_keep = t; let z_keep = z_; t = v; z_ = pz_;\n` + precalc + `    ${b.def.code(b.w, b.p, A)}\n${pzLine}      t = t_keep; z_ = z_keep;\n    }\n` + precalc;
          continue;
        }
        if (b.forcedZero) {
          // a post-priority function at normal priority: it rewrites the output; its "affine" argument is a copy of the output
          snips += `    {\n      let t_keep = t; let z_keep = z_; t = v; z_ = pz_;\n` + precalc + `    ${b.def.code(b.w, b.p, A)}\n      t = t_keep; z_ = z_keep;\n    }\n` + precalc;
          continue;
        }
        snips += '    ' + snippet + '\n';
        if (prio === 0) snips += pzLine; else snips += pzPrePost;
      }
    }
    return `  {
    var t = ${input};
    var z_ = ${zIn};
    var pz_ = 0.0;
    var r2 = max(dot(t, t), 1e-12);
    var r = sqrt(r2);
    var th = atan2(t.x, t.y);
    var ph = atan2(t.y, t.x);
    var v = ${initV};
${snips}    ${output} = v;${zOut ? `\n    ${zOut} = pz_;` : ''}
  }
`;
  };

  // JWildfire TransformationAffineFullStep: xy linear part, then the yz plane,
  // then the zx plane, and all translations added at the end.
  let body = `  let ax = ${A(0)}*pin.x + ${A(1)}*pin.y;
  let ay = ${A(3)}*pin.x + ${A(4)}*pin.y;
  let ay2 = ${A(16)}*ay + ${A(17)}*pin.z;
  let az2 = ${A(19)}*ay + ${A(20)}*pin.z;
  let ax3 = ${A(22)}*ax + ${A(23)}*az2;
  let az3 = ${A(25)}*ax + ${A(26)}*az2;
  var t0 = vec2f(ax3 + ${A(2)} + ${A(24)}, ay2 + ${A(5)} + ${A(18)});
  let zin = az3 + ${A(21)} + ${A(27)};
`;
  body += '  var zout = zin;\n';
  if (wf) {
    const wfArgs = `i32(${A(48)}), i32(${A(53)}), ${A(54)}, i32(${A(55)}), ${A(56)}, ${A(57)}, i32(${A(58)}), i32(${A(59)}), i32(${A(60)})`;
    body += `  let wfp = select(vec3f(t0, zin), pin, ${A(49)} > 0.5);\n  let wfv = wfieldValue(${wfArgs}, wfp.x, wfp.y, wfp.z);\n  let wfAmp = 1.0 + wfv * ${A(50)};\n`;
  }
  if (pre.length) body += genStage(pre, 't0', 't0', 'vec2f(0.0, 0.0)', 'zin', null);
  body += '  var vout = vec2f(0.0, 0.0);\n';
  body += genStage(main, 't0', 'vout', 'vec2f(0.0, 0.0)', 'zin', 'zout');
  if (post.length) body += genStage(post, 'vout', 'vout', 'vec2f(0.0, 0.0)', 'zout', null);

  let tail = `  return vec3f(px3 + ${A(8)} + ${A(36)}, py2 + ${A(11)} + ${A(30)}, pz3 + ${A(33)} + ${A(39)});`;
  if (wf) {
    // JWildfire: after the post affine the field jitters the output (×0.1, x/y/z each from a permuted lookup) and scales the colour (×0.1)
    const wfArgs = `i32(${A(48)}), i32(${A(53)}), ${A(54)}, i32(${A(55)}), ${A(56)}, ${A(57)}, i32(${A(58)}), i32(${A(59)}), i32(${A(60)})`;
    tail = `  var out_ = vec3f(px3 + ${A(8)} + ${A(36)}, py2 + ${A(11)} + ${A(30)}, pz3 + ${A(33)} + ${A(39)});
  if (abs(${A(52)}) > 1e-9) {
    let ji = 0.1 * ${A(52)};
    out_.x += wfieldValue(${wfArgs}, out_.x, out_.y, out_.z) * ji;
    out_.y += wfieldValue(${wfArgs}, out_.y, out_.x, out_.z) * ji;
    out_.z += wfieldValue(${wfArgs}, out_.z, out_.x, out_.y) * ji;
  }
  if (abs(${A(51)}) > 1e-9) { *cp = *cp * (1.0 + wfv * ${A(51)} * 0.1); }
  return out_;`;
  }
  return `fn ${name}(pin: vec3f, cp: ptr<function, f32>, rs: ptr<function, u32>, hd: ptr<function, bool>, rgb: ptr<function, vec4f>) -> vec3f {
  let PALB_: u32 = ${palBase}u; // this layer's palette base (direct-colour variations read it)
${body}  let px1 = ${A(6)}*vout.x + ${A(7)}*vout.y;
  let py1 = ${A(9)}*vout.x + ${A(10)}*vout.y;
  let py2 = ${A(28)}*py1 + ${A(29)}*zout;
  let pz2 = ${A(31)}*py1 + ${A(32)}*zout;
  let px3 = ${A(34)}*px1 + ${A(35)}*pz2;
  let pz3 = ${A(37)}*px1 + ${A(38)}*pz2;
${tail}
}`;
}

// JWildfire WeightingFieldType ordinals (as wfieldValue() expects them); IMAGE_MAP is not supported (needs an image)
const WFIELD_TYPE_ORD: Record<string, number> = { CELLULAR_NOISE: 1, CUBIC_NOISE: 2, CUBIC_FRACTAL_NOISE: 3, PERLIN_NOISE: 4, PERLIN_FRACTAL_NOISE: 5, SIMPLEX_NOISE: 6, SIMPLEX_FRACTAL_NOISE: 7, VALUE_NOISE: 8, VALUE_FRACTAL_NOISE: 9, WHITE_NOISE: 10 };

// JWildfire per-transform colour modifiers (DefaultRenderIterationState.transformPlotColor + its
// HSLRGBConverter, ported literally incl. quirks): the four modifier values travel with the point and
// are blended per transform like the colour; at plot time they reshape the palette RGB, which
// JWildfire holds on a 0..199.2 scale (palette·200/256) — the gamma and contrast formulas depend on that scale.
const MODS_WGSL = `@group(0) @binding(6) var<storage, read_write> mods: array<vec4f>; // per-point [gamma, contrast, saturation, hue]

fn modBlend(m: vec4f, b: u32) -> vec4f {
  // m' = m·(1+speed)/2 + value·(1−speed)/2 (block slots 40..47: value/speed pairs)
  let v = vec4f(xd[b + 40u], xd[b + 42u], xd[b + 44u], xd[b + 46u]);
  let s = vec4f(xd[b + 41u], xd[b + 43u], xd[b + 45u], xd[b + 47u]);
  return m * (1.0 + s) * 0.5 + v * (1.0 - s) * 0.5;
}

fn hslFromRgb(rgb: vec3f) -> vec3f {
  let r = clamp(rgb.x, 0.0, 1.0); let g = clamp(rgb.y, 0.0, 1.0); let bl = clamp(rgb.z, 0.0, 1.0);
  let mx = max(r, max(g, bl)); let mn = min(r, min(g, bl));
  let l = (mn + mx) * 0.5;
  var h = 1.0; var s = 0.0;
  if (abs(l) > 1e-10) {
    s = mx - mn;
    if (abs(s) > 1e-10) {
      s = s / select(2.0 - mx - mn, mn + mx, l <= 0.5);
      if (abs(r - mx) < 1e-10) { h = select(1.0 - (mx - g) / (mx - mn), 5.0 + (mx - bl) / (mx - mn), g == mn); }
      else if (abs(g - mx) < 1e-10) { h = select(3.0 - (mx - bl) / (mx - mn), 1.0 + (mx - r) / (mx - mn), bl == mn); }
      else { h = select(5.0 - (mx - r) / (mx - mn), 3.0 + (mx - g) / (mx - mn), r == mn); }
      h = h / 6.0;
    }
  }
  return vec3f(h, s, l);
}

fn rgbFromHsl(hsl: vec3f) -> vec3f {
  var h = clamp(hsl.x, 0.0, 1.0); let s = clamp(hsl.y, 0.0, 1.0); let l = clamp(hsl.z, 0.0, 1.0);
  let v = select(l + s - l * s, l * (1.0 + s), l <= 0.5);
  if (v <= 0.0) { return vec3f(0.0); }
  h = clamp(h * 6.0, 0.0, 6.0);
  let hi = floor(h);
  let y = l + l - v;
  let x = y + (v - y) * (h - hi);
  let z = v - (v - y) * (h - hi);
  let i = i32(hi);
  if (i == 1) { return vec3f(z, v, y); }
  if (i == 2) { return vec3f(y, v, x); }
  if (i == 3) { return vec3f(y, z, v); }
  if (i == 4) { return vec3f(x, y, v); }
  if (i == 5) { return vec3f(v, y, z); }
  return vec3f(v, x, y);
}

fn applyColorMods(col: vec3f, m: vec4f) -> vec3f {
  const S: f32 = 255.0 * 200.0 / 256.0; // JWildfire RenderColor scale
  var c = col * S;
  if (abs(m.x) > 1e-10) {
    let gamma = 4.2 / (4.2 - m.x);
    let alpha = c.x * 0.299 + c.y * 0.588 + c.z * 0.113;
    if (alpha > 1e-10) { c = c * (pow(alpha, gamma) / alpha); }
  }
  if (abs(m.y) > 1e-10) {
    let gamma = 1.2 / (1.2 - m.y * 0.5);
    c = (c - 127.5) * gamma + 127.5;
  }
  if (abs(m.z) > 1e-10) {
    let avg = c.x * 0.299 + c.y * 0.588 + c.z * 0.113;
    c = c + (c - avg) * m.z;
  }
  if (abs(m.w) > 1e-10) {
    let hsl = hslFromRgb(c / 255.0);
    c = clamp(round(rgbFromHsl(vec3f(hsl.x + m.w, hsl.y, hsl.z)) * 255.0), vec3f(0.0), vec3f(255.0));
  }
  return max(c, vec3f(0.0)) / S;
}
`;

// ---- subflame_wf: the nested flame (first layer) an instance renders ----
interface SubFlame { layer: Layer; preserveZ: boolean }
const subCache = new Map<string, SubFlame | null>();
/** Parse a sub-flame XML the way SubFlameWFFunc sees it: first layer, final xforms without a colour type recolour as
 *  DIFFUSION (JWildfire sets UNSET finals to DIFFUSION there — the opposite of a normal render's NONE); nested
 *  subflame_wf instances are dropped (no recursion). null when nothing renders. */
export function parseSubFlame(xml: string | undefined, fallbackPalette: Flame['layers'][number]['palette']): SubFlame | null {
  const text = xml || DFLT_SUBFLAME_XML;
  const have = subCache.get(text);
  if (have !== undefined) return have;
  let out: SubFlame | null = null;
  try {
    const fixed = text.replace(/<finalxform\b([^>]*)>/g, (m, attrs: string) => (/\bcolor_type=/.test(attrs) ? m : `<finalxform color_type="DIFFUSION"${attrs}>`));
    const { flame } = importFlameText(fixed, fallbackPalette);
    const layer = flame.layers[0];
    for (const x of [...layer.xforms, ...(layer.final ? [layer.final] : []), ...layer.moreFinals]) {
      const strip = (l?: VarInstance[]) => l?.filter((vi) => !VARIATIONS[vi.name]?.flags?.includes('subflame'));
      x.variations = strip(x.variations) ?? [];
      if (!x.variations.length) x.variations = [{ name: 'linear', weight: 0, params: {} }];
      x.preVariations = strip(x.preVariations); x.postVariations = strip(x.postVariations);
    }
    if (layer.xforms.length) out = { layer, preserveZ: !!flame.preserveZ };
  } catch (e) { console.warn('subflame_wf: sub-flame not parsed:', e); }
  subCache.set(text, out);
  return out;
}
/** subflame_wf instances of a flame in kernel order (layers → xforms → pre/main/post lists) — the data hook's index. */
function subflameInstances(layers: Layer[]): VarInstance[] {
  const out: VarInstance[] = [];
  for (const ly of layers) for (const x of [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals]) for (const l of varLists(x)) for (const vi of l) if (VARIATIONS[vi.name]?.flags?.includes('subflame')) out.push(vi);
  return out;
}

/** WGSL for JWildfire's post-symmetry projectors: copy `i` of a plotted world point.
 *  Axis modes offset by half the distance, mirror about the centre, then rotate the two
 *  copies in opposite directions by `rotation` degrees. POINT rotates by i·2π/order. */
function symmetryWgsl(p: PostSymmetry, n: number): string {
  const cx = p.centreX, cy = p.centreY;
  const a = (p.rotation * Math.PI) / 180;
  const doRotate = Math.abs(a) > 1e-9;
  const hd = p.distance / 2;
  const f = (v: number) => (Number.isFinite(v) ? v : 0).toPrecision(9);
  if (p.type === 'POINT') {
    const order = n - 1;
    return `
fn symApply(q: vec3f, i: u32) -> vec3f {
  if (i == 0u) { return q; } // JWildfire plots the original, then `+ order + ` rotated copies (its first repeats it)
  let ang = f32(i - 1u) * ${f((2 * Math.PI) / order)};
  let ca = cos(ang); let sa = sin(ang);
  let dx = q.x - ${f(cx)}; let dy = q.y - ${f(cy)};
  return vec3f(${f(cx)} + dx * ca - dy * sa, ${f(cy)} + dy * ca + dx * sa, q.z);
}
`;
  }
  const axisX = p.type === 'X_AXIS';
  const mirror = axisX
    ? `var x = select(${f(cx)} - dx - ${f(hd)}, ${f(cx)} + dx + ${f(hd)}, i == 0u);\n  var y = q.y;`
    : `var x = q.x;\n  var y = select(${f(cy)} - dy - ${f(hd)}, ${f(cy)} + dy + ${f(hd)}, i == 0u);`;
  const rot = !doRotate ? '' : `
  let rx = x - ${f(cx)}; let ry = y - ${f(cy)};
  let ca = ${f(Math.cos(a))}; let sa = ${f(Math.sin(a))};
  if (i == 0u) { x = ${f(cx)} + rx * ca + ry * sa; y = ${f(cy)} + ry * ca - rx * sa; }
  else { x = ${f(cx)} + rx * ca - ry * sa; y = ${f(cy)} + ry * ca + rx * sa; }`;
  return `
fn symApply(q: vec3f, i: u32) -> vec3f {
  let dx = q.x - ${f(cx)};
  let dy = q.y - ${f(cy)};
  ${mirror}${rot}
  return vec3f(x, y, q.z);
}
`;
}

export function compileFlame(flame: Flame, nPoints: number): CompiledFlame {
  const layers = visibleLayers(flame);
  const usesMods = layers.some((ly) => [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals].some((x) => x.colorMods?.some((v) => v !== 0)));
  const solid = !!flame.solid?.enabled;
  // JWildfire post symmetry (DefaultRenderIterationState's projector chain): every plotted point
  // is duplicated. Axis modes plot two mirrored copies; POINT plots the original plus `order`
  // rotated copies (its i = 0 copy repeats the original, exactly as JWildfire does).
  const psym = flame.postSymmetry;
  const symN = !psym ? 1 : psym.type === 'POINT' ? Math.min(64, Math.max(1, Math.round(psym.order))) + 1 : 2;
  const symWgsl = !psym ? '' : symmetryWgsl(psym, symN);
  const usesMat = solid && usesMaterials(flame);
  const usesMesh = layers.some((ly) => [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals].some((x) => varLists(x).some((l) => l.some((vi) => VARIATIONS[vi.name]?.flags?.includes('mesh')))));
  const L = layers.length;

  // ---- Layout ----
  let off = XD_HEADER;
  const infos = layers.map((ly) => {
    const n = ly.xforms.length;
    const cdfBase = off;
    off += CDF_ROW * (n + 1);
    const bases = ly.xforms.map((x) => { const b = off; off += blockSize(x); return b; });
    let finalBase = -1;
    if (ly.final) { finalBase = off; off += blockSize(ly.final); }
    const moreBases = ly.moreFinals.map((x) => { const b = off; off += blockSize(x); return b; });
    return { n, cdfBase, bases, finalBase, moreBases };
  });
  // subflame_wf: every instance's sub-flame gets its own weight table rows, xform blocks and palette (256 RGB) after the layers
  const subs = subflameInstances(layers).map((vi) => {
    const sf = parseSubFlame(vi.res?.flame, layers[0].palette);
    if (!sf) return null;
    const n = sf.layer.xforms.length;
    const cdfBase = off; off += CDF_ROW * (n + 1);
    const bases = sf.layer.xforms.map((x) => { const b = off; off += blockSize(x); return b; });
    const finals = [...(sf.layer.final ? [sf.layer.final] : []), ...sf.layer.moreFinals];
    const finalBases = finals.map((x) => { const b = off; off += blockSize(x); return b; });
    const palBase = off; off += 768;
    return { sf, n, cdfBase, bases, finals, finalBases, palBase };
  });
  const dataSize = Math.max(off, 64);

  // ---- Per-layer code ----
  let funcs = '';
  let iterFns = '';
  layers.forEach((ly, li) => {
    const info = infos[li];
    ly.xforms.forEach((x, i) => {
      funcs += genXformFn(`applyX${li}_${i}`, x, info.bases[i], li * 256) + '\n\n';
    });
    if (ly.final) funcs += genXformFn(`applyF${li}`, ly.final, info.finalBase, li * 256) + '\n\n';
    ly.moreFinals.forEach((x, k) => { funcs += genXformFn(`applyF${li}_${k}`, x, info.moreBases[k], li * 256) + '\n\n'; });

    let sel = `    let cb = ${info.cdfBase}u + (prev + 1u) * ${CDF_ROW}u;\n    var xi = 0u;\n`;
    for (let i = 0; i < info.n - 1; i++) {
      sel += `    if (rw > xd[cb + ${i}u]) { xi = ${i + 1}u; }\n`;
    }
    const cases = ly.xforms.map((_, i) => {
      const b = info.bases[i];
      // Color-speed blend runs first so direct-color variations get the last word.
      const ct = ly.xforms[i].colorType;
      // JWildfire ColorType: DIFFUSION blends the index (default); CYCLIC adds the symmetry to the index (mod 1); DISTANCE keeps
      // the index and paints the palette entry at color + |Δp|·(symmetry+1) — a plot colour that a later gradient step
      // (DIFFUSION/CYCLIC final) replaces, unlike a direct-colour variation's (rgbo.w 0.5 vs 1)
      const cstep = ct === 'CYCLIC' ? `        c = fract(c + (1.0 - 2.0 * xd[${b + 13}u]));`
        : ct === 'DISTANCE' ? '' : `        let cs = xd[${b + 13}u];\n        c = c * (1.0 - cs) + xd[${b + 12}u] * cs;`;
      const dstep = ct === 'DISTANCE' ? `\n        if (rgbo.w < 0.75) { let dci = i32((xd[${b + 12}u] + length(np - p) * (2.0 - 2.0 * xd[${b + 13}u])) * 254.0 + 0.5) % 256; rgbo = vec4f(pal[${li * 256}u + u32(dci)].xyz, 0.5); }` : '';
      return `      case ${i}u: {
${cstep}
        op = xd[${b + 14}u];${usesMods ? `\n        m = modBlend(m, ${b}u);` : ''}${usesMat ? `\n        mt = mt * (1.0 + xd[${b + 65}u]) * 0.5 + xd[${b + 64}u] * (1.0 - xd[${b + 65}u]) * 0.5;` : ''}
        np = applyX${li}_${i}(p, &c, &rs, &hide, &rgbo);${dstep}
      }`;
    }).join('\n');
    // final transforms run in sequence (JWildfire: each further final takes the previous output)
    const finalStep = (fn: string, base: number, input: string, fx: XForm) => `
      {
        ${fx.colorType === 'CYCLIC' ? `dc = fract(dc + (1.0 - 2.0 * xd[${base + 13}u]));` : fx.colorType === 'DISTANCE' ? '' : `let fcs = xd[${base + 13}u];\n        dc = dc * (1.0 - fcs) + xd[${base + 12}u] * fcs;`}${usesMods ? `\n        dm = modBlend(dm, ${base}u);` : ''}${fx.colorType !== 'DISTANCE' && (fx.colorType === 'CYCLIC' || fx.colorSpeed > 0) ? '\n        if (rgbo.w < 0.75) { rgbo = vec4f(0.0); } // gradient step: a DISTANCE plot colour is replaced by palette[index]' : ''}
      }
      dp = ${fn}(${input}, &dc, &rs, &hide, &rgbo);${fx.colorType === 'DISTANCE' ? `\n      if (rgbo.w < 0.75) { let dci = i32((xd[${base + 12}u] + length(dp - ${input}) * (2.0 - 2.0 * xd[${base + 13}u])) * 254.0 + 0.5) % 256; rgbo = vec4f(pal[${li * 256}u + u32(dci)].xyz, 0.5); }` : ''}`;
    let finalBlock = ly.final ? finalStep(`applyF${li}`, info.finalBase, 'p', ly.final) : '';
    ly.moreFinals.forEach((fx, k) => { finalBlock += finalStep(`applyF${li}_${k}`, info.moreBases[k], 'dp', fx); }); // dp starts as p

    iterFns += `
fn iterLayer${li}(idx: u32) {
  var pt = pts[idx];
  var rs = rngs[idx].x;
  var prev = min(rngs[idx].y & 255u, ${info.n - 1}u);
  var fuse = f32((rngs[idx].y >> 8u) & 0xFFFFFu);
  // walker age since (re)start, saturating at 7 (bits 28..30): its first few iterations count as JWildfire's "first call" of a
  // variation instance (pre_stabilize resets there) — a few iterations, so an xform with a small weight is reached too
  var age = (rngs[idx].y >> 28u) & 7u;${solid ? '\n  var bdone = (rngs[idx].y & 0x80000000u) != 0u; // shadow maps: this walker already contributed its light-space bounds sample' : ''}
  var p = pt.xyz;
  var c = pt.w;${usesMods ? '\n  var m = mods[idx];' : ''}${usesMat ? '\n  var mt = mats[idx];' : ''}
  let ca = cos(P.rotation);
  let sa = sin(P.rotation);
  let offX = P.fullW * 0.5 - P.tileX;
  let offY = P.fullH * 0.5 - P.tileY;
  let cam3d = (P.flags & 2u) != 0u;

  for (var it = 0u; it < P.iters; it = it + 1u) {
    wstart_ = age < 7u;
    age = min(age + 1u, 7u);
    let rw = rnd(&rs);
${sel}
    var np = p;
    var op = 1.0;
    var hide = false;
    var rgbo = vec4f(0.0); // direct RGB colour from dc_* variations (w = 1 when set)
    switch xi {
${cases}
      default: {}
    }
    prev = xi;
    p = np;

    // JWildfire re-fuses a point only when a coordinate is NaN/infinite (it never limits the magnitude,
    // and its doubles give z ~1e300 of headroom under preserve_z growth). In f32 z would overflow where
    // JWildfire's is merely huge, and an infinite z poisons x/y through the 0·z terms of the identity
    // 3D affines — so z is kept finite (|z| ≤ 1e18, NaN → 0) and only x/y decide whether the point restarts.
    // JWildfire notices the bad point only at its next validateState() (every 1000 iterations, so ~500
    // wasted-but-counted iterations on average) and then fuses 20 more; the same expected loss here.
    if (p.x != p.x || p.y != p.y || abs(p.x) > 3.0e38 || abs(p.y) > 3.0e38) {
      p = vec3f(rnd(&rs) * 2.0 - 1.0, rnd(&rs) * 2.0 - 1.0, 0.0);
      c = rnd(&rs);${usesMods ? '\n      m = vec4f(0.0);' : ''}
      fuse = 21.0 + floor(rnd(&rs) * 1000.0);
      age = 0u;
      continue;
    }
    if (p.z != p.z) { p.z = 0.0; }
    p.z = clamp(p.z, -1.0e18, 1.0e18);
    if (fuse > 0.0) {
      fuse = fuse - 1.0;
      continue;
    }

    var dp = p;
    var dc = c;${usesMods ? '\n    var dm = m;' : ''}${finalBlock}
    if (hide) { op = 0.0; }
${symN > 1 ? `    // post symmetry: plot this point once per symmetry copy (shadowing dp inside the loop
    // keeps the projection below untouched — its initialiser still reads the outer point)
    for (var si = 0u; si < ${symN}u; si = si + 1u) {
    let dp = symApply(dp, si);` : ''}
    var rx: f32;
    var ry: f32;
    var visible = true;
    var dz = 1.0; // dimish-z intensity
    var cz = 0.0; // camera-space depth (solid rendering's z-buffer key)
    if (cam3d) {
      // JWildfire 3D camera: rotate by the camera matrix (yaw/pitch/roll), offset,
      // then perspective-divide; the centre offset applies after projection.
      let cx = P.m0.x * dp.x + P.m0.y * dp.y + P.m0.z * dp.z + P.camPos.x;
      let cy = P.m1.x * dp.x + P.m1.y * dp.y + P.m1.z * dp.z + P.camPos.y;
      cz = P.m2.x * dp.x + P.m2.y * dp.y + P.m2.z * dp.z + P.camPos.z;
      let zr = 1.0 - P.camPersp * cz + P.camPos.z;
      visible = zr >= 1e-9;
      // dimish-z: fade points beyond dimZDist toward the dim colour
      if (P.dimZ.x > 1e-9) {
        let zd = P.dimZ.y - cz;
        if (zd > 0.0) { dz = exp(-zd * zd * P.dimZ.x); }
      }
      var px = cx;
      var py = cy;
      // depth of field (bubble shape): random disc offset ∝ distance from focus.
      // Solid flames skip the jitter entirely — JWildfire's view calls applyOnlyCamera and defers
      // the blur to the post-process DOF pass (PostDOFCalculator), which we run after the tonemap.
      if (${solid ? 'false' : 'P.dof.x != 0.0'}) {
        var dist = 0.0;
        if (P.dof.w > 0.5) {
          let fx_ = cx - P.focus.x; let fy_ = cy - P.focus.y; let fz_ = cz - P.focus.z;
          let d = pow(fx_ * fx_ + fy_ * fy_ + fz_ * fz_, 1.0 / P.dof.y);
          let area = P.focus.w; let fade = area / 2.25; let amf = area - fade;
          if (d > area) { dist = d; }
          else if (d > amf) { let u = clamp((d - amf) / fade, 0.0, 1.0); dist = u * u * u * (u * (u * 6.0 - 15.0) + 10.0) * d; }
        } else {
          let zd_ = P.camZ - cz;
          if (zd_ > 0.0) { dist = zd_; }
        }
        if (dist > 0.0) {
          var ff = 1.0;
          if (P.dof.z >= 0.999999) { ff = rnd(&rs); }
          else if (P.dof.z > 1e-6) { if (rnd(&rs) <= P.dof.z) { ff = rnd(&rs); } }
          let dr = ff * P.dof.x * dist;
          let ang = 6.28318530718 * rnd(&rs);
          px = cx + dr * cos(ang);
          py = cy + dr * sin(ang);
        }
      }
      rx = px / zr - P.centerX;
      ry = py / zr - P.centerY;
    } else {
      let ox = dp.x - P.centerX;
      let oy = dp.y - P.centerY;
      rx = ox * ca - oy * sa;
      ry = ox * sa + oy * ca;
    }
    // flam3/JWildfire raster convention: +y grows downward on the image.
    var fx = rx * P.ppu + offX;
    var fy = ry * P.ppu + offY;
    // JWildfire antialiasing: some samples get a random jitter (raster pixels)
    if (P.aa.x > 0.0 && rnd(&rs) > 1.0 - P.aa.x) {
      let dr = exp(P.aa.y * sqrt(-log(max(rnd(&rs), 1e-12)))) - 1.0;
      let da = 6.28318530718 * rnd(&rs);
      fx = fx + dr * cos(da);
      fy = fy + dr * sin(da);
    }
${solid ? `    // JWildfire solid rendering: no density — the nearest point per raster cell wins (z-buffer on the camera-space
    // depth), carrying its untransformed position (for the normals), palette colour × layer weight and material.
    // OPAQUE draw mode drops a point with probability 1−opacity; hidden points never plot; colour modifiers do not apply.
    // Every drawn point also splats into the shadow maps (light-space z, whether or not the camera sees it).
    let drawn = !hide && (op >= 1.0 || rnd(&rs) <= op);
    // mode 1 takes ONE bounds sample per walker (its first plotted point after the fuse: ~65k samples, like
    // JWildfire's first 40960); more would let rare far points stretch the map
    if (drawn && P.shadow.x == 2u) { shadowSplat(dp); }
    else if (drawn && P.shadow.x == 1u && !bdone) { bdone = true; shadowSplat(dp); }
    if (visible && fx >= 0.0 && fy >= 0.0 && fx < f32(P.width) && fy < f32(P.height) && drawn) {
      var col = pal[${li * 256}u + min(u32(clamp(dc, 0.0, 1.0) * 255.99), 255u)];
      if (rgbo.w > 0.25) { col = vec4f(clamp(rgbo.xyz, vec3f(0.0), vec3f(1.0)), 1.0); }
      if (dz < 1.0) { col = vec4f(mix(P.dimColor.xyz, col.xyz, dz), col.w); }
      let lw = xd[${8 + li}u] * (200.0 / 256.0); // JWildfire RenderColor scale: the shading sees palette·200/256/255
      solidSplat(u32(fy) * P.width + u32(fx), cz, dp, col.xyz * lw, ${usesMat ? 'mt' : '0.0'});
    }` : `    if (visible && fx >= 0.0 && fy >= 0.0 && fx < f32(P.width) && fy < f32(P.height)) {
      let hi = (u32(fy) * P.width + u32(fx)) * 4u;
      var col = pal[${li * 256}u + min(u32(clamp(dc, 0.0, 1.0) * 255.99), 255u)];
      if (rgbo.w > 0.25) { col = vec4f(clamp(rgbo.xyz, vec3f(0.0), vec3f(1.0)), 1.0); }${usesMods ? '\n      col = vec4f(applyColorMods(col.xyz, dm), col.w);' : ''}
      if (dz < 1.0) { col = vec4f(mix(P.dimColor.xyz, col.xyz, dz), col.w); }
      let lw = op * xd[${8 + li}u]; // JWildfire layer weight: colour intensity multiplier (the density count is unaffected)
      // fixed-point colour accumulation is dithered so dim contributions (small weights/opacities) stay unbiased
      let dth = rnd(&rs);
      atomicAdd(&hist[hi + 0u], u32(col.x * lw * 255.0 + dth));
      atomicAdd(&hist[hi + 1u], u32(col.y * lw * 255.0 + dth));
      atomicAdd(&hist[hi + 2u], u32(col.z * lw * 255.0 + dth));
      atomicAdd(&hist[hi + 3u], u32(op * 256.0));
    }`}
${symN > 1 ? '    }' : ''}
  }

  pts[idx] = vec4f(p, c);${usesMods ? '\n  mods[idx] = m;' : ''}${usesMat ? '\n  mats[idx] = mt;' : ''}
  rngs[idx] = vec2u(rs, prev | (u32(fuse) << 8u) | (age << 28u)${solid ? ' | select(0u, 0x80000000u, bdone)' : ''});
}
`;
  });

  // ---- Layer dispatch by thread index ----
  let ldis = '  var layer = 0u;\n';
  for (let li = 0; li < L - 1; li++) {
    ldis += `  if (f32(idx) >= xd[${li}u]) { layer = ${li + 1}u; }\n`;
  }
  // ---- subflame_wf: nested chaos games (SubFlameWFFunc.subflameIter/prefuseIter) ----
  let subFuncs = '';
  if (subs.length) {
    const subFns: string[] = [];
    subs.forEach((sb, k) => {
      if (!sb) return;
      const pz = sb.sf.preserveZ ? 'true' : 'false';
      const ly = sb.sf.layer;
      ly.xforms.forEach((x, i) => { funcs += genXformFn(`applyS${k}_${i}`, x, sb.bases[i], sb.palBase, pz) + '\n\n'; });
      sb.finals.forEach((x, j) => { funcs += genXformFn(`applySF${k}_${j}`, x, sb.finalBases[j], sb.palBase, pz) + '\n\n'; });
      let sel = `  let cb = ${sb.cdfBase}u + (sub${k}_xi + 1u) * ${CDF_ROW}u;\n  var xi = 0u;\n`;
      for (let i = 0; i < sb.n - 1; i++) sel += `  if (rw > xd[cb + ${i}u]) { xi = ${i + 1}u; }\n`;
      const cases = ly.xforms.map((x, i) => {
        const b = sb.bases[i];
        const cstep = x.colorType === 'CYCLIC' ? `c = fract(c + (1.0 - 2.0 * xd[${b + 13}u]));` : x.colorType === 'DISTANCE' ? '' : `let cs = xd[${b + 13}u]; c = c * (1.0 - cs) + xd[${b + 12}u] * cs;`;
        return `    case ${i}u: { ${cstep} op = xd[${b + 14}u]; np = applyS${k}_${i}(sub${k}_p, &c, rs, &hide, &rgbo); }`;
      }).join('\n');
      const fsteps = sb.finals.map((fx, j) => {
        const b = sb.finalBases[j];
        const cstep = fx.colorType === 'CYCLIC' ? `qc = fract(qc + (1.0 - 2.0 * xd[${b + 13}u]));` : fx.colorType === 'DISTANCE' ? '' : `{ let fcs = xd[${b + 13}u]; qc = qc * (1.0 - fcs) + xd[${b + 12}u] * fcs; }`;
        return `    ${cstep} q = applySF${k}_${j}(q, &qc, rs, &qhide, &qrgb);`;
      }).join('\n');
      subFuncs += `
var<private> sub${k}_p: vec3f = vec3f(0.0);
var<private> sub${k}_c: f32 = 0.0;
var<private> sub${k}_xi: u32 = 0u;
var<private> sub${k}_init: bool = false;
// one chaos-game step of sub-flame ${k}: picks the next xform from the current one's weight row, moves the persistent point; false when the xform is hidden/opaque-skipped
fn sub${k}_step(rs: ptr<function, u32>, hd: ptr<function, bool>, rgb: ptr<function, vec4f>) -> bool {
  let rw = rnd(rs);
${sel}  var c = sub${k}_c;
  var np = sub${k}_p;
  var op = 1.0;
  var hide = false;
  var rgbo = vec4f(0.0);
  switch xi {
${cases}
    default: {}
  }
  sub${k}_xi = xi;
  sub${k}_p = np;
  sub${k}_c = c;
  if (abs(op) <= 1e-9) { return false; }                       // DrawMode.HIDDEN
  if (abs(op - 1.0) > 1e-9 && rnd(rs) > op) { return false; }  // DrawMode.OPAQUE
  *hd = hide; *rgb = rgbo;
  return true;
}
fn subflame${k}(rs: ptr<function, u32>, cp: ptr<function, f32>, hd: ptr<function, bool>, rgb: ptr<function, vec4f>, scale: f32, angle: f32, ox: f32, oy: f32, oz: f32, cscz: f32, cmode: i32) -> vec3f {
  var hh = false; var rr = vec4f(0.0);
  if (!sub${k}_init || any(sub${k}_p != sub${k}_p) || any(abs(sub${k}_p) > vec3f(3.0e38))) {
    // prefuseIter: a fresh point, 42 unplotted steps
    sub${k}_p = vec3f(rnd(rs) - 0.5, rnd(rs) - 0.5, 0.0); sub${k}_c = rnd(rs); sub${k}_xi = 0u; sub${k}_init = true;
    for (var i = 0u; i < 42u; i = i + 1u) { sub${k}_step(rs, &hh, &rr); }
  }
  var q = sub${k}_p; var qc = sub${k}_c; var qhide = false; var qrgb = vec4f(0.0);
  var drawn = false;
  for (var it = 0u; it < 100u; it = it + 1u) {   // MAX_ITER
    if (sub${k}_step(rs, &qhide, &qrgb)) { drawn = true; break; }
  }
  if (!drawn) { return vec3f(0.0); }
  q = sub${k}_p; qc = sub${k}_c;
${fsteps}
  // colour: the sub-flame's palette entry (GradientColorStep, index·254 + 0.5) unless a direct-colour variation set one.
  // JWildfire holds palette colours on the RenderColor scale (·200/256) and direct colours on 0..255; our plot scale is the
  // palette's, so a palette colour passes through unscaled as an RGB colour but reads ·200/256 as a channel value
  var srgb = qrgb.xyz; var chan = qrgb.xyz;
  if (qrgb.w < 0.5) { let ci = u32(clamp(i32(qc * 254.0 + 0.5), 0, 255)) * 3u; srgb = vec3f(xd[${sb.palBase}u + ci], xd[${sb.palBase}u + ci + 1u], xd[${sb.palBase}u + ci + 2u]); chan = srgb * (200.0 / 256.0); }
  *hd = qhide;
  if (cmode != -1) {
    if (qrgb.w > 0.5 || cmode == -2) { *rgb = vec4f(srgb, 1.0); *cp = qc; }
    if (cmode == 0) { *cp = qc; }
    else if (cmode == 1) { *cp = chan.x; }
    else if (cmode == 2) { *cp = chan.y; }
    else if (cmode == 3) { *cp = chan.z; }
    else if (cmode == 4) { *cp = 0.2990 * chan.x + 0.5880 * chan.y + 0.1130 * chan.z; }
  }
  let ca_ = cos(angle * PI / 180.0); let sa_ = sin(angle * PI / 180.0);
  let x = scale * q.x; let y = scale * q.y;
  return vec3f(x * ca_ - y * sa_ + ox, x * sa_ + y * ca_ + oy, scale * q.z + oz + cscz * qc);
}
`;
      subFns.push(`    case ${k}u: { return subflame${k}(rs, cp, hd, rgb, scale, angle, ox, oy, oz, cscz, cmode); }`);
    });
    subFuncs += `
fn subflameAny(k: u32, rs: ptr<function, u32>, cp: ptr<function, f32>, hd: ptr<function, bool>, rgb: ptr<function, vec4f>, scale: f32, angle: f32, ox: f32, oy: f32, oz: f32, cscz: f32, cmode: i32) -> vec3f {
  switch k {
${subFns.join('\n')}
    default: { return vec3f(0.0); }
  }
}
`;
  }

  const lcases = layers.map((_, li) => `    case ${li}u: { iterLayer${li}(idx); }`).join('\n');

  const wgsl = `// Auto-generated WilderFire iteration kernel (${L} layer${L > 1 ? 's' : ''})
const PI: f32 = 3.14159265358979;

struct Params {
  width: u32,   // bounds of the target (tile) in pixels
  height: u32,
  iters: u32,
  flags: u32,   // bit0: preserve_z, bit1: 3D camera active
  centerX: f32,
  centerY: f32,
  ppu: f32,
  rotation: f32,
  tileX: f32,   // tile origin within the full image (0 for on-screen render)
  tileY: f32,
  fullW: f32,   // full image dimensions the camera maps onto
  fullH: f32,
  m0: vec4f,    // camera matrix rows (JWildfire yaw/pitch/bank/roll), w unused
  m1: vec4f,
  m2: vec4f,
  camPos: vec4f, // xyz: camera position offset, w: camPersp
  camPersp: f32,
  camZ: f32,     // legacy DOF focus plane
  _p2: f32,
  _p3: f32,
  focus: vec4f,  // xyz: DOF focus point, w: DOF area
  dof: vec4f,    // x: 0.1·camDOF·scale, y: exponent, z: fade, w: newDOF flag
  dimZ: vec4f,   // x: dimishZ, y: dimZDist
  dimColor: vec4f,
  aa: vec4f,     // x: antialias amount, y: antialias radius
  shadow: vec4u, // solid rendering shadow maps — x: mode (0 off, 1 collect light-space bounds, 2 splat), y: map size, z: casting light count
  lm: array<vec4f, 12>, // light-space projection rows (3 per casting light; LightViewCalculator matrix a)
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> xd: array<f32>;
@group(0) @binding(2) var<storage, read_write> pts: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> rngs: array<vec2u>; // x: rng state, y: prev xform
@group(0) @binding(4) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> pal: array<vec4f>;
${symWgsl}${usesMods ? MODS_WGSL : ''}${solid ? SOLID_KERNEL_WGSL : ''}${usesMat ? '\n@group(0) @binding(9) var<storage, read_write> mats: array<f32>; // per-point material index (JWildfire p.material)\n' : ''}${usesMesh ? '\n@group(0) @binding(12) var<storage, read> mesh: array<f32>; // mesh samplers: face CDFs + triangles (src/core/meshes.ts)\n' : ''}

fn rnd(state: ptr<function, u32>) -> f32 {
  var x = *state;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  *state = x;
  return f32(x) * 2.3283064365386963e-10;
}

fn mmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }

// runtime 0u (set opaquely in main): the double-float helpers (hsin_) route intermediates through an
// integer add of it so the shader compiler's fast-math reassociation cannot fold their error terms
var<private> df_zero: u32 = 0u;
// true during a walker's first iterations after a (re)start — JWildfire's "first call" of a variation instance (pre_stabilize)
var<private> wstart_: bool = false;

${collectFuncs([...layers, ...subs.flatMap((sb) => (sb ? [sb.sf.layer] : []))])}

${funcs}${subFuncs}${iterFns}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&rngs)) { return; }
  df_zero = bitcast<u32>(P.ppu) >> 31u;
${ldis}
  switch layer {
${lcases}
    default: {}
  }
}
`;

  // ---- Data writer (must mirror the layout above) ----
  const writeData = (fl: Flame, out: Float32Array) => {
    out.fill(0);
    const ls = visibleLayers(fl);

    // Layer thread cutoffs: like JWildfire every layer iterates equally often (one iteration state per
    // layer, round-robin); the layer weight scales the plotted colour, not the sample share (slots 8+li).
    let acc = 0;
    ls.forEach((_, i) => {
      acc += 1 / ls.length;
      out[i] = Math.round(acc * nPoints);
    });
    if (ls.length) out[ls.length - 1] = nPoints + 1;
    ls.forEach((l, i) => { out[8 + i] = Math.max(l.weight, 0); });

    // weight-table rows of an xform list: the plain row, then one row per "previous" xform (xaos)
    const writeCdf = (xforms: XForm[], cdfBase: number) => {
      const m = Math.min(xforms.length, CDF_ROW);
      const base = xforms.map((x) => Math.max(x.weight, 1e-6));
      const writeRow = (rowBase: number, mult: (j: number) => number) => {
        const wj = base.map((bw, j) => bw * Math.max(mult(j), 0));
        let rtot = wj.reduce((a, b) => a + b, 0);
        if (rtot <= 1e-12) {
          for (let j = 0; j < m; j++) wj[j] = base[j];
          rtot = base.reduce((a, b) => a + b, 0);
        }
        let racc = 0;
        for (let j = 0; j < m; j++) {
          racc += wj[j] / rtot;
          out[rowBase + j] = racc;
        }
        out[rowBase + m - 1] = 1.0001;
      };
      writeRow(cdfBase, () => 1);
      for (let i = 0; i < m; i++) {
        writeRow(cdfBase + CDF_ROW * (i + 1), (j) => xforms[i].xaos?.[j] ?? 1);
      }
    };
    let subIndex = 0; // subflame_wf instances in kernel order (matches `subs`)
    const writeBlock = (x: XForm, B: number) => {
      for (let i = 0; i < 6; i++) out[B + i] = x.affine[i];
      for (let i = 0; i < 6; i++) out[B + 6 + i] = x.post[i];
      out[B + 12] = x.color;
      out[B + 13] = x.colorSpeed;
      out[B + 14] = x.opacity;
      const I = [1, 0, 0, 0, 1, 0];
      for (let i = 0; i < 6; i++) out[B + 16 + i] = x.yz?.[i] ?? I[i];
      for (let i = 0; i < 6; i++) out[B + 22 + i] = x.zx?.[i] ?? I[i];
      for (let i = 0; i < 6; i++) out[B + 28 + i] = x.yzPost?.[i] ?? I[i];
      for (let i = 0; i < 6; i++) out[B + 34 + i] = x.zxPost?.[i] ?? I[i];
      for (let i = 0; i < 8; i++) out[B + 40 + i] = x.colorMods?.[i] ?? 0;
      for (let i = 0; i < 16; i++) out[B + 48 + i] = 0;
      out[B + 64] = x.material ?? 0;
      out[B + 65] = x.materialSpeed ?? 0;
      if (x.wfield && WFIELD_TYPE_ORD[x.wfield.type]) {
        const w = x.wfield;
        out[B + 48] = WFIELD_TYPE_ORD[w.type]; out[B + 49] = w.input === 'POSITION' ? 1 : 0;
        out[B + 50] = w.varAmount; out[B + 51] = w.color; out[B + 52] = w.jitter;
        out[B + 53] = w.seed; out[B + 54] = w.frequency; out[B + 55] = w.octaves; out[B + 56] = w.gain; out[B + 57] = w.lacunarity;
        out[B + 58] = w.fractalType === 'BILLOW' ? 1 : w.fractalType === 'RIGID_MULTI' ? 2 : 0;
        out[B + 59] = w.cellDistance === 'MANHATTAN' ? 1 : w.cellDistance === 'NATURAL' ? 2 : 0;
        out[B + 60] = ['CELL_VALUE', 'DISTANCE', 'DISTANCE2', 'DISTANCE_ADD', 'DISTANCE_SUB', 'DISTANCE_MUL', 'DISTANCE_DIV'].indexOf(w.cellReturn);
        if (out[B + 60] < 0) out[B + 60] = 2;
        w.params.slice(0, 3).forEach((pp, k) => { out[B + 61 + k] = pp.intensity; });
      }
      let o = B + HEADER;
      for (const list of varLists(x)) {
        for (const vi of list) {
          const def = VARIATIONS[vi.name];
          if (!def) continue;
          out[o++] = vi.weight;
          for (const pd of def.params ?? []) {
            out[o++] = vi.params[pd.name] ?? pd.def;
          }
          if (def.extra) {
            let k = 0;
            if (def.flags?.includes('mesh')) {
              // data hook: obj_mesh_primitive_wf / obj_mesh_wf → [cdf base, face count, triangle base] of its sampler in the mesh buffer (0 faces until loaded)
              const mk = meshKeyFor(vi);
              const lay = mk ? meshLayout.get(mk) : undefined;
              out[o++] = lay?.cdfBase ?? 0; out[o++] = lay?.faces ?? 0; out[o++] = lay?.triBase ?? 0; k = 3;
            } else if (def.flags?.includes('subflame')) {
              // data hook: subflame_wf → its compiled sub-flame index (a sub-flame that did not parse renders nothing)
              const si = subIndex++;
              out[o++] = si < subs.length && subs[si] ? si : 1e9; k = 1;
            }
            for (; k < def.extra; k++) out[o++] = 0;
          }
        }
      }
    };

    ls.forEach((ly, li) => {
      if (li >= infos.length) return;
      const info = infos[li];
      writeCdf(ly.xforms, info.cdfBase);
      ly.xforms.forEach((x, i) => { if (i < info.bases.length) writeBlock(x, info.bases[i]); });
      if (ly.final && info.finalBase >= 0) writeBlock(ly.final, info.finalBase);
      ly.moreFinals.forEach((x, k) => { if (k < info.moreBases.length) writeBlock(x, info.moreBases[k]); });
    });
    // subflame_wf: the sub-flames' tables/blocks/palettes (their structure is fixed at compile time; the sub-flame of an
    // instance whose XML changed since is a different signature → recompiled, so `subs` still matches)
    subs.forEach((sb) => {
      if (!sb) return;
      const ly = sb.sf.layer;
      writeCdf(ly.xforms, sb.cdfBase);
      ly.xforms.forEach((x, i) => writeBlock(x, sb.bases[i]));
      sb.finals.forEach((x, j) => writeBlock(x, sb.finalBases[j]));
      for (let i = 0; i < 256; i++) { const c = ly.palette[i] ?? [0, 0, 0]; out[sb.palBase + i * 3] = c[0]; out[sb.palBase + i * 3 + 1] = c[1]; out[sb.palBase + i * 3 + 2] = c[2]; }
    });
  };

  return { wgsl, dataSize, writeData, usesMods, solid, usesMat, usesMesh };
}

export const TONEMAP_WGSL = `// WilderFire tonemap: density-estimation filter + log-density + gamma/vibrancy
struct TP {
  width: u32,
  height: u32,
  filterN: u32, // colour filter size (odd; <= 1 = off), weights at sfilt[0..]
  filterNI: u32, // intensity filter size (JWildfire: gaussian 0.75 for sharpening kernels), weights at sfilt[128..]
  brightness: f32,
  gamma: f32,
  vibrancy: f32,
  spp: f32,
  de: vec4f, // x: DE estimator radius in px (0 = off), y: deCurve, z: transparent bg flag, w: oversample factor
  bg: vec4f, // rgb: background, w: gamma threshold
  jw: vec4f, // x: world area of the image (W·H/ppu²), y: contrast, z: colour scale 199.2/whiteLevel, w: low-density brightness
  bgUL: vec4f, // JWildfire background gradient corners (w of bgUL: 0 single colour, 1 GRADIENT_2X2, 2 GRADIENT_2X2_C)
  bgUR: vec4f,
  bgLL: vec4f,
  bgLR: vec4f,
  bgCC: vec4f,
  bgGeom: vec4f, // x, y: this tile's origin in the full image; z, w: full image size (the gradient spans the full image)
  post: vec4f, // x: modSaturation (saturation − 1), y: alphaScale (foreground opacity), z: filter_low_density, w: filter_sharpness
  adapt: vec4f, // adaptive filtering (MITCHELL_SINEPOW): x on/off, y/z/w = low-density/smoothing/detail kernel sizes
};

// Two passes: fsA = DE + log-scale per pixel into an rgba16float texture,
// fsB = spatial filter + gamma/vibrancy/background from that texture.
@group(0) @binding(0) var<uniform> T: TP;
@group(0) @binding(1) var<storage, read> hist: array<u32>;
@group(0) @binding(2) var<storage, read> sfilt: array<f32>;
@group(0) @binding(3) var midTex: texture_2d<f32>;

// JWildfire GammaCorrectionFilter.applyModSaturation: the finished pixel (background already
// composited in) goes through HSL with saturation shifted by (saturation − 1), clamped to 0..1.
fn modSaturation(c: vec3f, shift: f32) -> vec3f {
  if (abs(shift) < 1e-6) { return c; } // 'mod' is a WGSL reserved keyword — hence 'shift'
  let mx = max(c.r, max(c.g, c.b));
  let mn = min(c.r, min(c.g, c.b));
  let l = (mx + mn) * 0.5;
  let d = mx - mn;
  if (d < 1e-9) { return c; } // grey has no hue to preserve
  let s0 = select(d / (2.0 - mx - mn), d / (mx + mn), l <= 0.5);
  let s = clamp(s0 + shift, 0.0, 1.0);
  var h: f32;
  if (mx == c.r) { h = (c.g - c.b) / d + select(0.0, 6.0, c.g < c.b); }
  else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
  else { h = (c.r - c.g) / d + 4.0; }
  h = h / 6.0;
  let v = select(l + s - l * s, l * (1.0 + s), l <= 0.5);
  if (v <= 0.0) { return vec3f(0.0); }
  let m = l + l - v;
  let sv = (v - m) / v;
  let h6 = fract(h) * 6.0;
  let sextant = i32(floor(h6));
  let fr = h6 - floor(h6);
  let vsf = v * sv * fr;
  let mid1 = m + vsf;
  let mid2 = v - vsf;
  switch (sextant) {
    case 0: { return vec3f(v, mid1, m); }
    case 1: { return vec3f(mid2, v, m); }
    case 2: { return vec3f(m, v, mid1); }
    case 3: { return vec3f(m, mid2, v); }
    case 4: { return vec3f(mid1, m, v); }
    default: { return vec3f(v, m, mid2); }
  }
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(pos[vi], 0.0, 1.0);
}

// Abramowitz–Stegun 7.1.26 erf (max err 1.5e-7)
fn erf1(x: f32) -> f32 {
  let s = sign(x);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return s * y;
}

// (r, g, b, opacity-weighted count) at a clamped supersampled cell
fn cell(cx: i32, cy: i32) -> vec4f {
  let os = max(u32(T.de.w + 0.5), 1u);
  let px = clamp(cx, 0, i32(T.width * os) - 1);
  let py = clamp(cy, 0, i32(T.height * os) - 1);
  let i = (u32(py) * T.width * os + u32(px)) * 4u;
  return vec4f(f32(hist[i]), f32(hist[i + 1u]), f32(hist[i + 2u]), f32(hist[i + 3u]) / 256.0);
}

// Box-filtered os×os block of supersampled cells for output pixel (bx, by)
fn blockAt(bx: i32, by: i32) -> vec4f {
  let os = max(i32(T.de.w + 0.5), 1);
  if (os == 1) { return cell(bx, by); }
  var acc = vec4f(0.0);
  for (var j = 0; j < os; j = j + 1) {
    for (var i = 0; i < os; i = i + 1) {
      acc += cell(bx * os + i, by * os + j);
    }
  }
  return acc / f32(os * os);
}

// ---- Adaptive filtering (JWildfire LogDensityFilter.getFilter, MITCHELL_SINEPOW) ----
// Per pixel: sparse cells get a wide SINEPOW10, flat cells a 1× SINEPOW10, and cells with a
// strong Scharr edge response keep the narrow Mitchell-smooth primary.
const FILT_ADAPT_LOW: u32 = 256u;
const FILT_ADAPT_SMOOTH: u32 = 384u;
const FILT_ADAPT_DETAIL: u32 = 512u;

// The JWildfire raster keeps colours on the RenderColor scale (palette·200/256); ours are ·255,
// so rescale before the log10 or the edge threshold would sit ~0.1 decades off.
fn adaptLum(cx: i32, cy: i32) -> f32 {
  let c = cell(cx, cy);
  return (0.299 * c.r + 0.588 * c.g + 0.113 * c.b) * (199.21875 / 255.0 / 255.0);
}

// Scharr magnitude, reproducing JWildfire's kernel verbatim — including its x-pass sampling
// (x+1, y−1) twice where (x+1, y) would be symmetric. Parity beats tidiness here.
fn adaptScharr(x: i32, y: i32) -> f32 {
  let sx = 3.0 * adaptLum(x - 1, y - 1) - 3.0 * adaptLum(x + 1, y - 1)
         + 10.0 * adaptLum(x - 1, y) - 10.0 * adaptLum(x + 1, y - 1)
         + 3.0 * adaptLum(x - 1, y + 1) - 3.0 * adaptLum(x + 1, y + 1);
  let sy = 3.0 * adaptLum(x - 1, y - 1) + 10.0 * adaptLum(x, y - 1) + 3.0 * adaptLum(x + 1, y - 1)
         - 3.0 * adaptLum(x - 1, y + 1) - 10.0 * adaptLum(x, y + 1) - 3.0 * adaptLum(x + 1, y + 1);
  return sqrt(sx * sx + sy * sy);
}

// LogScaleCalculator for one raw cell count (no density estimation, no palette scaling).
fn adaptIntensity(count: f32) -> f32 {
  let contrast = max(T.jw.y, 1e-3);
  let d = count / T.spp;
  let glow = T.jw.w / (contrast * T.spp) / (count + 1.0);
  return 2.0 * contrast * T.brightness * 0.43429448 * log(1.0 + d / (contrast * T.jw.x)) + glow;
}

/** Returns (weight offset in sfilt, kernel size) for output pixel (x, y). */
fn adaptSelect(x: i32, y: i32) -> vec2u {
  let os = max(i32(T.de.w + 0.5), 1);
  let rx = x * os;
  let ry = y * os;
  let c = cell(rx, ry);
  if (c.w < 1.0) { return vec2u(0u, T.filterN); } // empty cell keeps the primary kernel
  let whiteLevel = 199.21875 / max(T.jw.z, 1e-9);
  if (adaptIntensity(c.w) * whiteLevel < T.post.z) { return vec2u(FILT_ADAPT_LOW, u32(T.adapt.y)); }
  if (log(1.0 + adaptScharr(rx, ry)) * 0.43429448 < T.post.w) { return vec2u(FILT_ADAPT_SMOOTH, u32(T.adapt.z)); }
  return vec2u(FILT_ADAPT_DETAIL, u32(T.adapt.w));
}

// Density-estimation + JWildfire log scale for output pixel (x, y):
// returns (r, g, b, intensity), all zero where nothing landed.
fn logScaled(x: i32, y: i32) -> vec4f {
  let c0 = blockAt(x, y);
  var rgb = c0.rgb;
  var cnt = c0.w;

  // Density estimation (JWildfire DeCalculator): average neighbours of
  // *similar* density, weighted 1/(d²+1); a neighbour is accepted when
  // |erf((n - c)/m1)| <= deCurve^dist · m2 with m1 = sqrt(8c)+5, m2 = (c+1)^-¼.
  // Empty/sparse pixels gather from a wide area (soft glow), dense ones stay sharp.
  let R = i32(T.de.x + 0.5);
  if (R > 0) {
    let os2 = max(T.de.w, 1.0) * max(T.de.w, 1.0);
    let cw = c0.w * os2;               // raw hits (per output pixel)
    let m1 = sqrt(8.0 * cw) + 5.0;
    let m2 = pow(cw + 1.0, -0.25);
    let lnc = log(max(T.de.y, 1e-4));  // deCurve
    var sumRGB = vec3f(0.0);
    var sumA = 0.0;
    var wsum = 0.0;
    for (var dy = -R; dy <= R; dy = dy + 1) {
      for (var dx = -R; dx <= R; dx = dx + 1) {
        let d2 = f32(dx * dx + dy * dy);
        let cc = blockAt(x + dx, y + dy);
        let la = cc.w * os2;
        let dev = abs(erf1((la - cw) / m1));
        if (dev <= exp(lnc * sqrt(d2)) * m2) {
          let wgt = 1.0 / (d2 + 1.0);
          sumRGB += cc.rgb * wgt;
          sumA += cc.w * wgt;
          wsum += wgt;
        }
      }
    }
    if (wsum > 1e-9) {
      rgb = sumRGB / wsum;
      // JWildfire stores the estimated density as an int: deCount = (int)(sumA + 0.5).
      // Below half a hit the pixel is empty — that is what keeps isolated stray samples
      // from spreading into a visible speckle haze; the colour sums stay unrounded.
      cnt = floor(sumA / wsum * os2 + 0.5) / os2;
    }
  }

  if (cnt <= 0.0) {
    return vec4f(0.0);
  }
  // ---- JWildfire / flam3 log-density (LogScaleCalculator) ----
  // hits per output pixel relative to the expected count (quality), then
  // divided by the world area the image covers, so the curve is zoom-invariant:
  //   intensity = 2·contrast·brightness·log10(1 + d/(contrast·area)) + glow/(hits+1)
  let os = max(T.de.w, 1.0);
  let hits = cnt * os * os;             // blockAt averages the os×os block
  let d = hits / T.spp;
  let contrast = max(T.jw.y, 1e-3);
  let k1 = 2.0 * contrast * T.brightness;
  let glow = T.jw.w / (contrast * T.spp) / (hits + 1.0);
  let a = k1 * 0.43429448 * log(1.0 + d / (contrast * T.jw.x)) + glow;
  let avg = rgb / (255.0 * cnt);        // mean palette color, 0..1
  return vec4f(avg * a * T.jw.z, a);    // ×199.2/whiteLevel (JWildfire palette scaling)
}

// LogDensityFilter.calculateBGColor: single colour, bilinear 2×2 gradient, or the 2×2 + centre variant (four bilinear
// quadrants meeting at the centre colour; JWildfire's w2 = W/2 − 1 quadrant size), on 0..255 rounded values
fn bgAt(x: i32, y: i32) -> vec3f {
  let kind = i32(T.bgUL.w + 0.5);
  if (kind == 0) { return T.bg.rgb; }
  let px = f32(x) + T.bgGeom.x; let py = f32(y) + T.bgGeom.y;
  let W = T.bgGeom.z; let H = T.bgGeom.w;
  let UL = floor(T.bgUL.rgb * 255.0 + 0.5); let UR = floor(T.bgUR.rgb * 255.0 + 0.5);
  let LL = floor(T.bgLL.rgb * 255.0 + 0.5); let LR = floor(T.bgLR.rgb * 255.0 + 0.5); let CC = floor(T.bgCC.rgb * 255.0 + 0.5);
  var c00 = UL; var c10 = UR; var c01 = LL; var c11 = LR;
  var tx: f32; var ty: f32;
  if (kind == 1) {
    tx = px / (W - 1.0); ty = py / (H - 1.0);
  } else {
    let w2 = floor(W / 2.0) - 1.0; let h2 = floor(H / 2.0) - 1.0;
    let left = px <= w2; let top = py <= h2;
    tx = select((px - w2) / w2, px / w2, left); ty = select((py - h2) / h2, py / h2, top);
    if (left && top) { c00 = UL; c10 = mix(UL, UR, 0.5); c01 = mix(LL, UL, 0.5); c11 = CC; }
    else if (left) { c00 = mix(UL, LL, 0.5); c10 = CC; c01 = LL; c11 = mix(LL, LR, 0.5); }
    else if (top) { c00 = mix(UL, UR, 0.5); c10 = UR; c01 = CC; c11 = mix(UR, LR, 0.5); }
    else { c00 = CC; c10 = mix(UR, LR, 0.5); c01 = mix(LL, LR, 0.5); c11 = LR; }
  }
  return floor(mix(mix(c00, c10, tx), mix(c01, c11, tx), ty) + 0.5) / 255.0;
}

@fragment
fn fsA(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let x = i32(fragPos.x);
  let y = i32(fragPos.y);
  if (x >= i32(T.width) || y >= i32(T.height) || T.spp <= 0.0) { return vec4f(0.0); }
  return logScaled(x, y);
}

fn mid(x: i32, y: i32) -> vec4f {
  let px = clamp(x, 0, i32(T.width) - 1);
  let py = clamp(y, 0, i32(T.height) - 1);
  return textureLoad(midTex, vec2i(px, py), 0);
}

@fragment
fn fsB(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let transparent = T.de.z > 0.5;
  let x = i32(fragPos.x);
  let y = i32(fragPos.y);
  let bgc = bgAt(x, y);
  let bgOut = select(vec4f(bgc, 1.0), vec4f(0.0), transparent);
  if (x >= i32(T.width) || y >= i32(T.height) || T.spp <= 0.0) {
    return bgOut;
  }
  // Spatial filter (JWildfire LogDensityFilter) over the log-scaled image:
  // colours with the primary kernel, intensity with its own (smoothing) kernel.
  var v = vec4f(0.0);
  var cOff = 0u;
  var NC = i32(T.filterN);
  if (T.adapt.x > 0.5) {
    let sel = adaptSelect(x, y);
    cOff = sel.x;
    NC = i32(sel.y);
  }
  let NI = i32(T.filterNI);
  if (NC <= 1 && NI <= 1) {
    v = mid(x, y);
  } else {
    let N = max(NC, NI);
    let h = N / 2;
    let hc = NC / 2;
    let hi = NI / 2;
    for (var j = -h; j <= h; j = j + 1) {
      for (var i = -h; i <= h; i = i + 1) {
        var wc = 0.0;
        var wi = 0.0;
        if (NC > 1 && abs(i) <= hc && abs(j) <= hc) { wc = sfilt[cOff + u32((j + hc) * NC + (i + hc))]; }
        if (NI > 1 && abs(i) <= hi && abs(j) <= hi) { wi = sfilt[128u + u32((j + hi) * NI + (i + hi))]; }
        if (NC <= 1 && i == 0 && j == 0) { wc = 1.0; }
        if (NI <= 1 && i == 0 && j == 0) { wi = 1.0; }
        if (wc != 0.0 || wi != 0.0) {
          let s = mid(x + i, y + j);
          v += vec4f(s.rgb * wc, s.w * wi);
        }
      }
    }
  }
  let a = v.w;
  if (a <= 0.0) {
    return bgOut;
  }
  let crgb = v.rgb;

  // Gamma with flam3's gamma_threshold: a linear ramp below the threshold so
  // gamma never amplifies single-sample speckle into visible noise.
  let g = 1.0 / max(T.gamma, 0.1);
  let thr = T.bg.w;
  var aG: f32;
  if (thr > 0.0 && a < thr) {
    let frac = a / thr;
    aG = (1.0 - frac) * a * (pow(thr, g) / thr) + frac * pow(a, g);
  } else {
    aG = pow(max(a, 0.0), g);
  }

  // flam3 vibrancy: scale channels by the gamma'd alpha ratio (saturation-
  // preserving) blended against per-channel gamma.
  let ls2 = select(0.0, aG / a, a > 0.0);
  var col = mix(
    pow(max(crgb, vec3f(0.0)), vec3f(g)),
    crgb * ls2,
    clamp(T.vibrancy, 0.0, 1.0)
  );

  // col is already alpha-scaled (flam3/JWildfire: logScl = alpha / intensity),
  // so composite as col + bg*(1 - alpha), NOT mix(bg, col, alpha), which would
  // apply alpha twice. Per-channel clip like JWildfire (whiteLevel < 255 fades to white).
  // JWildfire scales only the ALPHA by the foreground-opacity curve; the colour keeps the
  // unscaled log-scale factor, so fg_opacity shows more or less background through the flame.
  let alpha = clamp(aG * T.post.y, 0.0, 1.0);
  let ccol = clamp(col, vec3f(0.0), vec3f(1.0));
  if (transparent) {
    // straight (un-premultiplied) colour for the PNG alpha channel
    let straight = select(ccol, clamp(col / max(alpha, 1e-6), vec3f(0.0), vec3f(1.0)), alpha > 1e-6);
    return vec4f(modSaturation(straight, T.post.x), alpha);
  }
  return vec4f(modSaturation(clamp(ccol + bgc * (1.0 - alpha), vec3f(0.0), vec3f(1.0)), T.post.x), 1.0);
}
`;
