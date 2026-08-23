// .flame XML interop — the flam3 / Apophysis / JWildfire interchange format.
//
// Notes on the format:
//  - coefs / post attribute order is "a d b e c f" (flam3 stores the matrix
//    column-wise), while our Affine is [a,b,c,d,e,f] with x' = a·x + b·y + c.
//  - variations are plain attributes on <xform> (linear="1" spherical="0.5"),
//    parametric variation params are `${name}_${param}` attributes.
//  - color speed: modern files use color_speed, old ones symmetry
//    (color_speed = (1 - symmetry) / 2).
//  - xaos is the chaos="…" attribute (row of multipliers).
//  - palettes come in three flavors: <palette count format>hex</palette>,
//    <colors count data="…"/>, or 256 × <color index rgb="r g b"/>.

import type { Flame, XForm, Layer, Affine, RGB, WeightingField, LightDiffFunc } from './flame';
import { defaultXForm, defaultFlame, normalizeFlame, MAX_LAYERS, MAX_XFORMS, WFIELD_TYPES, defaultWeightingField, defaultSolidRender, defaultSolidMaterial, LIGHT_DIFF_FUNCS } from './flame';
import { VARIATIONS } from './variations';
import { normFilterKernel } from '../gpu/filters';
import type { MotionCurve, CurvePoint, CurveInterp } from './motion';

const NON_VARIATION_ATTRS = new Set([
  'weight', 'color', 'symmetry', 'color_speed', 'opacity', 'coefs', 'post',
  'chaos', 'animate', 'motion_frequency', 'motion_function', 'var_color',
  'name', 'mirror_pre_post_translations',
  // JWildfire per-xform extras we do not model (mod_* colour modifiers ARE modelled: colorMods)
  'material', 'material_speed', 'mod_gamma', 'mod_gamma_speed', 'mod_contrast', 'mod_contrast_speed',
  'mod_saturation', 'mod_saturation_speed', 'mod_hue', 'mod_hue_speed', 'color_type', 'target_color', 'targetcolor',
  'draw_mode', 'weighting_field_type', 'weighting_field_input', 'yzCoefs', 'zxCoefs', 'yzPost', 'zxPost',
]);
/** JWildfire writes `<var>_fx_priority` per instance (priority override); we keep definition priorities. */
const isFxPriorityAttr = (n: string) => n.endsWith('_fx_priority');
/** JWildfire "ressources" (`<var>_<name>`, the value hex-encoded UTF-8): file names of meshes/maps. Kept in
 *  `VarInstance.res` when the definition lists the name; we store the file's basename (JWildfire writes the
 *  author's full path, useless elsewhere; obj_mesh_wf looks the basename up in the browser's mesh store). */
const resNameOf = (attrName: string, vname: string): string | undefined => VARIATIONS[vname]?.res?.find((r) => attrName.endsWith('_' + r));
const decodeRes = (hex: string, name: string): string => {
  const h = hex.trim();
  let s = h;
  if (/^([0-9a-fA-F]{2})+$/.test(h)) {
    const bytes = new Uint8Array(h.length / 2);
    for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(h.slice(2 * i, 2 * i + 2), 16);
    s = new TextDecoder().decode(bytes);
  } // (else a plain string; be lenient)
  return name.endsWith('_filename') ? s.slice(Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\')) + 1) : s;
};
const encodeRes = (s: string): string => Array.from(new TextEncoder().encode(s), (b) => b.toString(16).toUpperCase().padStart(2, '0')).join('');

/** JWildfire 3D-flavored variation names whose z=0 projections map onto our 2D
 *  set — lets JWildfire's bundled sample flames load with close fidelity. */
const VAR_ALIASES: Record<string, string> = {
  linear3D: 'linear',
  spherical3D: 'spherical',
  blur3D: 'gaussian_blur',
  julia3D: 'julian',
  julia3Dz: 'julian',
  curl3D: 'curl',
  bubble2: 'bubble',
  boarders: 'linear',
};

const nums = (s: string | null): number[] =>
  (s ?? '').trim().split(/\s+/).map(parseFloat).filter((v) => isFinite(v));

function parseCoefs(s: string | null, fallback: Affine): Affine {
  const v = nums(s);
  if (v.length !== 6) return [...fallback] as Affine;
  // XML order: a d b e c f
  return [v[0], v[2], v[4], v[1], v[3], v[5]];
}

function coefsToXML(a: Affine): string {
  return [a[0], a[3], a[1], a[4], a[2], a[5]].map(fmt).join(' ');
}

const fmt = (v: number) => {
  const r = Math.round(v * 1e6) / 1e6;
  return Object.is(r, -0) ? '0' : String(r);
};

/** Variation names the last import could not resolve (reset per parseFlameXML call). */
export const lastImportUnknown: string[] = [];
const noteUnknown = (name: string) => { if (!lastImportUnknown.includes(name)) lastImportUnknown.push(name); };
/** Motion curves found by the last import (JWildfire `<prop>Curve_*` attributes), as WilderFire curves. */
export const lastImportCurves: MotionCurve[] = [];

// ---- JWildfire motion curves ----
// JWildfire serialises a curve as a family of attributes sharing a prefix:
//   <prefix>_enabled, _view_xmin/xmax/ymin/ymax, _interpolation (SPLINE|BEZIER|LINEAR),
//   _selected_idx, _locked, _point_count, _x<i> (frame, int), _y<i> (value)
// The prefix is `<javaField>Curve` for flame/layer/xform fields (camPitchCurve,
// weightCurve, xyCoeff00Curve, …), `<var>_amountCurve` for a variation weight and
// `<var>_<param>` for a variation parameter. Time is frames at the flame's `fps`.
const CURVE_SUFFIX_RE = /^(.+)_(enabled|view_xmin|view_xmax|view_ymin|view_ymax|interpolation|selected_idx|locked|point_count|parent_curve|x\d+|y\d+)$/;
interface RawCurve { prefix: string; points: CurvePoint[]; interp: CurveInterp; enabled: boolean }

/** Prefixes of every curve on the element (those with a `_point_count`). */
function curvePrefixes(elm: Element): Set<string> {
  const s = new Set<string>();
  for (const a of Array.from(elm.attributes)) if (a.name.endsWith('_point_count')) s.add(a.name.slice(0, -'_point_count'.length));
  return s;
}
function isCurveAttr(name: string, prefixes: Set<string>): boolean {
  const m = CURVE_SUFFIX_RE.exec(name);
  return !!m && prefixes.has(m[1]);
}
function readCurves(elm: Element, fps: number): RawCurve[] {
  const out: RawCurve[] = [];
  for (const prefix of curvePrefixes(elm)) {
    const get = (s: string) => elm.getAttribute(`${prefix}_${s}`);
    const n = parseInt(get('point_count') ?? '0');
    if (!isFinite(n) || n <= 0) continue;
    const points: CurvePoint[] = [];
    for (let i = 0; i < n; i++) {
      const x = parseFloat(get('x' + i) ?? ''), y = parseFloat(get('y' + i) ?? '');
      if (isFinite(x) && isFinite(y)) points.push({ t: x / fps, v: y });
    }
    if (!points.length) continue;
    const ji = (get('interpolation') ?? 'SPLINE').toUpperCase();
    const interp: CurveInterp = ji === 'LINEAR' ? 'linear' : 'spline';
    out.push({ prefix, points, interp, enabled: get('enabled') !== 'false' });
  }
  return out;
}
const pushCurve = (raw: RawCurve, path: string, map: (v: number) => number = (v) => v) => {
  lastImportCurves.push({ path, points: raw.points.map((p) => ({ t: p.t, v: map(p.v) })), interp: raw.interp, enabled: raw.enabled });
};
/** affine index for JWildfire xyCoeff{00,01,10,11,20,21} (XML "a d b e c f" order). */
const COEFF_IDX: Record<string, number> = { '00': 0, '01': 3, '10': 1, '11': 4, '20': 2, '21': 5 };
const XFORM_CURVE_MAP: Record<string, { key: string; map?: (v: number) => number }> = {
  weightCurve: { key: 'weight' }, colorCurve: { key: 'color' }, opacityCurve: { key: 'opacity' },
  colorSymmetryCurve: { key: 'colorSpeed', map: (v) => (1 - v) / 2 },
};
for (const [ij, idx] of Object.entries(COEFF_IDX)) {
  XFORM_CURVE_MAP[`xyCoeff${ij}Curve`] = { key: `affine.${idx}` };
  XFORM_CURVE_MAP[`xyPostCoeff${ij}Curve`] = { key: `post.${idx}` };
  XFORM_CURVE_MAP[`yzCoeff${ij}Curve`] = { key: `yz.${idx}` };
  XFORM_CURVE_MAP[`yzPostCoeff${ij}Curve`] = { key: `yzPost.${idx}` };
  XFORM_CURVE_MAP[`zxCoeff${ij}Curve`] = { key: `zx.${idx}` };
  XFORM_CURVE_MAP[`zxPostCoeff${ij}Curve`] = { key: `zxPost.${idx}` };
}

/** Curve-import context: where this element lives in the Flame + timebase. */
interface CurveCtx { pathBase: string; fps: number }

function parseXFormEl(elm: Element, ctx?: CurveCtx): XForm {
  const x = defaultXForm();
  x.variations = [];
  x.affine = parseCoefs(elm.getAttribute('coefs'), x.affine);
  x.post = parseCoefs(elm.getAttribute('post'), x.post);
  // JWildfire 3D affines (same "00 01 10 11 20 21" order); identity → omitted
  const ID: Affine = [1, 0, 0, 0, 1, 0];
  for (const [attr, key] of [['yzCoefs', 'yz'], ['zxCoefs', 'zx'], ['yzPost', 'yzPost'], ['zxPost', 'zxPost']] as const) {
    const a = parseCoefs(elm.getAttribute(attr), ID);
    if (a.some((v, i) => Math.abs(v - ID[i]) > 1e-12)) x[key] = a;
  }
  const wa = parseFloat(elm.getAttribute('weight') ?? '');
  if (isFinite(wa)) x.weight = Math.max(wa, 0);
  const ca = parseFloat(elm.getAttribute('color') ?? '');
  if (isFinite(ca)) x.color = Math.min(1, Math.max(0, ca));
  const oa = parseFloat(elm.getAttribute('opacity') ?? '');
  if (isFinite(oa)) x.opacity = Math.min(1, Math.max(0, oa));
  const csA = parseFloat(elm.getAttribute('color_speed') ?? '');
  const symA = parseFloat(elm.getAttribute('symmetry') ?? '');
  if (isFinite(csA)) x.colorSpeed = Math.min(1, Math.max(0, csA));
  else if (isFinite(symA)) x.colorSpeed = Math.min(1, Math.max(0, (1 - symA) / 2));
  // JWildfire colour types: an xform recolours (color/symmetry) only as DIFFUSION (the default for
  // normal xforms) — a *final* xform defaults to NONE and leaves the colour alone; explicit NONE
  // (and the target/distance/cyclic modes we do not model) never blend towards `color`
  const ctype = (elm.getAttribute('color_type') ?? '').toUpperCase();
  if (ctype === 'CYCLIC' || ctype === 'DISTANCE' || ctype === 'TARGET' || ctype === 'TARGETG') x.colorType = ctype; // symmetry (1 − 2·colorSpeed) is their parameter
  else if (ctype === 'NONE' || (elm.tagName.toLowerCase() === 'finalxform' && ctype !== 'DIFFUSION')) x.colorSpeed = 0;
  if (ctype === 'TARGET' || ctype === 'TARGETG') { const tc = nums(elm.getAttribute('targetcolor')); x.targetColor = tc.length === 3 ? tc.map((v) => Math.min(1, Math.max(0, v))) as [number, number, number] : [0, 0, 0]; }
  const chaos = nums(elm.getAttribute('chaos'));
  if (chaos.length) x.xaos = chaos.map((v) => Math.max(0, v));
  // JWildfire solid-rendering material index (+ blend speed), like colour/color_speed
  const matA = parseFloat(elm.getAttribute('material') ?? '');
  if (isFinite(matA) && matA !== 0) x.material = matA;
  const matS = parseFloat(elm.getAttribute('material_speed') ?? '');
  if (isFinite(matS) && matS !== 0) x.materialSpeed = Math.min(1, Math.max(-1, matS));
  // JWildfire colour modifiers (transform "Color" tab): gamma/contrast/saturation/hue with blend speeds
  const mods = ['mod_gamma', 'mod_gamma_speed', 'mod_contrast', 'mod_contrast_speed', 'mod_saturation', 'mod_saturation_speed', 'mod_hue', 'mod_hue_speed']
    .map((a) => { const v = parseFloat(elm.getAttribute(a) ?? ''); return isFinite(v) ? v : 0; });
  if (mods.some((v) => v !== 0)) x.colorMods = mods;
  // JWildfire weighting field (wfield_* attributes; enum names as written by JWildfire)
  const wft = (elm.getAttribute('wfield_type') ?? 'NONE').toUpperCase();
  if (WFIELD_TYPES.includes(wft)) {
    const numA = (a: string, d: number) => { const v = parseFloat(elm.getAttribute(a) ?? ''); return isFinite(v) ? v : d; };
    const strA = (a: string, d: string) => (elm.getAttribute(a) ?? d).toUpperCase();
    const wf = defaultWeightingField(wft);
    wf.input = strA('wfield_input', 'AFFINE') === 'POSITION' ? 'POSITION' : 'AFFINE';
    wf.varAmount = numA('wfield_var_amount_intensity', 0); wf.color = numA('wfield_color_intensity', 0); wf.jitter = numA('wfield_jitter_intensity', 0);
    wf.seed = Math.round(numA('wfield_noise_seed', 1337)); wf.frequency = numA('wfield_noise_frequency', 1);
    wf.fractalType = strA('wfield_fract_noise_fract_type', 'FBM') as WeightingField['fractalType'];
    wf.octaves = Math.round(numA('wfield_fract_noise_octaves', 3)); wf.gain = numA('wfield_fract_noise_gain', 0.5); wf.lacunarity = numA('wfield_fract_noise_lacunarity', 2);
    wf.cellReturn = strA('wfield_cell_noise_return_type', 'DISTANCE2') as WeightingField['cellReturn'];
    wf.cellDistance = strA('wfield_cell_noise_dist_function', 'EUCLIDIAN') as WeightingField['cellDistance'];
    for (let i = 1; i <= 3; i++) {
      const it = numA(`wfield_var_param${i}_intensity`, 0), vn = elm.getAttribute(`wfield_var_param${i}_var_name`) ?? '', pn = elm.getAttribute(`wfield_var_param${i}_param_name`) ?? '';
      if (Math.abs(it) > 1e-9 && vn) wf.params.push({ varName: vn, paramName: pn, intensity: it });
    }
    x.wfield = wf;
  }

  // Variation weights + params from remaining attributes. Names prefixed
  // pre_/post_ (flam3/JWildfire pre_blur, post_curl, …) route to the stages.
  type Stage = 'main' | 'pre' | 'post';
  const known = (n: string): string | null =>
    VARIATIONS[n] ? n : (VAR_ALIASES[n] && VARIATIONS[VAR_ALIASES[n]] ? VAR_ALIASES[n] : null);
  const resolve = (name: string): { stage: Stage; vname: string } | null => {
    const direct = known(name);
    if (direct) return { stage: 'main', vname: direct };
    if (name.startsWith('pre_')) {
      const v = known(name.slice(4));
      if (v) return { stage: 'pre', vname: v };
    }
    if (name.startsWith('post_')) {
      const v = known(name.slice(5));
      if (v) return { stage: 'post', vname: v };
    }
    return null;
  };
  const params: Record<string, Record<string, number>> = {}; // keyed by raw attr prefix
  const weights: { raw: string; stage: Stage; vname: string; weight: number }[] = [];
  const unresolved: string[] = [];
  const cprefixes = curvePrefixes(elm);
  const isResAttr = (n: string): boolean => {
    for (let us = n.lastIndexOf('_'); us > 0; us = n.lastIndexOf('_', us - 1)) {
      const r = resolve(n.slice(0, us));
      if (r && resNameOf(n, r.vname) === n.slice(us + 1)) return true;
    }
    return false;
  };
  for (const attr of Array.from(elm.attributes)) {
    // Duplicate variation instances on one xform were renamed name__dup<k> by
    // the lenient pre-parser; strip the marker to resolve them.
    const name = attr.name.replace(/__dup\d+/, '');
    if (NON_VARIATION_ATTRS.has(name) || isFxPriorityAttr(name) || name.startsWith('wfield_')) continue; // _fx_priority is read below
    if (isCurveAttr(attr.name, cprefixes)) continue;
    if (isResAttr(name)) continue; // string resources are read per instance below
    const val = parseFloat(attr.value);
    if (!isFinite(val)) continue;
    const direct = resolve(name);
    if (direct) {
      weights.push({ raw: attr.name, ...direct, weight: val });
      continue;
    }
    // `${var}_${param}` — params may themselves contain underscores, so try
    // every split point against the known variations' parameter lists.
    let matched = false;
    for (let us = name.lastIndexOf('_'); us > 0; us = name.lastIndexOf('_', us - 1)) {
      const vn = name.slice(0, us);
      const pn = name.slice(us + 1);
      const res = resolve(vn);
      // (a param name with spaces/punctuation — "Density Pixels" — arrives sanitised by the lenient pre-parser)
      const pd = VARIATIONS[res?.vname ?? '']?.params?.find((p) => p.name === pn || sanitizeAttrName(p.name) === pn);
      if (res && pd) {
        (params[attr.name.slice(0, attr.name.length - name.length + us)] ??= {})[pd.name] = val;
        matched = true;
        break;
      }
    }
    if (!matched) unresolved.push(name);
  }
  // Report unknown variations (drop their `${var}_${param}` attributes from the list).
  for (const u of unresolved) {
    if (unresolved.some((o) => o !== u && u.startsWith(o + '_'))) continue;
    noteUnknown(u);
  }
  const instByRaw = new Map<string, { inst: XForm['variations'][number]; list: 'variations' | 'preVariations' | 'postVariations' }>();
  for (const { raw, stage, vname, weight } of weights) {
    // JWildfire applies a variation whatever its amount; a zero amount only matters for variations whose effect does
    // not scale with it (pre/post steps that rewrite the point, hide/direct-colour/stateful ones — e.g. `pre_stabilize="0"`,
    // `post_mirror_wf="0"` in the wild) — plain sums contribute nothing and are dropped to keep flames tidy
    const vdef = VARIATIONS[vname];
    if (weight === 0 && (vdef.priority ?? 0) === 0 && !vdef.flags?.some((f) => f === 'hide' || f === 'dc' || f === 'state' || f === 'stateful')) continue;
    const p: Record<string, number> = {};
    for (const pd of VARIATIONS[vname].params ?? []) {
      p[pd.name] = params[raw]?.[pd.name] ?? pd.def;
    }
    const inst: XForm['variations'][number] = { name: vname, weight, params: p };
    // JWildfire per-instance priority override (`<var>_fx_priority`), kept only when it differs from the definition
    const fxp = parseFloat(elm.getAttribute(`${raw}_fx_priority`) ?? '');
    if (isFinite(fxp) && Math.round(fxp) !== (VARIATIONS[vname].priority ?? 0)) inst.priority = Math.round(fxp);
    for (const r of VARIATIONS[vname].res ?? []) {
      const rv = decodeRes(elm.getAttribute(`${raw}_${r}`) ?? '', r);
      if (rv && rv !== VARIATIONS[vname].resDef?.[r]) (inst.res ??= {})[r] = rv;
    }
    if (stage === 'pre') { (x.preVariations ??= []).push(inst); instByRaw.set(raw, { inst, list: 'preVariations' }); }
    else if (stage === 'post') { (x.postVariations ??= []).push(inst); instByRaw.set(raw, { inst, list: 'postVariations' }); }
    else { x.variations.push(inst); instByRaw.set(raw, { inst, list: 'variations' }); }
  }
  // flam3 pre/post variations ADD to a pass-through point; our stages are pure
  // sums, so inject linear 1 to reproduce that unless one is already present.
  for (const list of [x.preVariations, x.postVariations]) {
    if (list?.length && !list.some((v) => v.name === 'linear')) {
      list.unshift({ name: 'linear', weight: 1, params: {} });
    }
  }
  if (!x.variations.length) {
    x.variations.push({ name: 'linear', weight: 1, params: {} });
  }
  // Motion curves on this xform → WilderFire curve paths.
  if (ctx && cprefixes.size) {
    for (const rc of readCurves(elm, ctx.fps)) {
      const m = XFORM_CURVE_MAP[rc.prefix];
      if (m) { pushCurve(rc, `${ctx.pathBase}.${m.key}`, m.map); continue; }
      // `<raw>_amountCurve` (weight) or `<raw>_<param>`
      let raw: string | null = null, key = '';
      if (rc.prefix.endsWith('_amountCurve')) { raw = rc.prefix.slice(0, -'_amountCurve'.length); key = 'weight'; }
      else {
        for (const r of instByRaw.keys()) {
          if (rc.prefix.startsWith(r + '_')) {
            const pn = rc.prefix.slice(r.length + 1);
            const e = instByRaw.get(r)!;
            if (VARIATIONS[e.inst.name]?.params?.some((pd) => pd.name === pn)) { raw = r; key = `params.${pn}`; break; }
          }
        }
      }
      const e = raw ? instByRaw.get(raw) : undefined;
      if (!e) continue;
      const idx = (x[e.list] ?? []).indexOf(e.inst);
      if (idx >= 0) pushCurve(rc, `${ctx.pathBase}.${e.list}.${idx}.${key}`);
    }
  }
  return x;
}

function parsePaletteEl(flameEl: Element): RGB[] | null {
  // <palette count="256" format="RGB"> hex... </palette>
  const pal = flameEl.querySelector('palette');
  if (pal) {
    const hex = (pal.textContent ?? '').replace(/[^0-9a-fA-F]/g, '');
    const stride = (pal.getAttribute('format') ?? 'RGB').toUpperCase() === 'RGB' ? 6 : 8;
    const colors: RGB[] = [];
    for (let i = 0; i + stride <= hex.length; i += stride) {
      const s = stride === 8 ? hex.slice(i + 2, i + 8) : hex.slice(i, i + 6);
      colors.push([
        parseInt(s.slice(0, 2), 16) / 255,
        parseInt(s.slice(2, 4), 16) / 255,
        parseInt(s.slice(4, 6), 16) / 255,
      ]);
    }
    if (colors.length >= 2) return resample(colors);
  }
  // <colors count="256" data="00RRGGBB..."/>
  const colorsEl = flameEl.querySelector('colors');
  if (colorsEl) {
    const hex = (colorsEl.getAttribute('data') ?? '').replace(/[^0-9a-fA-F]/g, '');
    const colors: RGB[] = [];
    for (let i = 0; i + 8 <= hex.length; i += 8) {
      colors.push([
        parseInt(hex.slice(i + 2, i + 4), 16) / 255,
        parseInt(hex.slice(i + 4, i + 6), 16) / 255,
        parseInt(hex.slice(i + 6, i + 8), 16) / 255,
      ]);
    }
    if (colors.length >= 2) return resample(colors);
  }
  // 256 × <color index="i" rgb="r g b"/>
  const colorEls = Array.from(flameEl.querySelectorAll('color'));
  if (colorEls.length >= 2) {
    const out: RGB[] = new Array(256).fill(null).map(() => [0, 0, 0] as RGB);
    for (const ce of colorEls) {
      const idx = parseInt(ce.getAttribute('index') ?? '');
      const rgb = nums(ce.getAttribute('rgb'));
      if (isFinite(idx) && idx >= 0 && idx < 256 && rgb.length === 3) {
        out[idx] = [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255];
      }
    }
    return out;
  }
  return null;
}

function resample(colors: RGB[]): RGB[] {
  if (colors.length === 256) return colors;
  const out: RGB[] = [];
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (colors.length - 1);
    const k = Math.min(colors.length - 2, Math.floor(t));
    const u = t - k;
    out.push([0, 1, 2].map((c) => colors[k][c] + (colors[k + 1][c] - colors[k][c]) * u) as RGB);
  }
  return out;
}

/** What the lenient pre-parser turns a non-XML attribute name into (the importer matches parameter
 *  names through the same function). */
const sanitizeAttrName = (name: string) => name.trim().replace(/[^\w-]/g, '_');

/** JWildfire writes one attribute per variation instance, so an xform with two
 *  `bubble` variations yields a duplicate attribute (invalid XML). Rename the
 *  later occurrences (and their trailing params) to `name__dup<k>`. */
function dedupeAttributes(text: string): string {
  // JWildfire's own marker for repeated instances: bubble#1#="…" bubble#1#_param="…"
  text = text.replace(/([A-Za-z_]\w*)#(\d+)#/g, (_m, n: string, k: string) => `${n}__dup${Number(k) + 1}`);
  return text.replace(/<(xform|finalxform)\b([^>]*)>/g, (m, tag: string, attrs: string) => {
    // Attribute names may contain spaces or punctuation ("foo_show/hide(1/0)",
    // "glsl_x_Density Pixels", "glsl_x_Red Fac."): tokenize on name="value" pairs and sanitize
    // (dots too — legal in XML names, but not every parser agrees about a trailing one).
    attrs = attrs.replace(/([^\s"=][^"=]*?)\s*=\s*"([^"]*)"/g, (_m, name: string, val: string) => ` ${sanitizeAttrName(name)}="${val}"`);
    const seen = new Map<string, number>();
    let cur: { base: string; k: number } | null = null;
    const out = attrs.replace(/([A-Za-z_][\w.-]*)(\s*=\s*"[^"]*")/g, (_m, name: string, rest: string) => {
      const n = seen.get(name) ?? 0;
      seen.set(name, n + 1);
      if (n > 0 && !name.includes('_')) { cur = { base: name, k: n + 1 }; return `${name}__dup${n + 1}${rest}`; }
      if (n > 0 && cur && name.startsWith(cur.base + '_')) return `${cur.base}__dup${cur.k}${name.slice(cur.base.length)}${rest}`;
      if (n > 0) { const us = name.indexOf('_'); const base = us > 0 ? name.slice(0, us) : name; cur = { base, k: n + 1 }; return `${base}__dup${n + 1}${name.slice(base.length)}${rest}`; }
      if (cur && !name.startsWith(cur.base + '_') && !name.startsWith(cur.base + '__dup')) cur = null;
      return name + rest;
    });
    return `<${tag}${out}>`;
  });
}

export function parseFlameXML(text: string, fallbackPalette: RGB[]): Flame[] {
  lastImportUnknown.length = 0;
  lastImportCurves.length = 0;
  let doc = new DOMParser().parseFromString(text, 'application/xml');
  if (doc.querySelector('parsererror')) doc = new DOMParser().parseFromString(dedupeAttributes(text), 'application/xml');
  if (doc.querySelector('parsererror')) {
    throw new Error('Not valid XML: ' + (doc.querySelector('parsererror')?.textContent ?? '').slice(0, 120));
  }
  // flam3/Apophysis/JWildfire use <flame>; newer JWildfire files may use <jwf-flame>.
  const flameEls = Array.from(doc.querySelectorAll('flame, jwf-flame'));
  if (!flameEls.length) throw new Error('No <flame> element found.');

  return flameEls.map((fe) => {
    const f = defaultFlame(fallbackPalette);
    f.name = fe.getAttribute('name') ?? 'imported';
    const size = nums(fe.getAttribute('size'));
    const sizeMin = size.length === 2 ? Math.min(size[0], size[1]) : 1024;
    const center = nums(fe.getAttribute('center'));
    if (center.length === 2) { f.centerX = center[0]; f.centerY = center[1]; }
    const scale = parseFloat(fe.getAttribute('scale') ?? '');
    // JWildfire's effective pixels-per-unit is scale × cam_zoom (LogScaleCalculator).
    const camZoom = parseFloat(fe.getAttribute('cam_zoom') ?? '');
    const zoomMul = isFinite(camZoom) && camZoom > 0 ? camZoom : 1;
    if (isFinite(scale) && scale > 0) f.zoom = (scale * zoomMul) / (0.25 * sizeMin);
    const rot = parseFloat(fe.getAttribute('rotate') ?? '');
    if (isFinite(rot)) f.rotation = (rot * Math.PI) / 180;
    // JWildfire 3D camera
    const numAttr = (n: string) => { const v = parseFloat(fe.getAttribute(n) ?? ''); return isFinite(v) ? v : 0; };
    // cam_pitch / cam_yaw / cam_roll are stored in radians; cam_roll is the bank axis
    // (the flam3 `rotate` attribute is JWildfire's camRoll, our `rotation`).
    f.camPitch = (numAttr('cam_pitch') * 180) / Math.PI;
    f.camYaw = (numAttr('cam_yaw') * 180) / Math.PI;
    f.camBank = (numAttr('cam_roll') * 180) / Math.PI;
    f.camPersp = fe.hasAttribute('cam_persp') ? numAttr('cam_persp') : numAttr('cam_perspective'); // JWildfire also reads the old cam_perspective
    f.camPosX = numAttr('cam_pos_x');
    f.camPosY = numAttr('cam_pos_y');
    f.camPosZ = numAttr('cam_pos_z');
    f.preserveZ = numAttr('preserve_z') !== 0;
    // Depth of field + dimish-z (JWildfire); defaults follow Flame's constructor
    const numOr = (n: string, d: number) => { const v = parseFloat(fe.getAttribute(n) ?? ''); return isFinite(v) ? v : d; };
    f.camDOF = Math.max(0, numAttr('cam_dof'));
    f.camDOFArea = Math.max(0, numOr('cam_dof_area', 0.5));
    f.camDOFExponent = Math.max(0.1, numOr('cam_dof_exponent', 2));
    f.camDOFScale = numOr('cam_dof_scale', 1);
    f.camDOFShape = (fe.getAttribute('cam_dof_shape') ?? 'BUBBLE').toUpperCase() || 'BUBBLE';
    f.camDOFRotate = numOr('cam_dof_rotate', 0);
    f.camDOFParams = [1, 2, 3, 4, 5, 6].map((i) => numOr('cam_dof_param' + i, 0));
    f.camDOFFade = Math.min(1, Math.max(0, numOr('cam_dof_fade', 1)));
    f.newDOF = numAttr('new_dof') !== 0;
    f.focusX = numAttr('cam_xfocus'); f.focusY = numAttr('cam_yfocus'); f.focusZ = numAttr('cam_zfocus');
    f.camZ = numAttr('cam_zpos');
    f.dimishZ = Math.max(0, numAttr('cam_zdimish'));
    f.dimZDist = numAttr('cam_zdimdist');
    const dzc = nums(fe.getAttribute('cam_zdimcolor'));
    if (dzc.length === 3) f.dimZColor = dzc.map((v) => Math.min(1, Math.max(0, v))) as RGB;
    const br = parseFloat(fe.getAttribute('brightness') ?? '');
    if (isFinite(br) && br > 0) f.brightness = Math.min(br, 1000); // JWildfire allows any value (files with 150 exist); the slider covers 0.1–8
    const ga = parseFloat(fe.getAttribute('gamma') ?? '');
    if (isFinite(ga) && ga >= 0) f.gamma = Math.min(ga, 8); // 0 = JWildfire's flat tonemap (exponent 0), kept as such
    const gt = parseFloat(fe.getAttribute('gamma_threshold') ?? '');
    if (isFinite(gt) && gt >= 0) f.gammaThreshold = Math.min(gt, 0.5);
    const vi = parseFloat(fe.getAttribute('vibrancy') ?? '');
    if (isFinite(vi)) f.vibrancy = Math.min(1, Math.max(0, vi));
    // JWildfire tonemap constants + spatial filter + antialiasing (defaults = JWildfire's)
    f.contrast = Math.max(0.05, numOr('contrast', 1));
    f.whiteLevel = Math.max(1, numOr('white_level', 220));
    f.lowDensityBrightness = Math.max(0, numOr('low_density_brightness', 0.24));
    f.filterRadius = Math.min(3, Math.max(0, numOr('filter', 0.75)));
    f.filterKernel = normFilterKernel(fe.getAttribute('filter_kernel') ?? 'MITCHELL_SMOOTH');
    // Adaptive filtering parameters (used only by the MITCHELL_SINEPOW kernel)
    f.filterSharpness = numOr('filter_sharpness', 4);
    f.filterLowDensity = numOr('filter_low_density', 0.025);
    f.antialiasAmount = Math.min(1, Math.max(0, numOr('antialias_amount', 0.25)));
    f.antialiasRadius = Math.max(0, numOr('antialias_radius', 0.5));
    f.deRadius = Math.min(2, Math.max(0, numOr('de_radius', 1)));
    f.deCurve = Math.min(1, Math.max(0.01, numOr('de_curve', 0.8)));
    // Colour/compositing settings JWildfire applies after the tonemap (GammaCorrectionFilter)
    f.saturation = Math.max(0, numOr('saturation', 1));
    f.fgOpacity = Math.max(0, numOr('fg_opacity', 1));
    f.bgTransparency = numAttr('bg_transparency') !== 0;
    f.oversample = Math.min(3, Math.max(1, Math.round(numOr('oversample', 1))));
    // Provenance (JWildfire writes these when the author filled the meta-info tab)
    const strOpt = (n: string) => { const v = (fe.getAttribute(n) ?? '').trim(); return v ? v : undefined; };
    const author = strOpt('meta_info_author'), created = strOpt('meta_info_creation_time'), uuid = strOpt('meta_info_uuid');
    if (author) f.author = author;
    if (created) f.created = created;
    if (uuid) f.uuid = uuid;
    // Post symmetry (DefaultRenderIterationState): plotted points are mirrored or rotated
    const pst = (fe.getAttribute('post_symmetry_type') ?? 'NONE').toUpperCase();
    if (pst === 'X_AXIS' || pst === 'Y_AXIS' || pst === 'POINT') {
      f.postSymmetry = {
        type: pst,
        order: Math.min(64, Math.max(1, Math.round(numOr('post_symmetry_order', 3)))),
        centreX: numOr('post_symmetry_centre_x', 0), centreY: numOr('post_symmetry_centre_y', 0),
        distance: numOr('post_symmetry_distance', 1.25), rotation: numOr('post_symmetry_rotation', 6),
      };
    }
    const bg = nums(fe.getAttribute('background'));
    if (bg.length === 3) {
      // Old files use 0-255, new ones 0-1 — sniff by magnitude.
      const mx = Math.max(...bg);
      f.background = bg.map((v) => (mx > 1 ? v / 255 : v)) as RGB;
    }
    // JWildfire background gradient (BGColorType GRADIENT_2X2 / GRADIENT_2X2_C: corner colours ul/ur/ll/lr + centre cc, 0..1)
    const bgt = (fe.getAttribute('background_type') ?? '').toUpperCase();
    if (bgt === 'GRADIENT_2X2' || bgt === 'GRADIENT_2X2_C') {
      const col = (n: string): RGB => { const c = nums(fe.getAttribute(n)); return (c.length === 3 ? c.map((v) => Math.min(1, Math.max(0, v))) : [0, 0, 0]) as RGB; };
      const g = { type: bgt as 'GRADIENT_2X2' | 'GRADIENT_2X2_C', ul: col('background_ul'), ur: col('background_ur'), ll: col('background_ll'), lr: col('background_lr'), cc: col('background_cc') };
      // a gradient whose colours are all equal (JWildfire's default file state) is just a single colour
      const flat = [g.ur, g.ll, g.lr, ...(bgt === 'GRADIENT_2X2_C' ? [g.cc] : [])].every((c) => c.every((v, i) => Math.abs(v - g.ul[i]) < 1e-9));
      if (!flat) f.bgGradient = g; else f.background = g.ul;
    }
    // JWildfire solid rendering (sld_render_* — attribute names as JWildfire writes them, typos included)
    if (numAttr('sld_render_enabled') === 1) {
      const s = defaultSolidRender(true);
      const strA = (n: string, d: string) => (fe.getAttribute(n) ?? d).toUpperCase();
      s.ao.enabled = fe.hasAttribute('sld_render_ao_enabled') ? numAttr('sld_render_ao_enabled') === 1 : s.ao.enabled;
      s.ao.intensity = numOr('sld_render_ao_intensity', s.ao.intensity);
      s.ao.searchRadius = numOr('sld_render_ao_search_radius', s.ao.searchRadius);
      s.ao.blurRadius = numOr('sld_render_ao_blur_radius', s.ao.blurRadius);
      s.ao.radiusSamples = Math.round(numOr('sld_render_ao_radius_samples', s.ao.radiusSamples));
      s.ao.azimuthSamples = Math.round(numOr('sld_render_ao_azimuth_samples', s.ao.azimuthSamples));
      s.ao.falloff = numOr('sld_render_ao_falloff', s.ao.falloff);
      s.ao.affectDiffuse = numOr('sld_render_ao_affect_diffuse', s.ao.affectDiffuse);
      const st = strA('sld_render_shadow_type', 'OFF');
      s.shadows.type = st === 'FAST' || st === 'SMOOTH' ? st : 'OFF';
      s.shadows.smoothRadius = numOr('sld_render_shadow_smooth_radius', s.shadows.smoothRadius);
      s.shadows.mapSize = Math.round(numOr('sld_render_shadowmap_size', s.shadows.mapSize));
      s.shadows.bias = numOr('sld_render_shadowmap_bias', s.shadows.bias);
      if (fe.hasAttribute('sld_render_material_count')) {
        const n = Math.max(0, Math.min(8, Math.round(numAttr('sld_render_material_count'))));
        s.materials = Array.from({ length: n }, (_, i) => {
          const m = defaultSolidMaterial();
          // JWildfire's MaterialSettings field defaults differ from setupDefaultMaterials(): a listed material starts from the field defaults
          m.diffuse = numOr(`sld_render_material_diffuse${i}`, 0.5); m.ambient = numOr(`sld_render_material_ambient${i}`, 1);
          m.phong = numOr(`sld_render_material_phong${i}`, 1); m.phongSize = numOr(`sld_render_material_phong_size${i}`, 24);
          m.phongColor = [numOr(`sld_render_material_phong_red${i}`, 0), numOr(`sld_render_material_phong_green${i}`, 0), numOr(`sld_render_material_phong_blue${i}`, 0)];
          const df = strA(`sld_render_material_light_diif_func${i}`, 'COSA');
          m.diffFunc = LIGHT_DIFF_FUNCS.includes(df as LightDiffFunc) ? df as LightDiffFunc : 'COSA';
          m.reflMapIntensity = numOr(`sld_render_material_refl_map_intensity${i}`, 0.5);
          m.reflMapping = strA(`sld_render_material_refl_mappping${i}`, 'BLINN_NEWELL') === 'SPHERICAL' ? 'SPHERICAL' : 'BLINN_NEWELL';
          // the author's full path is useless elsewhere: keep the file name, looked up in this browser's image store
          const rf = (fe.getAttribute(`sld_render_material_refl_map_filename${i}`) ?? '').replace(/^.*[\\/]/, '').trim(); // (strA upper-cases: enums only)
          if (rf) m.reflMapFilename = rf;
          return m;
        });
      }
      if (fe.hasAttribute('sld_render_ligtht_count')) {
        const n = Math.max(0, Math.min(4, Math.round(numAttr('sld_render_ligtht_count'))));
        s.lights = Array.from({ length: n }, (_, i) => ({
          // DistantLight field defaults: altitude/azimuth 0, intensity 0.5, black, shadows on, shadow intensity 0.8
          altitude: numOr(`sld_render_light_altitude${i}`, 0), azimuth: numOr(`sld_render_light_azimuth${i}`, 0),
          intensity: numOr(`sld_render_light_intensity${i}`, 0.5),
          color: [numOr(`sld_render_light_red${i}`, 0), numOr(`sld_render_light_green${i}`, 0), numOr(`sld_render_light_blue${i}`, 0)] as RGB,
          castShadows: fe.hasAttribute(`sld_render_light_shadows${i}`) ? numAttr(`sld_render_light_shadows${i}`) === 1 : true,
          shadowIntensity: numOr(`sld_render_light_shadow_intensity${i}`, 0.8),
        }));
      }
      // Post-bokeh (JWildfire PostDOFCalculator — solid post-process DOF's glint shaping; flame-level attrs)
      const bk = (fe.getAttribute('post_bokeh_filter_kernel') ?? '').toUpperCase();
      if (bk) s.postBokeh.filterKernel = bk;
      s.postBokeh.intensity = Math.max(0, numOr('post_bokeh_intensity', s.postBokeh.intensity));
      s.postBokeh.brightness = Math.max(0, numOr('post_bokeh_brightness', s.postBokeh.brightness));
      s.postBokeh.size = Math.max(0, numOr('post_bokeh_size', s.postBokeh.size));
      s.postBokeh.activation = Math.max(0, numOr('post_bokeh_activation', s.postBokeh.activation));
      f.solid = s;
    }
    // Motion curves (only the first flame's curves are surfaced)
    const fpsA = parseFloat(fe.getAttribute('fps') ?? '');
    const fps = isFinite(fpsA) && fpsA > 0 ? fpsA : 25;
    const isFirst = fe === flameEls[0];
    if (isFirst) {
      const zoomPerUnit = 1 / (0.25 * sizeMin);
      const FLAME_MAP: Record<string, { key: string; map?: (v: number) => number }> = {
        camPitchCurve: { key: 'camPitch' }, camYawCurve: { key: 'camYaw' }, camBankCurve: { key: 'camBank' },
        camPerspectiveCurve: { key: 'camPersp' }, camRollCurve: { key: 'rotation', map: (v) => (v * Math.PI) / 180 },
        centreXCurve: { key: 'centerX' }, centreYCurve: { key: 'centerY' },
        camPosXCurve: { key: 'camPosX' }, camPosYCurve: { key: 'camPosY' }, camPosZCurve: { key: 'camPosZ' },
        brightnessCurve: { key: 'brightness' }, gammaCurve: { key: 'gamma' }, gammaThresholdCurve: { key: 'gammaThreshold' },
        vibrancyCurve: { key: 'vibrancy' }, contrastCurve: { key: 'contrast' }, whiteLevelCurve: { key: 'whiteLevel' },
        camZoomCurve: { key: 'zoom', map: (v) => v * (isFinite(scale) && scale > 0 ? scale : 1) * zoomPerUnit },
        pixelsPerUnitCurve: { key: 'zoom', map: (v) => v * zoomMul * zoomPerUnit },
        camDOFCurve: { key: 'camDOF' }, camDOFAreaCurve: { key: 'camDOFArea' }, camDOFExponentCurve: { key: 'camDOFExponent' },
        camDOFScaleCurve: { key: 'camDOFScale' }, camDOFFadeCurve: { key: 'camDOFFade' },
        focusXCurve: { key: 'focusX' }, focusYCurve: { key: 'focusY' }, focusZCurve: { key: 'focusZ' }, camZCurve: { key: 'camZ' },
        dimishZCurve: { key: 'dimishZ' }, dimZDistanceCurve: { key: 'dimZDist' },
      };
      for (const rc of readCurves(fe, fps)) {
        const m = FLAME_MAP[rc.prefix];
        if (m) pushCurve(rc, m.key, m.map);
      }
    }
    const parseLayerContent = (elm: Element, li: number): Omit<Layer, 'weight' | 'visible'> => {
      const ctx = (base: string): CurveCtx | undefined => (isFirst ? { pathBase: base, fps } : undefined);
      const xf = Array.from(elm.querySelectorAll(':scope > xform')).slice(0, MAX_XFORMS)
        .map((xe, xi) => parseXFormEl(xe, ctx(`layers.${li}.xforms.${xi}`)));
      const fins = Array.from(elm.querySelectorAll(':scope > finalxform'));
      return {
        xforms: xf.length ? xf : [defaultXForm()],
        final: fins.length ? parseXFormEl(fins[0], ctx(`layers.${li}.final`)) : null,
        moreFinals: fins.slice(1).map((fe) => parseXFormEl(fe)), // JWildfire: further finals, applied in sequence
        palette: parsePaletteEl(elm) ?? fallbackPalette,
      };
    };

    // JWildfire layered flames use <layer> children; flam3/Apophysis are flat.
    const layerEls = Array.from(fe.querySelectorAll(':scope > layer')).slice(0, MAX_LAYERS);
    if (layerEls.length) {
      f.layers = layerEls.map((le, li) => {
        const w = parseFloat(le.getAttribute('weight') ?? le.getAttribute('density') ?? '');
        const visAttr = le.getAttribute('visible');
        if (isFirst) for (const rc of readCurves(le, fps)) if (rc.prefix === 'weightCurve') pushCurve(rc, `layers.${li}.weight`);
        return {
          ...parseLayerContent(le, li),
          weight: isFinite(w) && w >= 0 ? w : 1,
          visible: visAttr !== '0' && visAttr !== 'false',
        };
      });
    } else {
      f.layers = [{ ...parseLayerContent(fe, 0), weight: 1, visible: true }];
    }
    // Re-run through the normalizer for clamping / defaults.
    return normalizeFlame(f, fallbackPalette);
  });
}

// ---- export ----

export interface XMLExportOpts {
  /** motion curves to embed as JWildfire `*Curve_*` attributes */
  curves?: MotionCurve[];
  /** timebase for curve frames (JWildfire default 25) */
  fps?: number;
}

/** JWildfire attribute family for one curve. Frames are integers; duplicate frames collapse (last wins). */
function curveAttrs(prefix: string, c: MotionCurve, fps: number): string[] {
  const byFrame = new Map<number, number>();
  for (const p of [...c.points].sort((a, b) => a.t - b.t)) byFrame.set(Math.round(p.t * fps), p.v);
  const pts = Array.from(byFrame.entries());
  if (!pts.length) return [];
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const interp = c.interp === 'linear' || c.interp === 'step' ? 'LINEAR' : 'SPLINE';
  const a: string[] = [
    `${prefix}_enabled="${c.enabled === false ? 'false' : 'true'}"`,
    `${prefix}_view_xmin="${Math.min(0, ...xs) - 10}"`, `${prefix}_view_xmax="${Math.max(...xs) + 10}"`,
    `${prefix}_view_ymin="${fmt(Math.min(...ys) - 1)}"`, `${prefix}_view_ymax="${fmt(Math.max(...ys) + 1)}"`,
    `${prefix}_interpolation="${interp}"`, `${prefix}_selected_idx="0"`, `${prefix}_locked="false"`,
    `${prefix}_point_count="${pts.length}"`,
  ];
  pts.forEach(([x, y], i) => a.push(`${prefix}_x${i}="${x}"`, `${prefix}_y${i}="${fmt(y)}"`));
  return a;
}

/** Curves grouped by the element they belong to: '' (flame), 'layers.L', 'layers.L.xforms.i', 'layers.L.final'. */
type CurveBuckets = Map<string, { rest: string; curve: MotionCurve }[]>;
function bucketCurves(curves: MotionCurve[] | undefined): CurveBuckets {
  const b: CurveBuckets = new Map();
  for (const c of curves ?? []) {
    if (!c.points.length) continue;
    const m = /^layers\.(\d+)(?:\.(final|xforms\.\d+))?\.(.+)$/.exec(c.path);
    const key = m ? (m[2] ? `layers.${m[1]}.${m[2]}` : `layers.${m[1]}`) : '';
    const rest = m ? m[3] : c.path;
    (b.get(key) ?? b.set(key, []).get(key)!).push({ rest, curve: c });
  }
  return b;
}
const AFFINE_TO_COEFF = ['00', '10', '20', '01', '11', '21']; // affine idx → JWildfire xyCoeff{ij}
const FLAME_CURVE_PREFIX: Record<string, { prefix: string; map?: (v: number) => number }> = {
  camPitch: { prefix: 'camPitchCurve' }, camYaw: { prefix: 'camYawCurve' }, camBank: { prefix: 'camBankCurve' },
  camPersp: { prefix: 'camPerspectiveCurve' }, rotation: { prefix: 'camRollCurve', map: (v) => (v * 180) / Math.PI },
  centerX: { prefix: 'centreXCurve' }, centerY: { prefix: 'centreYCurve' },
  camPosX: { prefix: 'camPosXCurve' }, camPosY: { prefix: 'camPosYCurve' }, camPosZ: { prefix: 'camPosZCurve' },
  brightness: { prefix: 'brightnessCurve' }, gamma: { prefix: 'gammaCurve' }, gammaThreshold: { prefix: 'gammaThresholdCurve' },
  vibrancy: { prefix: 'vibrancyCurve' }, contrast: { prefix: 'contrastCurve' }, whiteLevel: { prefix: 'whiteLevelCurve' },
  camDOF: { prefix: 'camDOFCurve' }, camDOFArea: { prefix: 'camDOFAreaCurve' }, camDOFExponent: { prefix: 'camDOFExponentCurve' },
  camDOFScale: { prefix: 'camDOFScaleCurve' }, camDOFFade: { prefix: 'camDOFFadeCurve' },
  focusX: { prefix: 'focusXCurve' }, focusY: { prefix: 'focusYCurve' }, focusZ: { prefix: 'focusZCurve' }, camZ: { prefix: 'camZCurve' },
  dimishZ: { prefix: 'dimishZCurve' }, dimZDist: { prefix: 'dimZDistanceCurve' },
};
function mapped(c: MotionCurve, map?: (v: number) => number): MotionCurve {
  return map ? { ...c, points: c.points.map((p) => ({ t: p.t, v: map(p.v) })) } : c;
}
function xformCurveAttrs(x: XForm, items: { rest: string; curve: MotionCurve }[], fps: number): string[] {
  const out: string[] = [];
  const listPrefix: Record<string, string> = { variations: '', preVariations: 'pre_', postVariations: 'post_' };
  for (const { rest, curve } of items) {
    let m: RegExpExecArray | null;
    if (rest === 'weight') out.push(...curveAttrs('weightCurve', curve, fps));
    else if (rest === 'color') out.push(...curveAttrs('colorCurve', curve, fps));
    else if (rest === 'opacity') out.push(...curveAttrs('opacityCurve', curve, fps));
    else if (rest === 'colorSpeed') out.push(...curveAttrs('colorSymmetryCurve', mapped(curve, (v) => 1 - 2 * v), fps));
    else if ((m = /^(affine|post|yz|zx|yzPost|zxPost)\.([0-5])$/.exec(rest))) {
      const pfx: Record<string, string> = { affine: 'xyCoeff', post: 'xyPostCoeff', yz: 'yzCoeff', zx: 'zxCoeff', yzPost: 'yzPostCoeff', zxPost: 'zxPostCoeff' };
      out.push(...curveAttrs(`${pfx[m[1]]}${AFFINE_TO_COEFF[+m[2]]}Curve`, curve, fps));
    } else if ((m = /^(variations|preVariations|postVariations)\.(\d+)\.(weight|params\.(.+))$/.exec(rest))) {
      const vi = (x as any)[m[1]]?.[+m[2]];
      if (!vi || !VARIATIONS[vi.name]) continue;
      const vname = listPrefix[m[1]] + vi.name;
      out.push(...curveAttrs(m[3] === 'weight' ? `${vname}_amountCurve` : `${vname}_${m[4]}`, curve, fps));
    }
  }
  return out;
}

function xformToXML(x: XForm, tag: string, nXForms: number, extraAttrs: string[] = []): string {
  const attrs: string[] = [];
  if (tag === 'xform') attrs.push(`weight="${fmt(x.weight)}"`);
  attrs.push(`color="${fmt(x.color)}"`);
  attrs.push(`color_speed="${fmt(x.colorSpeed)}"`);
  attrs.push(`symmetry="${fmt(1 - 2 * x.colorSpeed)}"`);
  if (x.colorType) attrs.push(`color_type="${x.colorType}"`);
  if (x.colorType === 'TARGET' || x.colorType === 'TARGETG') attrs.push(`targetcolor="${(x.targetColor ?? [0, 0, 0]).map(fmt).join(' ')}"`);
  else if (tag === 'finalxform' && x.colorSpeed > 0) attrs.push('color_type="DIFFUSION"'); // JWildfire finals default to NONE (no recolouring)
  if (x.wfield) {
    const w = x.wfield;
    attrs.push(`wfield_type="${w.type}"`, `wfield_input="${w.input}"`, `wfield_color_intensity="${fmt(w.color)}"`, `wfield_var_amount_intensity="${fmt(w.varAmount)}"`,
      `wfield_jitter_intensity="${fmt(w.jitter)}"`, `wfield_noise_seed="${w.seed}"`, `wfield_noise_frequency="${fmt(w.frequency)}"`,
      `wfield_fract_noise_fract_type="${w.fractalType}"`, `wfield_fract_noise_octaves="${w.octaves}"`, `wfield_fract_noise_gain="${fmt(w.gain)}"`, `wfield_fract_noise_lacunarity="${fmt(w.lacunarity)}"`,
      `wfield_cell_noise_return_type="${w.cellReturn}"`, `wfield_cell_noise_dist_function="${w.cellDistance}"`);
    w.params.forEach((pp, i) => attrs.push(`wfield_var_param${i + 1}_intensity="${fmt(pp.intensity)}"`, `wfield_var_param${i + 1}_var_name="${pp.varName}"`, `wfield_var_param${i + 1}_param_name="${pp.paramName}"`));
  }
  if (x.colorMods?.some((v) => v !== 0)) {
    ['mod_gamma', 'mod_gamma_speed', 'mod_contrast', 'mod_contrast_speed', 'mod_saturation', 'mod_saturation_speed', 'mod_hue', 'mod_hue_speed']
      .forEach((a, i) => attrs.push(`${a}="${fmt(x.colorMods![i] ?? 0)}"`));
  }
  attrs.push(`opacity="${fmt(x.opacity)}"`);
  if (x.material || x.materialSpeed) attrs.push(`material="${fmt(x.material ?? 0)}"`, `material_speed="${fmt(x.materialSpeed ?? 0)}"`);
  const pushVars = (list: typeof x.variations | undefined, prefix: string) => {
    for (const vi of list ?? []) {
      if (!VARIATIONS[vi.name]) continue;
      attrs.push(`${prefix}${vi.name}="${fmt(vi.weight)}"`);
      if (vi.priority !== undefined) attrs.push(`${prefix}${vi.name}_fx_priority="${vi.priority}"`); // JWildfire per-instance priority
      // Parameter names are written as JWildfire writes them, spaces and dots included ("glsl_x_Density
      // Pixels", "crop_trapezoid_Base Sup."): JWildfire's reader matches them verbatim, so an encoded
      // form would silently drop the value there. Such a file is not strict XML; our importer's lenient
      // pre-parser (dedupeAttributes) takes it back, as does JWildfire.
      for (const pd of VARIATIONS[vi.name].params ?? []) {
        attrs.push(`${prefix}${vi.name}_${pd.name}="${fmt(vi.params[pd.name] ?? pd.def)}"`);
      }
      for (const r of VARIATIONS[vi.name].res ?? []) attrs.push(`${prefix}${vi.name}_${r}="${encodeRes(vi.res?.[r] ?? VARIATIONS[vi.name].resDef?.[r] ?? '')}"`); // JWildfire "ressource" (hex UTF-8)
    }
  };
  pushVars(x.variations, '');
  pushVars(x.preVariations, 'pre_');
  pushVars(x.postVariations, 'post_');
  attrs.push(`coefs="${coefsToXML(x.affine)}"`);
  const isIdentityPost = x.post.every((v, i) => Math.abs(v - [1, 0, 0, 0, 1, 0][i]) < 1e-9);
  if (!isIdentityPost) attrs.push(`post="${coefsToXML(x.post)}"`);
  if (x.yz) attrs.push(`yzCoefs="${coefsToXML(x.yz)}"`);
  if (x.zx) attrs.push(`zxCoefs="${coefsToXML(x.zx)}"`);
  if (x.yzPost) attrs.push(`yzPost="${coefsToXML(x.yzPost)}"`);
  if (x.zxPost) attrs.push(`zxPost="${coefsToXML(x.zxPost)}"`);
  if (tag === 'xform' && x.xaos && x.xaos.some((v) => v !== 1)) {
    const row = Array.from({ length: nXForms }, (_, j) => fmt(x.xaos![j] ?? 1));
    attrs.push(`chaos="${row.join(' ')}"`);
  }
  attrs.push(...extraAttrs);
  return `   <${tag} ${attrs.join(' ')}/>`;
}

function paletteToXML(palette: RGB[], indent: string): string {
  let hex = '';
  for (let i = 0; i < 256; i++) {
    const c = palette[i] ?? [0, 0, 0];
    hex += c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0')).join('');
    if (i % 8 === 7) hex += '\n' + indent + '   ';
  }
  return `${indent}<palette count="256" format="RGB">\n${indent}   ${hex.trimEnd()}\n${indent}</palette>`;
}

/** JWildfire background gradient attributes (only when a gradient is set; SINGLE_COLOR is the default). */
/** JWildfire post-symmetry attributes (always written, like JWildfire, so NONE round-trips). */
function psymAttrs(f: Flame): string {
  const p = f.postSymmetry;
  return `post_symmetry_type="${p?.type ?? 'NONE'}" post_symmetry_order="${p?.order ?? 3}" ` +
    `post_symmetry_centre_x="${fmt(p?.centreX ?? 0)}" post_symmetry_centre_y="${fmt(p?.centreY ?? 0)}" ` +
    `post_symmetry_distance="${fmt(p?.distance ?? 1.25)}" post_symmetry_rotation="${fmt(p?.rotation ?? 6)}" `;
}

function bgAttrs(f: Flame): string {
  const g = f.bgGradient;
  if (!g) return '';
  const c = (v: RGB) => v.map(fmt).join(' ');
  return ` background_type="${g.type}" background_ul="${c(g.ul)}" background_ur="${c(g.ur)}" background_ll="${c(g.ll)}" background_lr="${c(g.lr)}" background_cc="${c(g.cc)}"`;
}

/** JWildfire `sld_render_*` attributes (written only when solid rendering is on; names as JWildfire spells them). */
function solidAttrs(f: Flame): string {
  const s = f.solid;
  if (!s?.enabled) return '';
  const a: string[] = [
    'sld_render_enabled="1"',
    `sld_render_ao_enabled="${s.ao.enabled ? 1 : 0}"`, `sld_render_ao_intensity="${fmt(s.ao.intensity)}"`,
    `sld_render_ao_search_radius="${fmt(s.ao.searchRadius)}"`, `sld_render_ao_blur_radius="${fmt(s.ao.blurRadius)}"`,
    `sld_render_ao_radius_samples="${s.ao.radiusSamples}"`, `sld_render_ao_azimuth_samples="${s.ao.azimuthSamples}"`,
    `sld_render_ao_falloff="${fmt(s.ao.falloff)}"`, `sld_render_ao_affect_diffuse="${fmt(s.ao.affectDiffuse)}"`,
    `sld_render_shadow_type="${s.shadows.type}"`, `sld_render_shadow_smooth_radius="${fmt(s.shadows.smoothRadius)}"`,
    `sld_render_shadowmap_size="${s.shadows.mapSize}"`, `sld_render_shadowmap_bias="${fmt(s.shadows.bias)}"`,
    `sld_render_material_count="${s.materials.length}"`,
  ];
  s.materials.forEach((m, i) => a.push(
    `sld_render_material_diffuse${i}="${fmt(m.diffuse)}"`, `sld_render_material_ambient${i}="${fmt(m.ambient)}"`,
    `sld_render_material_phong${i}="${fmt(m.phong)}"`, `sld_render_material_phong_size${i}="${fmt(m.phongSize)}"`,
    `sld_render_material_phong_red${i}="${fmt(m.phongColor[0])}"`, `sld_render_material_phong_green${i}="${fmt(m.phongColor[1])}"`, `sld_render_material_phong_blue${i}="${fmt(m.phongColor[2])}"`,
    `sld_render_material_light_diif_func${i}="${m.diffFunc}"`, `sld_render_material_refl_map_intensity${i}="${fmt(m.reflMapIntensity)}"`,
    `sld_render_material_refl_mappping${i}="${m.reflMapping}"`,
    ...(m.reflMapFilename ? [`sld_render_material_refl_map_filename${i}="${m.reflMapFilename.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;')}"`] : []),
  ));
  a.push(`sld_render_ligtht_count="${s.lights.length}"`);
  s.lights.forEach((l, i) => a.push(
    `sld_render_light_altitude${i}="${fmt(l.altitude)}"`, `sld_render_light_azimuth${i}="${fmt(l.azimuth)}"`,
    `sld_render_light_intensity${i}="${fmt(l.intensity)}"`, `sld_render_light_shadow_intensity${i}="${fmt(l.shadowIntensity)}"`,
    `sld_render_light_red${i}="${fmt(l.color[0])}"`, `sld_render_light_green${i}="${fmt(l.color[1])}"`, `sld_render_light_blue${i}="${fmt(l.color[2])}"`,
    `sld_render_light_shadows${i}="${l.castShadows ? 1 : 0}"`,
  ));
  a.push(
    `post_bokeh_filter_kernel="${s.postBokeh.filterKernel}"`,
    `post_bokeh_intensity="${fmt(s.postBokeh.intensity)}"`,
    `post_bokeh_brightness="${fmt(s.postBokeh.brightness)}"`,
    `post_bokeh_size="${fmt(s.postBokeh.size)}"`,
    `post_bokeh_activation="${fmt(s.postBokeh.activation)}"`,
  );
  return ' ' + a.join(' ');
}

export function flameToXML(f: Flame, opts: XMLExportOpts = {}): string {
  const size = 1024;
  const scale = 0.25 * size * f.zoom;
  const fps = opts.fps && opts.fps > 0 ? opts.fps : 25;
  const buckets = bucketCurves(opts.curves);
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
  const lines: string[] = [];
  // Flame-level curves + timebase
  const flameCurveAttrs: string[] = [];
  for (const { rest, curve } of buckets.get('') ?? []) {
    const m = FLAME_CURVE_PREFIX[rest];
    if (m) flameCurveAttrs.push(...curveAttrs(m.prefix, mapped(curve, m.map), fps));
    else if (rest === 'zoom') flameCurveAttrs.push(...curveAttrs('camZoomCurve', mapped(curve, (v) => v / f.zoom), fps));
  }
  let timeAttrs = '';
  if (opts.curves?.some((c) => c.points.length)) {
    let end = 0;
    for (const c of opts.curves) for (const p of c.points) end = Math.max(end, p.t);
    timeAttrs = ` fps="${fps}" frame="0" frame_count="${Math.max(1, Math.round(end * fps) + 1)}"`;
  }
  lines.push(
    `<flame version="WilderFire 0.1" name="${esc(f.name)}" size="${size} ${size}" ` +
    `center="${fmt(f.centerX)} ${fmt(f.centerY)}" scale="${fmt(scale)}" rotate="${fmt((f.rotation * 180) / Math.PI)}" ` +
    `cam_pitch="${fmt((f.camPitch * Math.PI) / 180)}" cam_yaw="${fmt((f.camYaw * Math.PI) / 180)}" cam_roll="${fmt((f.camBank * Math.PI) / 180)}" cam_persp="${fmt(f.camPersp)}" ` +
    `cam_pos_x="${fmt(f.camPosX)}" cam_pos_y="${fmt(f.camPosY)}" cam_pos_z="${fmt(f.camPosZ)}" preserve_z="${f.preserveZ ? 1 : 0}" ` +
    `cam_zpos="${fmt(f.camZ ?? 0)}" cam_xfocus="${fmt(f.focusX ?? 0)}" cam_yfocus="${fmt(f.focusY ?? 0)}" cam_zfocus="${fmt(f.focusZ ?? 0)}" ` +
    `cam_dof="${fmt(f.camDOF ?? 0)}" cam_dof_area="${fmt(f.camDOFArea ?? 0.5)}" cam_dof_exponent="${fmt(f.camDOFExponent ?? 2)}" new_dof="${f.newDOF ? 1 : 0}" ` +
    `cam_dof_shape="${f.camDOFShape ?? 'BUBBLE'}" cam_dof_scale="${fmt(f.camDOFScale ?? 1)}" cam_dof_rotate="${fmt(f.camDOFRotate ?? 0)}" cam_dof_fade="${fmt(f.camDOFFade ?? 1)}" ` +
    (f.camDOFParams ?? [0, 0, 0, 0, 0, 0]).map((v, i) => `cam_dof_param${i + 1}="${fmt(v)}" `).join('') +
    `cam_zdimish="${fmt(f.dimishZ ?? 0)}" cam_zdimdist="${fmt(f.dimZDist ?? 0)}" cam_zdimcolor="${(f.dimZColor ?? [0, 0, 0]).map(fmt).join(' ')}" ` +
    `filter="${fmt(f.filterRadius ?? 0)}" filter_kernel="${normFilterKernel(f.filterKernel)}" ` +
    `filter_sharpness="${fmt(f.filterSharpness ?? 4)}" filter_low_density="${fmt(f.filterLowDensity ?? 0.025)}" ` +
    `antialias_amount="${fmt(f.antialiasAmount ?? 0.25)}" antialias_radius="${fmt(f.antialiasRadius ?? 0.5)}" ` +
    `de_radius="${fmt(f.deRadius ?? 1)}" de_curve="${fmt(f.deCurve ?? 0.8)}" ` +
    `saturation="${fmt(f.saturation ?? 1)}" fg_opacity="${fmt(f.fgOpacity ?? 1)}" ` +
    `bg_transparency="${f.bgTransparency ? 1 : 0}" oversample="${f.oversample ?? 1}" ` +
    (f.author ? `meta_info_author="${esc(f.author)}" ` : '') + (f.created ? `meta_info_creation_time="${esc(f.created)}" ` : '') + (f.uuid ? `meta_info_uuid="${esc(f.uuid)}" ` : '') +
    psymAttrs(f) +
    `quality="200" brightness="${fmt(f.brightness)}" gamma="${fmt(f.gamma)}" gamma_threshold="${fmt(f.gammaThreshold)}" ` +
    `contrast="${fmt(f.contrast ?? 1)}" white_level="${fmt(f.whiteLevel ?? 220)}" low_density_brightness="${fmt(f.lowDensityBrightness ?? 0.24)}" ` +
    `vibrancy="${fmt(f.vibrancy)}" background="${fmt(f.background[0])} ${fmt(f.background[1])} ${fmt(f.background[2])}"` +
    bgAttrs(f) + solidAttrs(f) + timeAttrs + (flameCurveAttrs.length ? ' ' + flameCurveAttrs.join(' ') : '') + '>',
  );
  const writeLayerBody = (ly: Layer, li: number, indent: string) => {
    ly.xforms.forEach((x, xi) => {
      const extra = xformCurveAttrs(x, buckets.get(`layers.${li}.xforms.${xi}`) ?? [], fps);
      lines.push(indent + xformToXML(x, 'xform', ly.xforms.length, extra).trim());
    });
    if (ly.final) {
      const extra = xformCurveAttrs(ly.final, buckets.get(`layers.${li}.final`) ?? [], fps);
      lines.push(indent + xformToXML(ly.final, 'finalxform', ly.xforms.length, extra).trim());
      for (const mf of ly.moreFinals) lines.push(indent + xformToXML(mf, 'finalxform', ly.xforms.length).trim());
    }
    lines.push(paletteToXML(ly.palette, indent));
  };
  if (f.layers.length > 1) {
    // JWildfire layered form
    f.layers.forEach((ly, li) => {
      const lc = (buckets.get(`layers.${li}`) ?? []).filter((c) => c.rest === 'weight');
      const extra = lc.length ? ' ' + curveAttrs('weightCurve', lc[0].curve, fps).join(' ') : '';
      lines.push(`   <layer weight="${fmt(ly.weight)}" visible="${ly.visible ? 1 : 0}" density="1"${extra}>`);
      writeLayerBody(ly, li, '      ');
      lines.push('   </layer>');
    });
  } else {
    // Flat flam3/Apophysis-compatible form
    writeLayerBody(f.layers[0], 0, '   ');
  }
  lines.push('</flame>');
  return lines.join('\n');
}

/** Import from text: sniffs JSON vs .flame XML. Returns the first flame, plus every flame of a pack file. */
export function importFlameText(text: string, fallbackPalette: RGB[]): { flame: Flame; flames: Flame[]; count: number; unknown: string[]; curves: MotionCurve[] } {
  const trimmed = text.trim();
  if (trimmed.startsWith('<')) {
    const flames = parseFlameXML(trimmed, fallbackPalette);
    return { flame: flames[0], flames, count: flames.length, unknown: [...lastImportUnknown], curves: [...lastImportCurves] };
  }
  const obj = JSON.parse(trimmed);
  const flame = normalizeFlame(obj, fallbackPalette);
  return { flame, flames: [flame], count: 1, unknown: [], curves: [] };
}
