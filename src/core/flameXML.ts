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

import type { Flame, XForm, Layer, Affine, RGB } from './flame';
import { defaultXForm, defaultFlame, normalizeFlame, MAX_LAYERS, MAX_XFORMS } from './flame';
import { VARIATIONS } from './variations';
import type { MotionCurve, CurvePoint, CurveInterp } from './motion';

const NON_VARIATION_ATTRS = new Set([
  'weight', 'color', 'symmetry', 'color_speed', 'opacity', 'coefs', 'post',
  'chaos', 'animate', 'motion_frequency', 'motion_function', 'var_color',
  'name', 'mirror_pre_post_translations',
  // JWildfire per-xform extras we do not model
  'material', 'material_speed', 'mod_gamma', 'mod_gamma_speed', 'mod_contrast', 'mod_contrast_speed',
  'mod_saturation', 'mod_saturation_speed', 'mod_hue', 'mod_hue_speed', 'color_type', 'target_color',
  'draw_mode', 'weighting_field_type', 'weighting_field_input', 'yzCoefs', 'zxCoefs', 'yzPost', 'zxPost',
]);
/** JWildfire writes `<var>_fx_priority` per instance (priority override); we keep definition priorities. */
const isFxPriorityAttr = (n: string) => n.endsWith('_fx_priority');

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
  const chaos = nums(elm.getAttribute('chaos'));
  if (chaos.length) x.xaos = chaos.map((v) => Math.max(0, v));

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
  for (const attr of Array.from(elm.attributes)) {
    // Duplicate variation instances on one xform were renamed name__dup<k> by
    // the lenient pre-parser; strip the marker to resolve them.
    const name = attr.name.replace(/__dup\d+/, '');
    if (NON_VARIATION_ATTRS.has(name) || isFxPriorityAttr(name) || name.startsWith('wfield_')) continue;
    if (isCurveAttr(attr.name, cprefixes)) continue;
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
      if (res && VARIATIONS[res.vname]?.params?.some((p) => p.name === pn)) {
        (params[attr.name.slice(0, attr.name.length - name.length + us)] ??= {})[pn] = val;
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
    if (weight === 0) continue;
    const p: Record<string, number> = {};
    for (const pd of VARIATIONS[vname].params ?? []) {
      p[pd.name] = params[raw]?.[pd.name] ?? pd.def;
    }
    const inst = { name: vname, weight, params: p };
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

/** JWildfire writes one attribute per variation instance, so an xform with two
 *  `bubble` variations yields a duplicate attribute (invalid XML). Rename the
 *  later occurrences (and their trailing params) to `name__dup<k>`. */
function dedupeAttributes(text: string): string {
  // JWildfire's own marker for repeated instances: bubble#1#="…" bubble#1#_param="…"
  text = text.replace(/([A-Za-z_]\w*)#(\d+)#/g, (_m, n: string, k: string) => `${n}__dup${Number(k) + 1}`);
  return text.replace(/<(xform|finalxform)\b([^>]*)>/g, (m, tag: string, attrs: string) => {
    // Attribute names may contain spaces or punctuation ("foo_show/hide(1/0)",
    // "glsl_x_Density Pixels"): tokenize on name="value" pairs and sanitize.
    attrs = attrs.replace(/([^\s"=][^"=]*?)\s*=\s*"([^"]*)"/g, (_m, name: string, val: string) => ` ${name.trim().replace(/[^\w.-]/g, '_')}="${val}"`);
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
    f.camPersp = numAttr('cam_persp');
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
    f.camDOFFade = Math.min(1, Math.max(0, numOr('cam_dof_fade', 1)));
    f.newDOF = numAttr('new_dof') !== 0;
    f.focusX = numAttr('cam_xfocus'); f.focusY = numAttr('cam_yfocus'); f.focusZ = numAttr('cam_zfocus');
    f.camZ = numAttr('cam_zpos');
    f.dimishZ = Math.max(0, numAttr('cam_zdimish'));
    f.dimZDist = numAttr('cam_zdimdist');
    const dzc = nums(fe.getAttribute('cam_zdimcolor'));
    if (dzc.length === 3) f.dimZColor = dzc.map((v) => Math.min(1, Math.max(0, v))) as RGB;
    const br = parseFloat(fe.getAttribute('brightness') ?? '');
    if (isFinite(br) && br > 0) f.brightness = Math.min(br, 6);
    const ga = parseFloat(fe.getAttribute('gamma') ?? '');
    if (isFinite(ga) && ga > 0) f.gamma = Math.min(ga, 8);
    const gt = parseFloat(fe.getAttribute('gamma_threshold') ?? '');
    if (isFinite(gt) && gt >= 0) f.gammaThreshold = Math.min(gt, 0.5);
    const vi = parseFloat(fe.getAttribute('vibrancy') ?? '');
    if (isFinite(vi)) f.vibrancy = Math.min(1, Math.max(0, vi));
    // JWildfire tonemap constants + spatial filter + antialiasing (defaults = JWildfire's)
    f.contrast = Math.max(0.05, numOr('contrast', 1));
    f.whiteLevel = Math.max(1, numOr('white_level', 220));
    f.lowDensityBrightness = Math.max(0, numOr('low_density_brightness', 0.24));
    f.filterRadius = Math.min(3, Math.max(0, numOr('filter', 0.75)));
    f.filterKernel = /GAUSS/i.test(fe.getAttribute('filter_kernel') ?? '') ? 'gaussian' : 'mitchell';
    f.antialiasAmount = Math.min(1, Math.max(0, numOr('antialias_amount', 0.25)));
    f.antialiasRadius = Math.max(0, numOr('antialias_radius', 0.5));
    f.deRadius = Math.min(2, Math.max(0, numOr('de_radius', 1)));
    f.deCurve = Math.min(1, Math.max(0.01, numOr('de_curve', 0.8)));
    const bg = nums(fe.getAttribute('background'));
    if (bg.length === 3) {
      // Old files use 0-255, new ones 0-1 — sniff by magnitude.
      const mx = Math.max(...bg);
      f.background = bg.map((v) => (mx > 1 ? v / 255 : v)) as RGB;
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
      const fin = elm.querySelector(':scope > finalxform');
      return {
        xforms: xf.length ? xf : [defaultXForm()],
        final: fin ? parseXFormEl(fin, ctx(`layers.${li}.final`)) : null,
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
  attrs.push(`opacity="${fmt(x.opacity)}"`);
  const pushVars = (list: typeof x.variations | undefined, prefix: string) => {
    for (const vi of list ?? []) {
      if (!VARIATIONS[vi.name]) continue;
      attrs.push(`${prefix}${vi.name}="${fmt(vi.weight)}"`);
      for (const pd of VARIATIONS[vi.name].params ?? []) {
        attrs.push(`${prefix}${vi.name}_${pd.name}="${fmt(vi.params[pd.name] ?? pd.def)}"`);
      }
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
    `cam_dof_shape="BUBBLE" cam_dof_scale="${fmt(f.camDOFScale ?? 1)}" cam_dof_rotate="0" cam_dof_fade="${fmt(f.camDOFFade ?? 1)}" ` +
    `cam_zdimish="${fmt(f.dimishZ ?? 0)}" cam_zdimdist="${fmt(f.dimZDist ?? 0)}" cam_zdimcolor="${(f.dimZColor ?? [0, 0, 0]).map(fmt).join(' ')}" ` +
    `filter="${fmt(f.filterRadius ?? 0)}" filter_kernel="${(f.filterKernel ?? 'mitchell') === 'gaussian' ? 'GAUSSIAN' : 'MITCHELL_SMOOTH'}" ` +
    `antialias_amount="${fmt(f.antialiasAmount ?? 0.25)}" antialias_radius="${fmt(f.antialiasRadius ?? 0.5)}" ` +
    `de_radius="${fmt(f.deRadius ?? 1)}" de_curve="${fmt(f.deCurve ?? 0.8)}" ` +
    `quality="200" brightness="${fmt(f.brightness)}" gamma="${fmt(f.gamma)}" gamma_threshold="${fmt(f.gammaThreshold)}" ` +
    `contrast="${fmt(f.contrast ?? 1)}" white_level="${fmt(f.whiteLevel ?? 220)}" low_density_brightness="${fmt(f.lowDensityBrightness ?? 0.24)}" ` +
    `vibrancy="${fmt(f.vibrancy)}" background="${fmt(f.background[0])} ${fmt(f.background[1])} ${fmt(f.background[2])}"` +
    timeAttrs + (flameCurveAttrs.length ? ' ' + flameCurveAttrs.join(' ') : '') + '>',
  );
  const writeLayerBody = (ly: Layer, li: number, indent: string) => {
    ly.xforms.forEach((x, xi) => {
      const extra = xformCurveAttrs(x, buckets.get(`layers.${li}.xforms.${xi}`) ?? [], fps);
      lines.push(indent + xformToXML(x, 'xform', ly.xforms.length, extra).trim());
    });
    if (ly.final) {
      const extra = xformCurveAttrs(ly.final, buckets.get(`layers.${li}.final`) ?? [], fps);
      lines.push(indent + xformToXML(ly.final, 'finalxform', ly.xforms.length, extra).trim());
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

/** Import from text: sniffs JSON vs .flame XML. Returns the first flame. */
export function importFlameText(text: string, fallbackPalette: RGB[]): { flame: Flame; count: number; unknown: string[]; curves: MotionCurve[] } {
  const trimmed = text.trim();
  if (trimmed.startsWith('<')) {
    const flames = parseFlameXML(trimmed, fallbackPalette);
    return { flame: flames[0], count: flames.length, unknown: [...lastImportUnknown], curves: [...lastImportCurves] };
  }
  const obj = JSON.parse(trimmed);
  return { flame: normalizeFlame(obj, fallbackPalette), count: 1, unknown: [], curves: [] };
}
