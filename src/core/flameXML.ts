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

const NON_VARIATION_ATTRS = new Set([
  'weight', 'color', 'symmetry', 'color_speed', 'opacity', 'coefs', 'post',
  'chaos', 'animate', 'motion_frequency', 'motion_function', 'var_color',
  'name', 'mirror_pre_post_translations',
]);

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

function parseXFormEl(elm: Element): XForm {
  const x = defaultXForm();
  x.variations = [];
  x.affine = parseCoefs(elm.getAttribute('coefs'), x.affine);
  x.post = parseCoefs(elm.getAttribute('post'), x.post);
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
  for (const attr of Array.from(elm.attributes)) {
    const name = attr.name;
    if (NON_VARIATION_ATTRS.has(name)) continue;
    const val = parseFloat(attr.value);
    if (!isFinite(val)) continue;
    const direct = resolve(name);
    if (direct) {
      weights.push({ raw: name, ...direct, weight: val });
      continue;
    }
    // `${var}_${param}` — match against the parameter list (raw name keeps its prefix)
    const us = name.lastIndexOf('_');
    if (us > 0) {
      const vn = name.slice(0, us);
      const pn = name.slice(us + 1);
      const res = resolve(vn);
      if (res && VARIATIONS[res.vname]?.params?.some((p) => p.name === pn)) {
        (params[vn] ??= {})[pn] = val;
      }
    }
  }
  for (const { raw, stage, vname, weight } of weights) {
    if (weight === 0) continue;
    const p: Record<string, number> = {};
    for (const pd of VARIATIONS[vname].params ?? []) {
      p[pd.name] = params[raw]?.[pd.name] ?? pd.def;
    }
    const inst = { name: vname, weight, params: p };
    if (stage === 'pre') (x.preVariations ??= []).push(inst);
    else if (stage === 'post') (x.postVariations ??= []).push(inst);
    else x.variations.push(inst);
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

export function parseFlameXML(text: string, fallbackPalette: RGB[]): Flame[] {
  const doc = new DOMParser().parseFromString(text, 'application/xml');
  if (doc.querySelector('parsererror')) {
    throw new Error('Not valid XML: ' + (doc.querySelector('parsererror')?.textContent ?? '').slice(0, 120));
  }
  const flameEls = Array.from(doc.querySelectorAll('flame'));
  if (!flameEls.length) throw new Error('No <flame> element found.');

  return flameEls.map((fe) => {
    const f = defaultFlame(fallbackPalette);
    f.name = fe.getAttribute('name') ?? 'imported';
    const size = nums(fe.getAttribute('size'));
    const sizeMin = size.length === 2 ? Math.min(size[0], size[1]) : 1024;
    const center = nums(fe.getAttribute('center'));
    if (center.length === 2) { f.centerX = center[0]; f.centerY = center[1]; }
    const scale = parseFloat(fe.getAttribute('scale') ?? '');
    if (isFinite(scale) && scale > 0) f.zoom = scale / (0.25 * sizeMin);
    const rot = parseFloat(fe.getAttribute('rotate') ?? '');
    if (isFinite(rot)) f.rotation = (rot * Math.PI) / 180;
    const br = parseFloat(fe.getAttribute('brightness') ?? '');
    if (isFinite(br) && br > 0) f.brightness = Math.min(br, 6);
    const ga = parseFloat(fe.getAttribute('gamma') ?? '');
    if (isFinite(ga) && ga > 0) f.gamma = Math.min(ga, 8);
    const vi = parseFloat(fe.getAttribute('vibrancy') ?? '');
    if (isFinite(vi)) f.vibrancy = Math.min(1, Math.max(0, vi));
    const bg = nums(fe.getAttribute('background'));
    if (bg.length === 3) {
      // Old files use 0-255, new ones 0-1 — sniff by magnitude.
      const mx = Math.max(...bg);
      f.background = bg.map((v) => (mx > 1 ? v / 255 : v)) as RGB;
    }
    const parseLayerContent = (elm: Element): Omit<Layer, 'weight' | 'visible'> => {
      const xf = Array.from(elm.querySelectorAll(':scope > xform')).slice(0, MAX_XFORMS).map(parseXFormEl);
      const fin = elm.querySelector(':scope > finalxform');
      return {
        xforms: xf.length ? xf : [defaultXForm()],
        final: fin ? parseXFormEl(fin) : null,
        palette: parsePaletteEl(elm) ?? fallbackPalette,
      };
    };

    // JWildfire layered flames use <layer> children; flam3/Apophysis are flat.
    const layerEls = Array.from(fe.querySelectorAll(':scope > layer')).slice(0, MAX_LAYERS);
    if (layerEls.length) {
      f.layers = layerEls.map((le) => {
        const w = parseFloat(le.getAttribute('weight') ?? le.getAttribute('density') ?? '');
        const visAttr = le.getAttribute('visible');
        return {
          ...parseLayerContent(le),
          weight: isFinite(w) && w >= 0 ? w : 1,
          visible: visAttr !== '0' && visAttr !== 'false',
        };
      });
    } else {
      f.layers = [{ ...parseLayerContent(fe), weight: 1, visible: true }];
    }
    // Re-run through the normalizer for clamping / defaults.
    return normalizeFlame(f, fallbackPalette);
  });
}

function xformToXML(x: XForm, tag: string, nXForms: number): string {
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
  if (tag === 'xform' && x.xaos && x.xaos.some((v) => v !== 1)) {
    const row = Array.from({ length: nXForms }, (_, j) => fmt(x.xaos![j] ?? 1));
    attrs.push(`chaos="${row.join(' ')}"`);
  }
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

export function flameToXML(f: Flame): string {
  const size = 1024;
  const scale = 0.25 * size * f.zoom;
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
  const lines: string[] = [];
  lines.push(
    `<flame version="WilderFire 0.1" name="${esc(f.name)}" size="${size} ${size}" ` +
    `center="${fmt(f.centerX)} ${fmt(f.centerY)}" scale="${fmt(scale)}" rotate="${fmt((f.rotation * 180) / Math.PI)}" ` +
    `filter="0.5" quality="200" brightness="${fmt(f.brightness)}" gamma="${fmt(f.gamma)}" ` +
    `vibrancy="${fmt(f.vibrancy)}" background="${fmt(f.background[0])} ${fmt(f.background[1])} ${fmt(f.background[2])}">`,
  );
  const writeLayerBody = (ly: Layer, indent: string) => {
    for (const x of ly.xforms) lines.push(indent + xformToXML(x, 'xform', ly.xforms.length).trim());
    if (ly.final) lines.push(indent + xformToXML(ly.final, 'finalxform', ly.xforms.length).trim());
    lines.push(paletteToXML(ly.palette, indent));
  };
  if (f.layers.length > 1) {
    // JWildfire layered form
    for (const ly of f.layers) {
      lines.push(`   <layer weight="${fmt(ly.weight)}" visible="${ly.visible ? 1 : 0}" density="1">`);
      writeLayerBody(ly, '      ');
      lines.push('   </layer>');
    }
  } else {
    // Flat flam3/Apophysis-compatible form
    writeLayerBody(f.layers[0], '   ');
  }
  lines.push('</flame>');
  return lines.join('\n');
}

/** Import from text: sniffs JSON vs .flame XML. Returns the first flame. */
export function importFlameText(text: string, fallbackPalette: RGB[]): { flame: Flame; count: number } {
  const trimmed = text.trim();
  if (trimmed.startsWith('<')) {
    const flames = parseFlameXML(trimmed, fallbackPalette);
    return { flame: flames[0], count: flames.length };
  }
  const obj = JSON.parse(trimmed);
  return { flame: normalizeFlame(obj, fallbackPalette), count: 1 };
}
