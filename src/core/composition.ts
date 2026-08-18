// A composition: an ordered stack of image layers blended into one picture. Today every layer is a flame
// (rendered by its own FlameRenderer into an offscreen texture); other layer kinds (escape-time fractals,
// images) plug into the same stack later. A plain flame is a one-layer composition, so single-flame files,
// library entries and exports keep working unchanged.
//
// Blending follows the W3C compositing spec (the same formulas as CSS mix-blend-mode / Canvas 2D
// globalCompositeOperation): straight-alpha source-over with a separable blend function B(cb, cs) —
//   co = cs·αs·(1−αb) + cb·αb·(1−αs) + B(cb,cs)·αs·αb,  αo = αs + αb·(1−αs)   (co premultiplied)
// `blendPixel` (TS, used to composite export tiles) and `BLEND_WGSL` (the live compositor) implement it once each;
// tests/composition.test.ts keeps them in agreement.

import type { Flame, RGB } from './flame';
import { normalizeFlame } from './flame';
import { type EscapeLayerData, normalizeEscape, defaultEscape } from './escape';

export const BLEND_MODES = ['normal', 'add', 'multiply', 'screen', 'overlay', 'darken', 'lighten', 'color-dodge', 'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion'] as const;
export type BlendMode = typeof BLEND_MODES[number];

export interface CompLayerBase {
  id: string;
  name: string;
  visible: boolean;
  /** 0..1 */
  opacity: number;
  blend: BlendMode;
  /** The layer draws its own background (opaque, its bg colour/gradient); off = transparent where nothing is plotted. */
  ownBackground: boolean;
  /** Clip to the layer below: this layer's alpha is multiplied by the alpha of what is already composited under it (Photoshop's clipping mask). */
  clip: boolean;
}
export interface FlameCompLayer extends CompLayerBase { kind: 'flame'; flame: Flame }
/** an escape-time fractal (Mandelbrot/Julia/… families, custom formulas), see escape.ts */
export interface EscapeCompLayer extends CompLayerBase { kind: 'escape'; escape: EscapeLayerData }
/** a picture (PNG/JPEG/WebP…) from the browser's image store, placed on the canvas */
export interface ImageLayerData {
  /** content hash — the key in the image store (and in a composition file's `assets`) */
  key: string;
  /** natural size (kept so the layout works before the image is loaded) */
  w: number;
  h: number;
  fit: 'contain' | 'cover' | 'stretch' | 'none';
  /** extra scale on top of the fit, offset as a fraction of the canvas, rotation in radians */
  scale: number;
  offsetX: number;
  offsetY: number;
  rotation: number;
  /** repeat the picture beyond its edges */
  tile: boolean;
}
export interface ImageCompLayer extends CompLayerBase { kind: 'image'; image: ImageLayerData }
export type CompLayer = FlameCompLayer | EscapeCompLayer | ImageCompLayer;

export interface Composition {
  version: 1;
  name: string;
  /** Composition background (under the bottom layer; only visible where the stack is transparent). */
  background: RGB;
  layers: CompLayer[];
}

export const MAX_COMP_LAYERS = 8;

let idSeq = 0;
export function newLayerId(): string { return `L${Date.now().toString(36)}${(idSeq++).toString(36)}`; }

export function flameLayer(flame: Flame, opts: Partial<Omit<FlameCompLayer, 'kind' | 'flame'>> = {}): FlameCompLayer {
  return { kind: 'flame', id: newLayerId(), name: flame.name || 'Flame', visible: true, opacity: 1, blend: 'normal', ownBackground: true, clip: false, flame, ...opts };
}

export function escapeLayer(escape: EscapeLayerData, opts: Partial<Omit<EscapeCompLayer, 'kind' | 'escape'>> = {}): EscapeCompLayer {
  return { kind: 'escape', id: newLayerId(), name: 'Escape', visible: true, opacity: 1, blend: 'normal', ownBackground: true, clip: false, escape, ...opts };
}
export function imageLayer(image: ImageLayerData, opts: Partial<Omit<ImageCompLayer, 'kind' | 'image'>> = {}): ImageCompLayer {
  return { kind: 'image', id: newLayerId(), name: 'Image', visible: true, opacity: 1, blend: 'normal', ownBackground: false, clip: false, image, ...opts };
}
export function defaultImage(key: string, w: number, h: number): ImageLayerData {
  return { key, w, h, fit: 'contain', scale: 1, offsetX: 0, offsetY: 0, rotation: 0, tile: false };
}
function normalizeImage(obj: any): ImageLayerData | null {
  if (!obj || typeof obj.key !== 'string' || !obj.key) return null;
  return {
    key: obj.key, w: Math.max(1, Math.round(num(obj.w, 1))), h: Math.max(1, Math.round(num(obj.h, 1))),
    fit: ['contain', 'cover', 'stretch', 'none'].includes(obj.fit) ? obj.fit : 'contain',
    scale: Math.max(1e-6, num(obj.scale, 1)), offsetX: num(obj.offsetX, 0), offsetY: num(obj.offsetY, 0), rotation: num(obj.rotation, 0), tile: obj.tile === true,
  };
}
export { defaultEscape };

/** A one-layer composition around a flame (what every existing file/entry becomes). */
export function wrapFlame(flame: Flame): Composition {
  return { version: 1, name: flame.name || 'Untitled', background: [0, 0, 0], layers: [flameLayer(flame)] };
}

/** True when the composition is just one opaque flame (the historical single-flame document). */
export function isSingleFlame(c: Composition): boolean {
  return c.layers.length === 1 && c.layers[0].kind === 'flame';
}

const num = (v: unknown, d: number) => (typeof v === 'number' && isFinite(v) ? v : d);
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/** Tolerant reader for stored/imported JSON: a composition, or a bare flame (wrapped). */
export function normalizeComposition(obj: any, fallbackPalette: RGB[]): Composition {
  if (obj && typeof obj === 'object' && Array.isArray(obj.layers) && obj.layers.length && obj.layers.every((l: any) => l && typeof l === 'object' && 'kind' in l)) {
    const layers: CompLayer[] = [];
    for (const l of obj.layers.slice(0, MAX_COMP_LAYERS)) {
      if (l.kind !== 'flame' && l.kind !== 'escape' && l.kind !== 'image') continue; // unknown kinds (future files) are dropped
      const base = {
        id: typeof l.id === 'string' && l.id ? l.id : newLayerId(),
        visible: l.visible !== false, opacity: clamp01(num(l.opacity, 1)),
        blend: ((BLEND_MODES as readonly string[]).includes(l.blend) ? l.blend : 'normal') as BlendMode,
        ownBackground: l.ownBackground !== false, clip: l.clip === true,
      };
      if (l.kind === 'flame') {
        const flame = normalizeFlame(l.flame, fallbackPalette);
        layers.push({ kind: 'flame', ...base, name: typeof l.name === 'string' ? l.name : flame.name || 'Flame', flame });
      } else if (l.kind === 'escape') {
        layers.push({ kind: 'escape', ...base, name: typeof l.name === 'string' ? l.name : 'Escape', escape: normalizeEscape(l.escape, fallbackPalette) });
      } else {
        const image = normalizeImage(l.image);
        if (image) layers.push({ kind: 'image', ...base, name: typeof l.name === 'string' ? l.name : 'Image', image });
      }
    }
    if (!layers.length) layers.push(flameLayer(normalizeFlame(null, fallbackPalette)));
    const bg = Array.isArray(obj.background) && obj.background.length === 3 ? obj.background.map((v: unknown) => clamp01(num(v, 0))) as RGB : [0, 0, 0] as RGB;
    return { version: 1, name: typeof obj.name === 'string' ? obj.name : layers[0].name, background: bg, layers };
  }
  return wrapFlame(normalizeFlame(obj, fallbackPalette));
}

export function compositionToJSON(c: Composition): string { return JSON.stringify(c); }

// ---- blend maths (W3C compositing, separable modes) ----

function blendChannel(mode: BlendMode, cb: number, cs: number): number {
  switch (mode) {
    case 'normal': return cs;
    case 'add': return Math.min(1, cb + cs);
    case 'multiply': return cb * cs;
    case 'screen': return cb + cs - cb * cs;
    case 'overlay': return blendChannel('hard-light', cs, cb);
    case 'darken': return Math.min(cb, cs);
    case 'lighten': return Math.max(cb, cs);
    case 'color-dodge': return cb === 0 ? 0 : cs >= 1 ? 1 : Math.min(1, cb / (1 - cs));
    case 'color-burn': return cb >= 1 ? 1 : cs <= 0 ? 0 : 1 - Math.min(1, (1 - cb) / cs);
    case 'hard-light': return cs <= 0.5 ? cb * 2 * cs : blendChannel('screen', cb, 2 * cs - 1);
    case 'soft-light': {
      if (cs <= 0.5) return cb - (1 - 2 * cs) * cb * (1 - cb);
      const d = cb <= 0.25 ? ((16 * cb - 12) * cb + 4) * cb : Math.sqrt(cb);
      return cb + (2 * cs - 1) * (d - cb);
    }
    case 'difference': return Math.abs(cb - cs);
    case 'exclusion': return cb + cs - 2 * cb * cs;
  }
}

/** Composite one straight-alpha source pixel over a straight-alpha backdrop pixel; returns straight rgba (0..1). */
export function blendPixel(mode: BlendMode, backdrop: [number, number, number, number], source: [number, number, number, number], opacity = 1): [number, number, number, number] {
  const ab = backdrop[3], as = source[3] * opacity;
  const ao = as + ab * (1 - as);
  if (ao <= 1e-12) return [0, 0, 0, 0];
  const out: [number, number, number, number] = [0, 0, 0, ao];
  for (let i = 0; i < 3; i++) {
    const cb = backdrop[i], cs = source[i];
    const co = cs * as * (1 - ab) + cb * ab * (1 - as) + blendChannel(mode, cb, cs) * as * ab;
    out[i] = Math.min(1, Math.max(0, co / ao));
  }
  return out;
}

/** WGSL twin of blendPixel: `blendOver(mode, backdrop, source, opacity)` on straight-alpha vec4f. Mode ids = BLEND_MODES order. */
export const BLEND_WGSL = `
fn blendCh(mode: u32, cb: f32, cs: f32) -> f32 {
  switch mode {
    case 0u: { return cs; }
    case 1u: { return min(1.0, cb + cs); }
    case 2u: { return cb * cs; }
    case 3u: { return cb + cs - cb * cs; }
    case 4u: { return blendHard(cs, cb); }
    case 5u: { return min(cb, cs); }
    case 6u: { return max(cb, cs); }
    case 7u: { if (cb == 0.0) { return 0.0; } if (cs >= 1.0) { return 1.0; } return min(1.0, cb / (1.0 - cs)); }
    case 8u: { if (cb >= 1.0) { return 1.0; } if (cs <= 0.0) { return 0.0; } return 1.0 - min(1.0, (1.0 - cb) / cs); }
    case 9u: { return blendHard(cb, cs); }
    case 10u: {
      if (cs <= 0.5) { return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb); }
      var d = sqrt(cb);
      if (cb <= 0.25) { d = ((16.0 * cb - 12.0) * cb + 4.0) * cb; }
      return cb + (2.0 * cs - 1.0) * (d - cb);
    }
    case 11u: { return abs(cb - cs); }
    case 12u: { return cb + cs - 2.0 * cb * cs; }
    default: { return cs; }
  }
}
fn blendHard(cb: f32, cs: f32) -> f32 {
  if (cs <= 0.5) { return cb * 2.0 * cs; }
  let s = 2.0 * cs - 1.0;
  return cb + s - cb * s;
}
fn blendOver(mode: u32, backdrop: vec4f, source: vec4f, opacity: f32) -> vec4f {
  let ab = backdrop.a; let as_ = source.a * opacity;
  let ao = as_ + ab * (1.0 - as_);
  if (ao <= 1e-12) { return vec4f(0.0); }
  var co = vec3f(0.0);
  for (var i = 0u; i < 3u; i = i + 1u) {
    let cb = backdrop[i]; let cs = source[i];
    co[i] = cs * as_ * (1.0 - ab) + cb * ab * (1.0 - as_) + blendCh(mode, cb, cs) * as_ * ab;
  }
  return vec4f(clamp(co / ao, vec3f(0.0), vec3f(1.0)), ao);
}
`;
