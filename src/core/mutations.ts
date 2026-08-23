// MutaGen's typed mutations, the random symmetry / weighting-field generators and the sampler's coverage measures —
// JWildfire's org.jwildfire.create.tina.mutagen / randomsymmetry / randomweightingfield / RandomFlameGeneratorSampler,
// transcribed (LGPL 2.1+, © Andreas Maschke and contributors; see NOTICE.md). Everything here works on the Flame model
// directly; the random draws, constants and type lists are JWildfire's.
import type { Flame, Layer, XForm, VarInstance, Affine, RGB } from './flame';
import { defaultXForm, defaultWeightingField, visibleLayers } from './flame';
import { VARIATIONS, defaultParams } from './variations';
import { randomPalette, rotatePalette } from './palette';
import { experimental, randomVariationName, varMutate, randomFlameInStyle } from './randomStyles';

const rnd = Math.random;
const randomInt = (n: number) => Math.floor(rnd() * n);
const FTOI = (v: number) => (v >= 0 ? Math.floor(v + 0.5) : -Math.floor(-v + 0.5));
const known = (name: string) => name in VARIATIONS;
const mk = (name: string, weight: number, params: Record<string, number> = {}): VarInstance => {
  if (!known(name)) name = 'linear';
  return { name, weight, params: { ...defaultParams(name), ...params } };
};
const allXForms = (ly: Layer) => [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals];
const finalsOf = (ly: Layer) => [...(ly.final ? [ly.final] : []), ...ly.moreFinals];
const varsOf = (x: XForm) => [...(x.preVariations ?? []), ...x.variations, ...(x.postVariations ?? [])];

// ---------- XFormTransformService on the Flame model (affine arrays are [a, b, c, d, e, f] = c00 c10 c20 c01 c11 c21) ----------
type Plane = 'xy' | 'yz' | 'zx';
const ID: Affine = [1, 0, 0, 0, 1, 0];
/** the coefficient set an edit plane + post flag address (created on demand for the 3D planes) */
function aff(x: XForm, plane: Plane, post: boolean): Affine {
  if (plane === 'xy') return post ? x.post : x.affine;
  const k = (plane === 'yz' ? (post ? 'yzPost' : 'yz') : (post ? 'zxPost' : 'zx')) as 'yz' | 'zx' | 'yzPost' | 'zxPost';
  return (x[k] ??= [...ID] as Affine);
}
function rotateX(x: XForm, angle: number, post: boolean, plane: Plane = 'xy') {
  if (Math.abs(angle) < 1e-12) return;
  const a = (angle * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  const m = aff(x, plane, post);
  // JWildfire: M2 = [[c00 c01 c20],[c10 c11 c21]] · R  (R = [[cos −sin],[sin cos]])
  const [c00, c10, c20, c01, c11, c21] = m;
  m[0] = c00 * c + c01 * s; m[3] = -c00 * s + c01 * c;
  m[1] = c10 * c + c11 * s; m[4] = -c10 * s + c11 * c;
  m[2] = c20; m[5] = c21;
}
function globalTranslateX(x: XForm, dx: number, dy: number, post: boolean, plane: Plane = 'xy') {
  const m = aff(x, plane, post); m[2] += dx; m[5] += dy;
}
function scaleX(x: XForm, s: number, xs: boolean, ys: boolean, post: boolean, plane: Plane = 'xy') {
  if (Math.abs(s - 1) < 1e-12) return;
  const m = aff(x, plane, post);
  if (xs) { m[0] *= s; m[3] *= s; }
  if (ys) { m[1] *= s; m[4] *= s; }
}
const flipHorizontal = (x: XForm, post: boolean, plane: Plane = 'xy') => { const m = aff(x, plane, post); m[0] = -m[0]; m[1] = -m[1]; };
const flipVertical = (x: XForm, post: boolean, plane: Plane = 'xy') => { const m = aff(x, plane, post); m[3] = -m[3]; m[4] = -m[4]; };

// ---------- the mutations ----------
export type MutationType = 'all' | 'add_transform' | 'add_variation' | 'change_weight' | 'gradient_position' | 'local_gamma' | 'affine' | 'affine_3d' | 'bokeh'
  | 'random_bg_color' | 'random_flame' | 'random_ztransform' | 'random_gradient' | 'random_parameter' | 'similar_gradient' | 'weighting_field' | 'color_type';
export const MUTATION_TYPES: { id: MutationType; name: string; hint: string }[] = [
  { id: 'all', name: 'All', hint: 'one of the mutations below, drawn with JWildfire\'s weights' },
  { id: 'add_transform', name: 'Add transform', hint: 'a new transform (sometimes replacing one) with a random variation — or a new final' },
  { id: 'add_variation', name: 'Add variation', hint: 'a random variation added to a transform (often replacing one of its own)' },
  { id: 'change_weight', name: 'Change weight (xaos)', hint: 'toggles random entries of the xaos (modified weights) table' },
  { id: 'gradient_position', name: 'Gradient position', hint: 'new colour indices for the transforms, or the gradient rotated' },
  { id: 'local_gamma', name: 'Local gamma', hint: 'random per-transform colour modifiers (gamma, contrast, saturation, hue)' },
  { id: 'affine', name: 'Affine', hint: 'nudges, flips, moves, rotates or scales a few transforms' },
  { id: 'affine_3d', name: 'Affine 3D', hint: 'the same in the YZ / ZX planes, with a 3D camera and post_zscale_wf' },
  { id: 'bokeh', name: 'Bokeh', hint: 'a depth-of-field setup with a random blur shape and a crackle transform' },
  { id: 'random_bg_color', name: 'Background colour', hint: 'a single colour or a 2×2 gradient from a random palette' },
  { id: 'random_flame', name: 'Random flame', hint: 'replaces the layer with a random flame (any style)' },
  { id: 'random_ztransform', name: 'Random Z transform', hint: 'small moves in the YZ / ZX planes, with a 3D camera when the flame is flat' },
  { id: 'random_gradient', name: 'Random gradient', hint: 'a new random gradient' },
  { id: 'random_parameter', name: 'Random parameter', hint: 'three random variation parameters nudged' },
  { id: 'similar_gradient', name: 'Similar gradient', hint: 'a gradient rebuilt from the current one\'s colour clusters' },
  { id: 'weighting_field', name: 'Weighting field', hint: 'random noise fields on the heavier transforms' },
  { id: 'color_type', name: 'Colour type', hint: 'random colour types (DIFFUSION / NONE / DISTANCE / TARGET / TARGETG) per transform' },
];
// AllMutation's list (repeats weight the draw); PAINTERLY_STYLE (post_brush_stroke_wf) and RANDOM_BACKGROUND (background
// transforms) are not in this build and fall back to ALL's neighbours
const ALL_TYPES: MutationType[] = ['bokeh', 'add_transform', 'affine', 'change_weight', 'random_gradient', 'all', 'affine_3d', 'local_gamma', 'color_type', 'random_parameter', 'affine', 'weighting_field',
  'add_transform', 'local_gamma', 'similar_gradient', 'random_bg_color', 'all', 'affine', 'change_weight', 'affine_3d', 'add_variation', 'change_weight', 'weighting_field', 'add_transform', 'local_gamma', 'all',
  'color_type', 'affine', 'random_parameter', 'affine_3d', 'bokeh', 'similar_gradient', 'gradient_position', 'color_type', 'local_gamma', 'random_bg_color', 'all', 'add_transform', 'affine', 'all', 'change_weight',
  'affine', 'weighting_field', 'change_weight', 'affine_3d', 'random_parameter'];

/** Apply one mutation of `type` to every visible layer of `f` (MutaGenController.modifyFlame); returns the type applied. */
export function applyMutation(f: Flame, type: MutationType, strength = 1): MutationType {
  let t = type;
  while (t === 'all') t = ALL_TYPES[randomInt(ALL_TYPES.length)];
  for (const ly of visibleLayers(f)) MUTATIONS[t](f, ly, strength);
  return t;
}

const pickXForm = (ly: Layer, finalChance: number): XForm => (!finalsOf(ly).length || rnd() >= finalChance ? ly.xforms[randomInt(ly.xforms.length)] : finalsOf(ly)[randomInt(finalsOf(ly).length)]);
const addFinal = (ly: Layer, x: XForm) => { if (!ly.final) ly.final = x; else ly.moreFinals.push(x); };
const removeFinal = (ly: Layer, i: number) => { if (i === 0) { ly.final = ly.moreFinals.shift() ?? null; } else ly.moreFinals.splice(i - 1, 1); };
const newXForm = (weight: number): XForm => { const x = defaultXForm(); x.weight = weight; x.variations = []; return x; };
const fixXaos = (ly: Layer) => { const n = ly.xforms.length; for (const x of ly.xforms) if (x.xaos) x.xaos = Array.from({ length: n }, (_, i) => x.xaos![i] ?? 1); };

/** AffineMutation.applyToXForm / Affine3DMutation.apply's shared body (the plane decides which coefficients) */
function affineNudge(x: XForm, amount: number, plane: Plane) {
  if (rnd() < 0.5) {
    const m = aff(x, 'xy', rnd() >= 0.75); // the coefficient nudges address the XY set explicitly (setXYCoeff…), whatever the edit plane
    for (const i of [0, 3, 1, 4, 2, 5]) if (rnd() < 0.5) m[i] += amount * (-0.25 + 0.5 * rnd());
  } else {
    if (rnd() < 0.33) flipHorizontal(x, rnd() < 0.5, plane);
    if (rnd() > 0.67) flipVertical(x, rnd() < 0.5, plane);
    if (rnd() < 0.75) globalTranslateX(x, -amount + 2 * rnd() * amount, -amount + 2 * rnd() * amount, rnd() < 0.5, plane);
    if (rnd() < 0.5) rotateX(x, -amount * 10 + 20 * rnd() * amount, rnd() < 0.5, plane);
    if (rnd() < 0.5) scaleX(x, -amount + 2 * rnd() * amount, rnd() < 0.75, rnd() < 0.75, false, plane);
  }
}

const MUTATIONS: Record<Exclude<MutationType, 'all'>, (f: Flame, ly: Layer, s: number) => void> = {
  add_transform: (_f, ly, s) => {
    if (rnd() < 0.75) {
      if (rnd() < 0.5 && ly.xforms.length > 0) ly.xforms.splice(randomInt(ly.xforms.length), 1);
      const x = newXForm((0.1 + rnd() * 2) * s); ly.xforms.push(x);
      x.variations.push(mk(rnd() < 0.33 ? experimental() : randomVariationName(), (0.01 + rnd() * 10) * s));
      fixXaos(ly);
    } else {
      const fin = finalsOf(ly);
      if (rnd() < 0.5 && fin.length) removeFinal(ly, randomInt(fin.length));
      const x = newXForm((0.1 + rnd() * 2) * s); x.colorSpeed = 0; x.colorType = 'NONE'; addFinal(ly, x);
      let name: string;
      if (rnd() < 0.33) name = experimental();
      else for (;;) { name = randomVariationName(); if (!/blur/i.test(name) && !name.startsWith('fract_')) break; }
      x.variations.push(mk(name, (0.01 + rnd() * 10) * s));
    }
  },
  add_variation: (_f, ly, s) => {
    const x = pickXForm(ly, 0.25);
    if (rnd() < 0.75) {
      if (rnd() < 0.67 && x.variations.length > 0) x.variations.splice(randomInt(x.variations.length), 1);
      x.variations.push(mk(rnd() < 0.33 ? experimental() : randomVariationName(), (0.01 + rnd() * 10) * s));
    }
  },
  change_weight: (_f, ly) => {
    const n = ly.xforms.length;
    for (const x of ly.xforms) {
      const idx = randomInt(n);
      x.xaos ??= Array.from({ length: n }, () => 1);
      if (x.xaos[idx] > 0) x.xaos[idx] = 0; else x.xaos[idx] += 1;
    }
  },
  gradient_position: (_f, ly) => {
    if (rnd() < 0.5) { for (const x of ly.xforms) { x.color = rnd(); if (rnd() < 0.25) x.colorSpeed = (1 - rnd()) / 2; } } // setColorSymmetry(random)
    else ly.palette = rotatePalette(ly.palette, (-256 + randomInt(512)) / 256);
  },
  local_gamma: (_f, ly) => {
    for (const x of ly.xforms) { const pair = (): [number, number] => [1 - 2 * rnd(), rnd() < 0.33 ? 1 - 2 * rnd() : 0]; x.colorMods = [...pair(), ...pair(), ...pair(), ...pair()]; }
  },
  affine: (_f, ly, s) => { for (const a of [0.75, 0.2, 0.1]) affineNudge(pickXForm(ly, 0.5), a * s, 'xy'); },
  affine_3d: (f, ly, s) => {
    f.camPitch = 30 + rnd() * 20; f.camYaw = 15 - rnd() * 30; f.camBank = 15 - rnd() * 30; f.camPersp = 0.05 + rnd() * 0.2; f.camDOFArea = 0.2 + rnd() * 0.5;
    if (!f.preserveZ && rnd() > 0.33) f.preserveZ = true;
    for (const a of [0.5, 0.2, 0.1]) {
      const plane: Plane = rnd() < 0.5 ? 'yz' : 'zx';
      const x = pickXForm(ly, 0.5);
      if (f.preserveZ && rnd() > 0.25 && !varsOf(x).some((v) => v.name === 'post_zscale_wf')) (x.postVariations ??= []).push(mk('post_zscale_wf', 0.001 + rnd() * 0.1));
      affineNudge(x, a * s, plane);
    }
  },
  bokeh: (f, ly) => bokehMutation(f, ly),
  random_bg_color: (f) => {
    const pal = randomPalette();
    const at = (i: number): RGB => [...pal[Math.max(0, Math.min(255, i))]] as RGB;
    const mode = rnd();
    if (mode < 0.5) { f.background = at(randomInt(256)); f.bgGradient = undefined; }
    else {
      const i1 = randomInt(254), i2 = randomInt(254);
      f.bgGradient = { type: mode < 0.8 ? 'GRADIENT_2X2' : 'GRADIENT_2X2_C', ll: at(i1), lr: at(i1 + 1), ul: at(i2), ur: at(i2 + 1), cc: at(randomInt(256)) };
    }
  },
  random_flame: (_f, ly) => {
    // RandomFlameMutation: the layer becomes the first layer of a random flame (any style, sparse symmetry / field)
    const r = randomFlameInStyle('any');
    addRandomSymmetry(r, 'sparse'); addRandomWeightingField(r, 'sparse');
    const src = r.layers[0];
    ly.xforms = src.xforms; ly.final = src.final; ly.moreFinals = src.moreFinals; ly.palette = src.palette; ly.weight = src.weight;
  },
  random_ztransform: (f, ly, s) => {
    if (Math.abs(f.camPitch) < 2.5 && Math.abs(f.camBank) < 2.5) {
      f.camPitch = 30 + rnd() * 20; f.camYaw = 15 - rnd() * 30; f.camBank = 15 - rnd() * 30;
      if (rnd() < 0.5) { f.camPersp = 0.05 + rnd() * 0.2; f.camDOFArea = 0.2 + rnd() * 0.5; }
    }
    if (!f.preserveZ && rnd() > 0.33) f.preserveZ = true;
    for (const a of [0.2, 0.1, 0.05]) {
      const plane: Plane = rnd() < 0.5 ? 'yz' : 'zx';
      const x = pickXForm(ly, 0.25);
      const amount = a * s;
      if (rnd() < 0.75) globalTranslateX(x, -amount + 2 * rnd() * amount, -amount + 2 * rnd() * amount, rnd() < 0.25, plane);
      if (rnd() < 0.5) rotateX(x, -amount * 10 + 20 * rnd() * amount, rnd() < 0.25, plane);
      if (rnd() < 0.5) scaleX(x, -amount + 2 * rnd() * amount, rnd() < 0.75, rnd() < 0.75, rnd() < 0.25, plane);
    }
  },
  random_gradient: (_f, ly) => { ly.palette = randomPalette(); },
  random_parameter: (_f, ly, s) => {
    for (const k of [6, 5, 2]) {
      const amount = k * (0.25 + 0.75 * rnd()) * s;
      const all = allXForms(ly).flatMap(varsOf).filter((v) => Object.keys(v.params).length && !PARAM_BLACKLIST.has(v.name.toLowerCase()));
      if (all.length) varMutate(all[randomInt(all.length)], amount);
    }
  },
  similar_gradient: (_f, ly) => { ly.palette = similarGradient(ly.palette); },
  weighting_field: (_f, ly, s) => weightingFieldMutation(ly, s, DEFAULT_WFIELD_TYPES),
  color_type: (_f, ly) => {
    const X_TYPES: (XForm['colorType'] | 'DIFFUSION')[] = ['DIFFUSION', undefined, 'DISTANCE', 'TARGET', undefined, 'DIFFUSION', 'TARGETG', undefined, 'NONE', undefined, undefined, 'DIFFUSION'];
    const F_TYPES: (XForm['colorType'] | 'DIFFUSION')[] = ['NONE', 'DIFFUSION', 'NONE', undefined, undefined, 'DISTANCE', 'NONE', 'TARGET', undefined, 'DIFFUSION', undefined, undefined, 'NONE', 'TARGETG', undefined, undefined, 'DIFFUSION', 'NONE'];
    const set = (xs: XForm[], types: (XForm['colorType'] | 'DIFFUSION')[], isFinal: boolean) => {
      for (const x of xs) {
        const t = types[Math.min(randomInt(types.length), types.length - 1)];
        // UNSET resolves like JWildfire's Layer does: DIFFUSION for a transform, NONE for a final
        const ct = t === undefined ? (isFinal ? 'NONE' : 'DIFFUSION') : t;
        if (ct === 'DIFFUSION') { delete x.colorType; if (isFinal && x.colorSpeed === 0) x.colorSpeed = 0.5; }
        else { x.colorType = ct; if (ct === 'NONE') x.colorSpeed = 0; }
        if (ct === 'TARGET') x.targetColor = [randomInt(256) / 255, randomInt(256) / 255, randomInt(256) / 255];
      }
    };
    set(ly.xforms, X_TYPES, false); set(finalsOf(ly), F_TYPES, true);
  },
};
const PARAM_BLACKLIST = new Set(['truchet', 'mandelbrot', 'fract_formula_julia_wf', 'fract_formula_mand_wf', 'dc_perlin', 'snowflake_wf', 'crob', 'tree_js', 'brownian_js', 'dragon_js', 'maurer_lines', 'htree_js', 'gosperisland_js', 'rsquares_js', 'hilbert_js', 'koch_js', 'bubblet3d']);

/** BokehMutation.execute (strength 1): a DOF setup with a random blur shape, plus a scale-0 crackle transform the layer gets if it has none */
export function bokehMutation(f: Flame, ly: Layer) {
  f.camDOF = 0.1 + rnd() * 0.3; f.newDOF = true; f.camPitch = 30 + rnd() * 20; f.camYaw = 15 - rnd() * 30; f.camBank = 15 - rnd() * 30; f.camPersp = 0.05 + rnd() * 0.2; f.camDOFArea = 0.2 + rnd() * 0.5;
  let crackle: VarInstance | undefined;
  for (const x of ly.xforms) { for (let i = x.variations.length - 1; i >= 0; i--) { const v = x.variations[i]; if (v.name === 'crackle' && Math.abs(v.params.scale ?? 0) < 1e-10) { crackle = v; x.color = rnd(); break; } } if (crackle) break; }
  if (!crackle) { const x = newXForm(0.5); ly.xforms.push(x); crackle = mk('crackle', 1, { scale: 0 }); x.variations.push(crackle); fixXaos(ly); }
  crackle.weight = 1 + rnd() * 2; crackle.params.distort = 1.5 + rnd() * 1.5; crackle.params.cellsize = 0.5 + rnd() * 2;
  if (rnd() < 0.33) { f.focusX = 0.33 - rnd() * 0.66; f.focusY = 0.25 - rnd() * 0.5; f.focusZ = 0.1 - rnd() * 0.2; } else { f.focusX = 0; f.focusY = 0; f.focusZ = 0; }
  f.camDOFScale = 1.5 + rnd() * 2; f.camDOFRotate = 20 * rnd();
  const p = [0, 0, 0, 0, 0, 0]; let shape = 'BUBBLE', fade = 0;
  const r = rnd();
  if (r < 0.08) { shape = 'BUBBLE'; fade = 0.6 + rnd() * 0.4; }
  else if (r < 0.16) { shape = 'HEART'; fade = rnd() < 0.25 ? 0.2 + rnd() * 0.8 : 0; }
  else if (r < 0.2) { shape = 'CANNABISCURVE'; fade = rnd() < 0.25 ? 0.2 + rnd() * 0.8 : 0; }
  else if (r < 0.28) { shape = 'NBLUR'; p[0] = 3 + rnd() * 5; if (rnd() < 0.33) { p[1] = 2 + rnd() * 5; p[2] = 1; p[3] = 0; p[4] = rnd() < 0.33 ? 1 : 0; } }
  else if (r < 0.36) { shape = 'FLOWER'; p[0] = 0.3 + rnd() * 0.2; p[1] = 5 + rnd() * 5; }
  else if (r < 0.44) { shape = 'CLOVERLEAF'; fade = rnd() < 0.25 ? 0.2 + rnd() * 0.8 : 0; }
  else if (r < 0.52) { shape = 'SINEBLUR'; p[0] = 1.2 + rnd(); }
  else if (r < 0.6) { shape = 'PERLIN_NOISE'; p[0] = rnd(); p[1] = 1.2 + rnd() * 1.8; p[2] = 0.1 + rnd() * 0.4; }
  else if (r < 0.64) { shape = 'STARBLUR'; fade = 0.2 + rnd() * 0.8; p[0] = 4 + rnd() * 6; p[1] = 0.40162283177245455973959534526548; }
  else if (r < 0.72) { shape = 'TAURUS'; f.camDOFRotate = 0; p[0] = 2.5 + rnd(); p[1] = 4 + rnd() * 3; p[2] = 1.25 * rnd() * 0.5; p[3] = 0.9 + rnd() * 0.2; }
  else if (r < 0.8) { shape = 'BRUSH_STROKE'; f.camDOFRotate = 0; p[0] = 1 + randomInt(30); p[1] = rnd() > 0.25 ? 1 + randomInt(30) : 0; p[2] = rnd() > 0.75 ? 1 + randomInt(30) : 0; }
  else if (r < 0.92) { shape = 'RECT'; p[0] = 0.4 + rnd() * 0.4; }
  f.camDOFShape = shape; f.camDOFFade = fade; f.camDOFParams = p;
}

// ---------- gradients ----------
type HSL = { h: number; s: number; l: number };
/** HSLTransformer.rgb2hsl on 0..255 ints (hue 0..1, 1 for greys) */
function rgb2hsl(r8: number, g8: number, b8: number): HSL {
  const r = r8 / 255, g = g8 / 255, b = b8 / 255, max = Math.max(r, g, b), min = Math.min(r, g, b);
  const o: HSL = { h: 1, s: 0, l: (min + max) / 2 };
  if (Math.abs(o.l) <= 1e-10) return o;
  o.s = max - min;
  if (Math.abs(o.s) <= 1e-10) return o;
  o.s /= o.l <= 0.5 ? min + max : 2 - max - min;
  if (Math.abs(r - max) < 1e-10) o.h = g === min ? 5 + (max - b) / (max - min) : 1 - (max - g) / (max - min);
  else if (Math.abs(g - max) < 1e-10) o.h = b === min ? 1 + (max - r) / (max - min) : 3 - (max - b) / (max - min);
  else o.h = r === min ? 3 + (max - g) / (max - min) : 5 - (max - r) / (max - min);
  o.h /= 6;
  return o;
}
const roundColor = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
/** HSLTransformer.hsl2rgb → 0..255 ints */
function hsl2rgb(c: HSL): [number, number, number] {
  const v = c.l <= 0.5 ? c.l * (1 + c.s) : c.l + c.s - c.l * c.s;
  if (v <= 0) return [0, 0, 0];
  let hue = Math.max(0, Math.min(6, c.h * 6));
  const y = c.l + c.l - v, x = y + (v - y) * (hue - Math.trunc(hue)), z = v - (v - y) * (hue - Math.trunc(hue));
  const t = [[v, x, y], [z, v, y], [y, v, x], [y, z, v], [x, y, v], [v, y, z]][Math.min(5, Math.trunc(hue))];
  return [roundColor(t[0] * 255), roundColor(t[1] * 255), roundColor(t[2] * 255)];
}
/** SimilarGradientCreator.createKeyFrames + RandomGradientGenerator.generatePalette(keys, fade, not uniform) */
export function similarGradient(palette: RGB[]): RGB[] {
  type GC = { rgb: [number, number, number]; hsl: HSL };
  const gc = (r: number, g: number, b: number): GC => ({ rgb: [r, g, b], hsl: rgb2hsl(r, g, b) });
  const keyFrames = 20 + Math.trunc(rnd() * 10 + 0.5), weightedKeyFrames = 3 + Math.trunc(rnd() * 3 + 0.5), maxSize = 160;
  let clusters: GC[][] = palette.map((c) => [gc(roundColor(c[0] * 255), roundColor(c[1] * 255), roundColor(c[2] * 255))]);
  const avgHsl = (cs: GC[]): HSL => { let h = 0, s = 0, l = 0; for (const c of cs) { h += c.hsl.h; s += c.hsl.s; l += c.hsl.l; } return { h: h / cs.length, s: s / cs.length, l: l / cs.length }; };
  while (clusters.length > keyFrames) {
    const avgs = clusters.map(avgHsl);
    let minDist = Infinity, ii = -1, jj = -1;
    for (let i = 0; i < clusters.length; i++) for (let j = 0; j < clusters.length; j++) if (i !== j) {
      const d = Math.sqrt(avgs[i].h * avgs[j].h + avgs[i].s * avgs[j].s + avgs[i].l * avgs[j].l); // (JWildfire's "distance", kept as is)
      if (d < minDist) { minDist = d; ii = i; jj = j; }
    }
    const merged = [...clusters[ii], ...clusters[jj]];
    clusters = clusters.filter((_, k) => k !== ii && k !== jj); clusters.push(merged);
  }
  const maxSub = Math.max(...clusters.map((c) => c.length));
  let res: [number, number, number][] = [];
  for (const sub of clusters) {
    const maxCount = Math.trunc(weightedKeyFrames * sub.length / maxSub + 0.5);
    for (let i = 0; i < maxCount; i++) {
      if (!sub.length) break;
      const k = randomInt(sub.length), c = sub[k];
      if (rnd() < 0.125) res.push(hsl2rgb({ ...c.hsl, h: rnd() }));
      else if (rnd() < 0.25) res.push(hsl2rgb({ ...c.hsl, l: rnd() }));
      else res.push([...c.rgb]);
      sub.splice(k, 1);
    }
  }
  if (rnd() < 0.25) for (let i = res.length - 1; i > 0; i--) { const j = randomInt(i + 1); [res[i], res[j]] = [res[j], res[i]]; }
  while (res.length > maxSize) res.splice(randomInt(res.length), 1);
  return classicPalette(res, true, false);
}
/** RandomGradientGenerator.generateClassicPalette: key colours spread over 256 entries, optionally faded, widths jittered unless uniform */
export function classicPalette(keys: [number, number, number][], fade: boolean, uniform: boolean): RGB[] {
  if (keys.length === 0) return Array.from({ length: 256 }, () => [0, 0, 0] as RGB);
  if (keys.length === 1) return Array.from({ length: 256 }, () => [keys[0][0] / 255, keys[0][1] / 255, keys[0][2] / 255] as RGB);
  const n = keys.length - (fade ? 1 : 0);
  const cnt = new Array<number>(n).fill(0);
  let sum1 = 0, sum2 = 0;
  const idxScl = 256 / n;
  for (let i = 0; i < n; i++) { cnt[i] = Math.trunc(idxScl); sum1 += cnt[i]; sum2 += idxScl; if (sum2 - sum1 > 1) { cnt[i]++; sum1++; } }
  cnt[n - 1] += 256 - sum1;
  if (!uniform && idxScl > 3) for (let i = 0; i < n; i++) { const j = randomInt(n); const min = Math.min(cnt[i], cnt[j]); const r = Math.trunc(rnd() * min - min / 2); cnt[i] += r; cnt[j] -= r; }
  const out: RGB[] = [];
  let j = 0, l = 0;
  for (let i = 0; i < 256; i++) {
    const L = keys[l];
    let c: [number, number, number];
    if (fade) { const R = keys[l + 1], t = j / cnt[l]; c = [roundColor(L[0] + (R[0] - L[0]) * t), roundColor(L[1] + (R[1] - L[1]) * t), roundColor(L[2] + (R[2] - L[2]) * t)]; }
    else c = L;
    out.push([c[0] / 255, c[1] / 255, c[2] / 255]);
    j++;
    if (j >= cnt[l]) { j = 0; if (l < n - 1) l++; }
  }
  return out;
}

// ---------- weighting fields ----------
const DEFAULT_WFIELD_TYPES = ['CELLULAR_NOISE', 'CELLULAR_NOISE', 'CELLULAR_NOISE', 'CUBIC_NOISE', 'CUBIC_FRACTAL_NOISE', 'CUBIC_FRACTAL_NOISE', 'PERLIN_NOISE', 'PERLIN_FRACTAL_NOISE', 'PERLIN_FRACTAL_NOISE', 'PERLIN_FRACTAL_NOISE', 'PERLIN_FRACTAL_NOISE', 'PERLIN_FRACTAL_NOISE', 'WHITE_NOISE', 'SIMPLEX_NOISE', 'SIMPLEX_FRACTAL_NOISE', 'SIMPLEX_FRACTAL_NOISE', 'VALUE_NOISE', 'VALUE_FRACTAL_NOISE', 'VALUE_FRACTAL_NOISE'];
const CELL_RETURNS = ['CELL_VALUE', 'DISTANCE2', 'DISTANCE2', 'DISTANCE2', 'DISTANCE', 'DISTANCE_ADD', 'DISTANCE_ADD', 'DISTANCE_DIV', 'DISTANCE2', 'DISTANCE_DIV', 'DISTANCE_MUL', 'DISTANCE_MUL', 'DISTANCE_DIV', 'DISTANCE_DIV'] as const;
const CELL_DISTANCES = ['EUCLIDIAN', 'MANHATTAN', 'EUCLIDIAN', 'NATURAL'] as const;
const FRACTAL_TYPES = ['FBM', 'BILLOW', 'FBM', 'RIGID_MULTI', 'FBM'] as const;
const pickFrom = <T,>(l: readonly T[]): T => l[Math.min(randomInt(l.length), l.length - 1)];
/** WeightingFieldMutation.execute: the heavier transforms (at most half of them) get random fields, the rest lose theirs */
export function weightingFieldMutation(ly: Layer, strength: number, types: string[]) {
  if (!ly.xforms.length) return;
  const chosen = new Set<XForm>();
  if (ly.xforms.length === 1) chosen.add(ly.xforms[0]);
  else {
    const maxW = Math.max(...ly.xforms.map((x) => x.weight));
    while (chosen.size < 1) for (const x of ly.xforms) { if ((x.weight / maxW) * rnd() > 0.25) { chosen.add(x); if (chosen.size > ly.xforms.length / 2) break; } }
  }
  for (const x of ly.xforms) { if (chosen.has(x)) applyRandomWeightingField(x, strength, types); else delete x.wfield; }
}
function applyRandomWeightingField(x: XForm, s: number, types: string[]) {
  const type = pickFrom(types);
  const wf = defaultWeightingField(type);
  wf.input = rnd() < 0.25 ? 'POSITION' : 'AFFINE';
  if (rnd() > 0.42) wf.color = 0.5 - rnd();
  if (rnd() > 0.65) wf.jitter = rnd() > 0.65 ? 0.5 - rnd() : 0.25 - 0.5 * rnd();
  wf.params = [];
  const intensity = () => (rnd() > 0.33 ? (0.05 + rnd() * 0.2) * s : (0.25 - rnd() * 0.5) * s);
  let hasVarParam = false;
  const vars = varsOf(x);
  for (const v of vars) {
    const names = Object.keys(v.params);
    if (names.length && rnd() > 0.33) {
      // JWildfire writes param slots 1 and 2 for every match: the last matching variation wins
      const i1 = Math.min(randomInt(names.length), names.length - 1);
      wf.params = [{ varName: v.name, paramName: names[i1], intensity: intensity() }];
      if (names.length > 2) { let i2 = i1; while (i2 === i1) i2 = Math.min(randomInt(names.length), names.length - 1); wf.params.push({ varName: v.name, paramName: names[i2], intensity: intensity() }); }
      hasVarParam = true;
    }
  }
  if (!hasVarParam && vars.length > 1 && rnd() > 0.5) { const v = vars[Math.min(randomInt(vars.length), vars.length - 1)]; wf.params.push({ varName: v.name, paramName: 'amount', intensity: intensity() }); hasVarParam = true; }
  if ((hasVarParam && rnd() > 0.66) || (!hasVarParam && rnd() > 0.33)) wf.varAmount = rnd() > 0.33 ? (0.05 + rnd() * 0.5) * s : (0.25 - rnd() * 0.5) * s;
  else wf.varAmount = (0.01 + rnd() * 0.15) * s;
  const seedFreq = () => { wf.seed = 1 + Math.trunc(rnd() * 30000); wf.frequency = 0.75 + rnd() * 3; };
  if (type === 'CELLULAR_NOISE') { seedFreq(); wf.cellDistance = pickFrom(CELL_DISTANCES); wf.cellReturn = pickFrom(CELL_RETURNS); }
  else if (['CUBIC_NOISE', 'PERLIN_NOISE', 'SIMPLEX_NOISE', 'VALUE_NOISE'].includes(type)) seedFreq();
  else if (type.endsWith('FRACTAL_NOISE')) { seedFreq(); wf.octaves = 2 + Math.trunc(rnd() * 4); wf.lacunarity = 1.25 + rnd() * 3; wf.gain = 0.2 + rnd() * 0.75; wf.fractalType = pickFrom(FRACTAL_TYPES); }
  x.wfield = wf;
}

// ---------- the random symmetry / weighting-field generators (applied to a freshly generated flame) ----------
export type SymmetryKind = 'none' | 'all' | 'sparse' | 'xaxis' | 'yaxis' | 'point';
export const SYMMETRY_KINDS: { id: SymmetryKind; name: string }[] = [{ id: 'none', name: 'None' }, { id: 'all', name: '(All)' }, { id: 'sparse', name: '(All, sparse)' }, { id: 'xaxis', name: 'X axis' }, { id: 'yaxis', name: 'Y axis' }, { id: 'point', name: 'Point' }];
export function addRandomSymmetry(f: Flame, kind: SymmetryKind) {
  let k = kind;
  if (k === 'sparse') k = rnd() > 0.66 ? 'all' : 'none';
  if (k === 'all') k = (['none', 'xaxis', 'yaxis', 'point'] as SymmetryKind[])[randomInt(4)];
  if (k === 'none') { f.postSymmetry = undefined; return; }
  const centreX = 0.5 - rnd(), centreY = 0.5 - rnd();
  if (k === 'point') f.postSymmetry = { type: 'POINT', order: Math.trunc(2 + rnd() * 6), centreX, centreY, distance: 0, rotation: 0 };
  else if (k === 'xaxis') f.postSymmetry = { type: 'X_AXIS', order: 3, centreX, centreY, distance: -0.25 + rnd() * 1.75, rotation: -30 + rnd() * 60 };
  else f.postSymmetry = { type: 'Y_AXIS', order: 3, centreX, centreY, distance: -0.25 + rnd() * 0.75, rotation: 0 };
}
export type WFieldKind = 'none' | 'all' | 'sparse' | 'basic' | 'cellular' | 'fractal';
export const WFIELD_KINDS: { id: WFieldKind; name: string }[] = [{ id: 'none', name: 'None' }, { id: 'all', name: '(All)' }, { id: 'sparse', name: '(All, sparse)' }, { id: 'basic', name: 'Basic noise' }, { id: 'cellular', name: 'Cellular noise' }, { id: 'fractal', name: 'Fractal noise' }];
const WFIELD_LISTS: Record<'basic' | 'cellular' | 'fractal', string[]> = {
  basic: ['CELLULAR_NOISE', 'CUBIC_NOISE', 'PERLIN_NOISE', 'SIMPLEX_NOISE', 'VALUE_NOISE'],
  cellular: ['CELLULAR_NOISE'],
  fractal: ['CUBIC_FRACTAL_NOISE', 'PERLIN_FRACTAL_NOISE', 'SIMPLEX_FRACTAL_NOISE', 'VALUE_FRACTAL_NOISE'],
};
export function addRandomWeightingField(f: Flame, kind: WFieldKind) {
  let k = kind;
  if (k === 'sparse') k = rnd() > 0.66 ? 'all' : 'none';
  if (k === 'all') k = (['cellular', 'basic', 'fractal'] as const)[randomInt(3)]; // (the image-map generator needs an image)
  if (k === 'none') { for (const ly of f.layers) for (const x of ly.xforms) delete x.wfield; return; }
  for (const ly of f.layers) weightingFieldMutation(ly, 1, WFIELD_LISTS[k]);
}

// ---------- RandomFlameGeneratorSampler's coverage measures, on RGBA pixels ----------
/** calculateCoverage: the share of pixels clearly off the background (after a 3×3 Sobel when `useFilter`) */
export function coverage(px: Uint8ClampedArray, w: number, h: number, bg: RGB, useFilter: boolean): number {
  let img = px;
  if (useFilter) {
    // ConvolveTransformer SOBEL_3X3 per channel: magnitude of the two 3×3 gradients, clamped to 0..255
    const out = new Uint8ClampedArray(px.length);
    const at = (x: number, y: number, c: number) => px[((Math.max(0, Math.min(h - 1, y))) * w + Math.max(0, Math.min(w - 1, x))) * 4 + c];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      const gx = -at(x - 1, y - 1, c) - 2 * at(x - 1, y, c) - at(x - 1, y + 1, c) + at(x + 1, y - 1, c) + 2 * at(x + 1, y, c) + at(x + 1, y + 1, c);
      const gy = -at(x - 1, y - 1, c) - 2 * at(x, y - 1, c) - at(x + 1, y - 1, c) + at(x - 1, y + 1, c) + 2 * at(x, y + 1, c) + at(x + 1, y + 1, c);
      out[(y * w + x) * 4 + c] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
    }
    img = out;
  }
  const [br, bgc, bb] = [Math.round(bg[0] * 255), Math.round(bg[1] * 255), Math.round(bg[2] * 255)];
  let n = 0;
  if (br === 0 && bgc === 0 && bb === 0) { for (let i = 0; i < w * h; i++) if (img[i * 4] > 29 || img[i * 4 + 1] > 15 || img[i * 4 + 2] > 78) n++; }
  else { for (let i = 0; i < w * h; i++) if (Math.abs(img[i * 4] - br) > 29 && Math.abs(img[i * 4 + 1] - bgc) > 15 && Math.abs(img[i * 4 + 2] - bb) > 78) n++; }
  return n / (w * h);
}
/** createSimplifiedRefImage: PixelizeTransformer with a 5-px grid (every pixel takes its 5×5 block's top-left value) */
export function pixelize(px: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(px.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const sx = x - (x % 5), sy = y - (y % 5); for (let c = 0; c < 4; c++) out[(y * w + x) * 4 + c] = px[(sy * w + sx) * 4 + c]; }
  return out;
}
/** calculateDiffCoverage: the share of pixels whose 5-px block differs clearly from the reference's */
export function diffCoverage(px: Uint8ClampedArray, ref: Uint8ClampedArray, w: number, h: number): number {
  const img = pixelize(px, w, h);
  let n = 0;
  for (let i = 0; i < w * h; i++) if (Math.abs(img[i * 4] - ref[i * 4]) > 29 || Math.abs(img[i * 4 + 1] - ref[i * 4 + 1]) > 15 || Math.abs(img[i * 4 + 2] - ref[i * 4 + 2]) > 78) n++;
  return n / (w * h);
}
