// Random flame STYLES — JWildfire's RandomFlameGenerator classes, transcribed (LGPL 2.1+, © Andreas
// Maschke and contributors; see NOTICE.md). Each style builds transforms the way its generator does:
// the same variations, weights, probabilities and affine moves, through a port of
// XFormTransformService. The gradient is ours (JWildfire assigns a random one separately).
import type { Flame, XForm, VarInstance, RGB } from './flame';
import { defaultFlame, defaultXForm, normalizeFlame } from './flame';
import { VARIATIONS, defaultParams } from './variations';
import { randomPalette } from './palette';

const rnd = Math.random;
const randomInt = (n: number) => Math.floor(rnd() * n);
const FTOI = (v: number) => (v >= 0 ? Math.floor(v + 0.5) : -Math.floor(-v + 0.5));

// ---------- JWildfire's XForm coefficient model ----------
// coeff00..coeff21 are flam3's a d b e c f; our affine is [a, b, c, d, e, f] = [c00, c10, c20, c01, c11, c21].
interface JX {
  c00: number; c01: number; c10: number; c11: number; c20: number; c21: number;
  p00: number; p01: number; p10: number; p11: number; p20: number; p21: number;
  weight: number; color: number; colorSymmetry: number;
  vars: VarInstance[];
  xaos: Record<number, number>;
  colorMods?: number[];
}
const newXForm = (): JX => ({ c00: 1, c01: 0, c10: 0, c11: 1, c20: 0, c21: 0, p00: 1, p01: 0, p10: 0, p11: 1, p20: 0, p21: 0, weight: 0.5, color: 0, colorSymmetry: 0, vars: [], xaos: {} });

const SMALL_EPSILON = 1e-12;
type M3 = [[number, number, number], [number, number, number], [number, number, number]];
const identity = (): M3 => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const mul = (a: M3, b: M3): M3 => {
  const r = identity();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return r;
};
const getM = (x: JX, post: boolean): M3 => post
  ? [[x.p00, x.p01, x.p20], [x.p10, x.p11, x.p21], [0, 0, 1]]
  : [[x.c00, x.c01, x.c20], [x.c10, x.c11, x.c21], [0, 0, 1]];
const setM = (x: JX, post: boolean, m: M3) => {
  if (post) { x.p00 = m[0][0]; x.p01 = m[0][1]; x.p10 = m[1][0]; x.p11 = m[1][1]; x.p20 = m[0][2]; x.p21 = m[1][2]; }
  else { x.c00 = m[0][0]; x.c01 = m[0][1]; x.c10 = m[1][0]; x.c11 = m[1][1]; x.c20 = m[0][2]; x.c21 = m[1][2]; }
};
/** XFormTransformService.rotate */
function rotate(x: JX, angle: number, post = false) {
  if (Math.abs(angle) < SMALL_EPSILON) return;
  const a = (angle * Math.PI) / 180;
  const m1 = identity();
  m1[0][0] = Math.cos(a); m1[0][1] = -Math.sin(a); m1[1][0] = Math.sin(a); m1[1][1] = Math.cos(a);
  setM(x, post, mul(getM(x, post), m1));
}
/** XFormTransformService.localTranslate (in the transform's own frame) */
function localTranslate(x: JX, dx: number, dy: number, post = false) {
  if (Math.abs(dx) < SMALL_EPSILON && Math.abs(dy) < SMALL_EPSILON) return;
  const m1 = identity();
  m1[0][2] = dx; m1[1][2] = dy;
  setM(x, post, mul(getM(x, post), m1));
}
/** XFormTransformService.globalTranslate */
function globalTranslate(x: JX, dx: number, dy: number, post = false) {
  if (Math.abs(dx) < SMALL_EPSILON && Math.abs(dy) < SMALL_EPSILON) return;
  if (post) { x.p20 += dx; x.p21 += dy; } else { x.c20 += dx; x.c21 += dy; }
}
/** XFormTransformService.scale */
function scale(x: JX, s: number, xs: boolean, ys: boolean, post = false) {
  if (Math.abs(s - 1) < SMALL_EPSILON) return;
  if (xs) { if (post) { x.p00 *= s; x.p01 *= s; } else { x.c00 *= s; x.c01 *= s; } }
  if (ys) { if (post) { x.p10 *= s; x.p11 *= s; } else { x.c10 *= s; x.c11 *= s; } }
}

// ---------- variations ----------
const known = (name: string) => name in VARIATIONS;
// A few exotic variations (glsl_*, crop_trapezoid…) carry parameter NAMES with spaces, which cannot be
// XML attributes: a flame using them could never be saved as .flame. The random pickers avoid them.
const XML_NAME = /^[A-Za-z_][\w.-]*$/;
const saveable = (name: string) => ((VARIATIONS[name] as { params?: { name: string }[] })?.params ?? []).every((p) => XML_NAME.test(p.name));
const flagsOf = (name: string): string[] => ((VARIATIONS[name] as { flags?: string[] })?.flags ?? []);
/** VariationFuncList.filterVariations: keep the names this build knows */
const filterVariations = (names: string[]) => names.filter((n) => known(n) && saveable(n));

/** ExperimentalSimpleRandomFlameGenerator.FNCLST_EXPERIMENTAL (duplicates kept on purpose — they weight the draw) */
const FNCLST_EXPERIMENTAL_RAW = ['blur3D', 'bubble', 'escher', 'rays', 'epispiral_wf', 'curl3D', 'diamond', 'juliaq', 'julia3Dq', 'post_juliaq', 'post_julia3Dq',
  'cloverleaf_wf', 'disc', 'sech', 'loonie', 'exp', 'cosh', 'split', 'waves2_3D', 'wedge_sph', 'circlize', 'heart_wf', 'bwraps7', 'colorscale_wf', 'gdoffs', 'taurus', 'dc_crackle_wf',
  'mandelbrot', 'spirograph', 'target', 'eclipse', 'butterfly3D', 'cpow', 'pre_subflame_wf', 'conic', 'julia3D', 'cell', 'dc_hexes_wf', 'stripes', 'post_mirror_wf', 'flipcircle', 'waves2_3D', 'juliac',
  'colorscale_wf', 'crackle', 'truchet', 'cannabiscurve_wf', 'cpow', 'subflame_wf', 'post_smartcrop', 'glynnSim3', 'flower', 'fourth', 'heart', 'julia3D', 'disc2', 'polar2', 'farblur', 'waves3_wf', 'waves2b',
  'foci', 'scry', 'flux', 'bwraps7', 'splitbrdr', 'checks', 'colorscale_wf', 'falloff2', 'sinusoidal3d', 'cloverleaf_wf', 'lazyTravis', 'kaleidoscope', 'eclipse', 'hemisphere', 'flipy', 'phoenix_julia',
  'popcorn2', 'sec', 'lazysusan', 'sin', 'separation', 'bi_linear', 'hexnix3D', 'popcorn2_3D', 'julian3Dx', 'post_mirror_wf', 'heart_wf', 'mcarpet', 'mandelbrot', 'cannabiscurve_wf', 'colormap_wf', 'juliac',
  'rose_wf', 'edisc', 'blocky', 'octagon', 'murl', 'waves2', 'twintrian', 'coth', 'super_shape', 'post_colormap_wf', 'waves2_3D', 'auger', 'pre_wave3D_wf', 'hexes', 'dc_hexes_wf', 'barycentroid', 'spirograph',
  'truchet', 'epispiral', 'waves4_wf', 'glynnSim2', 'tanh', 'bipolar', 'cot', 'horseshoe', 'target', 'wedge', 'unpolar', 'pre_boarders2', 'modulus', 'mobius', 'bubble2', 'bwraps7', 'colorscale_wf', 'truchet',
  'collideoscope', 'xheart', 'waves2b', 'kaleidoscope', 'glynnSim2', 'twoface', 'cross', 'tangent3D', 'csc', 'curve', 'boarders2', 'julian3Dx', 'csch', 'bent2', 'splits', 'julian3Dx', 'whorl', 'xtrb',
  'post_mirror_wf', 'mandelbrot', 'sphericalN', 'waves2_3D', 'cloverleaf_wf', 'cannabiscurve_wf', 'tan', 'blob3D', 'julia3D', 'hypertile1', 'svf', 'dc_crackle_wf', 'log', 'cos', 'oscilloscope', 'wedge_julia',
  'bwraps7', 'heart_wf', 'linearT3D', 'juliac', 'hexes', 'truchet', 'spirograph', 'glynnSim3', 'pdj', 'popcorn', 'hypertile2', 'waves2_3D', 'parabola', 'rings2', 'spherical3D', 'spiral', 'rectangles', 'foci_3D',
  'sintrange', 'waves2b', 'elliptic', 'waves', 'swirl', 'glynnSim1', 'eclipse', 'bwraps7', 'layered_spiral', 'heart_wf', 'colorscale_wf', 'boarders', 'secant2', 'waffle', 'lissajous', 'hypertile', 'circus',
  'lazyTravis', 'ovoid3d', 'circleblur', 'sineblur', 'starblur', 'lace_js', 'japanese_maple_leaf', 'fdisc'];
let FNCLST_EXPERIMENTAL: string[] | null = null; // filtered on first use: the JWildfire registry loads lazily at startup
const experimental = () => { if (!FNCLST_EXPERIMENTAL) FNCLST_EXPERIMENTAL = filterVariations(FNCLST_EXPERIMENTAL_RAW); return FNCLST_EXPERIMENTAL[randomInt(FNCLST_EXPERIMENTAL.length)]; };

/** VariationFuncList.getRandomVariationname: any variation this build renders (the registry minus image/text ones,
 *  which are not in it anyway); optionally only 2D (no 3d/z flag) or 3D ones. */
let allNames: string[] | null = null;
function randomVariationName(type?: '2d' | '3d'): string {
  if (!allNames) allNames = Object.keys(VARIATIONS).filter(saveable);
  for (let tries = 0; tries < 200; tries++) {
    const n = allNames[randomInt(allNames.length)];
    const is3d = flagsOf(n).includes('3d');
    if (type === '2d' && is3d) continue;
    if (type === '3d' && !is3d) continue;
    return n;
  }
  return 'linear';
}
/** Galaxies/Duality/JulianDisc exclude the odd ones out */
function randomVariationNamePlain(type?: '2d' | '3d'): string {
  for (;;) {
    const n = randomVariationName(type);
    if (!n.startsWith('fract') && !n.startsWith('inflate') && !n.startsWith('pre_') && !n.startsWith('post_') && !n.startsWith('prepost_') && n !== 'flatten') return n;
  }
}

const v = (name: string, weight: number, params: Record<string, number> = {}, priority?: number): VarInstance => {
  if (!known(name)) name = 'linear';
  return { name, weight, params: { ...defaultParams(name), ...params }, ...(priority !== undefined ? { priority } : {}) };
};
const addVar = (x: JX, weight: number, name: string, params?: Record<string, number>, priority?: number) => { x.vars.push(v(name, weight, params, priority)); return x.vars[x.vars.length - 1]; };

/** VariationFunc.mutate on a random variation of the layer (RandomParamMutation.setRandomFlameProperty) */
function mutateRandomParam(xs: JX[], amount: number) {
  const all = xs.flatMap((x) => x.vars).filter((vi) => Object.keys(vi.params).length);
  if (!all.length) return;
  const vi = all[randomInt(all.length)];
  const keys = Object.keys(vi.params);
  const k = keys[randomInt(keys.length)];
  const def = (VARIATIONS[vi.name] as { params?: { name: string; int?: boolean }[] })?.params?.find((p) => p.name === k);
  const o = vi.params[k];
  if (def?.int) { let da = FTOI(amount); if (da < 1) da = 1; vi.params[k] = o >= 0 ? o + da : o - da; }
  else {
    let step: number;
    if (o < 1e-6 || rnd() < 0.3) step = o >= 0 ? 0.1 * amount : -0.1 * amount;
    else step = o >= 0 ? (o / 100) * amount : (o / -100) * amount;
    vi.params[k] = o + step;
  }
}
const randomParamMutation = (xs: JX[], strength = 1) => {
  mutateRandomParam(xs, 6 * (0.25 + 0.75 * rnd()) * strength);
  mutateRandomParam(xs, 5 * (0.25 + 0.75 * rnd()) * strength);
  mutateRandomParam(xs, 2 * (0.25 + 0.75 * rnd()) * strength);
};
/** XForm.randomizeModColorEffects */
const randomizeModColorEffects = (x: JX) => {
  const pair = (): [number, number] => [1 - 2 * rnd(), rnd() < 0.33 ? 1 - 2 * rnd() : 0];
  x.colorMods = [...pair(), ...pair(), ...pair(), ...pair()];
};

// ---------- the flame being built ----------
interface JFlame {
  centreX: number; centreY: number; camRoll: number; camPitch: number; camYaw: number; camPersp: number;
  camZoom: number; ppu: number; width: number; height: number; gamma?: number; whiteBg?: boolean;
  xforms: JX[]; finals: JX[];
}
const newFlame = (): JFlame => ({ centreX: 0, centreY: 0, camRoll: 0, camPitch: 0, camYaw: 0, camPersp: 0, camZoom: 1, ppu: 50, width: 800, height: 600, xforms: [], finals: [] });
const distributeColors = (f: JFlame) => { const n = f.xforms.length; if (n > 1) f.xforms.forEach((x, i) => { x.color = i / (n - 1); }); };
const randomizeColors = (f: JFlame) => { for (const x of f.xforms) x.color = rnd(); };

function toXForm(j: JX, n: number): XForm {
  const x = defaultXForm();
  x.affine = [j.c00, j.c10, j.c20, j.c01, j.c11, j.c21];
  x.post = [j.p00, j.p10, j.p20, j.p01, j.p11, j.p21];
  x.weight = j.weight;
  x.color = Math.min(1, Math.max(0, j.color));
  x.colorSpeed = Math.min(1, Math.max(0, (1 - j.colorSymmetry) / 2)); // JWildfire colour symmetry ↔ colour speed
  x.variations = j.vars.length ? j.vars : [v('linear', 1)];
  const keys = Object.keys(j.xaos);
  if (keys.length) x.xaos = Array.from({ length: n }, (_, i) => (i in j.xaos ? j.xaos[i] : 1));
  if (j.colorMods) x.colorMods = j.colorMods;
  return x;
}
function toFlame(j: JFlame, styleName: string, useFilter: boolean, palette: RGB[]): Flame {
  const f = defaultFlame(palette);
  f.name = `${styleName} - ${Math.floor(rnd() * 1e9)}`;
  f.centerX = j.centreX; f.centerY = j.centreY;
  f.rotation = (j.camRoll * Math.PI) / 180;
  f.camPitch = j.camPitch; f.camYaw = j.camYaw; f.camPersp = j.camPersp;
  // JWildfire: pixels per unit × zoom on its own frame; ours is relative to ¼ of the shorter side
  f.zoom = (j.ppu * j.camZoom) / (0.25 * Math.min(j.width, j.height));
  f.brightness = 4; f.gamma = j.gamma ?? 4; f.gammaThreshold = 0.01; f.vibrancy = 1; f.contrast = 1;
  f.filterRadius = useFilter ? 0.75 : 0;
  if (j.whiteBg) f.background = [1, 1, 1];
  const n = j.xforms.length;
  f.layers[0].xforms = j.xforms.map((x) => toXForm(x, n));
  f.layers[0].final = j.finals.length ? toXForm(j.finals[0], n) : null;
  f.layers[0].moreFinals = j.finals.slice(1).map((x) => toXForm(x, n));
  return normalizeFlame(f, palette);
}

// ---------- the styles ----------
interface Style { id: string; name: string; filter: boolean; build: () => JFlame }

const STYLES: Style[] = [
  { id: 'bubbles', name: 'Bubbles', filter: true, build: () => {
    const f = newFlame(); f.camZoom = 0.5; f.ppu = 75 + rnd() * 42;
    { const x = newXForm(); f.xforms.push(x); x.weight = 12 + rnd() * 80;
      addVar(x, 2 + rnd() * 4, 'spherical');
      addVar(x, 0.05 + rnd() * 0.5, rnd() < 0.15 ? experimental() : rnd() < 0.8 ? 'eyefish' : 'fisheye');
      x.colorSymmetry = 0.991 + rnd() * 0.08;
      scale(x, 0.5 - rnd() * 0.5, true, true); rotate(x, 180 - rnd() * 360); localTranslate(x, 3 - 6 * rnd(), 3 - 6 * rnd());
      if (rnd() < 0.33) localTranslate(x, 0.75 - 1.5 * rnd(), 0.75 - 1.5 * rnd(), true); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + rnd() * 0.8;
      if (rnd() < 0.05) { addVar(x, 0.01 + rnd() * 0.4, 'bubble'); addVar(x, 0.01 + rnd() * 0.04, 'checks', { size: 5, x: 3, y: 3 }); }
      else addVar(x, 0.1 + rnd() * 0.5, 'bubble');
      addVar(x, 4 + rnd() * 2, 'pre_blur');
      x.colorSymmetry = -0.5;
      scale(x, 1.1 + rnd() * 1.9, true, true); localTranslate(x, 0.75 - 1.5 * rnd(), 0.75 - 1.5 * rnd()); rotate(x, 30 - rnd() * 60); }
    if (rnd() > 0.25) { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + rnd() * 1.5;
      addVar(x, 0.5, rnd() > 0.8 ? experimental() : rnd() > 0.5 ? 'linear3D' : 'noise');
      x.colorSymmetry = -0.5; rotate(x, 30 - rnd() * 60); if (rnd() < 0.5) scale(x, 0.5 + rnd() * 1.5, true, true); }
    if (rnd() > 0.5) { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + rnd() * 1.5;
      addVar(x, 0.5, rnd() > 0.8 ? (rnd() > 0.75 ? randomVariationName() : experimental()) : rnd() > 0.5 ? 'linear3D' : 'gaussian_blur');
      x.colorSymmetry = -0.5; rotate(x, 30 - rnd() * 60); if (rnd() < 0.5) scale(x, 0.15 + rnd() * 1.25, true, true); }
    distributeColors(f);
    return f;
  } },
  { id: 'julians', name: 'Julians', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200;
    const primary = rnd() < 0.666 ? 'julian' : 'juliascope';
    const rr = (a: number, b: number) => a + rnd() * (b - a);
    const ri = (a: number, b: number) => FTOI(a + rnd() * (b - a));
    const randomAffine = (x: JX) => {
      const s = rnd();
      if (s > 0.2 && s <= 0.4) rotate(x, rr(0, 360));
      if (s > 0.4 && s <= 0.6) localTranslate(x, rr(0, 1.5), rr(0, 1.5));
      if (s > 0.6 && s <= 0.8) scale(x, rr(0.25, 1.5), true, true);
      else if (s > 0.8) { rotate(x, rr(0, 360)); localTranslate(x, rr(0, 1.5), rr(0, 1.5)); scale(x, rr(0, 1.2), true, true); }
    };
    const randomPostAffine = (x: JX) => {
      const s = rnd();
      if (s > 0.6 && s <= 0.7) rotate(x, rr(0, 360), true);
      if (s > 0.7 && s <= 0.8) localTranslate(x, rr(0, 1.5), rr(0, 1.5), true);
      if (s > 0.8 && s <= 0.9) scale(x, rr(0.25, 1.5), true, true, true);
      else if (s > 0.9) { rotate(x, rr(0, 360), true); localTranslate(x, rr(0, 1.5), rr(0, 1.5), true); scale(x, rr(0, 1.2), true, true, true); }
    };
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.00001 + rnd() * 20; x.color = rnd();
      if (rnd() < 0.5) addVar(x, -1 + rnd() * 2, randomVariationName());
      addVar(x, -2 + rnd() * 4, primary, { power: ri(1, 7), dist: -2 + 4 * rnd() }); randomAffine(x); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.00001 + rnd() * 6; x.color = rnd();
      addVar(x, -2 + rnd() * 4, primary, { power: ri(1, 7), dist: -2 + 4 * rnd() });
      if (rnd() < 0.8) { addVar(x, -1 + rnd() * 2, 'linear'); addVar(x, -1 + rnd() * 2, randomVariationName()); }
      x.xaos[1] = rnd() < 0.5 ? 0 : rnd();
      scale(x, 0.5 + rnd() * 0.5, rnd() < 0.5, rnd() < 0.5); randomAffine(x); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.00001 + rnd() * 2; x.color = rnd();
      addVar(x, -2 + rnd() * 4, primary, { power: ri(1, 5), dist: -2 + 4 * rnd() }); randomAffine(x); randomPostAffine(x); }
    return f;
  } },
  { id: 'splits', name: 'Splits', filter: true, build: () => {
    const TX1 = filterVariations(['bipolar', 'boarders', 'boarders2', 'bubble', 'butterfly', 'bwraps7', 'cannabiscurve_wf', 'circlize', 'cloverleaf_wf', 'collideoscope', 'cos', 'cosh', 'cot', 'coth', 'csc', 'csch',
      'cubic3D', 'cubicLattice_3D', 'dc_perlin', 'diamond', 'disc2', 'eJulia', 'edisc', 'elliptic', 'epispiral', 'epispiral_wf', 'ex', 'exp', 'exponential', 'eyefish', 'flower', 'foci', 'xtrb', 'glynnSim1', 'glynnia',
      'heart', 'hexnix3D', 'julia', 'julian', 'julia3D', 'julia3Dz', 'juliascope', 'layered_spiral', 'mobius', 'ngon', 'phoenix_julia', 'pie', 'popcorn2_3D', 'radial_blur', 'rays', 'ripple', 'spherical', 'sphericalN',
      'spiral', 'tan', 'tanh', 'waves2_3D', 'wedge_sph', 'whorl', 'xheart']);
    const f = newFlame(); f.camZoom = 2; f.gamma = 2.7; f.ppu = 200;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + 0.4 * rnd(); addVar(x, 1, TX1[randomInt(TX1.length)]); rotate(x, rnd() < 0.5 ? 90 : -90); x.color = 0.6 + rnd() * 0.2; x.colorSymmetry = rnd() * 0.2; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5;
      addVar(x, 1, 'splits', rnd() < 0.25 ? { x: -0.5 + rnd(), y: -0.5 + rnd() } : { x: -1 + rnd() * 4, y: 0 });
      scale(x, 0.6 + rnd() * 0.8, true, true); scale(x, 1 + rnd() * 0.8, true, true, true); x.color = 0.4 + rnd() * 0.2; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.2 + 0.4 * rnd(); addVar(x, 1, experimental()); rotate(x, rnd() < 0.5 ? 90 : -90); x.color = 0.6 + rnd() * 0.2; x.colorSymmetry = rnd() * 0.2; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.3 + 0.8 * rnd(); addVar(x, 1, rnd() < 0.5 ? 'noise' : 'cylinder');
      scale(x, 5 + rnd() * 10, false, true, true); scale(x, 0.5 + rnd() * 0.75, true, false, true); scale(x, 0.2 + rnd() * 0.8, true, true); x.color = 0.6 + rnd() * 0.2; x.colorSymmetry = rnd() * 0.2; }
    return f;
  } },
  { id: 'spherical', name: 'Spherical', filter: false, build: () => {
    const f = newFlame(); f.camRoll = 90; f.camZoom = 2.4; f.camPersp = 0.32; f.ppu = 200;
    const turn = () => (rnd() < 0.5 ? (rnd() < 0.5 ? 180 : 90) : -90);
    { const x = newXForm(); f.xforms.push(x); x.weight = 4 + 12 * rnd(); addVar(x, 1, 'spherical3D'); rotate(x, turn()); globalTranslate(x, 1, 0); x.color = 1; x.colorSymmetry = 0.9 + rnd() * 0.2; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 3 + 11 * rnd(); addVar(x, 1, 'spherical3D'); rotate(x, turn()); x.color = 0.5; x.colorSymmetry = 0.9 + rnd() * 0.2; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.3 + 0.2 * rnd(); addVar(x, 1, 'linear3D'); rotate(x, 90); globalTranslate(x, Math.trunc(2 + rnd() * 2), 0); x.color = rnd(); x.colorSymmetry = 0; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.2 + 0.15 * rnd(); addVar(x, 1, 'linear3D'); rotate(x, 90); globalTranslate(x, -Math.trunc(2 + rnd() * 2), 0); x.color = rnd(); x.colorSymmetry = 0; }
    const max = randomInt(4);
    for (let i = 0; i < max; i++) { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + 0.3 * rnd(); addVar(x, 1, experimental());
      rotate(x, 90 - rnd() * 180); scale(x, 0.2 + 0.2 * rnd(), true, true); globalTranslate(x, 0.25 - rnd() * 0.5, 0.25 - rnd() * 0.5); x.color = rnd(); x.colorSymmetry = 0; }
    randomizeColors(f);
    return f;
  } },
  { id: 'ghosts', name: 'Ghosts', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200; f.camZoom = 2; f.camRoll = -90;
    { const x = newXForm(); f.xforms.push(x); x.weight = 1.5 + rnd();
      addVar(x, 0.5 + rnd(), 'spherical');
      addVar(x, 0.05 + rnd() * 0.15, randomVariationName(rnd() > 0.75 ? '2d' : '3d'), {}, rnd() < 0.5 ? -1 : 1);
      x.color = 0.4 + rnd() * 0.2; x.colorSymmetry = 0.82 + rnd() * 0.16;
      rotate(x, 180); scale(x, 2 + rnd() * 25, true, true); localTranslate(x, 0.5 * (0.5 - rnd()), 0.5 - rnd()); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 1.5 + rnd(); x.color = rnd();
      addVar(x, 0.05 + rnd() * 0.25, 'linear');
      if (rnd() < 0.66) addVar(x, 0.025 + rnd() * 0.1, 'radial_blur', { angle: rnd() * Math.PI }); else addVar(x, 0.05 + rnd() * 0.5, randomVariationName());
      addVar(x, 0.05 + rnd() * 0.5, randomVariationName(), {}, rnd() < 0.5 ? -1 : 1);
      scale(x, 0.1 + rnd() * 0.8, false, true); scale(x, 1 + rnd() * 3, true, false); scale(x, 0.1 + rnd() * 0.8, true, true);
      rotate(x, 180); localTranslate(x, 2 - 4 * rnd(), 0.5 * (2 - 4 * rnd())); x.colorSymmetry = rnd(); }
    return f;
  } },
  { id: 'tentacle', name: 'Tentacle', filter: true, build: () => {
    const f = newFlame(); f.ppu = 200 + rnd() * 100;
    const nx = Math.trunc(2 + rnd() * 3), ny = Math.trunc(2 + rnd() * 3);
    const xMin = -nx * 0.5, yMin = -ny * 0.5 + 1;
    let scl = 1;
    for (let y = 0; y < ny; y++) for (let x0 = 0; x0 < nx; x0++) {
      const x = newXForm(); x.weight = 0.5 + rnd() * 99.5; f.xforms.push(x);
      globalTranslate(x, xMin + x0, yMin + y);
      rotate(x, (rnd() < 0.5 ? 360 : -360) * rnd(), true);
      localTranslate(x, -1 + 2 * rnd(), -1 + 2 * rnd(), true);
      scl *= 0.75 + rnd() / 4; scale(x, scl, true, true, true);
      addVar(x, rnd() * 0.9 + 0.1, rnd() > 0.25 ? experimental() : 'linear3D');
      x.color = rnd();
    }
    return f;
  } },
  { id: 'linear', name: 'Linear', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200;
    const n = Math.trunc(1 + rnd() * 5);
    let scl = 1, tscl = 2, r0 = 0;
    const contRot = rnd() < 0.5;
    for (let i = 0; i < n; i++) {
      const x = newXForm(); f.xforms.push(x);
      if (contRot) { r0 += 45 * rnd() - 9 * rnd(); rotate(x, r0); } else rotate(x, (rnd() < 0.5 ? 360 : -360) * rnd());
      localTranslate(x, (2 * rnd() - 1) * tscl, (2 * rnd() - 1) * tscl);
      scl *= 0.8 + rnd() * 0.1; tscl *= 0.8 + rnd() * 0.1;
      scale(x, scl, true, true);
      x.color = rnd(); addVar(x, rnd() * 0.5 + 0.5, 'linear3D'); x.weight = scl * rnd() * 19.9 + 0.1;
    }
    return f;
  } },
  { id: 'sierpinsky', name: 'Sierpinsky', filter: false, build: () => {
    const f = newFlame(); f.camRoll = randomInt(8) * -45; f.ppu = 200; f.camZoom = 4.56;
    const posx = [-0.5, 0.5, 0.5, -0.5], posy = [-0.5, -0.5, 0.5, 0.5];
    for (let i = 0; i < 4; i++) {
      const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.c20 = posx[i]; x.c21 = posy[i]; x.color = rnd(); x.colorSymmetry = rnd();
      if (i === 3) {
        if (rnd() < 0.5) { addVar(x, 0.5, rnd() < 0.25 ? 'dc_linear' : 'linear3D'); rotate(x, 5 - 10 * rnd()); localTranslate(x, 0.1 - 0.2 * rnd(), 0.1 - 0.2 * rnd()); scale(x, 1 - 0.1 * rnd(), rnd() < 0.75, rnd() > 0.25); }
        const amount = rnd() < 0.33 ? 0.5 - rnd() : rnd() < 0.67 ? 0.25 + 0.5 * rnd() : 0.5;
        addVar(x, amount, experimental());
        if (rnd() < 0.5) scale(x, 1 - 0.1 * rnd(), rnd() < 0.75, rnd() > 0.25);
      } else addVar(x, 0.5, rnd() < 0.25 ? 'dc_linear' : 'linear3D');
    }
    return f;
  } },
  { id: 'galaxies', name: 'Galaxies', filter: false, build: () => {
    const f = newFlame(); f.camRoll = 1.49758722; f.width = 601; f.height = 338; f.ppu = 92.48366013; f.camZoom = 0.72 + rnd() * 0.42; f.centreX = 1.5357526; f.centreY = -0.4416446;
    { const x = newXForm(); f.xforms.push(x); x.weight = 25.75871591; x.color = 0.74488914; x.colorSymmetry = 0;
      x.c00 = 1; x.c10 = 0; x.c20 = 1.09171281; x.c01 = 0; x.c11 = 1; x.c21 = -1.22115911;
      addVar(x, rnd() > 0.25 ? 1 - 2 * rnd() : 1, randomVariationNamePlain());
      if (rnd() > 0.5) mutateRandomParam(f.xforms, 1); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 286.87636036; x.color = 0.90312262; x.colorSymmetry = 0.95;
      x.c00 = 0.96333808; x.c10 = 0.12845865; x.c20 = 0.31387449; x.c01 = -0.12845865; x.c11 = 0.96333808; x.c21 = 0.08003269;
      addVar(x, 1, 'sec'); }
    if (rnd() > 0.5) for (const x of f.xforms) randomizeModColorEffects(x);
    return f;
  } },
  { id: 'machine', name: 'Machine', filter: true, build: () => {
    const f = newFlame(); f.centreX = rnd() - 0.5; f.centreY = rnd() - 0.5; f.ppu = 200;
    { const x = newXForm(); f.xforms.push(x); x.weight = 32.16406573; x.color = 0.74488914; x.colorSymmetry = 0;
      x.c00 = 1; x.c10 = 0; x.c20 = 1.82382673; x.c01 = 0; x.c11 = 1; x.c21 = -1.8206164;
      addVar(x, 1, randomVariationName());
      scale(x, 1.25 - rnd() * 0.5, true, true); rotate(x, 360 * rnd()); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 257.3608644; x.color = 0.90312262; x.colorSymmetry = 0.95;
      x.c00 = 0.82496229; x.c10 = 0.55902149; x.c20 = 0.05455807; x.c01 = -0.55902149; x.c11 = 0.82496229; x.c21 = 0.73105973;
      x.xaos[0] = 0.54127492;
      addVar(x, 0.75 + rnd() * 0.5, 'yin_yang', { radius: 0.25 + rnd() * 0.5, ang1: 0, ang2: 0.1 + rnd() * 0.5, dual_t: 1, outside: 1 });
      scale(x, 1.25 - rnd() * 0.5, true, true); rotate(x, 360 * rnd()); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); }
    randomizeColors(f);
    return f;
  } },
  { id: 'brokat', name: 'Brokat', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200; f.camZoom = 2; f.camRoll = -90;
    { const x = newXForm(); f.xforms.push(x); x.weight = 1.5 + rnd();
      addVar(x, 1.6 + rnd() * 0.8, 'curl', { c1: -1, c2: 0.001 + rnd() * 0.0199 });
      x.color = 0.4 + rnd() * 0.2; x.colorSymmetry = 0.82 + rnd() * 0.16; rotate(x, 180); localTranslate(x, 1, 0, true);
      x.xaos[0] = 0; x.xaos[1] = 1; x.xaos[2] = 0; x.xaos[3] = 0; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.05 + rnd() * 0.35;
      const names = filterVariations(['juliascope', 'julia3D', 'julia3Dz', 'julian']);
      addVar(x, 1, names[randomInt(names.length)], { power: rnd() < 0.33 ? 2 : rnd() < 0.5 ? 3 : 4 });
      x.color = 0.5 + rnd() * 0.5; x.colorSymmetry = 0.5; x.xaos[0] = 1; x.xaos[1] = 0; x.xaos[2] = 1; x.xaos[3] = 1; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.4 + rnd() * 0.2;
      addVar(x, 0.01 + rnd() * 0.04, rnd() < 0.33 ? 'bubble' : experimental()); addVar(x, 5 + rnd() * 10, 'pre_blur');
      x.color = 0.1 + rnd() * 0.3; x.colorSymmetry = 0; localTranslate(x, -1, 0, true); x.xaos[0] = 1; x.xaos[1] = 1; x.xaos[2] = 0; x.xaos[3] = 0; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.4 + rnd() * 0.2;
      addVar(x, 0.01 + rnd() * 0.04, experimental());
      if (rnd() > 0.5) addVar(x, (0.01 + rnd() * 0.04) * 0.5, randomVariationName(), {}, -1);
      x.color = 0.1 + rnd() * 0.3; x.colorSymmetry = 0; rotate(x, rnd() * 360, true); scale(x, 1.1 + rnd() * 3, true, true, true);
      x.xaos[0] = 1; x.xaos[1] = 1; x.xaos[2] = 1; x.xaos[3] = 1; }
    return f;
  } },
  { id: 'spirals', name: 'Spirals', filter: false, build: () => {
    const f = newFlame(); f.centreX = -rnd() * 0.5; f.centreY = -rnd() * 0.5; f.ppu = 200; f.camRoll = 90 - rnd() * 180; f.camZoom = 2;
    { const x = newXForm(); f.xforms.push(x); x.weight = 25 + rnd() * 55;
      x.c00 = 0.23168009; x.c10 = -0.87153216; x.c20 = -1.09851548; x.c01 = 1.01859563; x.c11 = 0.23718475; x.c21 = 0.30609214;
      addVar(x, 1, 'waves2', { scalex: 0.04933602 + rnd() * 0.04, scaley: 0.06933602, freqx: 2.98088993, freqy: 2.98088993 });
      addVar(x, 0.001 + rnd() * 0.001, randomVariationName());
      if (rnd() < 0.33) addVar(x, 0.0001 + rnd() * 0.0001, randomVariationName(), {}, -1);
      x.color = 0.4 + rnd() * 0.2; x.colorSymmetry = 0.82 + rnd() * 0.16; localTranslate(x, 0.5 * (0.5 - rnd()), 0.5 - rnd(), true); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5;
      x.c00 = 1.1144261; x.c10 = -0.1144261; x.c20 = -0.1144261; x.c01 = 0.03033403; x.c11 = 0.96966597; x.c21 = -0.03033403;
      addVar(x, 0.2 + rnd() * 0.2, randomVariationName());
      if (rnd() > 0.42) addVar(x, 0.1 + rnd() * 0.1, randomVariationName(), {}, -1);
      if (rnd() > 0.42) addVar(x, 0.0001 + rnd() * 0.0001, randomVariationName(), {}, 1);
      x.color = rnd(); x.colorSymmetry = rnd(); }
    return f;
  } },
  { id: 'phoenix', name: 'Phoenix', filter: false, build: () => {
    const f = newFlame(); f.centreX = -0.43687754; f.centreY = -0.84902392; f.width = 1920; f.height = 1080; f.camZoom = 0.6 + rnd(); f.ppu = 200;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = 0; x.colorSymmetry = 0;
      x.c00 = 0.465496; x.c10 = -0.702038; x.c20 = -0.554347; x.c01 = 0.174105; x.c11 = 0.956244; x.c21 = -0.366901;
      addVar(x, 1, 'linear'); scale(x, 1.125 - rnd() * 0.25, true, true); rotate(x, 360 * rnd());
      if (rnd() > 0.5) { scale(x, 1.125 - rnd() * 0.25, true, true); rotate(x, 360 * rnd()); } }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = 1; x.colorSymmetry = 0;
      x.c00 = 0.706669; x.c10 = -0.515362; x.c20 = -1.045926; x.c01 = 0.515362; x.c11 = 0.706669; x.c21 = 0.383008;
      x.xaos[0] = 0.25 + rnd() * 0.1; x.xaos[1] = 1.15 + rnd() * 0.2;
      addVar(x, 1, 'linear'); addVar(x, 1, 'curl', { c1: 0.05 + rnd() * 0.1, c2: 0.86 + rnd() * 0.1 });
      localTranslate(x, 0.5 - rnd(), 0.5 - rnd()); if (rnd() > 0.5) localTranslate(x, 0.5 - rnd(), 0.5 - rnd()); if (rnd() > 0.666) rotate(x, -12 + 24 * rnd()); }
    randomizeColors(f);
    return f;
  } },
  { id: 'juliandisc', name: 'Julian disc', filter: false, build: () => {
    const nonBlur = () => { for (;;) { const r = experimental(); if (!r.includes('blur') && !r.includes('pre_') && !r.includes('post_') && !r.includes('prepost_') && r !== 'flatten' && !r.includes('inflate')) return r; } };
    const f = newFlame(); f.ppu = 200 + rnd() * 60;
    const place = (x: JX) => {
      if (rnd() < 0.33) globalTranslate(x, -0.0125 + 0.025 * rnd(), -0.0125 + 0.025 * rnd()); else if (rnd() < 0.75) globalTranslate(x, -0.125 + 0.25 * rnd(), -0.125 + 0.25 * rnd());
      if (rnd() < 0.15) rotate(x, 6 - rnd() * 12); else if (rnd() < 0.3) rotate(x, -45); else if (rnd() < 0.75) rotate(x, 90 - rnd() * 180);
    };
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + rnd() * 0.5;
      let power = FTOI(200 - rnd() * 400); if (power === 0 || power === 1 || power === -1) power = -30;
      addVar(x, [0.25, 0.5, 0.75][randomInt(3)], 'julian', { power });
      if (rnd() < 0.5) addVar(x, 0.001 + rnd() * 0.039, rnd() > 0.5 ? nonBlur() : 'gaussian_blur');
      x.colorSymmetry = -1; x.color = 0; place(x);
      if (rnd() < 0.5) scale(x, 1.2 - rnd() * 0.4, true, true); if (rnd() < 0.5) scale(x, 1.2 - rnd() * 0.4, true, true, true); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 2 + rnd() * 24; addVar(x, 1, 'disc');
      if (rnd() < 0.125) addVar(x, 0.001 + rnd() * 0.039, rnd() > 0.5 ? nonBlur() : 'gaussian_blur');
      x.color = 0.5 + rnd() * 0.5; x.colorSymmetry = 0.6 + rnd() * 0.33; place(x); }
    if (rnd() < 0.15 && rnd() < 0.5) {
      const fin = newXForm(); f.finals.push(fin);
      const list = filterVariations(['auger', 'bent', 'bent2', 'boarders', 'bubble', 'butterfly', 'bwraps7', 'cosine', 'curve', 'cylinder', 'diamond', 'disc', 'eclipse', 'edisc', 'elliptic', 'ex', 'exp', 'exponential',
        'eyefish', 'fan', 'fan2', 'fisheye', 'heart_wf', 'hemisphere', 'horseshoe', 'hyperbolic', 'julia', 'julia3D', 'julia3Dz', 'julian', 'juliascope', 'linearT', 'log', 'mobius', 'ngon', 'oscilloscope',
        'rings', 'rings2', 'scry', 'xtrb', 'sec', 'sin', 'sinh', 'sinusoidal', 'spherical', 'swirl', 'tan', 'tangent', 'tanh', 'boarders2', 'polar']);
      addVar(fin, 2 + rnd() * 2, list[randomInt(list.length)]);
    }
    randomizeColors(f);
    return f;
  } },
  { id: 'julianrings', name: 'Julian rings', filter: false, build: () => {
    const f = newFlame(); f.camRoll = rnd() * 360; f.width = 962; f.height = 541; f.ppu = 100 + rnd() * 200; f.camZoom = 3 + rnd() * 2; f.whiteBg = rnd() > 0.5;
    { const x = newXForm(); f.xforms.push(x); x.weight = 1 + rnd(); x.color = rnd(); x.colorSymmetry = -1 + 2 * rnd();
      x.c00 = 1.87375; x.c10 = 0; x.c20 = 0; x.c01 = 0; x.c11 = 1.873754; x.c21 = 0;
      const names = filterVariations(['julia3D', 'jubiQ', 'julia3Dq', 'julia3Dz', 'julian', 'julian2', 'julian3Dx', 'juliaq', 'juliascope']);
      addVar(x, 1, names[Math.min(randomInt(names.length), names.length - 1)], { power: rnd() > 0.5 ? 10 + rnd() * 5000 : -2 - rnd() * 100 });
      if (rnd() > 0.75) addVar(x, -0.125 + rnd() * 0.25, randomVariationName()); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 25 + rnd() * 25; x.color = rnd(); x.colorSymmetry = -1 + 2 * rnd();
      x.c00 = 0.990702; x.c10 = 0.154938; x.c20 = -0.140437; x.c01 = -0.136053; x.c11 = 0.92994; x.c21 = 0.159563;
      addVar(x, rnd() > 0.33 ? 1 : 0.9 + rnd() * 0.3, 'rings2', { val: 1 });
      if (rnd() > 0.5) addVar(x, -0.25 + rnd() * 0.5, randomVariationName());
      rotate(x, 45 - 90 * rnd()); localTranslate(x, -0.125 + 0.25 * rnd(), -0.125 + 0.25 * rnd()); }
    return f;
  } },
  { id: 'xenomorph', name: 'Xenomorph', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.1 + rnd() * 0.4; x.color = 0.74488914; x.colorSymmetry = 0;
      x.c00 = 0.75610133; x.c10 = -0.74186252; x.c20 = 5.25778565; x.c01 = 0.74186252; x.c11 = 0.75610133; x.c21 = -0.34949139;
      x.p00 = -0.42606416; x.p10 = 0.44290131; x.p01 = -0.10610689; x.p11 = -0.40885976; x.p20 = -2.81712; x.p21 = 7.390367;
      addVar(x, 1.43, 'bubble'); addVar(x, 0.012, 'linear');
      if (rnd() > 0.6) addVar(x, -0.249, 'radial_blur', { angle: 0.609835 }); else addVar(x, rnd() < 0.5 ? -0.249 : 0.5 - rnd(), randomVariationName());
      addVar(x, 1, 'power'); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 10 + rnd() * 8; x.color = 0.90312262; x.colorSymmetry = 0.95;
      x.c00 = -0.85421287; x.c10 = -0.63343313; x.c20 = 1.09379129; x.c01 = -0.63343313; x.c11 = 0.85421287; x.c21 = -0.20406326;
      x.p00 = 0.84389756; x.p10 = -0.35800434; x.p01 = 0.43174917; x.p11 = 0.89637273; x.p20 = -0.945758; x.p21 = -0.4502584;
      x.xaos[0] = 2.05; x.xaos[1] = 1.25; x.xaos[2] = 0.9;
      addVar(x, 0.008, 'linear'); addVar(x, 10.72, 'spherical');
      scale(x, 1.25 - rnd() * 0.5, true, true); rotate(x, 36 * rnd()); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.7; x.color = 0.47272985; x.colorSymmetry = 0;
      x.c00 = 0.68724662; x.c10 = -0.72642418; x.c20 = -2.14812602; x.c01 = 0.72642418; x.c11 = 0.68724662; x.c21 = 2.39994214;
      x.p00 = 0.60646395; x.p10 = 0.79511098; x.p01 = -0.79511098; x.p11 = 0.60646395; x.p20 = -1.06135064; x.p21 = -0.6369509;
      addVar(x, 1, 'bipolar', { shift: 0 }); }
    return f;
  } },
  { id: 'outlines', name: 'Outlines', filter: false, build: () => {
    const f = newFlame(); f.centreX = 0.1 - rnd() * 0.2; f.centreY = 0.1 - rnd() * 0.2; f.camZoom = 1.05 + rnd() * 0.15; f.ppu = 200;
    { const x = newXForm(); f.xforms.push(x); x.weight = 26.32600695; x.color = 0.74488914; x.colorSymmetry = 0;
      x.c00 = 1; x.c10 = 0; x.c20 = 1.25325706; x.c01 = 0; x.c11 = 1; x.c21 = -0.93791588; addVar(x, 1, 'spherical3D'); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 224.48613855; x.color = 0.90312262; x.colorSymmetry = 0.95;
      x.c00 = 1.07039553; x.c10 = 0.4803511; x.c20 = -0.2742276; x.c01 = -0.4803511; x.c11 = 1.07039553; x.c21 = -0.04097021;
      if (rnd() > 0.1) addVar(x, 1, 'loonie2', { sides: 3 + Math.trunc(rnd() * 6), star: 0.1 + rnd() * 0.3, circle: 0.1 + rnd() * 0.3 }); else addVar(x, 1, randomVariationName());
      rotate(x, 360 * rnd(), true); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 146.3720822; x.color = 0.6312262; x.colorSymmetry = 0.12;
      x.c00 = 0.9031166; x.c10 = 0.38124873; x.c20 = 0.37716179; x.c01 = -0.38124873; x.c11 = 0.9031166; x.c21 = -0.53412667;
      x.xaos[2] = 0; addVar(x, 0.4 + rnd() * 0.2, randomVariationName()); rotate(x, 360 * rnd(), true); }
    return f;
  } },
  { id: 'duality', name: 'Duality', filter: true, build: () => {
    const f = newFlame(); f.ppu = 200;
    const pick = () => randomVariationNamePlain(rnd() > 0.75 ? '2d' : '3d');
    const jitter = () => { if (rnd() > 0.25) randomParamMutation(f.xforms); if (rnd() > 0.5) randomParamMutation(f.xforms); if (rnd() > 0.75) randomParamMutation(f.xforms); };
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.01 + rnd() * 50; x.color = 0.74488914; x.colorSymmetry = 0; addVar(x, 1, pick()); localTranslate(x, 2 - 4 * rnd(), 2 - 4 * rnd()); }
    jitter();
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.1 + rnd() * 400; x.color = 0.90312262; x.colorSymmetry = 0.95;
      addVar(x, rnd() < 0.33 ? 0.5 + rnd() * 0.5 : 1, pick());
      scale(x, 1.25 - rnd() * 0.5, true, true); rotate(x, 36 * rnd()); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd());
      if (rnd() < 0.5) x.xaos[0] = rnd() < 0.5 ? 0.1 + rnd() * 0.5 : 3 + rnd() * 7; }
    jitter();
    if (rnd() < 0.1) { const x = newXForm(); f.xforms.push(x); x.weight = 0.01 + rnd() * 200; x.color = 0.6312262; x.colorSymmetry = 0.12;
      addVar(x, 0.5 + rnd() * 0.75, pick()); x.xaos[2] = 0; scale(x, 1.05 - rnd() * 0.45, true, true); rotate(x, 24 * rnd()); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); }
    jitter();
    return f;
  } },
];

export interface RandomStyleInfo { id: string; name: string }
export const RANDOM_STYLES: RandomStyleInfo[] = STYLES.map(({ id, name }) => ({ id, name }));

/** A random flame in the given style ('any' picks a style at random, like JWildfire's "All"). */
export function randomFlameInStyle(styleId: string, palette: RGB[] = randomPalette()): Flame {
  const style = STYLES.find((s) => s.id === styleId) ?? STYLES[randomInt(STYLES.length)];
  return toFlame(style.build(), style.name, style.filter, palette);
}
