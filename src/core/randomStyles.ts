// Random flame STYLES — JWildfire's RandomFlameGenerator classes, transcribed (LGPL 2.1+, © Andreas
// Maschke and contributors; see NOTICE.md). Each style builds transforms the way its generator does:
// the same variations, weights, probabilities and affine moves, through a port of
// XFormTransformService. The gradient is ours (JWildfire assigns a random one separately).
import type { Flame, XForm, VarInstance, RGB } from './flame';
import { defaultFlame, defaultXForm, defaultLayer, defaultSolidRender, normalizeFlame } from './flame';
import { VARIATIONS, defaultParams } from './variations';
import { randomPalette } from './palette';
import { flameToXML } from './flameXML';

const rnd = Math.random;
const randomInt = (n: number) => Math.floor(rnd() * n);
const FTOI = (v: number) => (v >= 0 ? Math.floor(v + 0.5) : -Math.floor(-v + 0.5));

// ---------- JWildfire's XForm coefficient model ----------
// coeff00..coeff21 are flam3's a d b e c f; our affine is [a, b, c, d, e, f] = [c00, c10, c20, c01, c11, c21].
interface JX {
  c00: number; c01: number; c10: number; c11: number; c20: number; c21: number;
  p00: number; p01: number; p10: number; p11: number; p20: number; p21: number;
  /** the YZ / ZX-plane affines (JWildfire EditPlane.YZ / ZX: yzCoeff00…21, zxCoeff00…21) and their posts, identity when absent */
  yz?: Aff6; zx?: Aff6; yzPost?: Aff6; zxPost?: Aff6;
  weight: number; color: number; colorSymmetry: number;
  vars: VarInstance[];
  xaos: Record<number, number>;
  colorMods?: number[];
  /** JWildfire DrawMode OPAQUE / HIDDEN: opacity */
  opacity?: number;
  colorType?: XForm['colorType']; targetColor?: [number, number, number];
}
const newXForm = (): JX => ({ c00: 1, c01: 0, c10: 0, c11: 1, c20: 0, c21: 0, p00: 1, p01: 0, p10: 0, p11: 1, p20: 0, p21: 0, weight: 0.5, color: 0, colorSymmetry: 0, vars: [], xaos: {} });
type Aff6 = [number, number, number, number, number, number]; // [c00, c10, c20, c01, c11, c21] like our Affine
/** the edit plane XFormTransformService works in: 'yz' / 'zx' route rotate/scale/translate to those coefficient sets (same layout as the XY ones) */
let editPlane: 'xy' | 'yz' | 'zx' = 'xy';
const ID6: Aff6 = [1, 0, 0, 0, 1, 0];
const planeOf = (x: JX, post: boolean): Aff6 => {
  const k = (editPlane === 'yz' ? (post ? 'yzPost' : 'yz') : (post ? 'zxPost' : 'zx')) as 'yz' | 'zx' | 'yzPost' | 'zxPost';
  return (x[k] ??= [...ID6] as Aff6);
};

const SMALL_EPSILON = 1e-12;
type M3 = [[number, number, number], [number, number, number], [number, number, number]];
const identity = (): M3 => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const mul = (a: M3, b: M3): M3 => {
  const r = identity();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return r;
};
const getM = (x: JX, post: boolean): M3 => {
  if (editPlane !== 'xy') { const z = planeOf(x, post); return [[z[0], z[3], z[2]], [z[1], z[4], z[5]], [0, 0, 1]]; }
  return post
    ? [[x.p00, x.p01, x.p20], [x.p10, x.p11, x.p21], [0, 0, 1]]
    : [[x.c00, x.c01, x.c20], [x.c10, x.c11, x.c21], [0, 0, 1]];
};
const setM = (x: JX, post: boolean, m: M3) => {
  if (editPlane !== 'xy') { const z = planeOf(x, post); z[0] = m[0][0]; z[1] = m[1][0]; z[2] = m[0][2]; z[3] = m[0][1]; z[4] = m[1][1]; z[5] = m[1][2]; return; }
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
  if (editPlane !== 'xy') { const z = planeOf(x, post); z[2] += dx; z[5] += dy; return; }
  if (post) { x.p20 += dx; x.p21 += dy; } else { x.c20 += dx; x.c21 += dy; }
}
/** XFormTransformService.scale */
function scale(x: JX, s: number, xs: boolean, ys: boolean, post = false) {
  if (Math.abs(s - 1) < SMALL_EPSILON) return;
  if (editPlane !== 'xy') { const z = planeOf(x, post); if (xs) { z[0] *= s; z[3] *= s; } if (ys) { z[1] *= s; z[4] *= s; } return; }
  if (xs) { if (post) { x.p00 *= s; x.p01 *= s; } else { x.c00 *= s; x.c01 *= s; } }
  if (ys) { if (post) { x.p10 *= s; x.p11 *= s; } else { x.c10 *= s; x.c11 *= s; } }
}
/** run `fn` with XFormTransformService working in the ZX plane (AbstractAffine3DRandomFlameGenerator.rotateXForm / scaleXForm) */
const inZX = (fn: () => void) => { editPlane = 'zx'; try { fn(); } finally { editPlane = 'xy'; } };
const inPlane = (plane: 'xy' | 'yz' | 'zx', fn: () => void) => { editPlane = plane; try { fn(); } finally { editPlane = 'xy'; } };

// ---------- variations ----------
const known = (name: string) => name in VARIATIONS;
// A few exotic variations (glsl_*, crop_trapezoid…) carry parameter NAMES with spaces, which JWildfire
// writes verbatim — their .flame files are not strict XML (WilderFire and JWildfire read them, other
// tools may not). The random pickers avoid them; a flame that arrives with one is kept and saved as is.
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
export const experimental = () => { if (!FNCLST_EXPERIMENTAL) FNCLST_EXPERIMENTAL = filterVariations(FNCLST_EXPERIMENTAL_RAW); return FNCLST_EXPERIMENTAL[randomInt(FNCLST_EXPERIMENTAL.length)]; };

/** VariationFuncList.getRandomVariationname: any variation this build renders (the registry minus image/text ones,
 *  which are not in it anyway); optionally only 2D (no 3d/z flag) or 3D ones. */
let allNames: string[] | null = null;
export function randomVariationName(type?: '2d' | '3d'): string {
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

/** VariationFunc.mutate: nudge one random parameter of the instance (ints by ≥1, doubles by a tenth of the amount or a percentage) */
export function varMutate(vi: VarInstance, amount: number) {
  const keys = Object.keys(vi.params);
  if (!keys.length) return;
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
/** VariationFunc.mutate on a random variation of the layer (RandomParamMutation.setRandomFlameProperty) */
function mutateRandomParam(xs: JX[], amount: number) {
  const all = xs.flatMap((x) => x.vars).filter((vi) => Object.keys(vi.params).length);
  if (!all.length) return;
  varMutate(all[randomInt(all.length)], amount);
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
  centreX: number; centreY: number; camRoll: number; camPitch: number; camYaw: number; camBank: number; camPersp: number;
  camZoom: number; ppu: number; width: number; height: number; gamma?: number; whiteBg?: boolean;
  camDOF: number; newDOF: boolean; preserveZ: boolean; camPosX: number; camPosY: number; camPosZ: number; camZ: number;
  xforms: JX[]; finals: JX[];
  /** the generator's choice of spatial filter (isUseFilter), decided while building when it is random */
  filter?: boolean;
  /** flame-level settings beyond the fields above (background, tone, solid rendering, post symmetry…), applied before normalisation */
  extra?: (f: Flame) => void;
}
const newFlame = (): JFlame => ({ centreX: 0, centreY: 0, camRoll: 0, camPitch: 0, camYaw: 0, camBank: 0, camPersp: 0, camZoom: 1, ppu: 50, width: 800, height: 600, camDOF: 0, newDOF: false, preserveZ: false, camPosX: 0, camPosY: 0, camPosZ: 0, camZ: 0, xforms: [], finals: [] });
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
  const nonId = (a?: Aff6) => a && a.some((v, i) => Math.abs(v - ID6[i]) > 1e-12);
  if (nonId(j.yz)) x.yz = [...j.yz!]; if (nonId(j.zx)) x.zx = [...j.zx!]; if (nonId(j.yzPost)) x.yzPost = [...j.yzPost!]; if (nonId(j.zxPost)) x.zxPost = [...j.zxPost!];
  if (j.opacity !== undefined) x.opacity = j.opacity;
  if (j.colorType) x.colorType = j.colorType;
  if (j.targetColor) x.targetColor = j.targetColor;
  return x;
}
function toFlame(j: JFlame, styleName: string, useFilter: boolean, palette: RGB[]): Flame {
  const f = defaultFlame(palette);
  f.name = `${styleName} - ${Math.floor(rnd() * 1e9)}`;
  f.centerX = j.centreX; f.centerY = j.centreY;
  f.rotation = (j.camRoll * Math.PI) / 180;
  f.camPitch = j.camPitch; f.camYaw = j.camYaw; f.camBank = j.camBank; f.camPersp = j.camPersp;
  f.camDOF = j.camDOF; f.newDOF = j.newDOF; f.preserveZ = j.preserveZ;
  f.camPosX = j.camPosX; f.camPosY = j.camPosY; f.camPosZ = j.camPosZ; f.camZ = j.camZ;
  // JWildfire: pixels per unit × zoom on its own frame; ours is relative to ¼ of the shorter side
  f.zoom = (j.ppu * j.camZoom) / (0.25 * Math.min(j.width, j.height));
  f.brightness = 4; f.gamma = j.gamma ?? 4; f.gammaThreshold = 0.01; f.vibrancy = 1; f.contrast = 1;
  f.filterRadius = (j.filter ?? useFilter) ? 0.75 : 0;
  if (j.whiteBg) f.background = [1, 1, 1];
  const n = j.xforms.length;
  f.layers[0].xforms = j.xforms.map((x) => toXForm(x, n));
  const fin = (x: JX) => { const o = toXForm(x, n); if (!(o.weight > 0)) o.weight = 1; return o; }; // JWildfire gives finals weight 0 (unused there)
  f.layers[0].final = j.finals.length ? fin(j.finals[0]) : null;
  f.layers[0].moreFinals = j.finals.slice(1).map(fin);
  j.extra?.(f);
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

// ---------- the second batch (2026-08-23): the remaining generators ----------
/** SimpleRandomFlameGenerator.FNCLST_ORIGINAL (duplicates weight the draw) */
const FNCLST_ORIGINAL_RAW = ['blur3D', 'bubble', 'curl3D', 'diamond', 'waves2_3D', 'disc', 'julia3D', 'heart', 'julia3D', 'hemisphere', 'waffle', 'bwraps7', 'horseshoe', 'boarders', 'blob3D', 'xtrb', 'xheart', 'julia3D', 'pdj', 'popcorn', 'rings2', 'rose_wf',
  'spherical3D', 'spiral', 'rectangles', 'blur', 'waves', 'swirl', 'secant2', 'boarders2'];
let FNCLST_ORIGINAL: string[] | null = null;
const original = () => { if (!FNCLST_ORIGINAL) FNCLST_ORIGINAL = filterVariations(FNCLST_ORIGINAL_RAW); return FNCLST_ORIGINAL[randomInt(FNCLST_ORIGINAL.length)]; };
const pickOf = (names: string[]) => { const l = filterVariations(names); return l.length ? l[randomInt(l.length)] : 'linear'; };
const HIDDEN = 0; // DrawMode.HIDDEN → opacity 0

STYLES.push(
  { id: 'simple', name: 'Simple (stunning)', filter: true, build: () => {
    const f = newFlame(); f.ppu = 200;
    const maxXForms = Math.trunc(2 + rnd() * 5);
    let scl = 1;
    for (let i = 0; i < maxXForms; i++) {
      const x = newXForm(); f.xforms.push(x);
      if (rnd() < 0.5) rotate(x, 360 * rnd()); else rotate(x, -360 * rnd());
      localTranslate(x, rnd() - 1, rnd() - 1);
      scl *= 0.75 + rnd() / 4;
      scale(x, scl, true, true);
      x.color = rnd();
      addVar(x, rnd() * 0.8 + 0.2, 'linear3D');
      if (rnd() > 0.33) addVar(x, rnd() * 0.5, original());
      x.weight = rnd() * 0.9 + 0.1;
    }
    return f;
  } },
  { id: 'affine3d', name: 'Affine3D', filter: true, build: () => {
    const f = newFlame(); f.ppu = 200;
    f.camPitch = 49 - 100 * rnd(); f.camYaw = 22 - 44 * rnd(); f.camBank = 44 - 88 * rnd(); f.camPersp = 0.06 + 0.36 * rnd();
    const maxXForms = Math.trunc(1 + rnd() * 5);
    let scl = 1, tscl = 2, r0 = 0;
    const contRot = rnd() < 0.5, withShear = rnd() < 0.5;
    for (let i = 0; i < maxXForms; i++) {
      const x = newXForm();
      const a = addVar(x, rnd() * 0.5 + 0.5, 'affine3D');
      f.xforms.push(x);
      if (contRot) { r0 += 45 * rnd() - 9 * rnd(); a.params.rotateZ = r0; }
      else a.params.rotateZ = rnd() < 0.5 ? 360 * rnd() : -360 * rnd();
      a.params.rotateX = 18 - 36 * rnd(); a.params.rotateY = 18 - 36 * rnd();
      a.params.translateX = (2 * rnd() - 1) * tscl; a.params.translateY = (2 * rnd() - 1) * tscl; a.params.translateZ = (2 * rnd() - 1) * tscl;
      if (withShear) for (const k of ['shearXY', 'shearXZ', 'shearYX', 'shearYZ', 'shearZX', 'shearZY']) if (rnd() > 0.5) a.params[k] = (2 * rnd() - 1) * tscl * 0.1;
      scl *= 0.8 + rnd() * 0.1; tscl *= 0.8 + rnd() * 0.1;
      a.params.scaleX = scl; a.params.scaleY = scl; a.params.scaleZ = scl;
      x.color = rnd();
      x.weight = scl * rnd() * 19.9 + 0.1;
      if (rnd() < 0.56) addVar(x, rnd() * 0.25 + 0.25, randomVariationName());
    }
    f.filter = rnd() > 0.42;
    return f;
  } },
  { id: 'edisc', name: 'EDisc', filter: true, build: () => {
    const f = newFlame(); f.width = 638; f.height = 359; f.camRoll = 360 * rnd(); f.centreX = 0.5 - rnd(); f.centreY = 0.5 - rnd();
    f.ppu = 90; f.camZoom = 0.55 + rnd() * 0.25; f.preserveZ = true;
    { const x = newXForm(); f.xforms.push(x); x.weight = 3.5 + rnd(); x.color = rnd(); x.colorSymmetry = 0.9;
      x.c00 = 0.88067445; x.c10 = -0.50058358; x.c20 = 0.111166; x.c01 = 0.50058358; x.c11 = 0.88067445; x.c21 = -0.873736;
      x.p00 = -0.172546; x.p10 = -1.052875; x.p01 = 1.052875; x.p11 = -0.172546; x.p20 = -0.057973; x.p21 = 0.051023;
      x.xaos[0] = 1 + rnd() * 10; x.xaos[2] = 0;
      addVar(x, 15, 'edisc');
      scale(x, 1.25 - rnd() * 0.5, true, true); rotate(x, 360 * rnd()); }
    const withTX3 = rnd() > 0.666;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + rnd(); x.color = rnd(); x.colorSymmetry = 0; x.opacity = HIDDEN;
      x.c20 = 0.166332; x.c21 = -0.620971;
      if (withTX3) { x.xaos[0] = 0; x.xaos[1] = 0; } else { x.xaos[0] = 1; x.xaos[1] = 1 + rnd() * 2; }
      addVar(x, 5 + rnd() * 20, pickOf(['juliascope3Db', 'juliascope', 'julian', 'julian2', 'julian3Dx']), { power: 50 + rnd() * 100, dist: 2 + rnd() * 6 }); }
    if (withTX3) { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = rnd(); x.colorSymmetry = 1; x.colorType = 'NONE';
      x.xaos[1] = 1 + rnd() * 2; x.xaos[2] = 0;
      addVar(x, 0.25 - rnd() * 0.5, randomVariationName()); }
    return f;
  } },
);

const FNCLST_MANDELBROT_FINAL = ['bipolar', 'boarders', 'boarders2', 'bubble', 'butterfly', 'bwraps7', 'circlize', 'collideoscope', 'cosh', 'cross', 'curl', 'curl3D', 'curve', 'eJulia', 'edisc', 'elliptic',
  'eyefish', 'flux', 'foci', 'foci_3D', 'glynnia', 'heart_wf', 'xtrb', 'hemisphere', 'horseshoe', 'hypertile', 'hypertile1', 'hypertile2', 'julia', 'julian', 'juliascope', 'loonie', 'loonie_3D', 'mobius',
  'npolar', 'phoenix_julia', 'popcorn2_3D', 'power', 'ripple', 'scry', 'scry_3D', 'sec', 'sech', 'separation', 'spherical', 'spiral', 'stripes', 'unpolar', 'waves2', 'whorl', 'xheart'];
STYLES.push(
  { id: 'mandelbrot', name: 'Mandelbrot', filter: true, build: () => {
    const f = newFlame(); f.centreY = 0.45; f.camPitch = 49; f.camPersp = 0.05 + rnd() * 0.12; f.ppu = 200; f.preserveZ = true; f.gamma = 2;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5;
      const id = randomInt(6);
      const seeded = { scale: 2, xmin: -2, xmax: 2, ymin: -2, ymax: 2, xseed: -1 + 2 * rnd(), yseed: -1 + 2 * rnd() };
      const vi = id === 0 ? addVar(x, 1, 'fract_dragon_wf', seeded) : id === 1 ? addVar(x, 1, 'fract_julia_wf', seeded) : id === 2 ? addVar(x, 1, 'fract_pearls_wf', seeded)
        : id === 3 ? addVar(x, 1, 'fract_salamander_wf', seeded) : id === 4 ? addVar(x, 1, 'fract_mandelbrot_wf') : addVar(x, 1, 'fract_meteors_wf');
      if (rnd() < 0.8) {
        const sc = 3 + rnd() * 3, xmin = -1 + 2 * rnd(), ymin = -1 + 2 * rnd(), xmax = xmin + 4 / sc, ymax = ymin + 4 / sc;
        Object.assign(vi.params, { xmin, xmax, ymin, ymax, offsetx: -(xmax - xmin) * 0.5, offsety: -(ymax - ymin) * 0.5, scale: 2 * sc });
      }
      vi.params.scalez = 1 + rnd() * 10; }
    if (rnd() < 0.75) { const x = newXForm(); f.finals.push(x); addVar(x, 1, pickOf(FNCLST_MANDELBROT_FINAL)); }
    randomizeColors(f);
    return f;
  } },
  { id: 'orchids', name: 'Orchids', filter: true, build: () => {
    const f = newFlame(); f.width = 638; f.height = 359; f.ppu = 315.33902046; f.camZoom = 2.2 + rnd() * 0.6;
    { const x = newXForm(); f.xforms.push(x); x.weight = 4 + rnd() * 2; x.color = 0.6 * rnd(); addVar(x, 1.5 + rnd(), 'elliptic'); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 1.5 + rnd(); x.color = 0.5 + rnd() * 0.2;
      x.c00 = -0.00283201; x.c10 = 0.99999599; x.c01 = -0.99999599; x.c11 = -0.00283201;
      addVar(x, rnd() + 1, 'poincare3D', { r: 0, a: 0, b: 0 });
      addVar(x, 10 * rnd() - 5, randomVariationName());
      scale(x, 0.5 + 2 * rnd(), true, true); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 1.2 + rnd(); x.color = 0.2 + rnd() * 0.5; addVar(x, 0.75, 'rays'); }
    { const x = newXForm(); f.finals.push(x); x.weight = 0; x.color = 0.96; addVar(x, 0.3 + rnd() * 0.8, 'hypertile1', { p: 4, q: 6 }); }
    { const x = newXForm(); f.finals.push(x); x.weight = 0;
      if (rnd() > 0.25) addVar(x, 1, rnd() < 0.5 ? 'polar' : 'polar2'); else addVar(x, 1, randomVariationName(rnd() > 0.75 ? '2d' : '3d')); }
    if (rnd() > 0.666) { f.xforms[0].xaos[0] = 1 + rnd(); f.xforms[0].xaos[2] = 0.2 * rnd(); }
    return f;
  } },
);

STYLES.push(
  { id: 'simpletiling', name: 'Simple tiling', filter: true, build: () => {
    const f = newFlame(); f.ppu = 200;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.4 + rnd() * 50;
      addVar(x, 0.01 + 0.4 * rnd(), randomVariationName()); addVar(x, 0.02 + 0.2 * rnd(), 'linear3D'); x.color = 0.87 + rnd() * 0.1; }
    const twoPrimary = rnd() > 0.33, linked = rnd() < 0.4;
    if (twoPrimary) { const x = newXForm(); f.xforms.push(x); x.weight = 0.4 + rnd() * 72;
      addVar(x, 1, randomVariationName()); x.colorSymmetry = -1; x.color = 0.89 + rnd() * 0.06;
      scale(x, 0.75 + rnd() * 0.25, rnd() > 0.125, rnd() < 0.875); rotate(x, -180 + rnd() * 360); }
    const nForms = 2 + randomInt(5);
    for (let i = 0; i < nForms; i++) { const x = newXForm(); f.xforms.push(x); x.weight = 0.25 + rnd() * 0.5;
      addVar(x, 0.5 + rnd() * 0.5, 'linear3D');
      if (rnd() > 0.75) scale(x, 0.125 + rnd() * 0.5, rnd() > 0.5, rnd() < 0.5); else scale(x, 0.75 + rnd() * 0.25, rnd() > 0.5, rnd() < 0.5);
      rotate(x, -45 + rnd() * 90); localTranslate(x, -2 + i * 0.5 + rnd() * 4, -2 + i * 0.5 + rnd() * 4); x.color = rnd(); }
    const n = f.xforms.length;
    for (let i = 0; i < n; i++) {
      const x = f.xforms[i];
      if (linked) { for (let j = 0; j < n; j++) x.xaos[j] = i === 0 ? (j !== 1 ? 0 : 1) : (j !== 1 ? 1 : 0); }
      else { let wg = rnd() * 12; if (rnd() < 0.25) wg = 0; for (let j = 0; j < n; j++) x.xaos[j] = i === 0 ? (j !== 1 ? wg : 1) : (j !== 1 ? 1 : wg); }
    }
    return f;
  } },
  { id: 'synth', name: 'Synth', filter: true, build: () => {
    const f = newFlame(); f.ppu = 300 + rnd() * 300;
    const centre_synth = 0.7, centre_mode = randomInt(20), centre_noise = 0, centre_power = -1, centre_smooth = 1;
    const centre_color = 0.4 + 0.2 * rnd(), centre_symmetry = 0.6 + 0.4 * rnd();
    const x1 = newXForm(); f.xforms.push(x1); x1.weight = 1; x1.color = 0; x1.colorSymmetry = -1;
    const sy = addVar(x1, 1, 'synth');
    const P = sy.params;
    P.mode = randomInt(20); P.power = -0.2;
    const numWaves = Math.trunc(rnd() * 3.5 + 2);
    P.a = 0.8 + rnd() * 0.4;
    const wave = (k: string, amp: number) => {
      P[k] = amp; P[k + '_type'] = randomInt(7); P[k + '_frq'] = randomInt(7) + 1; P[k + '_phs'] = rnd() * Math.PI; P[k + '_layer'] = randomInt(4);
      if (rnd() < 0.2) P[k + '_skew'] = rnd() * 2 - 1;
      if (rnd() < 0.1) P[k + '_frq'] = randomInt(20) + 7;
      if (rnd() < 0.8) P[k] = P[k] / (1 + 0.3 * P[k + '_frq']);
    };
    wave('b', rnd() * 2); wave('c', rnd());
    if (numWaves >= 3) wave('d', rnd());
    if (numWaves >= 4) wave('e', rnd());
    if (numWaves >= 5) wave('f', rnd());
    { const x = newXForm(); f.xforms.push(x); x.weight = 1; x.color = centre_color; x.colorSymmetry = centre_symmetry;
      const sy2 = addVar(x, centre_synth, 'synth', { ...P, power: centre_power, mode: centre_mode, smooth: centre_smooth }); void sy2;
      addVar(x, centre_noise, 'noise'); }
    if (rnd() < 0.55) { const x = newXForm(); f.xforms.push(x); x.weight = 5 * rnd() + 0.125; x.color = centre_color + 0.2 * rnd(); x.colorSymmetry = centre_symmetry - 0.4 * rnd();
      addVar(x, 0.25 + rnd() * 1.5, randomVariationName()); x.xaos[1] = 0; }
    return f;
  } },
);

/** TileBallRandomFlameGenerator.randomizeParams: nudge a random number of a variation's parameters */
function randomizeParams(vi: VarInstance) {
  const names = Object.keys(vi.params);
  if (!names.length) return;
  const defs = (VARIATIONS[vi.name] as { params?: { name: string; int?: boolean }[] })?.params ?? [];
  const rndParams = FTOI(rnd() * names.length);
  for (let i = 0; i < rndParams; i++) {
    const k = names[randomInt(names.length)];
    const amount = 0.1 + rnd();
    let o = vi.params[k];
    if (defs.find((d) => d.name === k)?.int) { let da = FTOI(amount); if (da < 1) da = 1; o = o >= 0 ? o + da : o - da; }
    else if (o < 1e-6 || rnd() < 0.3) o = o >= 0 ? o + 0.1 * amount : o - 0.1 * amount;
    else o = o >= 0 ? o + (o / 100) * amount : o - (o / 100) * amount;
    vi.params[k] = o;
  }
}
STYLES.push(
  { id: 'tileball', name: 'Tile Ball', filter: false, build: () => {
    const f = newFlame(); f.centreX = 0.1 - rnd() * 0.2; f.centreY = -0.6 + rnd() * 0.3; f.ppu = 200; f.camZoom = 1 + rnd() * 0.5;
    f.camPitch = 25 + rnd() * 40; f.camPersp = 0.1 + rnd() * 0.2; f.preserveZ = true;
    const a = 0.5;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; addVar(x, 1, 'linear3D'); addVar(x, 1.5 + rnd() * 1.5, 'ztranslate'); x.color = rnd(); x.colorSymmetry = 1; x.xaos[10] = 0; }
    for (const [dx, dy] of [[0, a], [a, a], [a, 0], [a, -a], [0, -a], [-a, -a], [-a, 0], [-a, a]]) {
      const x = newXForm(); f.xforms.push(x); x.weight = 0.5; addVar(x, 1, 'linear3D'); x.colorSymmetry = 1; localTranslate(x, dx, dy); x.xaos[10] = 0;
    }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.25 + rnd() * 1.25;
      randomizeParams(addVar(x, 0.2 + rnd() * 0.6, randomVariationName())); addVar(x, 1, 'flatten'); x.color = rnd(); x.colorSymmetry = rnd();
      for (let j = 0; j < 10; j++) x.xaos[j] = 0; x.xaos[10] = 1; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.25 + rnd() * 1.25;
      randomizeParams(addVar(x, 0.1 + rnd() * 0.3, randomVariationName())); addVar(x, 1, 'flatten'); x.color = rnd(); x.colorSymmetry = rnd(); x.xaos[10] = 0; }
    { const x = newXForm(); f.finals.push(x); x.weight = 0.8 + rnd() * 0.5; addVar(x, 1, 'curl3D', { cx: -0.75 + rnd() * 1.5, cy: 0.8 + rnd() * 0.4, cz: 0.025 + rnd() * 0.05 }); }
    return f;
  } },
);

STYLES.push(
  { id: 'underwater', name: 'Underwater', filter: false, build: () => {
    const f = newFlame(); f.camRoll = -90 + rnd() * 180; f.width = 1920; f.height = 1080; f.ppu = 313.97436855;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.32019788; x.color = 0.075 + rnd() * 0.15; x.colorSymmetry = 0.3 + rnd() * 0.15;
      x.c00 = 0.75092904; x.c10 = 0.56527515; x.c20 = 0.3204811; x.c01 = -0.56527515; x.c11 = 0.75092904; x.c21 = -0.98334041;
      if (rnd() > 0.333) addVar(x, 0.41432335, 'linear3D'); else addVar(x, 0.25 + rnd() * 0.5, randomVariationName(rnd() < 0.65 ? '2d' : '3d'));
      if (rnd() > 0.5) addVar(x, 0.22103179, 'eclipse', { shift: 0 }); else addVar(x, 0.15 + rnd() * 0.15, randomVariationName(rnd() < 0.5 ? '2d' : '3d'));
      if (rnd() < 0.5) rotate(x, -30 + 60 * rnd()); else localTranslate(x, -0.05 + 0.1 * rnd(), -0.05 + 0.1 * rnd()); }
    { const x = newXForm(); f.xforms.push(x); x.weight = rnd() < 0.5 ? 0.64084507 : 0.5 + rnd() * 0.3; x.color = 0.8 + rnd() * 0.1; x.colorSymmetry = 0.87685786;
      x.c00 = -0.36319594; x.c10 = -0.64089754; x.c20 = -0.3641259; x.c01 = 0.64089754; x.c11 = -0.36319594; x.c21 = 0.40615455;
      if (rnd() > 0.42) addVar(x, 0.21369106, 'linear3D'); else addVar(x, 0.125 + rnd() * 0.25, randomVariationName(rnd() < 0.65 ? '2d' : '3d'));
      addVar(x, 0.61264366, 'foci');
      if (rnd() > 0.5) addVar(x, 0.05 + rnd() * 0.15, randomVariationName(rnd() < 0.5 ? '2d' : '3d'));
      if (rnd() < 0.5) rotate(x, -30 + 60 * rnd()); else localTranslate(x, -0.15 + 0.3 * rnd(), -0.15 + 0.3 * rnd()); }
    if (rnd() > 0.72) { const x = newXForm(); f.finals.push(x); x.weight = 0; x.color = 0.5;
      if (rnd() > 0.666) addVar(x, 1, 'log'); else addVar(x, 1, randomVariationName(rnd() > 0.25 ? '2d' : '3d'));
      scale(x, 1.25 - rnd() * 0.5, true, true); rotate(x, 360 * rnd()); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); }
    let g: number, b: number;
    if (rnd() < 0.125) { g = 102; b = 102; }
    else if (rnd() < 0.5) { g = Math.trunc(50 + 102 * rnd()); b = g; }
    else { g = Math.trunc(50 + 102 * rnd()); b = Math.trunc(50 + 102 * rnd()); }
    f.extra = (fl) => { fl.background = [0, g / 255, b / 255]; fl.bgTransparency = false; };
    return f;
  } },
);

// ---- Gnarl family: the waves variations (GnarlRandomFlameGenerator.createWaves2B/Waves2/CrazyWaves) ----
function createWaves2B(scaleX: number, scaleY: number, freqX: number, freqY: number): VarInstance {
  let pwx = 3 * rnd();
  if (pwx < 1) pwx = 1; else if (pwx < 2) pwx = (pwx - 1.5 > 0 ? -1 : 1) * (0.5 + 2 * rnd()); else pwx = (pwx - 2.5 > 0 ? -1 : 1) * 1e-6 * (0.9 - rnd() * 0.2);
  let pwy = 3 * rnd();
  if (pwy < 1.5) pwy = 1; else if (pwy < 2.25) pwy = (pwy - 1.5 > 0 ? -1 : 1) * (0.5 + 2 * rnd()); else pwy = (pwy - 2.5 > 0 ? -1 : 1) * 1e-6;
  const w2 = v('waves2b', 1, { pwx, pwy, scaleinfx: rnd() < 0.5 ? scaleX : 0.5 - rnd(), scaleinfy: rnd() < 0.5 ? scaleY : 0.5 - rnd() });
  if (rnd() > 0.75) { let unity = rnd() * 10; unity = unity < 5 ? unity - 1 : unity + 5; w2.params.unity = unity; }
  if (rnd() < 0.33) w2.params.jacok = 0.375 - rnd() * 0.75;
  Object.assign(w2.params, { freqx: freqX, scalex: scaleX, freqy: freqY, scaley: scaleY });
  return w2;
}
const createWaves2 = (scaleX: number, scaleY: number, freqX: number, freqY: number): VarInstance => v('waves2', 1, { freqx: freqX, scalex: scaleX, freqy: freqY, scaley: scaleY });
function createCrazyWaves(scaleX: number, scaleY: number, freqX: number, freqY: number): VarInstance {
  const r = rnd();
  const fs = { freqx: freqX, scalex: scaleX, freqy: freqY, scaley: scaleY };
  if (r < 0.16) return v('waves22', 1, { ...fs, modex: 1, modey: 1, powerx: 2 + rnd() * 7, powery: 2 + rnd() * 7 });
  if (r < 0.34) return v('waves23', 1, fs);
  if (r < 0.61) {
    const w = v('vibration2', 1, { dir: rnd() * 2 * Math.PI, angle: rnd() * 2 * Math.PI, freq: 0.5 + rnd(), amp: 0.25 + rnd() * 0.5, phase: rnd() * Math.PI,
      dir2: rnd() * 2 * Math.PI, angle2: rnd() * 2 * Math.PI, freq2: 0.5 + rnd(), amp2: 0.25 + rnd() * 0.5, phase2: rnd() * Math.PI });
    for (const k of ['dm', 'tm', 'fm', 'am', 'd2m', 't2m', 'f2m', 'a2m']) if (rnd() > 0.25) { w.params[k] = rnd() * 0.1; w.params[k + 'freq'] = 0.1 + rnd() * 0.6; }
    return w;
  }
  if (r < 0.821) return v('waves4', 1, { ...fs, cont: 1, yfact: 0.1 + rnd() * 0.49 });
  return v('waves42', 1, { ...fs, cont: 1, yfact: 0.1 + rnd() * 0.59, freqx2: 0.5 + rnd() * 0.25 });
}
const GNARL_SIDES: Record<number, [number, number]> = { 3: [-120, 4], 4: [-90, 4], 5: [-72, 3.5], 6: [-60, 3.5], 7: [-51.42857, 3], 8: [-135, 3], 9: [-40, 2.5], 10: [-36, 2] };
STYLES.push(
  { id: 'gnarl', name: 'Gnarl', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200;
    let scaleX = rnd() * 0.04 + 0.04; if (rnd() > 0.75) scaleX = -scaleX;
    let scaleY = rnd() * 0.04 + 0.04; if (rnd() > 0.75) scaleY = -scaleY; else if (rnd() < 0.25) scaleY = scaleX;
    let freqX: number, freqY: number;
    if (rnd() < 0.5) { freqX = rnd() * 2 + 2; freqY = rnd() > 0.5 ? freqX : rnd() * 2 + 2; }
    else { freqX = -rnd() * 2 + 2; freqY = rnd() > 0.5 ? freqX : -rnd() * 2 + 16; }
    let blurAmount = 0.0025 * rnd(); if (rnd() < 0.33) blurAmount = -blurAmount;
    const wavesWeight = rnd() * 15 + 135, symmetry = 0.7 + rnd() * 0.3, sides = Math.trunc(rnd() * 8 + 3);
    { const x = newXForm(); f.xforms.push(x); x.weight = wavesWeight;
      const w2 = rnd() > 0.333 ? (rnd() > 0.666 ? createWaves2(scaleX, scaleY, freqX, freqY) : createCrazyWaves(scaleX, scaleY, freqX, freqY)) : createWaves2B(scaleX, scaleY, freqX, freqY);
      w2.weight = 1 + rnd() * 0.001; x.vars.push(w2);
      const varName = rnd() > 0.25 ? pickOf(['blur', 'cos', 'exp', 'exponential', 'lazysusan', 'ngon', 'sech', 'sinh', 'epispiral_wf', 'tanh', 'twintrian', 'epispiral']) : randomVariationName();
      addVar(x, blurAmount, varName);
      x.colorSymmetry = symmetry; x.color = 0.9;
      if (rnd() > 0.5) scale(x, 0.995, true, true);
      const [angle, half] = GNARL_SIDES[sides];
      const tx = rnd() * 2 * half - half, ty = rnd() * 2 * half - half;
      rotate(x, angle); localTranslate(x, tx * 1.5, ty * 1.5); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5;
      const k = randomInt(3);
      if (k === 0) addVar(x, rnd() * 0.37 + 0.1, 'radial_blur');
      else if (k === 1) { addVar(x, rnd() * 0.37 + 0.3, 'bubble'); addVar(x, rnd() * 0.37 + 0.1, 'radial_blur'); }
      else { addVar(x, rnd() * 0.1, 'radial_blur'); addVar(x, rnd() * 0.06 + 0.1, 'julian', { power: 50 - rnd() * 100, dist: rnd() * 10 - 2 }); }
      x.colorSymmetry = -1; x.color = rnd(); }
    if (rnd() > 0.75) { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; addVar(x, 1, 'linear3D');
      rotate(x, 180 - rnd() * 360); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); x.colorSymmetry = -1; x.color = rnd(); }
    return f;
  } },
);

const EXPGNARL_SIDES: Record<number, [number, number, number]> = { 2: [-180, 8, 4], 3: [-120, 8, 4], 4: [-90, 8, 4], 5: [-72, 7, 3.5], 6: [-60, 7, 3.5], 7: [-51.42857, 6, 3], 8: [-135, 6, 3], 9: [-40, 5, 2.5], 10: [-36, 4, 2], 11: [-32.73, 4.2, 3], 12: [-30, 4.2, 3] };
function createExpWaves2(scaleX: number, scaleY: number, freqX: number, freqY: number, varRnd: number): VarInstance {
  const w2 = v(varRnd < 0.6 ? 'waves2_wf' : varRnd < 0.8 ? 'waves3_wf' : 'waves4_wf', 1, {});
  w2.params.use_cos_x = rnd() < 0.25 ? 1 : 0; w2.params.use_cos_y = rnd() < 0.25 ? 1 : 0;
  if (rnd() < 0.25) w2.params.dampx = -(0.01 + rnd() * 0.39);
  if (rnd() < 0.25) w2.params.dampy = -(0.01 + rnd() * 0.39);
  Object.assign(w2.params, { freqx: freqX, scalex: scaleX, freqy: freqY, scaley: scaleY });
  return w2;
}
STYLES.push(
  { id: 'gnarl_experimental', name: 'Gnarl (experimental)', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200;
    let scaleX = rnd() * 0.04 + 0.04; if (rnd() > 0.75) scaleX = -scaleX;
    let scaleY = rnd() * 0.04 + 0.04; if (rnd() > 0.75) scaleY = -scaleY; else if (rnd() < 0.25) scaleY = scaleX;
    const freqX = rnd() * 2 + 7, freqY = rnd() < 0.25 ? freqX : rnd() * 2 + 7;
    const blurAmount = 0.0025 * rnd(), nonBlurAmount = 0.25 * rnd(), wavesWeight = rnd() * 10 + 75, symmetry = 0.6 + rnd() * 0.4, sides = Math.trunc(rnd() * 11 + 2);
    { const x = newXForm(); f.xforms.push(x); x.weight = wavesWeight;
      const varRnd = rnd();
      let w2: VarInstance;
      if (rnd() > 0.666) w2 = createExpWaves2(scaleX, scaleY, freqX, freqY, varRnd);
      else { const r = rnd(); w2 = r < 0.25 ? createWaves2(scaleX, scaleY, freqX, freqY) : r < 0.666 ? createCrazyWaves(scaleX, scaleY, freqX, freqY) : createWaves2B(scaleX, scaleY, freqX, freqY); }
      w2.weight = 1; x.vars.push(w2);
      const k = randomInt(36);
      const nb = ['cos', 'exp', 'exponential', 'lazysusan', 'ngon', 'sech', 'sinh', 'epispiral_wf', 'tanh', 'twintrian'];
      if (k === 0) addVar(x, blurAmount, 'blur');
      else if (k >= 1 && k <= 10) addVar(x, nonBlurAmount, nb[k - 1]);
      else if (k === 11) addVar(x, blurAmount, 'bubble');
      else if (k === 12) addVar(x, nonBlurAmount, 'epispiral');
      else addVar(x, blurAmount, experimental());
      x.colorSymmetry = symmetry; x.color = 0.9;
      if (rnd() > 0.5) scale(x, 0.9 + rnd() * 0.09, true, true);
      const [angle, span, off] = EXPGNARL_SIDES[sides];
      rotate(x, angle); localTranslate(x, rnd() * span - off, rnd() * span - off); }
    const second = newXForm();
    { const x = second; f.xforms.push(x); x.weight = 0.5;
      const k = randomInt(4);
      if (k === 0) addVar(x, rnd() * 0.7 + 0.1, 'radial_blur');
      else if (k === 1) { addVar(x, rnd() * 0.7 + 0.3, 'bubble'); addVar(x, rnd() * 0.7 + 0.1, 'radial_blur'); }
      else if (k === 2) { addVar(x, rnd() * 0.1, 'radial_blur'); addVar(x, rnd() * 0.06 + 0.1, 'julian', { power: 50 - rnd() * 100, dist: rnd() * 10 - 2 }); }
      else addVar(x, rnd() * 0.7 + 0.3, 'spherical3D');
      rotate(x, 180 - rnd() * 360);
      if (rnd() > 0.5) scale(x, 0.5 + rnd(), true, true);
      x.colorSymmetry = -1; x.color = rnd(); }
    if (rnd() > 0.75) {
      if (rnd() > 0.5) second.weight = 5 + rnd() * 20;
      const x = newXForm(); f.xforms.push(x); x.weight = 0.5; addVar(x, 1, 'linear3D');
      rotate(x, 180 - rnd() * 360); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd()); x.colorSymmetry = -1; x.color = rnd(); }
    return f;
  } },
);

const BUBBLES3D_WORKING = ['fan2', 'blade', 'blade3D', 'blob', 'blob3D', 'bwraps7', 'cell', 'cannabiscurve_wf', 'cloverleaf_wf', 'cos', 'cot', 'coth', 'cross', 'csch', 'diamond', 'disc', 'edisc', 'epispiral_wf', 'fan', 'fisheye',
  'eyefish', 'flux', 'heart', 'julia', 'julian', 'juliascope', 'log', 'parabola', 'power', 'epispiral', 'pre_subflame_wf', 'rectangles', 'rose_wf', 'sech', 'separation', 'split', 'truchet', 'wedge', 'zcone'];
function buildBubbles3D(): JFlame {
  const f = newFlame(); f.camPitch = 49; f.camYaw = 12; f.camPersp = 0.12; f.ppu = 200;
  { const x = newXForm(); f.xforms.push(x); x.weight = 1 + rnd() * 100;
    if (rnd() < 0.75) addVar(x, 0.05 + rnd() * 0.5, 'bubble'); else addVar(x, 0.25 + rnd() * 0.5, 'bubble2', { z: -0.5 - rnd() * 2 });
    addVar(x, 0.5 + 1.5 * rnd(), 'pre_blur');
    const fromWorking = rnd() < 0.5;
    addVar(x, -0.02 + 0.04 * rnd(), rnd() < 0.5 ? randomVariationName() : fromWorking ? pickOf(BUBBLES3D_WORKING) : experimental());
    const shape = rnd();
    if (shape < 0.125) addVar(x, 0.001 + rnd() * 0.01, 'hexes'); else if (shape < 0.25) addVar(x, 0.001 + rnd() * 0.01, 'oscilloscope'); else if (shape < 0.5) addVar(x, 0.001 + rnd() * 0.01, randomVariationName());
    if (rnd() > 0.33) globalTranslate(x, -1 + 2 * rnd(), -1 + 2 * rnd(), true);
    if (rnd() > 0.33) rotate(x, -32 + rnd() * 72);
    if (rnd() > 0.33) scale(x, 0.8 + rnd() * 0.4, true, false);
    x.color = 0; x.colorSymmetry = 0; }
  { const x = newXForm(); f.xforms.push(x); x.weight = 1 + rnd() * 50;
    addVar(x, -1.5 + 2 * rnd() * 3, rnd() > 0.5 ? experimental() : 'eyefish');
    addVar(x, 1 - 3.5 * rnd(), 'hemisphere'); x.color = 0; x.colorSymmetry = 0;
    if (rnd() > 0.33) globalTranslate(x, -3 + 62 * rnd(), -3 + 6 * rnd());
    if (rnd() > 0.33) rotate(x, -32 + rnd() * 72);
    if (rnd() > 0.33) scale(x, 0.8 + rnd() * 0.4, true, false); }
  { const x = newXForm(); f.xforms.push(x); x.weight = 1 + rnd() * 100;
    let idx = rnd() > 0.5 ? 2 : Math.trunc(-20 + rnd() * 40); if (idx === 0 || idx === 1) idx = 2;
    addVar(x, 1, rnd() > 0.5 ? 'julia3Dz' : 'julia3D', { power: idx }); x.color = 0; x.colorSymmetry = 0;
    globalTranslate(x, -1 + 2 * rnd(), -3 + 6 * rnd()); globalTranslate(x, -0.1 + 0.2 * rnd(), -0.1 + 0.2 * rnd(), true);
    rotate(x, -20 + rnd() * 40); scale(x, 0.8 + rnd() * 0.4, true, true); }
  f.preserveZ = rnd() > 0.5;
  randomizeColors(f);
  return f;
}
const FLOWERS_FILLED_LAST = ['blade', 'blur', 'blur3D', 'bubble_wf', 'cannabiscurve_wf', 'circlecrop', 'cloverleaf_wf', 'conic', 'crop', 'cross', 'flower', 'flux', 'hemisphere', 'hyperbolic', 'julia3D', 'julia3Dz', 'lazysusan', 'lissajous', 'log', 'mandelbrot', 'mobius', 'npolar',
  'pdj', 'perspective', 'pie', 'pie3D', 'polar', 'polar2', 'power', 'pre_subflame_wf', 'radial_blur', 'scry', 'separation', 'spiral', 'spirograph', 'split', 'tangent', 'tangent3D', 'twintrian', 'unpolar', 'wedge_sph', 'zblur'];
STYLES.push(
  { id: 'simple_experimental', name: 'Simple (experimental)', filter: true, build: () => {
    const f = newFlame(); f.ppu = 200;
    const maxXForms = Math.trunc(1 + rnd() * 5);
    let scl = 1;
    for (let i = 0; i < maxXForms; i++) {
      const x = newXForm(); f.xforms.push(x);
      if (rnd() < 0.5) rotate(x, 360 * rnd()); else rotate(x, -360 * rnd());
      localTranslate(x, rnd() - 1, rnd() - 1);
      scl *= 0.75 + rnd() / 4; scale(x, scl, true, true);
      x.color = rnd();
      addVar(x, rnd() * 0.3 + 0.2, 'linear3D');
      if (rnd() > 0.1) addVar(x, 0.2 + rnd() * 0.6, experimental());
      x.weight = rnd() * 0.9 + 0.1;
    }
    return f;
  } },
  { id: 'bubbles3d', name: 'Bubbles3D', filter: true, build: buildBubbles3D },
  { id: 'bubbles3d_experimental', name: 'Bubbles3D (experimental)', filter: true, build: () => {
    const f = buildBubbles3D();
    const x = f.xforms[f.xforms.length - 1];
    scale(x, 0.5 + 5 * rnd(), rnd() < 0.75, rnd() < 0.75); rotate(x, 180 - rnd() * 360); localTranslate(x, 4 - 8 * rnd(), 4 - 8 * rnd());
    return f;
  } },
  { id: 'flowers3d_filled', name: 'Flowers3D (filled)', filter: true, build: () => {
    const f = newFlame(); f.camPitch = 37; f.ppu = 200; f.camZoom = 2; f.camPersp = 0.32;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.25 + rnd() * 0.5; addVar(x, 0.25 + rnd() * 0.5, 'gaussian_blur'); x.color = 0; x.colorSymmetry = 0; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 1 + rnd() * 50;
      localTranslate(x, 0.3 - 0.6 * rnd(), 0.3 - 0.6 * rnd()); rotate(x, 90 + rnd() * 180); scale(x, 1.25 + rnd() * 1.25, true, true);
      addVar(x, 1, 'linear3D'); addVar(x, 0.01 - rnd() * 0.02, 'epispiral_wf'); addVar(x, 0.1 + rnd() * 0.1, 'ztranslate'); addVar(x, 0.00001 - rnd() * 0.00002, 'zcone');
      addVar(x, 0.001 + rnd() * 0.099, pickOf(['bubble', 'log']));
      addVar(x, 0.001 + rnd() * 0.099, pickOf(['arch', 'bipolar', 'hyperbolic', 'butterfly', 'cannabiscurve_wf', 'cell', 'checks', 'circlize', 'conic', 'coth', 'cpow', 'ex', 'falloff2', 'fan', 'flux', 'foci', 'heart', 'kaleidoscope', 'log', 'mobius', 'ngon', 'pdj', 'oscilloscope', 'spherical', 'spiral']));
      addVar(x, 0.05 - rnd() * 0.1, pickOf(['waves2', 'waves2_wf', 'waves3_wf', 'waves4_wf']), { scalex: 0.5 + rnd(), scaley: 0.5 + rnd(), freqx: 1 + rnd() * 2, freqy: 1 + rnd() * 2 });
      addVar(x, 0.001 + rnd() * 0.015, pickOf(['cross', 'checks', 'conic', 'kaleidoscope', 'lazysusan', 'log']));
      x.color = 0.33; x.colorSymmetry = 0; }
    { const blurCount = randomInt(5); let weight = 0.25 + rnd() * 0.5;
      for (let i = 0; i < blurCount; i++) { const x = newXForm(); f.xforms.push(x); x.weight = weight; weight *= 0.75;
        addVar(x, 0.25 + rnd() * 0.5, 'gaussian_blur'); x.color = 0; x.colorSymmetry = 0; localTranslate(x, 0.3 - 0.6 * rnd(), 0.3 - 0.6 * rnd(), true); } }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.1 + rnd() * 1.4; localTranslate(x, 0.3 - 0.6 * rnd(), 0.3 - 0.6 * rnd(), true);
      const k = randomInt(3);
      if (k === 0) { addVar(x, 0.05 + rnd() * 0.15, 'bubble'); addVar(x, 0.01 + rnd() * 0.5, pickOf(FLOWERS_FILLED_LAST)); }
      else if (k === 1) addVar(x, 0.05 + rnd() * 0.15, 'bubble');
      else addVar(x, 0.05 + rnd() * 0.15, pickOf(FLOWERS_FILLED_LAST));
      addVar(x, 0.05 + rnd() * 0.15, 'bubble'); x.color = 0; x.colorSymmetry = 0.8 + rnd() * 0.2; }
    { const x = newXForm(); f.finals.push(x); addVar(x, 0.05 + rnd() * 0.2, 'zscale'); addVar(x, 1, 'julia3D', { power: rnd() < 0.5 ? -3 : -4 }); addVar(x, 1, 'pre_circlecrop', { radius: 10000, zero: rnd() < 0.5 ? 1 : 0 }); }
    return f;
  } },
);

STYLES.push(
  { id: 'flowers3d', name: 'Flowers3D (stunning)', filter: true, build: () => {
    const f = newFlame(); f.camPitch = 49; f.camYaw = 12; f.ppu = 200;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5 + rnd(); addVar(x, 0.5, 'gaussian_blur'); x.color = 0; x.colorSymmetry = 0; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 3 + rnd() * 10;
      if (rnd() < 0.33) addVar(x, 1, 'linear3D'); else { const p = 2 * rnd() - 0.5; addVar(x, 1, 'linearT3D', { powX: p, powY: p, powZ: 2 * rnd() - 0.5 }); }
      addVar(x, 0.1 + rnd() * 0.3, 'spherical'); addVar(x, 0.2 + rnd() * 0.9, 'zcone'); addVar(x, 0.01 + 0.045 * rnd(), 'cross');
      if (rnd() < 0.33) { addVar(x, 0.02 + 0.29 * rnd(), 'epispiral_wf', { waves: 3 + randomInt(10) });
        if (rnd() < 0.33) addVar(x, 0.01 + 0.14 * rnd(), 'epispiral', { thickness: 0.05 + rnd() * 0.15, n: 3 + rnd() * 10 }); }
      x.color = rnd(); x.colorSymmetry = rnd();
      scale(x, 1 + (0.1 - rnd() * 0.2), true, true); rotate(x, 45 - rnd() * 90); localTranslate(x, 0.01 - 0.02 * rnd(), 0.01 - 0.02 * rnd()); }
    const adv = rnd() > 0.25;
    if (adv) { const x = newXForm(); f.xforms.push(x); x.weight = 0.3 + rnd() * 0.3; addVar(x, 0.05, 'blob3D', { low: 0.1, high: 0.3, waves: 9 }); x.color = 0; x.colorSymmetry = 1; }
    if (adv && rnd() > 0.25) { const x = newXForm(); f.xforms.push(x); x.weight = 0.1 + rnd() * 0.3; addVar(x, 0.5, 'blur3D'); addVar(x, 15, 'ztranslate'); x.color = 0; x.colorSymmetry = 1; }
    { const x = newXForm(); f.finals.push(x); let power = -2; if (rnd() < 0.25) power -= rnd() * 4; addVar(x, 2 + (1 - 2 * rnd()), 'julia3D', { power }); x.color = 0; x.colorSymmetry = 0; }
    return f;
  } },
  { id: 'spherical3d', name: 'Spherical3D', filter: false, build: () => {
    const f = newFlame(); f.centreY = -0.2; f.camPitch = 48; f.camYaw = 112; f.camZoom = 3.6; f.camPersp = 0.32; f.ppu = 200; f.preserveZ = rnd() < 0.5;
    const invert = rnd() > 0.5;
    const x1 = newXForm(); f.xforms.push(x1); x1.weight = 1 + 4 * rnd(); addVar(x1, 1, 'spherical3D_wf', invert ? { invert: 1 } : {});
    rotate(x1, rnd() < 0.5 ? 90 : -90); globalTranslate(x1, 1, 0); x1.color = 1; x1.colorSymmetry = 0.9 + rnd() * 0.1;
    const x2 = newXForm(); f.xforms.push(x2); x2.weight = 0.5 + rnd() * 4.5; addVar(x2, 1, 'spherical3D_wf', invert ? { invert: 1 } : {});
    rotate(x2, 90); x2.color = 0.5; x2.colorSymmetry = 0.9 + rnd() * 0.1;
    const cylinderVar = rnd() < 0.5 ? 'cylinder' : 'cylinder_apo';
    const fncList = ['bipolar', 'blade', 'blur', 'blur3D', 'cannabiscurve_wf', 'crackle', 'cylinder', 'cylinder_apo', 'edisc', 'flower', 'glynnSim2', 'julia3D', 'mandelbrot', 'modulus', 'noise', 'parabola', 'pie', 'pie3D', 'checks', 'pre_subflame_wf', 'radial_blur', 'rays',
      'rings', 'rose_wf', 'secant2', 'sinusoidal', 'spiral', 'spirograph', 'splits', 'square', 'twintrian', 'wedge_julia'];
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; addVar(x, 0.25 + 0.25 * rnd(), 'pre_blur'); addVar(x, 0.01 + rnd() * 0.39, 'ztranslate');
      addVar(x, 0.01 + rnd() * 0.39, rnd() < 0.66 ? cylinderVar : pickOf(fncList)); scale(x, 5, false, true, true); x.color = 1; x.colorSymmetry = -1; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; addVar(x, 0.025 + 0.025 * rnd(), 'pre_blur'); addVar(x, 0.01 + rnd() * 0.39, 'ztranslate');
      addVar(x, 0.01 + rnd() * 0.39, rnd() < 0.33 ? cylinderVar : pickOf(fncList)); scale(x, 3, false, true, true); x.color = 1; x.colorSymmetry = -1; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.1; addVar(x, 0.005 + rnd() * 0.005, 'gaussian_blur'); addVar(x, 0.005 + rnd() * 0.005, 'ztranslate');
      globalTranslate(x, 0.3, 0, true); x.color = 1; x.colorSymmetry = -1; }
    if (rnd() < 0.5) { const x = newXForm(); f.finals.push(x); x.weight = 0.1;
      const style = randomInt(3);
      let vf: VarInstance;
      if (style < 2) { const power = -4 + randomInt(9); vf = v(style === 1 ? 'julia3D' : 'julia3Dz', 1, { power }); if (rnd() > 0.5) addVar(x, 0.005 + rnd() * 0.5, 'linear3D'); }
      else vf = v('spherical3D_wf', 1, rnd() < 0.5 ? { invert: 1 } : {});
      vf.weight = 0.25 + rnd() * 0.75; x.vars.push(vf);
      rotate(x, 45); globalTranslate(x, 0, 0.25); globalTranslate(x, 0, -0.25, true); x.color = 1; x.colorSymmetry = 0.5 + rnd() * 0.5; }
    if (rnd() > 0.5) { x1.xaos[3] = 0; x1.xaos[4] = 0; x2.xaos[3] = 0; x2.xaos[4] = 0; }
    randomizeColors(f);
    return f;
  } },
);

/** build another style's flame (the wrapping generators: Brokat3D, Gnarl3D, Spirals3D, Black&White, Bokeh, SubFlame) */
const buildOf = (id: string): JFlame => { const st = STYLES.find((x) => x.id === id); if (!st) throw new Error(`no style ${id}`); const j = st.build(); if (j.filter === undefined) j.filter = st.filter; return j; };
/** AbstractAffine3DRandomFlameGenerator.prepareFlame's camera, between the wrapped generator's pre- and post-processing */
function affine3DCamera(f: JFlame) {
  f.camRoll = 0; f.camPitch = 25 + rnd() * 30; f.camYaw = 10 + rnd() * 20; f.camBank = 5 + rnd() * 10; f.camPersp = 0.1 + rnd() * 0.3; f.camZoom = 0.5; f.preserveZ = true;
  if (rnd() < 0.5) f.camDOF = 0.1 + rnd() * 0.2;
}
const rotateXForm = (f: JFlame, idx: number, amp0: number) => { if (f.xforms.length > idx) inZX(() => rotate(f.xforms[idx], (0.5 - rnd()) * amp0)); };
const scaleXForm = (f: JFlame, idx: number, offset: number, amp0: number) => { if (f.xforms.length > idx) inZX(() => scale(f.xforms[idx], offset + (0.5 - rnd()) * amp0, true, true)); };
const addFlatten = (f: JFlame, idx: number) => { if (f.xforms.length > idx) { const x = f.xforms[idx]; if (!x.vars.some((vi) => vi.name === 'flatten')) addVar(x, 1, 'flatten'); } };
const CROSS_FINAL = ['cross', 'boarders2', 'boarders', 'butterfly', 'cos', 'cosh', 'cosine', 'csc', 'cylinder', 'cylinder_apo', 'dc_ztransl', 'elliptic', 'eyefish', 'fibonacci2', 'heart_wf', 'hypertile1', 'loonie', 'mobius', 'perspective', 'popcorn', 'popcorn2_3D', 'ripple', 'roundspher3D', 'scry_3D', 'sec', 'secant2', 'separation', 'shredlin', 'sin', 'spherical', 'spiral', 'stripes', 'unpolar', 'waves2', 'waves4_wf', 'whorl', 'xtrb', 'rays1', 'rays2', 'rays3'];
STYLES.push(
  { id: 'cross', name: 'Cross', filter: false, build: () => {
    const f = newFlame(); f.ppu = 200;
    if (rnd() > 0.6) { f.newDOF = true; f.camDOF = rnd() * 0.07; } else { f.newDOF = false; f.camDOF = rnd() * 0.2; }
    f.camPitch = 10 + rnd() * 50; f.preserveZ = rnd() > 0.33; f.camPersp = 0.1 + rnd() * 0.5;
    const corner = (sym: number, px: number, py: number, color?: number) => { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.colorSymmetry = sym; if (color !== undefined) x.color = color; x.p20 = px; x.p21 = py; addVar(x, 1, 'linear3D'); };
    corner(0.99 - rnd() * 0.2, 1, -1); corner(0.8 - rnd() * 0.1, -1, -1); corner(0.92, -1, 1, 0.1 + rnd() * 0.1); corner(0.9 - rnd() * 0.2, 1, 1);
    const x5 = newXForm(); f.xforms.push(x5); x5.weight = 0.05 + rnd() * 0.2; x5.colorSymmetry = 0.1 + rnd() * 0.2;
    x5.p00 = 0.70711; x5.p10 = 0.70711; x5.p01 = -0.70711; x5.p11 = 0.70711; addVar(x5, 0.45, randomVariationName());
    if (rnd() > 0.75) { const x = newXForm(); f.xforms.push(x); x.weight = 0.05 + rnd() * 0.2; x.colorSymmetry = 0.1 + rnd() * 0.2;
      x.p00 = 0.70711; x.p10 = 0.70711; x.p01 = -0.70711; x.p11 = 0.70711; x.xaos[5] = 0; for (let j = 0; j < 5; j++) x5.xaos[j] = 0;
      addVar(x, 0.1 + rnd() * 0.6, randomVariationName()); }
    { const x = newXForm(); f.finals.push(x); x.weight = 0; rotate(x, rnd() * 360, true); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd(), true);
      if (rnd() > 0.75) addVar(x, 1.57, rnd() < 0.25 ? 'cross' : rnd() < 0.25 ? 'rays2' : pickOf(CROSS_FINAL));
      else addVar(x, 1, randomVariationName(rnd() > 0.75 ? '2d' : '3d')); }
    randomizeColors(f);
    return f;
  } },
  { id: 'brokat3d', name: 'Brokat3D', filter: false, build: () => {
    const f = buildOf('brokat'); affine3DCamera(f);
    f.camZoom *= 2; f.camYaw += (0.5 - rnd()) * 75; f.camPitch += (0.5 - rnd()) * 135; f.camBank += (0.5 - rnd()) * 15; f.camPersp = 0.1 + rnd() * 0.4;
    rotateXForm(f, 0, 33);
    for (let i = 1; i <= 3; i++) if (rnd() > 0.5) { rotateXForm(f, 1, 15); if (rnd() > 0.67) addFlatten(f, 1); }
    f.filter = false;
    return f;
  } },
  { id: 'gnarl3d', name: 'Gnarl3D', filter: false, build: () => {
    const f = buildOf(rnd() > 0.33 ? 'gnarl' : 'gnarl_experimental');
    f.xforms.length = Math.min(f.xforms.length, 2); for (const x of f.xforms) for (const k of Object.keys(x.xaos)) if (+k >= 2) delete x.xaos[+k];
    affine3DCamera(f);
    rotateXForm(f, 0, 3); scaleXForm(f, 0, 0.95, 0.01);
    f.filter = false;
    return f;
  } },
  { id: 'spirals3d', name: 'Spirals3D', filter: false, build: () => {
    const f = buildOf('spirals');
    f.xforms.length = Math.min(f.xforms.length, 2); for (const x of f.xforms) for (const k of Object.keys(x.xaos)) if (+k >= 2) delete x.xaos[+k];
    affine3DCamera(f);
    f.camYaw += (0.5 - rnd()) * 75; f.camPitch += (0.5 - rnd()) * 135; f.camBank = (0.5 - rnd()) * 75 + f.camYaw; f.camPersp = 0.1 + rnd() * 0.4; f.camZoom = 2; // (camBank from camYaw: as in JWildfire)
    rotateXForm(f, 0, 30); scaleXForm(f, 0, 0.97, 0.01); rotateXForm(f, 1, 10); scaleXForm(f, 1, 0.97, 0.01);
    f.filter = false;
    return f;
  } },
);

/** AllRandomFlameGenerator's "simple generators" (everything but Layers, SubFlame, the image ones and the Solid family), by our ids */
const SIMPLE_GENERATORS = ['blackandwhite', 'bokeh', 'brokat', 'brokat3d', 'bubbles', 'bubbles3d', 'cross', 'duality', 'bubbles3d_experimental', 'gnarl_experimental', 'simple_experimental', 'flowers3d_filled', 'flowers3d', 'galaxies', 'ghosts', 'orchids', 'edisc', 'phoenix', 'spirals', 'spirals3d', 'gnarl', 'gnarl3d', 'juliandisc', 'julians', 'julianrings', 'linear', 'affine3d', 'machine', 'mandelbrot', 'outlines', 'simple', 'simpletiling', 'sierpinsky', 'spherical', 'spherical3d', 'splits', 'synth', 'tentacle', 'tileball', 'underwater', 'xenomorph'];
/** BlackAndWhiteRandomFlameGenerator's candidate list (its repeats weight the draw), by our ids */
const BW_GENERATORS = ['brokat', 'brokat3d', 'bubbles', 'duality', 'bubbles3d', 'cross', 'duality', 'galaxies', 'bubbles3d_experimental', 'gnarl_experimental', 'simple_experimental', 'flowers3d_filled', 'flowers3d', 'gnarl', 'duality', 'galaxies', 'gnarl3d', 'juliandisc', 'julians', 'affine3d', 'julianrings', 'linear', 'machine', 'mandelbrot', 'outlines', 'phoenix',
  'simple', 'simpletiling', 'sierpinsky', 'duality', 'solid_experimental', 'solid_stunning', 'solid_julia3d', 'solid_shadows', 'solid_labyrinth', 'solid_recursive', 'galaxies', 'spherical', 'spherical3d', 'ghosts', 'orchids', 'edisc', 'spirals', 'spirals3d', 'splits', 'subflame', 'synth', 'tentacle', 'tileball', 'duality', 'underwater', 'xenomorph'];
const pickStyle = (ids: string[]) => { const have = ids.filter((id) => STYLES.some((st) => st.id === id)); return have[randomInt(have.length)]; };
/** BokehMutation.execute (strength 1): a DOF setup plus a scale-0 crackle transform */
function bokehMutation(j: JFlame) {
  j.camDOF = 0.1 + rnd() * 0.3; j.newDOF = true; j.camPitch = 30 + rnd() * 20; j.camYaw = 15 - rnd() * 30; j.camBank = 15 - rnd() * 30; j.camPersp = 0.05 + rnd() * 0.2;
  const dofArea = 0.2 + rnd() * 0.5;
  let crackle: VarInstance | undefined;
  for (const x of j.xforms) { for (let i = x.vars.length - 1; i >= 0; i--) { const vi = x.vars[i]; if (vi.name === 'crackle' && Math.abs(vi.params.scale ?? 0) < 1e-10) { crackle = vi; x.color = rnd(); break; } } if (crackle) break; }
  if (!crackle) { const x = newXForm(); x.weight = 0.5; j.xforms.push(x); crackle = addVar(x, 1, 'crackle', { scale: 0 }); }
  crackle.weight = 1 + rnd() * 2; crackle.params.distort = 1.5 + rnd() * 1.5; crackle.params.cellsize = 0.5 + rnd() * 2;
  const focus: [number, number, number] = rnd() < 0.33 ? [0.33 - rnd() * 0.66, 0.25 - rnd() * 0.5, 0.1 - rnd() * 0.2] : [0, 0, 0];
  const dofScale = 1.5 + rnd() * 2;
  let angle = 20 * rnd(), fade = 0, shape = 'BUBBLE';
  const params = [0, 0, 0, 0, 0, 0];
  const r = rnd();
  if (r < 0.08) { shape = 'BUBBLE'; fade = 0.6 + rnd() * 0.4; }
  else if (r < 0.16) { shape = 'HEART'; fade = rnd() < 0.25 ? 0.2 + rnd() * 0.8 : 0; }
  else if (r < 0.2) { shape = 'CANNABISCURVE'; fade = rnd() < 0.25 ? 0.2 + rnd() * 0.8 : 0; }
  else if (r < 0.28) { shape = 'NBLUR'; params[0] = 3 + rnd() * 5; if (rnd() < 0.33) { params[1] = 2 + rnd() * 5; params[2] = 1; params[3] = 0; params[4] = rnd() < 0.33 ? 1 : 0; } }
  else if (r < 0.36) { shape = 'FLOWER'; params[0] = 0.3 + rnd() * 0.2; params[1] = 5 + rnd() * 5; }
  else if (r < 0.44) { shape = 'CLOVERLEAF'; fade = rnd() < 0.25 ? 0.2 + rnd() * 0.8 : 0; }
  else if (r < 0.52) { shape = 'SINEBLUR'; params[0] = 1.2 + rnd(); }
  else if (r < 0.6) { shape = 'PERLIN_NOISE'; params[0] = rnd(); params[1] = 1.2 + rnd() * 1.8; params[2] = 0.1 + rnd() * 0.4; }
  else if (r < 0.64) { shape = 'STARBLUR'; fade = 0.2 + rnd() * 0.8; params[0] = 4 + rnd() * 6; params[1] = 0.40162283177245455973959534526548; }
  else if (r < 0.72) { shape = 'TAURUS'; angle = 0; params[0] = 2.5 + rnd(); params[1] = 4 + rnd() * 3; params[2] = 1.25 * rnd() * 0.5; params[3] = 0.9 + rnd() * 0.2; }
  else if (r < 0.8) { shape = 'BRUSH_STROKE'; angle = 0; params[0] = 1 + randomInt(30); params[1] = rnd() > 0.25 ? 1 + randomInt(30) : 0; params[2] = rnd() > 0.75 ? 1 + randomInt(30) : 0; }
  else if (r < 0.92) { shape = 'RECT'; params[0] = 0.4 + rnd() * 0.4; }
  const prev = j.extra;
  j.extra = (f) => { prev?.(f); f.camDOFArea = dofArea; f.focusX = focus[0]; f.focusY = focus[1]; f.focusZ = focus[2]; f.camDOFScale = dofScale; f.camDOFRotate = angle; f.camDOFFade = fade; f.camDOFShape = shape; f.camDOFParams = params; };
}
/** BlackAndWhiteRandomFlameGenerator.postProcessFlameBeforeRendering: a flat white or black gradient on a dark / light background, punchier tone */
function blackAndWhite(j: JFlame) {
  const prev = j.extra;
  const dark = rnd() < 0.42;
  const bg: RGB = dark ? (rnd() < 0.5 ? [0, 0, 0] : [Math.trunc(rnd() * 64) / 255, Math.trunc(rnd() * 64) / 255, Math.trunc(rnd() * 64) / 255]) : (rnd() < 0.5 ? [1, 1, 1] : [Math.trunc(255 - rnd() * 64) / 255, Math.trunc(255 - rnd() * 64) / 255, Math.trunc(255 - rnd() * 64) / 255]);
  const gamma = 0.45 + rnd() * 1.5, thr = 0.2 * rnd(), bright = 3 + rnd() * 2;
  j.extra = (f) => {
    prev?.(f);
    f.background = bg;
    const c: RGB = dark ? [1, 1, 1] : [0, 0, 0];
    for (const l of f.layers) l.palette = Array.from({ length: 256 }, () => [...c] as RGB);
    f.gamma = gamma; f.gammaThreshold = thr; f.saturation = 1; f.brightness = bright; f.contrast = 2.5; f.bgTransparency = false;
  };
}
STYLES.push(
  { id: 'subflame', name: 'SubFlame', filter: true, build: () => {
    // the sub-flame: one of the simple generators, with its own gradient, embedded as a subflame_wf resource
    const subPalette = randomPalette();
    const subId = pickStyle(SIMPLE_GENERATORS.filter((id) => id !== 'subflame'));
    const subStyle = STYLES.find((st) => st.id === subId)!;
    const sub = toFlame(subStyle.build(), subStyle.name, subStyle.filter, subPalette);
    const f = newFlame(); f.centreX = 2; f.centreY = 1; f.camRoll = -2; f.ppu = 200;
    f.gamma = sub.gamma;
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; const vi = addVar(x, 1, 'subflame_wf'); vi.res = { flame: flameToXML(sub) }; x.color = 0; x.colorSymmetry = -0.22; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.25 + rnd() * 0.5; x.c00 = 0.17254603006834707; x.c01 = 0.6439505508593787; x.c10 = -0.6439505508593787; x.c11 = 0.17254603006834707;
      x.c20 = 1.5 + rnd() * 2.5; x.c21 = -0.25 - rnd() * 0.35; addVar(x, 1, 'linear3D'); x.color = rnd(); x.colorSymmetry = -0.62; }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.25 + rnd() * 0.5; x.c00 = 0.17254603006834707; x.c01 = 0.6439505508593787; x.c10 = -0.6439505508593787; x.c11 = 0.17254603006834707;
      x.c20 = -3; x.c21 = 0.3; addVar(x, 1, 'curl3D', { cx: -0.2 + 0.4 * rnd(), cy: 0, cz: 0 }); x.color = rnd(); x.colorSymmetry = 0; }
    f.extra = (fl) => { fl.layers[0].palette = sub.layers[0].palette; };
    return f;
  } },
  { id: 'blackandwhite', name: 'Black&White', filter: true, build: () => { const f = buildOf(pickStyle(BW_GENERATORS)); blackAndWhite(f); f.filter = true; return f; } },
  { id: 'bokeh', name: 'Bokeh', filter: true, build: () => { const f = buildOf(pickStyle(BW_GENERATORS)); bokehMutation(f); f.filter = true; return f; } },
);

// ---- the Solid family (SolidRandomFlameGenerator + its subclasses) ----
/** randomizeSolidRenderingSettings + the base's postProcessFlameBeforeRendering (oversampling 1, 0.5 Gaussian filter, light 0 re-aimed) */
function solidSettings(j: JFlame, enabled = true) {
  j.preserveZ = true;
  const mat = { diffuse: 0.3 + rnd() * 0.3, ambient: 0.7 + rnd() * 0.3, phong: 0.2 + rnd() * 0.8, phongSize: 2 + rnd() * 10, diffFunc: rnd() < 0.5 ? 'COSA_SQUARE' : 'COSA' };
  const bias = 0.05 + rnd() * 0.1;
  const alt = 60 - 120 * rnd(), az = 30 - 60 * rnd();
  const prev = j.extra;
  j.extra = (f) => {
    prev?.(f);
    const sr = defaultSolidRender(enabled);
    Object.assign(sr.materials[0], mat);
    sr.lights[0].altitude = alt; sr.lights[0].azimuth = az;
    sr.shadows = { ...sr.shadows, type: 'FAST', mapSize: 4096, bias };
    f.solid = sr;
    f.oversample = 1; f.filterRadius = 0.5; f.filterKernel = 'GAUSSIAN'; f.antialiasAmount = 0; f.antialiasRadius = 0;
  };
}
/** SolidRandomFlameGenerator.getRandomVariation: none of the fract, inflate, pre_, post_, prepost_ names nor flatten */
const solidRandomVariation = () => randomVariationNamePlain();
/** SolidRandomFlameGenerator.getRandom3DShape: the 23 cases; the plot / text / terrain / knots / dla families are not in this build, so their cases are re-drawn */
function random3DShape(): VarInstance {
  for (;;) {
    switch (randomInt(23)) {
      case 1: return v('seashell3D', 1, { height: 3 + rnd() * 3, inner_radius: 0.2 + rnd() * 0.4, final_radius: 0.2 + rnd() * 0.4 });
      case 5: return v('superShape3d', 1);
      case 7: return v('sunflower', 1);
      case 8: return v('rhodonea', 1);
      case 9: return v('dustpoint', 1);
      case 10: return v('klein_group', 1);
      case 12: return v('butterfly_fay', 1);
      case 14: return v('mandelbrot', 1);
      case 15: return v('pie3D', 1, { thickness: 0.2 + rnd() * 0.4 });
      case 18: return v('obj_mesh_primitive_wf', 1, { primitive: randomInt(16), scale_x: 1, scale_y: 1, scale_z: 1, offset_x: 0, offset_y: 0, offset_z: 0, subdiv_level: rnd() > 0.5 ? 0 : rnd() > 0.5 ? 2 : 1, subdiv_smooth_passes: 12, subdiv_smooth_lambda: 0.42, subdiv_smooth_mu: -0.45 });
      case 19: return v('sattractor3D', 1);
      case 20: return v('oscilloscope2', 1);
      case 21: return v('spirograph', 1);
      case 22: return v('primitives_wf', 1, { shape: randomInt(6) });
      default: continue; // yplot2d_wf, yplot3d_wf, text_wf, terrain3D, polarplot2d_wf, polarplot3d_wf, knots3D, parplot2d_wf, dla3d_wf
    }
  }
}
const randomPlane = (): 'xy' | 'yz' | 'zx' => (rnd() > 0.666 ? 'xy' : rnd() < 0.5 ? 'yz' : 'zx');
/** scale / rotate / localTranslate in the current plane (the shape transforms of the Solid generators) */
const jumble = (x: JX, post: boolean) => { scale(x, 1.25 - rnd() * 0.5, true, true, post); rotate(x, 360 * rnd(), post); localTranslate(x, 1 - 2 * rnd(), 1 - 2 * rnd(), post); };
const solidHeader = (f: JFlame, zoom: () => number, bank: () => number) => { f.camPitch = 90 - rnd() * 180; f.camYaw = 30 - rnd() * 60; f.camBank = bank(); f.camPersp = rnd() * 0.2; f.width = 601; f.height = 338; f.ppu = 100 + rnd() * 100; f.camZoom = zoom(); };
STYLES.push(
  { id: 'solid_experimental', name: 'Solid (experimental)', filter: false, build: () => {
    const f = newFlame(); solidHeader(f, () => 0.2 + rnd() * 0.4, () => 45 - rnd() * 90); solidSettings(f);
    { const x = newXForm(); f.xforms.push(x); x.weight = rnd() < 0.3 ? 0.5 : 0.25 + rnd(); x.color = rnd(); x.colorSymmetry = 1 - 2 * rnd();
      x.c00 = 0.58839186; x.c10 = -0.93477004; x.c20 = 0.88339811; x.c01 = 0.93477004; x.c11 = 0.58839186; x.c21 = 0.59357541;
      x.p00 = -1; x.p10 = 1e-16; x.p01 = -1e-16; x.p11 = -1;
      x.yzPost = [-0.00074102, 0.99999973, 0, -0.99999973, -0.00074102, 0];
      if (rnd() < 0.85) addVar(x, 0.25 + rnd() * 0.75, 'hypertile3D2', rnd() > 0.75 ? { p: 4 + rnd() * 3, q: 4 + rnd() * 2 } : { p: 5, q: 4 });
      else addVar(x, 0.25 + rnd() * 0.75, solidRandomVariation()); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = rnd(); x.colorSymmetry = 1 - 2 * rnd(); if (rnd() > 0.125) x.opacity = HIDDEN;
      const sh = random3DShape(); sh.weight = 0.2 + rnd() * 0.5; x.vars.push(sh);
      inPlane(randomPlane(), () => { jumble(x, false); if (rnd() > 0.5) jumble(x, true); }); }
    if (rnd() > 0.666) { const x = newXForm(); f.finals.push(x); x.weight = 0; addVar(x, 0.75 + rnd() * 0.25, solidRandomVariation());
      if (rnd() > 0.5) addVar(x, 0.25 + rnd() * 0.25, 'linear3D'); if (rnd() > 0.5) jumble(x, false); if (rnd() > 0.5) jumble(x, true); }
    return f;
  } },
  { id: 'solid_julia3d', name: 'Solid Julia3D', filter: false, build: () => {
    const f = newFlame(); solidHeader(f, () => 0.6 + rnd() * 0.7, () => 15 - rnd() * 30); solidSettings(f);
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = 0.16556899; const sh = random3DShape(); sh.weight = 0.5 + rnd() * 2; x.vars.push(sh);
      if (rnd() > 0.75) addVar(x, 0.2 + rnd() * 0.2, 'pre_wave3D_wf', { wavelen: 0.75 + rnd() * 0.5, phase: 0.5 + rnd(), damping: 0.01 }, 0); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = 0.02637356;
      x.c00 = 0.53041695; x.c10 = -0.14810786; x.c20 = 2.09422612; x.c01 = 0.14810786; x.c11 = 0.53041695; x.c21 = 0.23893855;
      x.yz = [1, 0, 0.01405521, 0, 1, -0.01405521]; x.zx = [0.74081053, -0.04366518, -0.07027604, 0.04366518, 0.74081053, -0.14055209];
      addVar(x, 1, 'linear3D'); addVar(x, -0.524, 'zscale'); }
    { const x = newXForm(); f.finals.push(x); x.weight = 0; addVar(x, 1, 'julia3Dq', { power: FTOI(3 + rnd() * 4), divisor: rnd() > 0.5 ? 2 : 1 + rnd() * 2 }); }
    return f;
  } },
  { id: 'solid_labyrinth', name: 'Solid Labyrinth', filter: false, build: () => {
    const f = newFlame(); f.camPitch = 30 + rnd() * 40; f.camYaw = 30 - rnd() * 60; f.camBank = 5 - rnd() * 10; f.camPersp = rnd() * 0.2; f.width = 601; f.height = 338; f.ppu = 100 + rnd() * 100; f.camZoom = 0.75 + rnd() * 0.5;
    solidSettings(f, !(rnd() < 0.333));
    // the sub-flame: one direct-colour texture transform with its own gradient
    const sub = newFlame(); sub.width = 711; sub.height = 400; sub.ppu = 38.88427164;
    { const x = newXForm(); sub.xforms.push(x); x.weight = 0.5; x.color = 0; x.colorSymmetry = 0;
      if (rnd() < 0.5) {
        if (rnd() < 0.45) addVar(x, 1 + rnd() * 2, 'dc_crackle_wf', { cellsize: 0.15 + rnd() * 0.5, power: 0.2 + rnd() * 1.2, distort: rnd() > 0.2 ? rnd() * 0.5 : 0, scale: rnd() > 0.667 ? 1.06 - rnd() * 0.12 : 1, z: rnd() * 0.5, color_scale: 0.5, color_offset: 0 });
        else if (rnd() < 0.45) addVar(x, 0.15 + rnd() * 0.75, 'metaballs3d_wf');
        else { addVar(x, 1, 'truchet', { extended: rnd() < 0.25 ? 1 : 0, exponent: 0.2 + rnd() * 1.3, arc_width: 0.5 + rnd() * 0.5, rotation: 0, size: 0.4 + rnd() * 0.3, seed: 50 + rnd() * 50, direct_color: 1 }); addVar(x, 2 + rnd() * 2, 'pre_blur'); }
      } else addVar(x, 3 + rnd() * 4, 'dc_perlin', { shape: Math.trunc(rnd() * 3), map: rnd() < 0.15 ? Math.trunc(rnd() * 6) : 2, select_centre: 0, select_range: 1, centre: 0.25 + rnd() * 0.5, range: 0.25 + rnd() * 0.5, edge: 0, scale: 1 + rnd() * 5, octaves: rnd() < 0.33 ? FTOI(2 + rnd() * 2) : 2, amps: rnd() < 0.33 ? FTOI(2 + rnd() * 2) : 2, freqs: rnd() < 0.33 ? FTOI(2 + rnd() * 2) : 2, z: 0, select_bailout: 10 }); }
    const subFlame = toFlame(sub, 'Solid Labyrinth sub', false, randomPalette());
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5;
      let cs = 1.5 - rnd() * 3; if (Math.abs(cs) < 0.4) cs = rnd() < 0.5 ? -0.3 - rnd() : 0.3 + rnd();
      const vi = addVar(x, 1, 'subflame_wf', { color_mode: 0, colorscale_z: cs }); vi.res = { flame: flameToXML(subFlame) }; x.color = 0; x.colorSymmetry = -0.22; }
    if (rnd() > 0.25) { const prev = f.extra; const t = rnd() > 0.25 ? 'FAST' : 'SMOOTH'; f.extra = (fl) => { prev?.(fl); if (fl.solid) fl.solid.shadows.type = t; }; }
    return f;
  } },
  { id: 'solid_stunning', name: 'Solid (stunning)', filter: false, build: () => {
    const f = newFlame(); solidHeader(f, () => 0.5 + rnd() * 0.7, () => 30 - rnd() * 60); solidSettings(f);
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = rnd(); x.colorSymmetry = 1 - 2 * rnd(); if (rnd() > 0.125) x.opacity = HIDDEN;
      const sh = random3DShape(); sh.weight = 0.2 + rnd(); x.vars.push(sh);
      inPlane(randomPlane(), () => { jumble(x, false); if (rnd() > 0.5) jumble(x, true); }); }
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.c00 = 0.54625622; x.c10 = 0.26758811; x.c01 = -0.26758811; x.c11 = 0.54625622;
      x.yz = [0.46864442, -0.17017929, 1.21536218, 0.17017929, 0.46864442, -0.02558657]; x.zx = [0.81078767, 0.12035676, 0.21748586, -0.12035676, 0.81078767, -0.05117314];
      addVar(x, 1, 'linear3D'); jumble(x, false); if (rnd() > 0.666) jumble(x, true); }
    if (rnd() > 0.666) { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.c00 = 0.54625622; x.c10 = 0.26758811; x.c01 = -0.26758811; x.c11 = 0.54625622;
      x.yz = [0.46864442, -0.17017929, 1.21536218, 0.17017929, 0.46864442, -0.02558657]; x.zx = [0.59207155, -0.56684538, 1.31770847, 0.56684538, 0.59207155, -0.84435688];
      addVar(x, 1, 'linear3D'); jumble(x, false); if (rnd() > 0.5) jumble(x, true); }
    if (rnd() > 0.42) { const x = newXForm(); f.finals.push(x); x.weight = 0; addVar(x, 0.75 + rnd() * 0.25, solidRandomVariation());
      if (rnd() > 0.5) addVar(x, 0.25 + rnd() * 0.25, 'linear3D'); if (rnd() > 0.5) jumble(x, false); if (rnd() > 0.5) jumble(x, true); }
    return f;
  } },
  { id: 'solid_recursive', name: 'Solid (recursive)', filter: false, build: () => {
    const f = newFlame(); solidHeader(f, () => 0.5 + rnd() * 0.25, () => 30 - rnd() * 60); solidSettings(f);
    { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = rnd(); x.colorSymmetry = 1 - 2 * rnd(); if (rnd() > 0.125) x.opacity = HIDDEN;
      const sh = random3DShape(); sh.weight = rnd() < 0.75 ? 0.1 + rnd() * 0.3 : 0.2 + rnd() * 0.5; x.vars.push(sh);
      inPlane(randomPlane(), () => { jumble(x, false); if (rnd() > 0.5) jumble(x, true); }); }
    const aff = (p: Record<string, number>) => { const x = newXForm(); f.xforms.push(x); x.weight = 0.5; x.color = rnd(); x.colorSymmetry = 1 - 2 * rnd(); addVar(x, 1, 'affine3D', { shearXY: 0, shearXZ: 0, shearYX: 0, shearYZ: 0, shearZX: 0, shearZY: 0, ...p }); };
    { const scl = 0.55 + rnd() * 0.2; aff({ translateX: 0.5 + rnd() * 2, translateY: 0.25 + rnd(), translateZ: 0, scaleX: scl, scaleY: scl, scaleZ: scl, rotateX: rnd() * 4, rotateY: 0, rotateZ: rnd() * 4 }); }
    { const scl = 0.45 + rnd() * 0.3; aff({ translateX: 0.1 + rnd() * 0.2, translateY: 0.5 + rnd() * 3, translateZ: 0.35 + rnd(), scaleX: scl, scaleY: scl, scaleZ: scl, rotateX: 1 + rnd() * 5, rotateY: 25 - 50 * rnd(), rotateZ: 2 - 4 * rnd() }); }
    if (rnd() < 0.25) { const scl = 0.35 + rnd() * 0.2; aff({ translateX: 0.5 - rnd(), translateY: 0.5 - rnd(), translateZ: 0.5 - rnd(), scaleX: scl, scaleY: scl, scaleZ: scl, rotateX: 3 - rnd() * 6, rotateY: 3 - rnd() * 6, rotateZ: 3 - rnd() * 6 }); }
    return f;
  } },
  { id: 'solid_shadows', name: 'Solid (shadows)', filter: false, build: () => {
    // one of the solid generators (Stunning ×4, Experimental, Julia3D, recursive), then a floor layer and shadows
    const f = buildOf(['solid_stunning', 'solid_experimental', 'solid_stunning', 'solid_stunning', 'solid_julia3d', 'solid_stunning', 'solid_recursive'][randomInt(7)]);
    f.camZoom = 0.5 + rnd() * 0.25; f.camPersp = 0.05 + rnd() * 0.15; f.camPitch = -5 - rnd() * 30; f.camYaw = -25 + rnd() * 50; f.camBank = -15 + rnd() * 30; f.camRoll = 0;
    const v0 = f.xforms[0].vars[0]; v0.weight *= 0.25 + rnd() * 0.25;
    const floor = newXForm(); floor.weight = 0.5;
    if (rnd() < 0.5) addVar(floor, 1 + rnd(), 'plane_wf', { axis: 2, position: rnd() * 2 - 0.25, color_mode: FTOI(4 * rnd()) });
    else addVar(floor, 1 + rnd(), 'checkerboard_wf', { axis: 2, position: rnd() * 2 - 0.25, checker_size: 0.05 + rnd() * 0.1, displ_amount: 0.005 + rnd() * 0.03, with_sides: rnd() > 0.25 ? 1 : 0 });
    const layerWeight = 0.25 + rnd() * 0.5, shadowType = rnd() < 0.5 ? 'SMOOTH' : 'FAST';
    const prev = f.extra;
    f.extra = (fl) => { prev?.(fl); const l = defaultLayer(randomPalette()); l.weight = layerWeight; l.xforms = [toXForm(floor, 1)]; fl.layers.push(l); if (fl.solid) fl.solid.shadows.type = shadowType; fl.postSymmetry = undefined; };
    f.filter = false;
    return f;
  } },
);

export interface RandomStyleInfo { id: string; name: string }
export const RANDOM_STYLES: RandomStyleInfo[] = STYLES.map(({ id, name }) => ({ id, name }));

/** A random flame in the given style ('any' picks a style at random, like JWildfire's "All"). */
export function randomFlameInStyle(styleId: string, palette: RGB[] = randomPalette()): Flame {
  const style = STYLES.find((s) => s.id === styleId) ?? STYLES[randomInt(STYLES.length)];
  return toFlame(style.build(), style.name, style.filter, palette);
}
