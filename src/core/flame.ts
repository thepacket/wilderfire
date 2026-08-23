// Flame data model — the CPU-side description of a fractal flame,
// closely following flam3 / JWildfire conventions. A flame is a stack of
// layers (JWildfire-style); each layer has its own transforms and gradient,
// all accumulating into the shared histogram. Camera and tone are global.

import { normFilterKernel, type FilterKernel } from '../gpu/filters';
import { VARIATIONS } from './variations';

export type Affine = [number, number, number, number, number, number]; // x' = a x + b y + c ; y' = d x + e y + f
export type RGB = [number, number, number]; // 0..1

export interface VarInstance {
  name: string;
  weight: number;
  params: Record<string, number>;
  /** JWildfire per-instance priority override (`<var>_fx_priority`): a normal variation forced to run as a
   *  pre step (-1: input ← input + w·f(input)) or a post step (1: output ← output + w·f(output)). Absent = the definition's priority. */
  priority?: number;
  /** String resources (JWildfire "ressources"): file names such as `obj_mesh_wf`'s `obj_filename` (basename of the
   *  user-loaded mesh, see src/core/meshes.ts) or the colour/displacement map names — kept for round-tripping. */
  res?: Record<string, string>;
}

export interface WeightingField {
  /** JWildfire WeightingFieldType name: CELLULAR_NOISE, CUBIC_NOISE, CUBIC_FRACTAL_NOISE, PERLIN_NOISE, PERLIN_FRACTAL_NOISE,
   *  SIMPLEX_NOISE, SIMPLEX_FRACTAL_NOISE, VALUE_NOISE, VALUE_FRACTAL_NOISE, WHITE_NOISE (IMAGE_MAP is not supported). */
  type: string;
  input: 'AFFINE' | 'POSITION';
  varAmount: number;
  color: number;
  jitter: number;
  seed: number;
  frequency: number;
  fractalType: 'FBM' | 'BILLOW' | 'RIGID_MULTI';
  octaves: number;
  gain: number;
  lacunarity: number;
  cellReturn: 'CELL_VALUE' | 'DISTANCE' | 'DISTANCE2' | 'DISTANCE_ADD' | 'DISTANCE_SUB' | 'DISTANCE_MUL' | 'DISTANCE_DIV';
  cellDistance: 'EUCLIDIAN' | 'MANHATTAN' | 'NATURAL';
  /** up to three (variation name, param name or "amount", intensity) modulations */
  params: { varName: string; paramName: string; intensity: number }[];
}
export const WFIELD_TYPES = ['CELLULAR_NOISE', 'CUBIC_NOISE', 'CUBIC_FRACTAL_NOISE', 'PERLIN_NOISE', 'PERLIN_FRACTAL_NOISE', 'SIMPLEX_NOISE', 'SIMPLEX_FRACTAL_NOISE', 'VALUE_NOISE', 'VALUE_FRACTAL_NOISE', 'WHITE_NOISE'];
export function defaultWeightingField(type = 'SIMPLEX_NOISE'): WeightingField {
  return { type, input: 'AFFINE', varAmount: 0, color: 0, jitter: 0, seed: 1337, frequency: 1, fractalType: 'FBM', octaves: 3, gain: 0.5, lacunarity: 2, cellReturn: 'DISTANCE2', cellDistance: 'EUCLIDIAN', params: [] };
}

/** JWildfire solid rendering: a distant light (direction from altitude/azimuth in degrees). */
export interface SolidLight {
  altitude: number;
  azimuth: number;
  intensity: number;
  color: RGB;
  castShadows: boolean;
  shadowIntensity: number;
}

export type LightDiffFunc = 'COSA' | 'COSA_SQUARE' | 'COSA_HALVE' | 'COSA_HALVE_SQUARE';
export const LIGHT_DIFF_FUNCS: LightDiffFunc[] = ['COSA', 'COSA_SQUARE', 'COSA_HALVE', 'COSA_HALVE_SQUARE'];

/** JWildfire solid rendering material (Phong model). */
export interface SolidMaterial {
  diffuse: number;
  ambient: number;
  phong: number;
  phongSize: number;
  phongColor: RGB;
  diffFunc: LightDiffFunc;
  /** reflection map: an image (by file name, looked up in the browser's image store — src/core/reflMaps.ts), its
   *  intensity and the mapping of the view reflection onto it */
  reflMapIntensity: number;
  reflMapping: 'BLINN_NEWELL' | 'SPHERICAL';
  reflMapFilename?: string;
}

/** JWildfire solid rendering settings (`sld_render_*`). Absent on a flame = off. */
export interface SolidRender {
  enabled: boolean;
  lights: SolidLight[];
  materials: SolidMaterial[];
  /** ambient occlusion from the z-buffer neighbourhood */
  ao: { enabled: boolean; intensity: number; searchRadius: number; blurRadius: number; radiusSamples: number; azimuthSamples: number; falloff: number; affectDiffuse: number };
  /** shadow maps per light */
  shadows: { type: 'OFF' | 'FAST' | 'SMOOTH'; smoothRadius: number; mapSize: number; bias: number };
  /** Post-process DOF (JWildfire PostDOFCalculator, runs when the flame is solid and cam_dof > 0):
   *  every tonemapped pixel scatters as a kernel disc of radius |dofDist|·10 px; rare bright pixels
   *  become enlarged, brightened glints — the post_bokeh_* attributes shape those. */
  postBokeh: { filterKernel: string; intensity: number; brightness: number; size: number; activation: number };
}

export function defaultSolidLight(i = 0): SolidLight {
  return i === 0
    ? { altitude: 55, azimuth: -22, intensity: 0.8, color: [1, 1, 1], castShadows: true, shadowIntensity: 0.8 }
    : { altitude: 64, azimuth: 55, intensity: 0.6, color: [1, 1, 1], castShadows: false, shadowIntensity: 0.7 };
}
export function defaultSolidMaterial(): SolidMaterial {
  return { diffuse: 0.7, ambient: 0.5, phong: 0.6, phongSize: 24, phongColor: [1, 1, 1], diffFunc: 'COSA', reflMapIntensity: 0.5, reflMapping: 'BLINN_NEWELL' };
}
/** JWildfire's `SolidRenderSettings.setupDefaults()`: two lights, one material, AO on, shadows off. */
export function defaultSolidRender(enabled = false): SolidRender {
  return {
    enabled,
    lights: [defaultSolidLight(0), defaultSolidLight(1)],
    materials: [defaultSolidMaterial()],
    ao: { enabled: true, intensity: 0.6, searchRadius: 4, blurRadius: 1.5, radiusSamples: 6, azimuthSamples: 7, falloff: 0.5, affectDiffuse: 0.1 },
    shadows: { type: 'OFF', smoothRadius: 1, mapSize: 2048, bias: 0.01 },
    // JWildfire SolidRenderSettings.setupDefaultPostBokehOptions
    postBokeh: { filterKernel: 'SINEPOW15', intensity: 0.005, brightness: 1, size: 2, activation: 0.2 },
  };
}

export interface XForm {
  affine: Affine;
  post: Affine;
  /** JWildfire 3D affines (optional; identity when absent). Same [a,b,c,d,e,f]
   *  layout as `affine`, applied in the yz plane (y' = a·y + b·z + c, z' = d·y + e·z + f)
   *  and the zx plane (x' = a·x + b·z + c, z' = d·x + e·z + f) after the xy affine;
   *  post variants after the post affine. */
  yz?: Affine;
  zx?: Affine;
  yzPost?: Affine;
  zxPost?: Affine;
  weight: number;      // relative selection probability
  color: number;       // palette index 0..1
  colorSpeed: number;  // blend speed toward this xform's color
  /** JWildfire per-transform colour modifiers [gamma, gammaSpeed, contrast, contrastSpeed,
   *  saturation, saturationSpeed, hue, hueSpeed]: each point carries four modifier values,
   *  blended per transform like the colour (m' = m·(1+speed)/2 + value·(1−speed)/2), and applied
   *  to the plotted RGB. Absent/zeros = no effect (the default). */
  colorMods?: number[];
  /** JWildfire weighting field: a noise value per point (of the affine result or the incoming position) that
   *  scales the variation amounts (`varAmount`), up to three named variation params (`params`), the colour
   *  (`color`, ×0.1) and jitters the output (`jitter`, ×0.1). Absent = none. */
  wfield?: WeightingField;
  /** JWildfire solid-rendering material index carried by the point, blended per transform like the
   *  colour (m' = m·(1+speed)/2 + material·(1−speed)/2). Absent = 0 (the first material). */
  material?: number;
  materialSpeed?: number;
  /** JWildfire colour type beyond DIFFUSION (undefined): NONE = no colour step at all — the point keeps the RGB it carries
   *  (finals default to it); CYCLIC = colour index += symmetry (mod 1); DISTANCE = the plotted colour is the palette entry
   *  at color + |Δposition|·(symmetry+1) while the index stays. symmetry = 1 − 2·colorSpeed. */
  colorType?: 'NONE' | 'CYCLIC' | 'DISTANCE' | 'TARGET' | 'TARGETG';
  /** TARGET: the point's RGB is lerped towards this colour (0..1) by (symmetry + 1)/2; TARGETG: towards the palette entry at `color` */
  targetColor?: [number, number, number];
  opacity: number;     // 0..1 plot opacity
  variations: VarInstance[];
  /** Optional variation stages evaluated BEFORE the main sum (transforming the
   *  affine result) and AFTER it (transforming the summed output). Each stage
   *  is a weighted sum like the main list — include `linear` weight 1 in a
   *  stage to keep a pass-through of its input. */
  preVariations?: VarInstance[];
  postVariations?: VarInstance[];
  /** Xaos row: multiplier applied to each target xform's weight when THIS
   *  xform was applied last (flam3 "chaos"). Missing entries default to 1. */
  xaos?: number[];
}

export interface Layer {
  xforms: XForm[];
  final: XForm | null;
  /** JWildfire allows several final transforms, applied in sequence after `final` (each takes the previous output). */
  moreFinals: XForm[];
  palette: RGB[]; // 256 entries
  weight: number;   // JWildfire layer weight: multiplies the plotted colour intensity (all layers iterate equally)
  visible: boolean;
  /** JWildfire smooth_gradient: the gradient step interpolates between neighbouring palette entries instead of stepping */
  smoothGradient?: boolean;
  /** JWildfire gradient map (TransformationGradientMapColorStep): the plot colour is an image sampled at the point's position;
   *  `file` = basename in the image store (like bgImage). */
  gradientMap?: { file: string; hOffset: number; hScale: number; vOffset: number; vScale: number; lcolorAdd: number; lcolorScale: number };
}

/** JWildfire channel mixer: curves over the per-pixel average raw colour (x = value·100 on a 0..25600 axis, y likewise).
 *  RGB uses rr/gg/bb, BRIGHTNESS scales by rr(luma)/luma, FULL mixes all nine. Missing curves are JWildfire's defaults
 *  (identity for rr/gg/bb, zero otherwise). */
export type MixerKey = 'rr' | 'rg' | 'rb' | 'gr' | 'gg' | 'gb' | 'br' | 'bg' | 'bb';
export const MIXER_KEYS: MixerKey[] = ['rr', 'rg', 'rb', 'gr', 'gg', 'gb', 'br', 'bg', 'bb'];
export interface MixerCurve { points: [number, number][]; interp: 'spline' | 'linear' }
export interface ChannelMixer { mode: 'RGB' | 'BRIGHTNESS' | 'FULL'; curves: Partial<Record<MixerKey, MixerCurve>> }

export interface Flame {
  name: string;
  layers: Layer[];
  centerX: number;
  centerY: number;
  zoom: number;      // 1 = world range ~[-2,2] fits the short canvas axis
  rotation: number;  // radians (camera roll)
  /** 3D camera (JWildfire semantics): pitch/yaw in degrees, perspective factor,
   *  camera position offset; all zero = flat 2D projection. */
  camPitch: number;
  camYaw: number;
  camBank: number;
  camPersp: number;
  camPosX: number;
  camPosY: number;
  camPosZ: number;
  /** JWildfire preserve_z: 2D variations pass the point's z through (scaled by weight). */
  preserveZ: boolean;
  /** Depth of field (JWildfire): amount (0 = off), blur-shape scale, fade (0..1),
   *  and either the "new" focus-point model (focusX/Y/Z, area, exponent) or the
   *  legacy focus plane camZ. Blur shape is always the bubble (disc). */
  camDOF: number;
  camDOFArea: number;
  camDOFExponent: number;
  camDOFScale: number;
  /** JWildfire DOF blur shape (BUBBLE = the plain disc); the shape's parameters come from camDOFParams (cam_dof_param1..6) */
  camDOFShape?: string;
  camDOFRotate?: number;
  camDOFParams?: number[];
  camDOFFade: number;
  newDOF: boolean;
  focusX: number;
  focusY: number;
  focusZ: number;
  camZ: number;
  /** Depth fade (JWildfire dimish-z): points farther than dimZDist along the camera
   *  z axis fade toward dimZColor with exp(-d²·dimishZ). */
  dimishZ: number;
  dimZDist: number;
  dimZColor: RGB;
  brightness: number;
  gamma: number;
  /** flam3 gamma_threshold: linear ramp below this alpha, pow(1/gamma) above —
   *  prevents gamma from amplifying single-sample speckle. */
  gammaThreshold: number;
  vibrancy: number;
  background: RGB;
  /** JWildfire log-density constants: intensity = 2·contrast·brightness·log10(1 + d/(contrast·area))
   *  + lowDensityBrightness glow; colours scaled by 255/whiteLevel (fade to white). */
  contrast: number;
  whiteLevel: number;
  lowDensityBrightness: number;
  /** Spatial filter over the log-scaled image (JWildfire `filter` radius, 0 = off) and kernel. */
  filterRadius: number;
  /** JWildfire FilterKernelType name (MITCHELL_SMOOTH default; SINEPOW15 etc.) */
  filterKernel: FilterKernel;
  /** Adaptive filtering (MITCHELL_SINEPOW only): edge threshold and the density below which
   *  a pixel counts as sparse (JWildfire `filter_sharpness` / `filter_low_density`). */
  filterSharpness: number;
  filterLowDensity: number;
  /** JWildfire antialiasing: a fraction of samples get a random sub-pixel-ish jitter. */
  antialiasAmount: number;
  antialiasRadius: number;
  /** JWildfire density estimation (DeCalculator): estimator radius = deRadius·9 px
   *  (0 = off), deCurve = acceptance falloff with distance (0.8 default). */
  deRadius: number;
  deCurve: number;
  /** JWildfire `saturation`: an HSL saturation shift of (saturation − 1) applied to the
   *  finished pixel, after the background is composited in (GammaCorrectionFilter). */
  saturation: number;
  /** JWildfire `fg_opacity`: scales the alpha channel only, by 1 − atan(3·(v − 1))/1.25. */
  fgOpacity: number;
  /** JWildfire `bg_transparency`: the background stays transparent in the saved image. */
  bgTransparency: boolean;
  mixer?: ChannelMixer;
  /** JWildfire background_image (file name only): the image, stretched to the full image, replaces the background colour;
   *  the picture itself comes from the browser's image store (Render tab → Background → ⬆ image) */
  bgImage?: string;
  /** JWildfire `oversample` (spatial oversampling, 1–3): histogram supersampling factor. */
  oversample: number;
  /** JWildfire post symmetry, applied to the plotted point (DefaultRenderIterationState). */
  postSymmetry?: PostSymmetry;
  /** JWildfire motion blur: `length` extra render packets at frame + length·timeStep/2 − p·timeStep (p = 1..length, in frames),
   *  each layer weight scaled by 1 − p²·decay·0.07/length (≥ 0.01); rendered here as weighted sub-frame averages */
  motionBlur?: { length: number; timeStep: number; decay: number };
  /** Provenance written by JWildfire (meta_info_author / meta_info_creation_time / meta_info_uuid) — kept through import/export. */
  author?: string;
  created?: string;
  uuid?: string;
  /** JWildfire solid rendering (z-buffer surface shading instead of density accumulation). Absent = off. */
  solid?: SolidRender;
  /** JWildfire background gradient (`background_type`): 2×2 corner colours, optionally with a centre colour. Absent = single colour `background`. */
  bgGradient?: { type: 'GRADIENT_2X2' | 'GRADIENT_2X2_C'; ul: RGB; ur: RGB; ll: RGB; lr: RGB; cc: RGB };
}

/** JWildfire post symmetry (`post_symmetry_*`): every plotted point is duplicated,
 *  mirrored about an axis through the centre, or rotated into `order` copies. */
export interface PostSymmetry {
  type: 'X_AXIS' | 'Y_AXIS' | 'POINT';
  /** rotational copies (POINT only) */
  order: number;
  centreX: number;
  centreY: number;
  /** mirror offset from the centre (axis modes) */
  distance: number;
  /** degrees of extra rotation applied to the mirrored copy (axis modes) */
  rotation: number;
}

export const IDENTITY: Affine = [1, 0, 0, 0, 1, 0];
export const MAX_LAYERS = 8;
export const MAX_XFORMS = 16;

export function defaultXForm(): XForm {
  return {
    affine: [...IDENTITY] as Affine,
    post: [...IDENTITY] as Affine,
    weight: 1,
    color: 0,
    colorSpeed: 0.5,
    opacity: 1,
    variations: [{ name: 'linear', weight: 1, params: {} }],
  };
}

export function defaultLayer(palette: RGB[]): Layer {
  return {
    xforms: [defaultXForm(), defaultXForm()],
    final: null,
    moreFinals: [],
    palette,
    weight: 1,
    visible: true,
  };
}

export function defaultFlame(palette: RGB[]): Flame {
  return {
    name: 'untitled',
    layers: [defaultLayer(palette)],
    centerX: 0,
    centerY: 0,
    zoom: 1,
    rotation: 0,
    camPitch: 0, camYaw: 0, camBank: 0, camPersp: 0, camPosX: 0, camPosY: 0, camPosZ: 0,
    preserveZ: false,
    camDOF: 0, camDOFArea: 0.5, camDOFExponent: 2, camDOFScale: 1, camDOFFade: 1, newDOF: false, camDOFShape: 'BUBBLE', camDOFRotate: 0, camDOFParams: [0, 0, 0, 0, 0, 0],
    focusX: 0, focusY: 0, focusZ: 0, camZ: 0,
    dimishZ: 0, dimZDist: 0, dimZColor: [0, 0, 0],
    brightness: 4,
    gamma: 4,
    gammaThreshold: 0.01,
    vibrancy: 1,
    background: [0, 0, 0],
    contrast: 1, whiteLevel: 220, lowDensityBrightness: 0.24,
    filterRadius: 0, filterKernel: 'MITCHELL_SMOOTH',
    antialiasAmount: 0.25, antialiasRadius: 0.5,
    deRadius: 1, deCurve: 0.8,
    saturation: 1, fgOpacity: 1, bgTransparency: false, oversample: 1,
    filterSharpness: 4, filterLowDensity: 0.025,
  };
}

export function cloneFlame(f: Flame): Flame {
  return structuredClone(f);
}

export function cloneXForm(x: XForm): XForm {
  return structuredClone(x);
}

export function cloneLayer(l: Layer): Layer {
  return structuredClone(l);
}

/** Layers the GPU kernel includes (falls back to the first layer if all are hidden). */
export function visibleLayers(f: Flame): Layer[] {
  const vis = f.layers.filter((l) => l.visible);
  return vis.length ? vis : [f.layers[0]];
}

function strHash(s: string): string { let h = 5381; for (let i = 0; i < s.length; i++) h = (Math.imul(h, 33) ^ s.charCodeAt(i)) >>> 0; return h.toString(36) + s.length.toString(36); }

/** Structural signature — when this changes, the WGSL shader must be regenerated. */
export function flameSignature(f: Flame): string {
  // (a subflame_wf instance's sub-flame is compiled into the kernel: its XML is part of the structure)
  // (a registry entry whose kernel code depends on the instance — a formula, a sub-flame — contributes its sigKey)
  const names = (l?: VarInstance[]) => (l ?? []).map((v) => v.name + (v.priority !== undefined ? '@' + v.priority : '') + (v.name === 'subflame_wf' ? '{' + strHash(v.res?.flame ?? '') + '}' : '') + (VARIATIONS[v.name]?.sigKey ? '{' + strHash(VARIATIONS[v.name].sigKey!(v)) + '}' : '')).join(',');
  const sig = (x: XForm) => `${names(x.preVariations)}<${names(x.variations)}>${names(x.postVariations)}` + (x.colorType ? '~' + x.colorType : '') + (x.wfield ? `~wf(${x.wfield.params.map((p) => p.varName + '.' + p.paramName).join(',')})` : '');
  return visibleLayers(f)
    .map((l) => l.xforms.map(sig).join('|') + '#' + [l.final, ...l.moreFinals].map((x) => (x ? sig(x) : '-')).join('#') + (l.smoothGradient ? '~smooth' : '') + (l.gradientMap ? `~gmap(${[l.gradientMap.hOffset, l.gradientMap.hScale, l.gradientMap.vOffset, l.gradientMap.vScale, l.gradientMap.lcolorAdd, l.gradientMap.lcolorScale].map((v) => +v.toPrecision(6)).join(',')})` : ''))
    .join('@@') + (visibleLayers(f).some((l) => [...l.xforms, l.final, ...l.moreFinals].some((x) => x?.colorMods?.some((v) => v !== 0))) ? '~mods' : '')
    + (f.solid?.enabled ? '~solid' : '') + (usesMaterials(f) ? '~mat' : '')
    + (f.camDOF > 0 && f.camDOFShape && f.camDOFShape !== 'BUBBLE' ? `~dof(${f.camDOFShape},${(f.camDOFParams ?? []).map((v) => +v.toPrecision(6)).join(',')})` : '')
    // post-symmetry constants are baked into the kernel, so every field belongs in the signature
    + (f.postSymmetry ? `~psym(${f.postSymmetry.type},${f.postSymmetry.order},${f.postSymmetry.centreX},${f.postSymmetry.centreY},${f.postSymmetry.distance},${f.postSymmetry.rotation})` : '');
}

/** True when any transform carries a non-default material blend (the kernel then tracks a per-point material). */
export function usesMaterials(f: Flame): boolean {
  return visibleLayers(f).some((l) => [...l.xforms, l.final, ...l.moreFinals].some((x) => x && ((x.material ?? 0) !== 0 || (x.materialSpeed ?? 0) !== 0)));
}

/** Rotate the linear part of an affine (rotates the triangle in world space). */
export function rotateAffine(a: Affine, rad: number): Affine {
  const c = Math.cos(rad), s = Math.sin(rad);
  return [
    a[0] * c + a[1] * s, -a[0] * s + a[1] * c, a[2],
    a[3] * c + a[4] * s, -a[3] * s + a[4] * c, a[5],
  ];
}

export function scaleAffine(a: Affine, s: number): Affine {
  return [a[0] * s, a[1] * s, a[2], a[3] * s, a[4] * s, a[5]];
}

const num = (v: unknown, d: number) => (typeof v === 'number' && isFinite(v) ? v : d);
function normMixer(m: unknown): ChannelMixer | undefined {
  const o = m as { mode?: unknown; curves?: Record<string, { points?: unknown; interp?: unknown }> } | undefined;
  if (!o || (o.mode !== 'RGB' && o.mode !== 'BRIGHTNESS' && o.mode !== 'FULL')) return undefined;
  const curves: Partial<Record<MixerKey, MixerCurve>> = {};
  for (const k of MIXER_KEYS) {
    const c = o.curves?.[k]; if (!c || !Array.isArray(c.points)) continue;
    const pts = (c.points as unknown[]).filter((p): p is [number, number] => Array.isArray(p) && p.length === 2 && isFinite(p[0]) && isFinite(p[1])).map((p) => [num(p[0], 0), num(p[1], 0)] as [number, number]);
    if (pts.length) curves[k] = { points: pts, interp: c.interp === 'linear' ? 'linear' : 'spline' };
  }
  return { mode: o.mode, curves };
}
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function normAffine(a: unknown, d: Affine): Affine {
  if (!Array.isArray(a) || a.length !== 6) return [...d] as Affine;
  return a.map((v, i) => num(v, d[i])) as Affine;
}

function normVarList(list: any): VarInstance[] | null {
  if (!Array.isArray(list)) return null;
  return list
    .filter((v: any) => v && typeof v.name === 'string')
    .map((v: any) => ({
      name: v.name,
      weight: num(v.weight, 1),
      params: typeof v.params === 'object' && v.params ? Object.fromEntries(
        Object.entries(v.params).map(([k, val]) => [k, num(val, 0)])
      ) : {},
      ...(v.priority === -1 || v.priority === 1 ? { priority: v.priority } : {}),
      ...(typeof v.res === 'object' && v.res && Object.values(v.res).some((r) => typeof r === 'string' && r) ? { res: Object.fromEntries(Object.entries(v.res).filter(([, r]) => typeof r === 'string' && r) as [string, string][]) } : {}),
    }));
}

function normXForm(x: any): XForm {
  const d = defaultXForm();
  const vars: VarInstance[] = normVarList(x?.variations) ?? d.variations;
  const out: XForm = {
    affine: normAffine(x?.affine, d.affine),
    post: normAffine(x?.post, d.post),
    weight: Math.max(0, num(x?.weight, 1)),
    color: clamp01(num(x?.color, 0)),
    colorSpeed: clamp01(num(x?.colorSpeed, 0.5)),
    ...(Array.isArray(x?.colorMods) && x.colorMods.some((v: unknown) => Number(v)) ? { colorMods: Array.from({ length: 8 }, (_, i) => num(x.colorMods[i], 0)) } : {}),
    ...(x?.wfield && typeof x.wfield.type === 'string' && WFIELD_TYPES.includes(x.wfield.type) ? { wfield: { ...defaultWeightingField(x.wfield.type), ...x.wfield, params: Array.isArray(x.wfield.params) ? x.wfield.params.slice(0, 3) : [] } } : {}),
    opacity: clamp01(num(x?.opacity, 1)),
    variations: vars.length ? vars : d.variations,
  };
  if (x?.colorType === 'NONE' || x?.colorType === 'CYCLIC' || x?.colorType === 'DISTANCE' || x?.colorType === 'TARGET' || x?.colorType === 'TARGETG') out.colorType = x.colorType;
  if (Array.isArray(x?.targetColor) && x.targetColor.length === 3) out.targetColor = x.targetColor.map((v: unknown) => clamp01(num(v, 0))) as [number, number, number];
  if (num(x?.material, 0) !== 0) out.material = num(x.material, 0);
  if (num(x?.materialSpeed, 0) !== 0) out.materialSpeed = Math.min(1, Math.max(-1, num(x.materialSpeed, 0)));
  for (const k of ['yz', 'zx', 'yzPost', 'zxPost'] as const) {
    if (Array.isArray(x?.[k]) && x[k].length === 6) {
      const a = normAffine(x[k], IDENTITY);
      if (a.some((v, i) => v !== IDENTITY[i])) out[k] = a;
    }
  }
  const pre = normVarList(x?.preVariations);
  const post = normVarList(x?.postVariations);
  if (pre?.length) out.preVariations = pre;
  if (post?.length) out.postVariations = post;
  if (Array.isArray(x?.xaos)) {
    out.xaos = x.xaos.slice(0, MAX_XFORMS).map((v: unknown) => Math.max(0, num(v, 1)));
  }
  return out;
}

export function expandStops(stops: [number, number, number, number][]): RGB[] {
  const sorted = [...stops].sort((a, b) => a[0] - b[0]);
  const out: RGB[] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = sorted[0], hi = sorted[sorted.length - 1];
    for (let k = 0; k < sorted.length - 1; k++) {
      if (t >= sorted[k][0] && t <= sorted[k + 1][0]) { lo = sorted[k]; hi = sorted[k + 1]; break; }
    }
    const span = hi[0] - lo[0];
    const u = span > 1e-9 ? (t - lo[0]) / span : 0;
    out.push([
      lo[1] + (hi[1] - lo[1]) * u,
      lo[2] + (hi[2] - lo[2]) * u,
      lo[3] + (hi[3] - lo[3]) * u,
    ]);
  }
  return out;
}

/** Palette from an object that may carry paletteStops or a palette array. */
function normPalette(obj: any, fallback: RGB[]): RGB[] {
  if (Array.isArray(obj?.paletteStops) && obj.paletteStops.length >= 2) {
    return expandStops(obj.paletteStops.map((s: any) => [
      num(s?.[0], 0), clamp01(num(s?.[1], 0)), clamp01(num(s?.[2], 0)), clamp01(num(s?.[3], 0)),
    ]));
  }
  if (Array.isArray(obj?.palette) && obj.palette.length >= 2) {
    const src = obj.palette.map((c: any) => [
      clamp01(num(c?.[0], 0)), clamp01(num(c?.[1], 0)), clamp01(num(c?.[2], 0)),
    ] as RGB);
    const out: RGB[] = [];
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (src.length - 1);
      const k = Math.min(src.length - 2, Math.floor(t));
      const u = t - k;
      out.push([
        src[k][0] + (src[k + 1][0] - src[k][0]) * u,
        src[k][1] + (src[k + 1][1] - src[k][1]) * u,
        src[k][2] + (src[k + 1][2] - src[k][2]) * u,
      ]);
    }
    return out;
  }
  return fallback;
}

/** A final transform saved before colour type NONE existed carried "no recolouring" as colorSpeed 0 (the importer's
 *  mapping of JWildfire's final default): that is NONE. */
function normFinal(x: any): XForm {
  const out = normXForm(x);
  if (out.colorType === undefined && out.colorSpeed === 0) out.colorType = 'NONE';
  return out;
}
function normLayer(obj: any, fallbackPalette: RGB[]): Layer {
  const xforms = Array.isArray(obj?.xforms) && obj.xforms.length
    ? obj.xforms.slice(0, MAX_XFORMS).map(normXForm)
    : [defaultXForm()];
  return {
    xforms,
    final: obj?.final ? normFinal(obj.final) : null,
    moreFinals: Array.isArray(obj?.moreFinals) ? obj.moreFinals.map(normFinal) : [],
    palette: normPalette(obj, fallbackPalette),
    weight: Math.max(0, num(obj?.weight, 1)),
    visible: obj?.visible !== false,
    ...(obj?.smoothGradient ? { smoothGradient: true } : {}),
    ...(obj?.gradientMap && typeof obj.gradientMap.file === 'string' ? { gradientMap: { file: obj.gradientMap.file, hOffset: num(obj.gradientMap.hOffset, 0), hScale: num(obj.gradientMap.hScale, 1), vOffset: num(obj.gradientMap.vOffset, 0), vScale: num(obj.gradientMap.vScale, 1), lcolorAdd: num(obj.gradientMap.lcolorAdd, 0), lcolorScale: num(obj.gradientMap.lcolorScale, 0) } } : {}),
  };
}

/** Coerce any loosely-shaped flame object (AI / file import / legacy flat JSON)
 *  into a valid layered Flame. */
export function normalizeFlame(obj: any, fallbackPalette: RGB[]): Flame {
  let layers: Layer[];
  if (Array.isArray(obj?.layers) && obj.layers.length) {
    layers = obj.layers.slice(0, MAX_LAYERS).map((l: any) => normLayer(l, fallbackPalette));
  } else {
    layers = [normLayer(obj, fallbackPalette)]; // flat legacy shape
  }
  const bg = Array.isArray(obj?.background)
    ? obj.background.map((v: unknown) => clamp01(num(v, 0))) as RGB
    : [0, 0, 0] as RGB;
  return {
    name: typeof obj?.name === 'string' ? obj.name : 'untitled',
    layers,
    centerX: num(obj?.centerX, 0),
    centerY: num(obj?.centerY, 0),
    zoom: Math.max(0.01, num(obj?.zoom, 1)),
    rotation: num(obj?.rotation, 0),
    camPitch: num(obj?.camPitch, 0), camYaw: num(obj?.camYaw, 0), camBank: num(obj?.camBank, 0), camPersp: num(obj?.camPersp, 0),
    camPosX: num(obj?.camPosX, 0), camPosY: num(obj?.camPosY, 0), camPosZ: num(obj?.camPosZ, 0),
    preserveZ: !!obj?.preserveZ,
    camDOF: Math.max(0, num(obj?.camDOF, 0)), camDOFArea: Math.max(0, num(obj?.camDOFArea, 0.5)),
    camDOFExponent: Math.max(0.1, num(obj?.camDOFExponent, 2)), camDOFScale: num(obj?.camDOFScale, 1),
    camDOFShape: typeof obj?.camDOFShape === 'string' ? obj.camDOFShape : 'BUBBLE', camDOFRotate: num(obj?.camDOFRotate, 0),
    mixer: normMixer(obj?.mixer),
    bgImage: typeof obj?.bgImage === 'string' && obj.bgImage ? obj.bgImage : undefined,
    camDOFParams: Array.isArray(obj?.camDOFParams) ? Array.from({ length: 6 }, (_, i) => num(obj.camDOFParams[i], 0)) : [0, 0, 0, 0, 0, 0],
    camDOFFade: clamp01(num(obj?.camDOFFade, 1)), newDOF: !!obj?.newDOF,
    focusX: num(obj?.focusX, 0), focusY: num(obj?.focusY, 0), focusZ: num(obj?.focusZ, 0), camZ: num(obj?.camZ, 0),
    dimishZ: Math.max(0, num(obj?.dimishZ, 0)), dimZDist: num(obj?.dimZDist, 0),
    dimZColor: (Array.isArray(obj?.dimZColor) && obj.dimZColor.length === 3 ? obj.dimZColor.map((v: unknown) => clamp01(num(v, 0))) : [0, 0, 0]) as RGB,
    brightness: Math.max(0.05, num(obj?.brightness, 4)),
    gamma: num(obj?.gamma, 4) === 0 ? 0 : Math.max(0.5, num(obj?.gamma, 4)), // 0 = JWildfire flat tonemap
    gammaThreshold: Math.min(0.5, Math.max(0, num(obj?.gammaThreshold, 0.01))),
    vibrancy: clamp01(num(obj?.vibrancy, 1)),
    background: bg,
    contrast: Math.max(0.05, num(obj?.contrast, 1)),
    whiteLevel: Math.max(1, num(obj?.whiteLevel, 220)),
    lowDensityBrightness: Math.max(0, num(obj?.lowDensityBrightness, 0.24)),
    filterRadius: Math.min(3, Math.max(0, num(obj?.filterRadius, 0))),
    filterKernel: normFilterKernel(obj?.filterKernel),
    antialiasAmount: clamp01(num(obj?.antialiasAmount, 0.25)),
    antialiasRadius: Math.max(0, num(obj?.antialiasRadius, 0.5)),
    deRadius: Math.min(2, Math.max(0, num(obj?.deRadius, 1))),
    deCurve: Math.min(1, Math.max(0.01, num(obj?.deCurve, 0.8))),
    // JWildfire clamps the saturation shift at −1 (fully desaturated); above 1 it saturates further.
    filterSharpness: num(obj?.filterSharpness, 4),
    filterLowDensity: num(obj?.filterLowDensity, 0.025),
    saturation: Math.max(0, num(obj?.saturation, 1)),
    fgOpacity: Math.max(0, num(obj?.fgOpacity, 1)),
    bgTransparency: !!obj?.bgTransparency,
    oversample: Math.min(3, Math.max(1, Math.round(num(obj?.oversample, 1)))),
    ...(obj?.solid && typeof obj.solid === 'object' ? { solid: normSolid(obj.solid) } : {}),
    ...(obj?.bgGradient && (obj.bgGradient.type === 'GRADIENT_2X2' || obj.bgGradient.type === 'GRADIENT_2X2_C') ? { bgGradient: { type: obj.bgGradient.type, ul: rgb(obj.bgGradient.ul, [0, 0, 0]), ur: rgb(obj.bgGradient.ur, [0, 0, 0]), ll: rgb(obj.bgGradient.ll, [0, 0, 0]), lr: rgb(obj.bgGradient.lr, [0, 0, 0]), cc: rgb(obj.bgGradient.cc, [0, 0, 0]) } } : {}),
    ...(typeof obj?.author === 'string' && obj.author.trim() ? { author: obj.author.trim() } : {}),
    ...(typeof obj?.created === 'string' && obj.created.trim() ? { created: obj.created.trim() } : {}),
    ...(typeof obj?.uuid === 'string' && obj.uuid.trim() ? { uuid: obj.uuid.trim() } : {}),
    ...(obj?.motionBlur && num(obj.motionBlur.length, 0) > 0 ? { motionBlur: { length: Math.min(64, Math.max(1, Math.round(num(obj.motionBlur.length, 0)))), timeStep: num(obj.motionBlur.timeStep, 0.05), decay: num(obj.motionBlur.decay, 0.03) } } : {}),
    ...(obj?.postSymmetry && ['X_AXIS', 'Y_AXIS', 'POINT'].includes(obj.postSymmetry.type) ? { postSymmetry: {
      type: obj.postSymmetry.type as 'X_AXIS' | 'Y_AXIS' | 'POINT',
      order: Math.min(64, Math.max(1, Math.round(num(obj.postSymmetry.order, 3)))),
      centreX: num(obj.postSymmetry.centreX, 0), centreY: num(obj.postSymmetry.centreY, 0),
      distance: num(obj.postSymmetry.distance, 1.25), rotation: num(obj.postSymmetry.rotation, 6),
    } } : {}),
  };
}

const rgb = (v: unknown, d: RGB): RGB => (Array.isArray(v) && v.length === 3 ? v.map((c) => clamp01(num(c, 0))) as RGB : [...d] as RGB);

/** Coerce a loosely-shaped solid-render block (JSON import / AI) into a valid one. */
export function normSolid(s: any): SolidRender {
  const d = defaultSolidRender(!!s?.enabled);
  const lights: SolidLight[] = Array.isArray(s?.lights) ? s.lights.slice(0, 4).map((l: any, i: number) => {
    const dl = defaultSolidLight(i);
    return {
      altitude: num(l?.altitude, dl.altitude), azimuth: num(l?.azimuth, dl.azimuth), intensity: Math.max(0, num(l?.intensity, dl.intensity)),
      color: rgb(l?.color, dl.color), castShadows: l?.castShadows === undefined ? dl.castShadows : !!l.castShadows,
      shadowIntensity: clamp01(num(l?.shadowIntensity, dl.shadowIntensity)),
    };
  }) : d.lights;
  const materials: SolidMaterial[] = Array.isArray(s?.materials) ? s.materials.slice(0, 8).map((m: any) => {
    const dm = defaultSolidMaterial();
    return {
      diffuse: Math.max(0, num(m?.diffuse, dm.diffuse)), ambient: Math.max(0, num(m?.ambient, dm.ambient)),
      phong: Math.max(0, num(m?.phong, dm.phong)), phongSize: Math.max(0, num(m?.phongSize, dm.phongSize)),
      phongColor: rgb(m?.phongColor, dm.phongColor),
      diffFunc: LIGHT_DIFF_FUNCS.includes(m?.diffFunc) ? m.diffFunc : 'COSA',
      reflMapIntensity: Math.max(0, num(m?.reflMapIntensity, dm.reflMapIntensity)),
      ...(typeof m?.reflMapFilename === 'string' && m.reflMapFilename ? { reflMapFilename: m.reflMapFilename } : {}),
      reflMapping: m?.reflMapping === 'SPHERICAL' ? 'SPHERICAL' : 'BLINN_NEWELL',
    };
  }) : d.materials;
  const a = s?.ao ?? {}, sh = s?.shadows ?? {};
  return {
    enabled: !!s?.enabled,
    lights, materials,
    ao: {
      enabled: a.enabled === undefined ? d.ao.enabled : !!a.enabled, intensity: Math.max(0, num(a.intensity, d.ao.intensity)),
      searchRadius: Math.max(0, num(a.searchRadius, d.ao.searchRadius)), blurRadius: Math.max(0, num(a.blurRadius, d.ao.blurRadius)),
      radiusSamples: Math.max(1, Math.round(num(a.radiusSamples, d.ao.radiusSamples))), azimuthSamples: Math.max(1, Math.round(num(a.azimuthSamples, d.ao.azimuthSamples))),
      falloff: Math.max(0, num(a.falloff, d.ao.falloff)), affectDiffuse: Math.max(0, num(a.affectDiffuse, d.ao.affectDiffuse)),
    },
    shadows: {
      type: sh.type === 'FAST' || sh.type === 'SMOOTH' ? sh.type : 'OFF', smoothRadius: Math.max(0, num(sh.smoothRadius, d.shadows.smoothRadius)),
      mapSize: Math.max(64, Math.round(num(sh.mapSize, d.shadows.mapSize))), bias: num(sh.bias, d.shadows.bias),
    },
    postBokeh: {
      filterKernel: typeof s?.postBokeh?.filterKernel === 'string' ? s.postBokeh.filterKernel : d.postBokeh.filterKernel,
      intensity: Math.max(0, num(s?.postBokeh?.intensity, d.postBokeh.intensity)),
      brightness: Math.max(0, num(s?.postBokeh?.brightness, d.postBokeh.brightness)),
      size: Math.max(0, num(s?.postBokeh?.size, d.postBokeh.size)),
      activation: Math.max(0, num(s?.postBokeh?.activation, d.postBokeh.activation)),
    },
  };
}

export function flameToJSON(f: Flame): string {
  return JSON.stringify(f, null, 1);
}
