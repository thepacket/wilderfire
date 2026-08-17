// Flame data model — the CPU-side description of a fractal flame,
// closely following flam3 / JWildfire conventions. A flame is a stack of
// layers (JWildfire-style); each layer has its own transforms and gradient,
// all accumulating into the shared histogram. Camera and tone are global.

export type Affine = [number, number, number, number, number, number]; // x' = a x + b y + c ; y' = d x + e y + f
export type RGB = [number, number, number]; // 0..1

export interface VarInstance {
  name: string;
  weight: number;
  params: Record<string, number>;
  /** JWildfire per-instance priority override (`<var>_fx_priority`): a normal variation forced to run as a
   *  pre step (-1: input ← input + w·f(input)) or a post step (1: output ← output + w·f(output)). Absent = the definition's priority. */
  priority?: number;
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
}

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
  filterKernel: 'mitchell' | 'gaussian';
  /** JWildfire antialiasing: a fraction of samples get a random sub-pixel-ish jitter. */
  antialiasAmount: number;
  antialiasRadius: number;
  /** JWildfire density estimation (DeCalculator): estimator radius = deRadius·9 px
   *  (0 = off), deCurve = acceptance falloff with distance (0.8 default). */
  deRadius: number;
  deCurve: number;
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
    camDOF: 0, camDOFArea: 0.5, camDOFExponent: 2, camDOFScale: 1, camDOFFade: 1, newDOF: false,
    focusX: 0, focusY: 0, focusZ: 0, camZ: 0,
    dimishZ: 0, dimZDist: 0, dimZColor: [0, 0, 0],
    brightness: 4,
    gamma: 4,
    gammaThreshold: 0.01,
    vibrancy: 1,
    background: [0, 0, 0],
    contrast: 1, whiteLevel: 220, lowDensityBrightness: 0.24,
    filterRadius: 0, filterKernel: 'mitchell',
    antialiasAmount: 0.25, antialiasRadius: 0.5,
    deRadius: 1, deCurve: 0.8,
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

/** Structural signature — when this changes, the WGSL shader must be regenerated. */
export function flameSignature(f: Flame): string {
  const names = (l?: VarInstance[]) => (l ?? []).map((v) => v.name + (v.priority !== undefined ? '@' + v.priority : '')).join(',');
  const sig = (x: XForm) => `${names(x.preVariations)}<${names(x.variations)}>${names(x.postVariations)}`;
  return visibleLayers(f)
    .map((l) => l.xforms.map(sig).join('|') + '#' + [l.final, ...l.moreFinals].map((x) => (x ? sig(x) : '-')).join('#'))
    .join('@@') + (visibleLayers(f).some((l) => [...l.xforms, l.final, ...l.moreFinals].some((x) => x?.colorMods?.some((v) => v !== 0))) ? '~mods' : '');
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
    opacity: clamp01(num(x?.opacity, 1)),
    variations: vars.length ? vars : d.variations,
  };
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

function normLayer(obj: any, fallbackPalette: RGB[]): Layer {
  const xforms = Array.isArray(obj?.xforms) && obj.xforms.length
    ? obj.xforms.slice(0, MAX_XFORMS).map(normXForm)
    : [defaultXForm()];
  return {
    xforms,
    final: obj?.final ? normXForm(obj.final) : null,
    moreFinals: Array.isArray(obj?.moreFinals) ? obj.moreFinals.map(normXForm) : [],
    palette: normPalette(obj, fallbackPalette),
    weight: Math.max(0, num(obj?.weight, 1)),
    visible: obj?.visible !== false,
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
    camDOFFade: clamp01(num(obj?.camDOFFade, 1)), newDOF: !!obj?.newDOF,
    focusX: num(obj?.focusX, 0), focusY: num(obj?.focusY, 0), focusZ: num(obj?.focusZ, 0), camZ: num(obj?.camZ, 0),
    dimishZ: Math.max(0, num(obj?.dimishZ, 0)), dimZDist: num(obj?.dimZDist, 0),
    dimZColor: (Array.isArray(obj?.dimZColor) && obj.dimZColor.length === 3 ? obj.dimZColor.map((v: unknown) => clamp01(num(v, 0))) : [0, 0, 0]) as RGB,
    brightness: Math.max(0.05, num(obj?.brightness, 4)),
    gamma: Math.max(0.5, num(obj?.gamma, 4)),
    gammaThreshold: Math.min(0.5, Math.max(0, num(obj?.gammaThreshold, 0.01))),
    vibrancy: clamp01(num(obj?.vibrancy, 1)),
    background: bg,
    contrast: Math.max(0.05, num(obj?.contrast, 1)),
    whiteLevel: Math.max(1, num(obj?.whiteLevel, 220)),
    lowDensityBrightness: Math.max(0, num(obj?.lowDensityBrightness, 0.24)),
    filterRadius: Math.min(3, Math.max(0, num(obj?.filterRadius, 0))),
    filterKernel: obj?.filterKernel === 'gaussian' ? 'gaussian' : 'mitchell',
    antialiasAmount: clamp01(num(obj?.antialiasAmount, 0.25)),
    antialiasRadius: Math.max(0, num(obj?.antialiasRadius, 0.5)),
    deRadius: Math.min(2, Math.max(0, num(obj?.deRadius, 1))),
    deCurve: Math.min(1, Math.max(0.01, num(obj?.deCurve, 0.8))),
  };
}

export function flameToJSON(f: Flame): string {
  return JSON.stringify(f, null, 1);
}
