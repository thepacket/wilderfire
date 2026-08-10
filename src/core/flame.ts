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
}

export interface XForm {
  affine: Affine;
  post: Affine;
  weight: number;      // relative selection probability
  color: number;       // palette index 0..1
  colorSpeed: number;  // blend speed toward this xform's color
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
  palette: RGB[]; // 256 entries
  weight: number;   // density share relative to other layers
  visible: boolean;
}

export interface Flame {
  name: string;
  layers: Layer[];
  centerX: number;
  centerY: number;
  zoom: number;      // 1 = world range ~[-2,2] fits the short canvas axis
  rotation: number;  // radians
  brightness: number;
  gamma: number;
  vibrancy: number;
  background: RGB;
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
    brightness: 1.4,
    gamma: 2.4,
    vibrancy: 1,
    background: [0, 0, 0],
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
  const names = (l?: VarInstance[]) => (l ?? []).map((v) => v.name).join(',');
  const sig = (x: XForm) => `${names(x.preVariations)}<${names(x.variations)}>${names(x.postVariations)}`;
  return visibleLayers(f)
    .map((l) => l.xforms.map(sig).join('|') + '#' + (l.final ? sig(l.final) : '-'))
    .join('@@');
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
    opacity: clamp01(num(x?.opacity, 1)),
    variations: vars.length ? vars : d.variations,
  };
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
    brightness: Math.max(0.05, num(obj?.brightness, 1.4)),
    gamma: Math.max(0.5, num(obj?.gamma, 2.4)),
    vibrancy: clamp01(num(obj?.vibrancy, 1)),
    background: bg,
  };
}

export function flameToJSON(f: Flame): string {
  return JSON.stringify(f, null, 1);
}
