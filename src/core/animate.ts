// Flame morphing: structure-merging interpolation between two flames, and a
// keyframe timeline. Between two fixed keyframes the interpolated flame keeps
// a constant structural signature (variation union), so the GPU kernel is
// compiled once per segment and playback stays hot.

import type { Flame, XForm, VarInstance, RGB, Affine, Layer } from './flame';
import { cloneFlame, cloneXForm, cloneLayer, IDENTITY } from './flame';
import { VARIATIONS, defaultParams } from './variations';

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Interpolate one column of the linear part in polar form, taking the
 *  shortest angular path — spins rotate instead of collapsing through zero. */
function lerpColumn(ax: number, ay: number, bx: number, by: number, t: number): [number, number] {
  const ma = Math.hypot(ax, ay), mb = Math.hypot(bx, by);
  if (ma < 1e-9 || mb < 1e-9) return [lerp(ax, bx, t), lerp(ay, by, t)];
  const aa = Math.atan2(ay, ax);
  let d = Math.atan2(by, bx) - aa;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  const ang = aa + d * t;
  const m = lerp(ma, mb, t);
  return [Math.cos(ang) * m, Math.sin(ang) * m];
}

function lerpAffine(a: Affine, b: Affine, t: number): Affine {
  const [c1x, c1y] = lerpColumn(a[0], a[3], b[0], b[3], t); // x column (a, d)
  const [c2x, c2y] = lerpColumn(a[1], a[4], b[1], b[4], t); // y column (b, e)
  return [c1x, c2x, lerp(a[2], b[2], t), c1y, c2y, lerp(a[5], b[5], t)];
}

/** A weight-0 stand-in used when one side has fewer xforms (fade in/out). */
function ghost(of: XForm): XForm {
  const g = cloneXForm(of);
  g.weight = 0;
  g.opacity = 0;
  return g;
}

/** Identity final used when only one side has a final transform. */
function identityFinal(like: XForm): XForm {
  return {
    affine: [...IDENTITY] as Affine,
    post: [...IDENTITY] as Affine,
    weight: 1,
    color: like.color,
    colorSpeed: 0,
    opacity: 1,
    variations: [{ name: 'linear', weight: 1, params: {} }],
  };
}

function lerpVariations(a: VarInstance[], b: VarInstance[], t: number): VarInstance[] {
  const names: string[] = [];
  for (const v of a) if (VARIATIONS[v.name] && !names.includes(v.name)) names.push(v.name);
  for (const v of b) if (VARIATIONS[v.name] && !names.includes(v.name)) names.push(v.name);
  if (!names.length) names.push('linear');
  return names.map((name) => {
    const va = a.find((v) => v.name === name);
    const vb = b.find((v) => v.name === name);
    const defs = defaultParams(name);
    const params: Record<string, number> = {};
    for (const k of Object.keys(defs)) {
      const pa = va?.params[k] ?? vb?.params[k] ?? defs[k];
      const pb = vb?.params[k] ?? va?.params[k] ?? defs[k];
      params[k] = lerp(pa, pb, t);
    }
    return { name, weight: lerp(va?.weight ?? 0, vb?.weight ?? 0, t), params };
  });
}

/** yz/zx (+post) planes: linear lerp, identity when absent; omitted from the result when both absent. */
function lerp3DAffines(a: XForm, b: XForm, t: number): Partial<Pick<XForm, 'yz' | 'zx' | 'yzPost' | 'zxPost'>> {
  const out: Partial<Pick<XForm, 'yz' | 'zx' | 'yzPost' | 'zxPost'>> = {};
  for (const k of ['yz', 'zx', 'yzPost', 'zxPost'] as const) {
    if (!a[k] && !b[k]) continue;
    const A = a[k] ?? IDENTITY, B = b[k] ?? IDENTITY;
    out[k] = A.map((v, i) => lerp(v, B[i], t)) as Affine;
  }
  return out;
}

function lerpXForm(a: XForm, b: XForm, t: number, n: number): XForm {
  const out: XForm = {
    affine: lerpAffine(a.affine, b.affine, t),
    post: lerpAffine(a.post, b.post, t),
    ...lerp3DAffines(a, b, t),
    weight: lerp(a.weight, b.weight, t),
    color: lerp(a.color, b.color, t),
    colorSpeed: lerp(a.colorSpeed, b.colorSpeed, t),
    opacity: lerp(a.opacity, b.opacity, t),
    variations: lerpVariations(a.variations, b.variations, t),
  };
  if (a.preVariations?.length || b.preVariations?.length) {
    out.preVariations = lerpVariations(a.preVariations ?? [], b.preVariations ?? [], t);
  }
  if (a.postVariations?.length || b.postVariations?.length) {
    out.postVariations = lerpVariations(a.postVariations ?? [], b.postVariations ?? [], t);
  }
  if (a.xaos || b.xaos) {
    out.xaos = Array.from({ length: n }, (_, j) =>
      lerp(a.xaos?.[j] ?? 1, b.xaos?.[j] ?? 1, t));
  }
  return out;
}

/** A weight-0 stand-in used when one side has fewer layers (fade in/out). */
function ghostLayer(of: Layer): Layer {
  const g = cloneLayer(of);
  g.weight = 0;
  return g;
}

function lerpLayer(a: Layer, b: Layer, t: number): Layer {
  const n = Math.max(a.xforms.length, b.xforms.length);
  const xforms = Array.from({ length: n }, (_, i) => {
    const xa = a.xforms[i] ?? ghost(b.xforms[i]);
    const xb = b.xforms[i] ?? ghost(a.xforms[i]);
    return lerpXForm(xa, xb, t, n);
  });
  let final: XForm | null = null;
  if (a.final || b.final) {
    final = lerpXForm(
      a.final ?? identityFinal(b.final!),
      b.final ?? identityFinal(a.final!),
      t, n,
    );
  }
  const palette = a.palette.map((c, i) => {
    const d = b.palette[i] ?? c;
    return [lerp(c[0], d[0], t), lerp(c[1], d[1], t), lerp(c[2], d[2], t)] as RGB;
  });
  return {
    xforms,
    final,
    palette,
    weight: lerp(a.weight, b.weight, t),
    visible: a.visible || b.visible,
  };
}

export function interpFlame(a: Flame, b: Flame, t: number): Flame {
  if (t <= 0) return cloneFlame(a);
  if (t >= 1) return cloneFlame(b);
  const nL = Math.max(a.layers.length, b.layers.length);
  const layers = Array.from({ length: nL }, (_, i) => {
    const la = a.layers[i] ?? ghostLayer(b.layers[i]);
    const lb = b.layers[i] ?? ghostLayer(a.layers[i]);
    return lerpLayer(la, lb, t);
  });
  return {
    name: a.name,
    layers,
    centerX: lerp(a.centerX, b.centerX, t),
    centerY: lerp(a.centerY, b.centerY, t),
    zoom: Math.exp(lerp(Math.log(Math.max(a.zoom, 1e-4)), Math.log(Math.max(b.zoom, 1e-4)), t)),
    rotation: lerp(a.rotation, b.rotation, t),
    camPitch: lerp(a.camPitch ?? 0, b.camPitch ?? 0, t), camYaw: lerp(a.camYaw ?? 0, b.camYaw ?? 0, t),
    camBank: lerp(a.camBank ?? 0, b.camBank ?? 0, t), camPersp: lerp(a.camPersp ?? 0, b.camPersp ?? 0, t), camPosX: lerp(a.camPosX ?? 0, b.camPosX ?? 0, t),
    camPosY: lerp(a.camPosY ?? 0, b.camPosY ?? 0, t), camPosZ: lerp(a.camPosZ ?? 0, b.camPosZ ?? 0, t),
    preserveZ: t < 0.5 ? a.preserveZ : b.preserveZ,
    camDOF: lerp(a.camDOF ?? 0, b.camDOF ?? 0, t), camDOFArea: lerp(a.camDOFArea ?? 0.5, b.camDOFArea ?? 0.5, t),
    camDOFExponent: lerp(a.camDOFExponent ?? 2, b.camDOFExponent ?? 2, t), camDOFScale: lerp(a.camDOFScale ?? 1, b.camDOFScale ?? 1, t),
    camDOFFade: lerp(a.camDOFFade ?? 1, b.camDOFFade ?? 1, t), newDOF: t < 0.5 ? !!a.newDOF : !!b.newDOF,
    focusX: lerp(a.focusX ?? 0, b.focusX ?? 0, t), focusY: lerp(a.focusY ?? 0, b.focusY ?? 0, t), focusZ: lerp(a.focusZ ?? 0, b.focusZ ?? 0, t),
    camZ: lerp(a.camZ ?? 0, b.camZ ?? 0, t),
    dimishZ: lerp(a.dimishZ ?? 0, b.dimishZ ?? 0, t), dimZDist: lerp(a.dimZDist ?? 0, b.dimZDist ?? 0, t),
    dimZColor: [0, 1, 2].map((i) => lerp(a.dimZColor?.[i] ?? 0, b.dimZColor?.[i] ?? 0, t)) as RGB,
    brightness: lerp(a.brightness, b.brightness, t),
    gamma: lerp(a.gamma, b.gamma, t),
    gammaThreshold: lerp(a.gammaThreshold ?? 0.04, b.gammaThreshold ?? 0.04, t),
    vibrancy: lerp(a.vibrancy, b.vibrancy, t),
    background: [
      lerp(a.background[0], b.background[0], t),
      lerp(a.background[1], b.background[1], t),
      lerp(a.background[2], b.background[2], t),
    ] as RGB,
  };
}

export type Easing = 'linear' | 'smooth' | 'in' | 'out';

export interface Keyframe {
  time: number; // seconds on the timeline
  flame: Flame;
  /** Easing of the segment leaving this keyframe; undefined = timeline default. */
  ease?: Easing;
}

export function applyEase(u: number, e: Easing): number {
  switch (e) {
    case 'smooth': return u * u * (3 - 2 * u);
    case 'in': return u * u;
    case 'out': return 1 - (1 - u) * (1 - u);
    default: return u;
  }
}

export function sortKeys(keys: Keyframe[]): Keyframe[] {
  return [...keys].sort((a, b) => a.time - b.time);
}

/** Evaluate the timeline at time t (keys must be sorted, non-empty). */
export function flameAt(keys: Keyframe[], t: number, easing: Easing = 'linear'): Flame {
  if (keys.length === 1 || t <= keys[0].time) return cloneFlame(keys[0].flame);
  const last = keys[keys.length - 1];
  if (t >= last.time) return cloneFlame(last.flame);
  for (let i = 0; i < keys.length - 1; i++) {
    if (t >= keys[i].time && t <= keys[i + 1].time) {
      const span = keys[i + 1].time - keys[i].time;
      const u = span > 1e-9 ? (t - keys[i].time) / span : 1;
      return interpFlame(keys[i].flame, keys[i + 1].flame, applyEase(u, keys[i].ease ?? easing));
    }
  }
  return cloneFlame(last.flame);
}
