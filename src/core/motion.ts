// Motion curves: per-parameter animation on top of the keyframe timeline.
// A curve targets one numeric parameter of the flame by path (see PARAM_PATHS
// / paramPaths) and holds (time, value) points; between points the value is
// interpolated (linear / smooth / Catmull-Rom spline / step). Curves only touch
// numbers, so the compiled kernel signature never changes while they animate.

import type { Flame } from './flame';
import { cloneFlame } from './flame';
import { VARIATIONS } from './variations';

export type CurveInterp = 'linear' | 'smooth' | 'spline' | 'step';
export interface CurvePoint { t: number; v: number }
export interface MotionCurve {
  /** dotted path into the Flame, e.g. "camPitch", "layers.0.xforms.1.weight",
   *  "layers.0.xforms.1.variations.0.params.power", "layers.0.final.affine.2" */
  path: string;
  points: CurvePoint[];
  interp: CurveInterp;
  enabled?: boolean;
}

export const INTERPS: CurveInterp[] = ['spline', 'linear', 'smooth', 'step'];

/** Value of a curve at time t (points sorted by t; ends clamp). */
export function evalCurve(c: MotionCurve, t: number): number | undefined {
  const P = c.points;
  if (!P.length) return undefined;
  if (P.length === 1 || t <= P[0].t) return P[0].v;
  const n = P.length;
  if (t >= P[n - 1].t) return P[n - 1].v;
  let i = 0;
  while (i < n - 2 && t > P[i + 1].t) i++;
  const a = P[i], b = P[i + 1];
  const span = b.t - a.t;
  const u = span > 1e-12 ? (t - a.t) / span : 1;
  switch (c.interp) {
    case 'step': return a.v;
    case 'smooth': { const s = u * u * (3 - 2 * u); return a.v + (b.v - a.v) * s; }
    case 'spline': {
      // JWildfire Envelope SPLINE: with 2 points it is linear; with ≥ 3 a
      // Catmull-Rom (B = 0.5) applied *parametrically* to both time and value
      // (end points duplicated), then evaluated at the parameter where the time
      // spline equals t — so uneven key spacing curves the time axis as well.
      if (n < 3) return a.v + (b.v - a.v) * u;
      const cr = (xa: number, xb: number, xc: number, xd: number, s: number) => {
        const B = 0.5, s2 = s * s, s3 = s2 * s;
        return s3 * (-B * xa + (2 - B) * xb + (B - 2) * xc + B * xd)
          + s2 * (2 * B * xa + (B - 3) * xb + (3 - 2 * B) * xc - B * xd)
          + s * (-B * xa + B * xc) + xb;
      };
      const q0 = P[Math.max(0, i - 1)], q1 = a, q2 = b, q3 = P[Math.min(n - 1, i + 2)];
      // invert the (monotone for sorted keys) time spline by bisection
      let lo = 0, hi = 1;
      for (let k = 0; k < 40; k++) {
        const mid = (lo + hi) / 2;
        if (cr(q0.t, q1.t, q2.t, q3.t, mid) < t) lo = mid; else hi = mid;
      }
      return cr(q0.v, q1.v, q2.v, q3.v, (lo + hi) / 2);
    }
    default: return a.v + (b.v - a.v) * u;
  }
}

export function sortPoints(c: MotionCurve): void { c.points.sort((x, y) => x.t - y.t); }

/** Insert or replace the point at time t (within 1 ms). */
export function setPoint(c: MotionCurve, t: number, v: number): void {
  const i = c.points.findIndex((p) => Math.abs(p.t - t) < 1e-3);
  if (i >= 0) c.points[i].v = v; else c.points.push({ t, v });
  sortPoints(c);
}

// ---- path access ----

function walk(obj: any, path: string): { parent: any; key: string } | null {
  const parts = path.split('.');
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur == null) return null;
    cur = cur[parts[i]];
  }
  if (cur == null || typeof cur !== 'object') return null;
  return { parent: cur, key: parts[parts.length - 1] };
}
export function getParam(flame: Flame, path: string): number | undefined {
  const w = walk(flame, path);
  const v = w ? w.parent[w.key] : undefined;
  return typeof v === 'number' ? v : undefined;
}
export function setParam(flame: Flame, path: string, v: number): boolean {
  const w = walk(flame, path);
  if (!w || typeof w.parent[w.key] !== 'number') return false;
  w.parent[w.key] = v;
  return true;
}

/** Curves applied onto a copy of `base` at time t. */
export function applyCurves(base: Flame, curves: MotionCurve[], t: number): Flame {
  if (!curves.length) return base;
  const f = cloneFlame(base);
  for (const c of curves) {
    if (c.enabled === false) continue;
    const v = evalCurve(c, t);
    if (v !== undefined) setParam(f, c.path, v);
  }
  return f;
}

export function curvesEnd(curves: MotionCurve[]): number {
  let end = 0;
  for (const c of curves) for (const p of c.points) end = Math.max(end, p.t);
  return end;
}

// ---- parameter catalogue for the picker ----

export interface ParamEntry { path: string; label: string; group: string }

const FLAME_PARAMS: [string, string][] = [
  ['zoom', 'Zoom'], ['rotation', 'Rotation (rad)'], ['centerX', 'Center X'], ['centerY', 'Center Y'],
  ['camPitch', 'Pitch'], ['camYaw', 'Yaw'], ['camBank', 'Bank'], ['camPersp', 'Perspective'],
  ['camPosX', 'Cam X'], ['camPosY', 'Cam Y'], ['camPosZ', 'Cam Z'],
  ['camDOF', 'DOF amount'], ['camDOFArea', 'DOF area'], ['focusZ', 'Focus Z'], ['camZ', 'Focus plane (legacy)'],
  ['dimishZ', 'Dimish Z'], ['dimZDist', 'Dimish distance'],
  ['brightness', 'Brightness'], ['gamma', 'Gamma'], ['gammaThreshold', 'Gamma threshold'], ['vibrancy', 'Vibrancy'],
  ['contrast', 'Contrast'], ['whiteLevel', 'White level'], ['filterRadius', 'Filter radius'], ['deRadius', 'DE radius'],
];
const AFF = ['a', 'b', 'c', 'd', 'e', 'f'];
const AFF3 = ['yz', 'zx', 'yzPost', 'zxPost'] as const;

/** Every animatable numeric parameter of the flame, grouped for a picker. */
export function paramPaths(flame: Flame): ParamEntry[] {
  const out: ParamEntry[] = [];
  for (const [p, l] of FLAME_PARAMS) out.push({ path: p, label: l, group: 'Camera / tone' });
  flame.layers.forEach((ly, li) => {
    const L = flame.layers.length > 1 ? `L${li + 1} ` : '';
    if (flame.layers.length > 1) out.push({ path: `layers.${li}.weight`, label: `${L}layer weight`, group: `${L}Layer` });
    const xs: [string, string, typeof ly.xforms[number]][] = ly.xforms.map((x, xi) => [`layers.${li}.xforms.${xi}`, `${L}T${xi + 1}`, x]);
    if (ly.final) xs.push([`layers.${li}.final`, `${L}Final`, ly.final]);
    for (const [base, name, x] of xs) {
      const g = name;
      out.push({ path: `${base}.weight`, label: `${name} weight`, group: g });
      out.push({ path: `${base}.color`, label: `${name} color`, group: g });
      out.push({ path: `${base}.colorSpeed`, label: `${name} color speed`, group: g });
      out.push({ path: `${base}.opacity`, label: `${name} opacity`, group: g });
      AFF.forEach((k, i) => out.push({ path: `${base}.affine.${i}`, label: `${name} affine ${k}`, group: g }));
      AFF.forEach((k, i) => out.push({ path: `${base}.post.${i}`, label: `${name} post ${k}`, group: g }));
      for (const pl of AFF3) if (x[pl]) AFF.forEach((k, i) => out.push({ path: `${base}.${pl}.${i}`, label: `${name} ${pl} ${k}`, group: g }));
      const lists: [string, typeof x.variations][] = [['variations', x.variations]];
      if (x.preVariations?.length) lists.push(['preVariations', x.preVariations]);
      if (x.postVariations?.length) lists.push(['postVariations', x.postVariations]);
      for (const [lname, list] of lists) {
        list.forEach((vi, vidx) => {
          const vb = `${base}.${lname}.${vidx}`;
          const tag = lname === 'preVariations' ? 'pre ' : lname === 'postVariations' ? 'post ' : '';
          out.push({ path: `${vb}.weight`, label: `${name} ${tag}${vi.name} weight`, group: g });
          for (const pd of VARIATIONS[vi.name]?.params ?? []) {
            out.push({ path: `${vb}.params.${pd.name}`, label: `${name} ${tag}${vi.name} · ${pd.name}`, group: g });
          }
        });
      }
    }
  });
  return out;
}

export function paramLabel(flame: Flame, path: string): string {
  return paramPaths(flame).find((p) => p.path === path)?.label ?? path;
}

// ---- composition-layer parameters (escape-time / image layers, layer opacity, effects) ----
// A curve path "comp:<layerId>:<dotted path into the layer>" targets a composition layer instead of the flame,
// e.g. comp:L1abc:escape.zoom, comp:L1abc:escape.c.0, comp:L1abc:opacity, comp:L1abc:effects.blur.
import type { Composition } from './composition';

export const isCompPath = (path: string) => path.startsWith('comp:');
function compWalk(comp: Composition, path: string): { parent: any; key: string } | null {
  const m = /^comp:([^:]+):(.+)$/.exec(path);
  if (!m) return null;
  const layer = comp.layers.find((l) => l.id === m[1]);
  return layer ? walk(layer, m[2]) : null;
}
export function getCompParam(comp: Composition, path: string): number | undefined {
  const w = compWalk(comp, path);
  const v = w ? w.parent[w.key] : undefined;
  return typeof v === 'number' ? v : undefined;
}
export function setCompParam(comp: Composition, path: string, v: number): boolean {
  const w = compWalk(comp, path);
  if (!w || typeof w.parent[w.key] !== 'number') return false;
  w.parent[w.key] = v;
  return true;
}
/** Apply the composition-layer curves at time t onto `comp` IN PLACE (the layers are small; the flame layers are
 *  never touched — flame curves go through applyCurves). Returns comp. */
export function applyCompCurves(comp: Composition, curves: MotionCurve[], t: number): Composition {
  for (const c of curves) {
    if (c.enabled === false || !isCompPath(c.path)) continue;
    const v = evalCurve(c, t);
    if (v !== undefined) {
      setCompParam(comp, c.path, v);
      // a deep escape centre animated by curve: keep the exact-centre strings in step (f64 wins while animating)
      const m = /^comp:([^:]+):escape\.center[XY]$/.exec(c.path);
      if (m) { const l = comp.layers.find((x) => x.id === m[1]); if (l?.kind === 'escape') delete l.escape.centerHi; }
    }
  }
  return comp;
}
/** Animatable parameters of the non-flame layers (and every layer's opacity/effects), grouped per layer. */
export function compParamPaths(comp: Composition): ParamEntry[] {
  const out: ParamEntry[] = [];
  for (const l of comp.layers) {
    const g = `Layer · ${l.name}`;
    const P = (sub: string, label: string) => out.push({ path: `comp:${l.id}:${sub}`, label: `${l.name} ${label}`, group: g });
    P('opacity', 'opacity');
    if (l.effects) { P('effects.blur', 'blur'); P('effects.brightness', 'brightness'); P('effects.contrast', 'contrast'); P('effects.saturation', 'saturation'); P('effects.hue', 'hue'); P('effects.gamma', 'gamma'); }
    if (l.kind === 'escape') {
      for (const [k, lab] of [['escape.zoom', 'zoom'], ['escape.centerX', 'centre re'], ['escape.centerY', 'centre im'], ['escape.rotation', 'rotation'], ['escape.power', 'power'], ['escape.maxIter', 'max iterations'], ['escape.bailout', 'bailout'],
        ['escape.seed.0', 'seed re'], ['escape.seed.1', 'seed im'], ['escape.c.0', 'c re'], ['escape.c.1', 'c im'],
        ['escape.params.0.0', 'p1 re'], ['escape.params.0.1', 'p1 im'], ['escape.params.1.0', 'p2 re'], ['escape.params.1.1', 'p2 im'], ['escape.params.2.0', 'p3 re'], ['escape.params.2.1', 'p3 im'], ['escape.params.3.0', 'p4 re'], ['escape.params.3.1', 'p4 im'],
        ['escape.coloring.density', 'colour density'], ['escape.coloring.offset', 'colour offset'], ['escape.coloring.insideAlpha', 'inside alpha'], ['escape.coloring.outsideAlpha', 'outside alpha'],
        ['escape.coloring.trap.x', 'trap x'], ['escape.coloring.trap.y', 'trap y'], ['escape.coloring.trap.size', 'trap size']]) P(k, lab);
    } else if (l.kind === 'image') {
      for (const [k, lab] of [['image.scale', 'scale'], ['image.offsetX', 'offset x'], ['image.offsetY', 'offset y'], ['image.rotation', 'rotation']]) P(k, lab);
    }
  }
  return out;
}
