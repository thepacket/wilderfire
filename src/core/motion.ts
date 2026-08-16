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
      // Catmull-Rom on values with clamped end tangents (JWildfire's default feel)
      const p0 = P[Math.max(0, i - 1)].v, p1 = a.v, p2 = b.v, p3 = P[Math.min(n - 1, i + 2)].v;
      const u2 = u * u, u3 = u2 * u;
      return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u3);
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
  ['brightness', 'Brightness'], ['gamma', 'Gamma'], ['vibrancy', 'Vibrancy'],
];
const AFF = ['a', 'b', 'c', 'd', 'e', 'f'];

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
