// Animation presets: one click writes motion curves that loop seamlessly over `duration` seconds —
// the same curves the Anim tab edits by hand, so a preset is a starting point, not a black box.
// Cyclic presets end exactly where they start; the video exporter's loop is therefore clean.
import type { Flame } from './flame';
import type { MotionCurve, CurvePoint } from './motion';
import { getParam } from './motion';

export interface AnimPreset {
  id: string;
  name: string;
  /** what it animates, for the tooltip */
  what: string;
  /** curves for this flame; empty when the flame has nothing the preset can move */
  make: (flame: Flame, duration: number) => MotionCurve[];
}

const TWO_PI = Math.PI * 2;
const samples = (duration: number, n: number, f: (phase: number) => number): CurvePoint[] =>
  Array.from({ length: n + 1 }, (_, i) => ({ t: (duration * i) / n, v: f(i / n) }));
/** a smooth cycle v0 → v0·(1 + amp·sin) → v0, sampled densely so the spline stays a sine */
const sine = (path: string, duration: number, v0: number, amp: number, phase = 0, n = 16): MotionCurve =>
  ({ path, points: samples(duration, n, (p) => v0 * (1 + amp * Math.sin(TWO_PI * (p + phase)))), interp: 'spline' });
const ramp = (path: string, duration: number, from: number, to: number): MotionCurve =>
  ({ path, points: [{ t: 0, v: from }, { t: duration, v: to }], interp: 'linear' });
const xformBases = (f: Flame): string[] => f.layers.flatMap((ly, li) => [
  ...ly.xforms.map((_, xi) => `layers.${li}.xforms.${xi}`),
  ...(ly.final ? [`layers.${li}.final`] : []),
]);
const num = (f: Flame, path: string, dflt: number) => { const v = getParam(f, path); return v !== undefined && Number.isFinite(v) ? v : dflt; };

export const ANIM_PRESETS: AnimPreset[] = [
  { id: 'spin', name: 'Spin', what: 'one full turn of the camera roll', make: (f, d) => [ramp('rotation', d, f.rotation, f.rotation + TWO_PI)] },
  { id: 'spin-ccw', name: 'Spin (counter-clockwise)', what: 'one full turn the other way', make: (f, d) => [ramp('rotation', d, f.rotation, f.rotation - TWO_PI)] },
  { id: 'zoom-pulse', name: 'Zoom pulse', what: 'zoom breathes in and out by 25 %', make: (f, d) => [sine('zoom', d, f.zoom, 0.25)] },
  { id: 'drift', name: 'Drift', what: 'the centre wanders on a small figure-eight', make: (f, d) => {
    const s = 0.12 / Math.max(f.zoom, 0.05);
    return [
      { path: 'centerX', points: samples(d, 24, (p) => f.centerX + s * Math.sin(TWO_PI * p)), interp: 'spline' },
      { path: 'centerY', points: samples(d, 24, (p) => f.centerY + s * 0.5 * Math.sin(2 * TWO_PI * p)), interp: 'spline' },
    ];
  } },
  { id: 'orbit', name: 'Orbit (3D)', what: 'the camera yaws once around the flame (pitch tilted a little if flat)', make: (f, d) => {
    const out: MotionCurve[] = [ramp('camYaw', d, f.camYaw, f.camYaw + 360)];
    if (Math.abs(f.camPitch) < 1e-6 && Math.abs(f.camPersp) < 1e-6) out.push({ path: 'camPitch', points: [{ t: 0, v: 35 }, { t: d, v: 35 }], interp: 'linear' });
    return out;
  } },
  { id: 'wobble', name: 'Wobble (3D)', what: 'pitch and yaw rock gently, out of phase', make: (f, d) => [
    { path: 'camPitch', points: samples(d, 16, (p) => f.camPitch + 12 * Math.sin(TWO_PI * p)), interp: 'spline' },
    { path: 'camYaw', points: samples(d, 16, (p) => f.camYaw + 12 * Math.sin(TWO_PI * p + Math.PI / 2)), interp: 'spline' },
  ] },
  { id: 'breathe', name: 'Breathe', what: 'every transform weight swells and relaxes, staggered', make: (f, d) => {
    const bases = f.layers.flatMap((ly, li) => ly.xforms.map((_, xi) => `layers.${li}.xforms.${xi}`));
    return bases.map((b, i) => sine(`${b}.weight`, d, Math.max(num(f, `${b}.weight`, 1), 1e-3), 0.3, i / Math.max(bases.length, 1)));
  } },
  { id: 'sway', name: 'Variation sway', what: 'the first variation of every transform sways by 25 %', make: (f, d) => {
    const out: MotionCurve[] = [];
    xformBases(f).forEach((b, i) => {
      const w = getParam(f, `${b}.variations.0.weight`);
      if (w !== undefined && Math.abs(w) > 1e-6) out.push(sine(`${b}.variations.0.weight`, d, w, 0.25, i * 0.17));
    });
    return out;
  } },
  { id: 'twirl', name: 'Twirl', what: 'each transform rotates once about its own origin (the picture keeps churning)', make: (f, d) => {
    const out: MotionCurve[] = [];
    for (const b of xformBases(f)) {
      const a = num(f, `${b}.affine.0`, 1), bb = num(f, `${b}.affine.1`, 0), dd = num(f, `${b}.affine.3`, 0), e = num(f, `${b}.affine.4`, 1);
      // rotate the linear part by θ: [a b; d e] · R(θ); 32 spline samples keep the path a circle (linear chords
      // would shrink the transform by ~1 % between samples)
      const n = 32;
      const rot = (idx: number, fn: (c: number, s: number) => number): MotionCurve =>
        ({ path: `${b}.affine.${idx}`, points: samples(d, n, (p) => fn(Math.cos(TWO_PI * p), Math.sin(TWO_PI * p))), interp: 'spline' });
      out.push(rot(0, (c, s) => a * c + bb * s), rot(1, (c, s) => -a * s + bb * c), rot(3, (c, s) => dd * c + e * s), rot(4, (c, s) => -dd * s + e * c));
    }
    return out;
  } },
  { id: 'julia-sweep', name: 'Julia sweep', what: 'julian / juliascope distance sweeps 0.5 → 1.5 and back', make: (f, d) => {
    const out: MotionCurve[] = [];
    f.layers.forEach((ly, li) => [...ly.xforms.map((x, xi) => [x, `layers.${li}.xforms.${xi}`] as const), ...(ly.final ? [[ly.final, `layers.${li}.final`] as const] : [])].forEach(([x, b]) => {
      x.variations.forEach((vi, vidx) => {
        if (vi.name === 'julian' || vi.name === 'juliascope' || vi.name === 'julia3D' || vi.name === 'julia3Dz') {
          const v0 = num(f, `${b}.variations.${vidx}.params.dist`, 1);
          out.push({ path: `${b}.variations.${vidx}.params.dist`, points: samples(d, 16, (p) => v0 + 0.5 * Math.sin(TWO_PI * p)), interp: 'spline' });
        }
      });
    }));
    return out;
  } },
  { id: 'fade', name: 'Fade in and out', what: 'brightness rises from dark and sinks back (a clean start and end for a clip)', make: (f, d) => [
    { path: 'brightness', points: [{ t: 0, v: 0.05 }, { t: d * 0.25, v: f.brightness }, { t: d * 0.75, v: f.brightness }, { t: d, v: 0.05 }], interp: 'smooth' },
  ] },
];

export const presetById = (id: string) => ANIM_PRESETS.find((p) => p.id === id);

/** Merge preset curves into an existing set: a path the preset animates is replaced, others are kept. */
export function mergeCurves(existing: MotionCurve[], added: MotionCurve[]): MotionCurve[] {
  const paths = new Set(added.map((c) => c.path));
  return [...existing.filter((c) => !paths.has(c.path)), ...added];
}
