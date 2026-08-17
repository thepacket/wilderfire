import { describe, it, expect } from 'vitest';
import { evalCurve, setPoint, getParam, setParam, applyCurves, curvesEnd, paramPaths, type MotionCurve } from '../src/core/motion';
import { preset } from './helpers';

const curve = (points: [number, number][], interp: MotionCurve['interp'] = 'linear'): MotionCurve =>
  ({ path: 'brightness', interp, points: points.map(([t, v]) => ({ t, v })) });

describe('evalCurve', () => {
  it('clamps at the ends and interpolates linearly', () => {
    const c = curve([[0, 0], [2, 10]]);
    expect(evalCurve(c, -1)).toBe(0);
    expect(evalCurve(c, 3)).toBe(10);
    expect(evalCurve(c, 1)).toBeCloseTo(5);
  });
  it('step holds the previous value', () => {
    const c = curve([[0, 1], [1, 2], [2, 3]], 'step');
    expect(evalCurve(c, 0.99)).toBe(1);
    expect(evalCurve(c, 1)).toBe(1); // t <= P[i+1].t picks segment [0,1]
    expect(evalCurve(c, 1.5)).toBe(2);
  });
  it('smooth is symmetric and passes through the keys', () => {
    const c = curve([[0, 0], [1, 1]], 'smooth');
    expect(evalCurve(c, 0.5)).toBeCloseTo(0.5);
    expect(evalCurve(c, 0.25)! + evalCurve(c, 0.75)!).toBeCloseTo(1);
  });
  it('spline matches JWildfire Envelope.SPLINE (2 points → linear; ≥3 parametric Catmull-Rom)', () => {
    // JWildfire evaluated curl amount (0.5 @0s, 1.5 @3s, SPLINE) at t = 1 s as 0.8333…
    expect(evalCurve(curve([[0, 0.5], [3, 1.5]], 'spline'), 1)).toBeCloseTo(0.83333, 4);
    // SplineCheck.java: keys (0,0) (1,1) (3,0) at 25 fps, evaluated at frames 10/40/50/60
    const c = curve([[0, 0], [1, 1], [3, 0]], 'spline');
    expect(evalCurve(c, 0.4)).toBeCloseTo(0.59662, 3);
    expect(evalCurve(c, 1.6)).toBeCloseTo(0.81060, 3);
    expect(evalCurve(c, 2.0)).toBeCloseTo(0.59841, 3);
    expect(evalCurve(c, 2.4)).toBeCloseTo(0.36064, 3);
    // keys (0,0) (1,1) (2,0.5) (4,2) at frames 12/37/75
    const d = curve([[0, 0], [1, 1], [2, 0.5], [4, 2]], 'spline');
    expect(evalCurve(d, 0.48)).toBeCloseTo(0.58037, 3);
    expect(evalCurve(d, 1.48)).toBeCloseTo(0.67405, 3);
    expect(evalCurve(d, 3.0)).toBeCloseTo(1.16799, 3);
  });
  it('empty / single point', () => {
    expect(evalCurve(curve([]), 1)).toBeUndefined();
    expect(evalCurve(curve([[5, 7]]), 0)).toBe(7);
  });
});

describe('setPoint / curvesEnd', () => {
  it('replaces within 1 ms, otherwise inserts sorted', () => {
    const c = curve([[1, 1]]);
    setPoint(c, 0, 0); setPoint(c, 1.0004, 5);
    expect(c.points.map((p) => [p.t, p.v])).toEqual([[0, 0], [1, 5]]);
    expect(curvesEnd([c])).toBe(1);
  });
});

describe('param paths', () => {
  it('get/set numeric leaves by dotted path, refuse non-numeric', () => {
    const f = preset('Clockwork');
    expect(getParam(f, 'layers.0.xforms.0.weight')).toBe(f.layers[0].xforms[0].weight);
    expect(setParam(f, 'layers.0.xforms.0.affine.2', 0.25)).toBe(true);
    expect(f.layers[0].xforms[0].affine[2]).toBe(0.25);
    expect(setParam(f, 'layers.0.xforms.0.variations', 1)).toBe(false);
    expect(setParam(f, 'nope.deep', 1)).toBe(false);
  });
  it('applyCurves leaves the base untouched and skips disabled curves', () => {
    const f = preset('Clockwork');
    const b0 = f.brightness;
    const out = applyCurves(f, [
      { path: 'brightness', interp: 'linear', points: [{ t: 0, v: 1 }, { t: 2, v: 3 }] },
      { path: 'gamma', interp: 'linear', points: [{ t: 0, v: 9 }], enabled: false },
    ], 1);
    expect(out.brightness).toBeCloseTo(2);
    expect(out.gamma).toBe(f.gamma);
    expect(f.brightness).toBe(b0);
  });
  it('every catalogued path resolves to a number', () => {
    const f = preset('Clockwork');
    for (const p of paramPaths(f)) expect(getParam(f, p.path), p.path).toBeTypeOf('number');
  });
});
