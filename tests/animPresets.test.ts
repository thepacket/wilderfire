import { describe, it, expect } from 'vitest';
import { ANIM_PRESETS, mergeCurves, presetById } from '../src/core/animPresets';
import { evalCurve, getParam, applyCurves } from '../src/core/motion';
import { importFlameText } from '../src/core/flameXML';

const GREY = Array.from({ length: 256 }, (_, i) => [i / 255, i / 255, i / 255] as [number, number, number]);
const flame = importFlameText('<flame name="p" size="800 600" scale="150" brightness="4"><xform weight="0.6" color="0.2" linear="1" julian="0.4" julian_power="3" julian_dist="1" coefs="0.7 0.1 -0.1 0.7 0.2 0.1"/><xform weight="0.4" color="0.8" spherical="1" coefs="0.5 0 0 0.5 -0.3 0.1"/><finalxform linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY).flame;

describe('animation presets', () => {
  it('every preset writes curves on paths that exist in the flame, spanning the duration', () => {
    for (const p of ANIM_PRESETS) {
      const cs = p.make(flame, 6);
      expect(cs.length, p.id).toBeGreaterThan(0);
      for (const c of cs) {
        expect(getParam(flame, c.path), `${p.id}: ${c.path}`).toBeDefined();
        expect(c.points[0].t).toBe(0);
        expect(c.points[c.points.length - 1].t).toBeCloseTo(6, 9);
        for (const pt of c.points) expect(Number.isFinite(pt.v)).toBe(true);
      }
    }
  });
  it('cyclic presets end where they start; spin ends one turn on', () => {
    for (const p of ANIM_PRESETS) {
      for (const c of p.make(flame, 4)) {
        const first = c.points[0].v, last = c.points[c.points.length - 1].v;
        if (p.id === 'spin') expect(last - first).toBeCloseTo(Math.PI * 2, 9);
        else if (p.id === 'spin-ccw') expect(last - first).toBeCloseTo(-Math.PI * 2, 9);
        else if (p.id === 'orbit' && c.path === 'camYaw') expect(last - first).toBeCloseTo(360, 9);
        else expect(last).toBeCloseTo(first, 6);
      }
    }
  });
  it('twirl rotates each transform\'s linear part without changing its scale', () => {
    const cs = presetById('twirl')!.make(flame, 4);
    const at = (t: number) => applyCurves(flame, cs, t).layers[0].xforms[0].affine;
    const det = (a: number[]) => a[0] * a[4] - a[1] * a[3];
    const d0 = det(at(0));
    for (const t of [0.5, 1, 2.2, 3.7]) expect(Math.abs(det(at(t)) - d0)).toBeLessThan(1e-3 * Math.abs(d0)); // spline through 32 samples ≈ circle
    expect(at(2)[0]).toBeCloseTo(-at(0)[0], 3); // half a turn: the linear part is negated
  });
  it('julia sweep finds the julian and only it; breathe staggers the phases', () => {
    const js = presetById('julia-sweep')!.make(flame, 6);
    expect(js.map((c) => c.path)).toEqual(['layers.0.xforms.0.variations.1.params.dist']);
    expect(evalCurve(js[0], 1.5)!).toBeCloseTo(1.5, 2); // quarter way: +0.5
    const br = presetById('breathe')!.make(flame, 6);
    expect(br).toHaveLength(2);
    expect(evalCurve(br[0], 1.5)).not.toBeCloseTo(evalCurve(br[1], 1.5)! / (0.4 / 0.6), 2);
  });
  it('mergeCurves replaces curves on the same path and keeps the rest', () => {
    const a = [{ path: 'zoom', points: [{ t: 0, v: 1 }], interp: 'linear' as const }, { path: 'rotation', points: [{ t: 0, v: 0 }], interp: 'linear' as const }];
    const b = presetById('zoom-pulse')!.make(flame, 6);
    const m = mergeCurves(a, b);
    expect(m.map((c) => c.path).sort()).toEqual(['rotation', 'zoom']);
    expect(m.find((c) => c.path === 'zoom')!.points.length).toBeGreaterThan(1);
  });
});
