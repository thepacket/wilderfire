import { describe, it, expect } from 'vitest';
import { flameToXML, importFlameText, parseFlameXML, lastImportUnknown } from '../src/core/flameXML';
import { normalizeFlame, flameToJSON, type Flame } from '../src/core/flame';
import { PRESETS } from '../src/core/presets';
import { GREY, preset, fixture } from './helpers';
import type { MotionCurve } from '../src/core/motion';

const roundTrip = (f: Flame, curves: MotionCurve[] = []) => importFlameText(flameToXML(f, { curves }), GREY);

describe('.flame round-trip', () => {
  it('every preset survives XML export → import (structure, tone, camera, palette)', () => {
    for (const p of PRESETS) {
      const f = p.make();
      const { flame: back, unknown } = roundTrip(f);
      expect(unknown, p.name).toEqual([]);
      expect(back.layers.length).toBe(f.layers.length);
      back.layers.forEach((ly, li) => {
        expect(ly.xforms.length).toBe(f.layers[li].xforms.length);
        ly.xforms.forEach((x, xi) => {
          const o = f.layers[li].xforms[xi];
          expect(x.variations.map((v) => v.name)).toEqual(o.variations.map((v) => v.name));
          x.affine.forEach((v, i) => expect(v).toBeCloseTo(o.affine[i], 5));
          expect(x.weight).toBeCloseTo(o.weight, 5);
          expect(x.color).toBeCloseTo(o.color, 5);
          expect(x.colorSpeed).toBeCloseTo(o.colorSpeed, 5);
        });
        expect(!!ly.final).toBe(!!f.layers[li].final);
        ly.palette.forEach((c, i) => c.forEach((v, k) => expect(v).toBeCloseTo(f.layers[li].palette[i][k], 2)));
      });
      for (const k of ['brightness', 'gamma', 'gammaThreshold', 'vibrancy', 'zoom', 'centerX', 'centerY', 'rotation', 'contrast', 'whiteLevel', 'deRadius', 'deCurve', 'filterRadius'] as const) {
        expect(back[k], `${p.name}.${k}`).toBeCloseTo(f[k] as number, 4);
      }
    }
  });

  it('carries 3D camera, DOF, dimish-z, 3D affines and tonemap fields', () => {
    const f = preset('Ember');
    Object.assign(f, { camPitch: 30, camYaw: -15, camBank: 5, camPersp: 0.3, camPosZ: 0.2, preserveZ: true,
      camDOF: 0.4, camDOFArea: 0.3, newDOF: true, focusZ: 0.15, dimishZ: 2, dimZDist: 0.5, dimZColor: [0.5, 0, 0.25],
      contrast: 1.2, whiteLevel: 200, filterRadius: 0.75, filterKernel: 'gaussian', antialiasAmount: 0.1, deRadius: 0.5, deCurve: 0.6 });
    f.layers[0].xforms[0].yz = [0.9, 0.1, 0.05, -0.1, 0.9, 0];
    f.layers[0].xforms[1].zxPost = [1, 0.2, 0, 0, 1, 0.1];
    const { flame: b } = roundTrip(f);
    for (const k of ['camPitch', 'camYaw', 'camBank', 'camPersp', 'camPosZ', 'camDOF', 'camDOFArea', 'focusZ', 'dimishZ', 'dimZDist', 'contrast', 'whiteLevel', 'filterRadius', 'antialiasAmount', 'deRadius', 'deCurve'] as const) {
      expect(b[k], k).toBeCloseTo(f[k] as number, 4);
    }
    expect(b.preserveZ).toBe(true); expect(b.newDOF).toBe(true); expect(b.filterKernel).toBe('gaussian');
    expect(b.dimZColor.map((v) => +v.toFixed(3))).toEqual([0.5, 0, 0.25]);
    b.layers[0].xforms[0].yz!.forEach((v, i) => expect(v).toBeCloseTo(f.layers[0].xforms[0].yz![i], 5));
    b.layers[0].xforms[1].zxPost!.forEach((v, i) => expect(v).toBeCloseTo(f.layers[0].xforms[1].zxPost![i], 5));
  });

  it('writes and reads motion curves in JWildfire\'s *Curve_* format', () => {
    const f = preset('Clockwork');
    const curves: MotionCurve[] = [
      { path: 'camPitch', interp: 'spline', points: [{ t: 0, v: 0 }, { t: 2, v: 60 }] },
      { path: 'rotation', interp: 'linear', points: [{ t: 0, v: 0 }, { t: 4, v: Math.PI / 2 }] },
      { path: 'layers.0.xforms.0.weight', interp: 'linear', points: [{ t: 0, v: 1 }, { t: 1, v: 2 }] },
      { path: 'layers.0.xforms.0.variations.0.weight', interp: 'spline', points: [{ t: 0, v: 0.5 }, { t: 3, v: 1.5 }] },
      { path: 'layers.0.xforms.0.affine.2', interp: 'spline', points: [{ t: 0, v: 0 }, { t: 2, v: 0.5 }], enabled: false },
      { path: 'layers.0.xforms.0.colorSpeed', interp: 'spline', points: [{ t: 0, v: 0.2 }, { t: 2, v: 0.8 }] },
    ];
    const xml = flameToXML(f, { curves });
    expect(xml).toContain('camPitchCurve_point_count="2"');
    expect(xml).toContain('camPitchCurve_x1="50"'); // 2 s × 25 fps
    expect(xml).toContain('camRollCurve_y1="90"');   // radians → degrees
    expect(xml).toContain('xyCoeff20Curve_enabled="false"');
    expect(xml).toContain('_amountCurve_point_count');
    const { curves: back, unknown } = importFlameText(xml, GREY);
    expect(unknown).toEqual([]);
    const byPath = Object.fromEntries(back.map((c) => [c.path, c]));
    for (const c of curves) {
      const b = byPath[c.path];
      expect(b, c.path).toBeTruthy();
      expect(b.points.length).toBe(c.points.length);
      c.points.forEach((p, i) => { expect(b.points[i].t).toBeCloseTo(p.t, 6); expect(b.points[i].v).toBeCloseTo(p.v, 5); });
      expect(b.enabled === false).toBe(c.enabled === false);
    }
  });

  it('imports the JWildfire-written curve fixture with no unknown variations', () => {
    const { curves, unknown, flame } = importFlameText(fixture('curves.flame'), GREY);
    expect(unknown).toEqual([]);
    expect(flame.layers[0].xforms[0].variations[0].name).toBe('curl');
    expect(curves.map((c) => c.path).sort()).toEqual([
      'camPitch', 'layers.0.xforms.0.affine.2', 'layers.0.xforms.0.colorSpeed', 'layers.0.xforms.0.variations.0.weight', 'layers.0.xforms.0.weight', 'rotation',
    ]);
  });
});

describe('JWildfire fixtures', () => {
  it('3D / DOF / synth fixtures import with the expected settings', () => {
    const a = importFlameText(fixture('aff3d.flame'), GREY).flame;
    expect(a.layers[0].xforms[0].yz).toBeTruthy();
    expect(a.layers[0].xforms[1].zx).toBeTruthy();
    expect(a.camPitch).toBeCloseTo(40, 3);
    expect(a.dimishZ).toBeCloseTo(3, 6);
    const d = importFlameText(fixture('dof.flame'), GREY).flame;
    expect(d.camDOF).toBeCloseTo(0.6, 6); expect(d.newDOF).toBe(true); expect(d.focusZ).toBeCloseTo(0.2, 6);
    const s = importFlameText(fixture('synth.flame'), GREY).flame;
    expect(s.layers[0].xforms[0].variations[0].name).toBe('blur');
    expect(s.zoom).toBeCloseTo(1, 6);           // scale 128 on a 512 canvas
    expect(s.deRadius).toBe(0); expect(s.filterRadius).toBe(0); expect(s.antialiasAmount).toBe(0);
  });
  it('cam_zoom folds into zoom; cam_pitch/yaw/roll are radians; roll is bank', () => {
    const xml = `<flame size="1000 1000" scale="250" cam_zoom="2" cam_pitch="0.5" cam_yaw="0.25" cam_roll="0.1"><xform weight="1" coefs="1 0 0 1 0 0" linear="1"/></flame>`;
    const f = parseFlameXML(xml, GREY)[0];
    expect(f.zoom).toBeCloseTo(2, 6);
    expect(f.camPitch).toBeCloseTo((0.5 * 180) / Math.PI, 5);
    expect(f.camYaw).toBeCloseTo((0.25 * 180) / Math.PI, 5);
    expect(f.camBank).toBeCloseTo((0.1 * 180) / Math.PI, 5);
  });
  it('coefs use flam3 "a d b e c f" order and unknown variations are reported', () => {
    const xml = `<flame size="100 100"><xform weight="1" coefs="1 2 3 4 5 6" linear="1" not_a_variation="0.5"/></flame>`;
    const f = parseFlameXML(xml, GREY)[0];
    expect(f.layers[0].xforms[0].affine).toEqual([1, 3, 5, 2, 4, 6]);
    expect(lastImportUnknown).toContain('not_a_variation');
  });
  it('sample flames bundled in public/ all import cleanly', async () => {
    const { JWF_SAMPLES } = await import('../src/core/samples');
    const { readFileSync } = await import('node:fs');
    for (const s of JWF_SAMPLES) {
      const txt = readFileSync(`public/flames/${s.file}`, 'utf8');
      const { unknown, flame } = importFlameText(txt, GREY);
      expect(unknown, s.file).toEqual([]);
      expect(flame.layers[0].xforms.length).toBeGreaterThan(0);
    }
  });
});
