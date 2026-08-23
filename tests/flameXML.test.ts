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
      contrast: 1.2, whiteLevel: 200, filterRadius: 0.75, filterKernel: 'GAUSSIAN', antialiasAmount: 0.1, deRadius: 0.5, deCurve: 0.6 });
    f.layers[0].xforms[0].yz = [0.9, 0.1, 0.05, -0.1, 0.9, 0];
    f.layers[0].xforms[1].zxPost = [1, 0.2, 0, 0, 1, 0.1];
    const { flame: b } = roundTrip(f);
    for (const k of ['camPitch', 'camYaw', 'camBank', 'camPersp', 'camPosZ', 'camDOF', 'camDOFArea', 'focusZ', 'dimishZ', 'dimZDist', 'contrast', 'whiteLevel', 'filterRadius', 'antialiasAmount', 'deRadius', 'deCurve'] as const) {
      expect(b[k], k).toBeCloseTo(f[k] as number, 4);
    }
    expect(b.preserveZ).toBe(true); expect(b.newDOF).toBe(true); expect(b.filterKernel).toBe('GAUSSIAN');
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
  it('JWildfire colour/compositing attributes import and round-trip', () => {
    const xml = '<flame name="c" size="100 100" scale="25" saturation="1.35" fg_opacity="0.4" bg_transparency="1" oversample="2" filter_sharpness="3.5" filter_low_density="0.01"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>';
    const { flame } = importFlameText(xml, GREY);
    expect(flame.saturation).toBeCloseTo(1.35, 6);
    expect(flame.fgOpacity).toBeCloseTo(0.4, 6);
    expect(flame.bgTransparency).toBe(true);
    expect(flame.oversample).toBe(2);
    expect(flame.filterSharpness).toBeCloseTo(3.5, 6);
    expect(flame.filterLowDensity).toBeCloseTo(0.01, 6);
    const back = roundTrip(flame).flame;
    expect(back.saturation).toBeCloseTo(1.35, 5);
    expect(back.fgOpacity).toBeCloseTo(0.4, 5);
    expect(back.bgTransparency).toBe(true);
    expect(back.oversample).toBe(2);
    expect(back.filterSharpness).toBeCloseTo(3.5, 5);
  });
  it('defaults match JWildfire when the attributes are absent', () => {
    const { flame } = importFlameText('<flame name="d" size="100 100" scale="25"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY);
    expect(flame.saturation).toBe(1);
    expect(flame.fgOpacity).toBe(1);
    expect(flame.bgTransparency).toBe(false);
    expect(flame.oversample).toBe(1);
    expect(flame.filterSharpness).toBe(4);
    expect(flame.filterLowDensity).toBeCloseTo(0.025, 6);
    expect(flame.postSymmetry).toBeUndefined();
  });
  it('post symmetry imports every mode and round-trips (NONE stays absent)', () => {
    const psym = (t: string, extra = '') =>
      `<flame name="s" size="100 100" scale="25" post_symmetry_type="${t}" ${extra}><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>`;
    expect(importFlameText(psym('NONE'), GREY).flame.postSymmetry).toBeUndefined();
    const pt = importFlameText(psym('POINT', 'post_symmetry_order="5" post_symmetry_centre_x="0.25" post_symmetry_centre_y="-0.5"'), GREY).flame;
    expect(pt.postSymmetry).toEqual({ type: 'POINT', order: 5, centreX: 0.25, centreY: -0.5, distance: 1.25, rotation: 6 });
    const ax = importFlameText(psym('X_AXIS', 'post_symmetry_distance="0.8" post_symmetry_rotation="12.5"'), GREY).flame;
    expect(ax.postSymmetry?.type).toBe('X_AXIS');
    expect(ax.postSymmetry?.distance).toBeCloseTo(0.8, 6);
    expect(ax.postSymmetry?.rotation).toBeCloseTo(12.5, 6);
    expect(roundTrip(ax).flame.postSymmetry).toEqual(ax.postSymmetry);
    expect(roundTrip(importFlameText(psym('Y_AXIS'), GREY).flame).flame.postSymmetry?.type).toBe('Y_AXIS');
  });
  it('post symmetry changes the compile signature (the kernel bakes its constants)', async () => {
    const { flameSignature } = await import('../src/core/flame');
    const plain = importFlameText('<flame name="p" size="100 100" scale="25"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY).flame;
    const sym = importFlameText('<flame name="p" size="100 100" scale="25" post_symmetry_type="POINT" post_symmetry_order="4"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY).flame;
    expect(flameSignature(plain)).not.toBe(flameSignature(sym));
    const sym6 = JSON.parse(JSON.stringify(sym));
    sym6.postSymmetry.order = 6;
    expect(flameSignature(sym)).not.toBe(flameSignature(sym6));
  });
  it('a pack file returns every flame, first one as the active flame', () => {
    const one = (n: string) => `<flame name="${n}" size="100 100" scale="25"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>`;
    const { flame, flames, count } = importFlameText(`<flames>${one('a')}${one('b')}${one('c')}</flames>`, GREY);
    expect(count).toBe(3);
    expect(flames.map((f) => f.name)).toEqual(['a', 'b', 'c']);
    expect(flame).toBe(flames[0]);
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

describe('JWildfire solid rendering (sld_render_*)', () => {
  // attribute names as JWildfire writes them (incl. its typos ligtht/diif/mappping), values non-default throughout
  const jwf = `<flame name="solid" size="200 200" center="0 0" scale="50" sld_render_enabled="1" sld_render_ao_enabled="0" sld_render_ao_intensity="0.3" sld_render_ao_search_radius="5.0" sld_render_ao_blur_radius="2.5" sld_render_ao_radius_samples="4" sld_render_ao_azimuth_samples="9" sld_render_ao_falloff="0.7" sld_render_ao_affect_diffuse="0.2" sld_render_shadow_type="SMOOTH" sld_render_shadow_smooth_radius="1.5" sld_render_shadowmap_size="1024" sld_render_shadowmap_bias="0.02" sld_render_material_count="2" sld_render_material_diffuse0="0.4" sld_render_material_ambient0="0.6" sld_render_material_phong0="0.0794" sld_render_material_phong_size0="12.0" sld_render_material_phong_red0="1.0" sld_render_material_phong_green0="0.5" sld_render_material_phong_blue0="0.25" sld_render_material_refl_map_intensity0="0.5" sld_render_material_refl_map_filename0="" sld_render_material_refl_mappping0="BLINN_NEWELL" sld_render_material_light_diif_func0="COSA_HALVE" sld_render_material_diffuse1="0.1" sld_render_material_ambient1="0.8" sld_render_material_phong1="0.6" sld_render_material_phong_size1="15.0" sld_render_material_phong_red1="1.0" sld_render_material_phong_green1="1.0" sld_render_material_phong_blue1="1.0" sld_render_material_refl_map_intensity1="0.5" sld_render_material_refl_map_filename1="" sld_render_material_refl_mappping1="SPHERICAL" sld_render_material_light_diif_func1="COSA" sld_render_ligtht_count="2" sld_render_light_altitude0="-123.8096" sld_render_light_altitude0_enabled="false" sld_render_light_altitude0_point_count="1" sld_render_light_altitude0_x0="1" sld_render_light_altitude0_y0="-123.8096" sld_render_light_azimuth0="-85.7142" sld_render_light_intensity0="0.9" sld_render_light_shadow_intensity0="0.7" sld_render_light_red0="1.0" sld_render_light_green0="0.9" sld_render_light_blue0="0.8" sld_render_light_shadows0="1" sld_render_light_altitude1="64.0" sld_render_light_azimuth1="55.0" sld_render_light_intensity1="0.6" sld_render_light_shadow_intensity1="0.7" sld_render_light_red1="1.0" sld_render_light_green1="1.0" sld_render_light_blue1="1.0" sld_render_light_shadows1="0">
  <xform weight="0.5" color="0.0" material="0.3" material_speed="0.0" linear3D="1.0" coefs="1.0 0.0 0.0 1.0 0.0 0.0"/>
  <xform weight="0.5" color="1.0" spherical3D="1.0" coefs="0.5 0.0 0.0 0.5 0.5 0.0"/>
</flame>`;

  it('imports the JWildfire attribute set (lights, materials, AO, shadows, per-xform material)', () => {
    const { flame: f, unknown } = importFlameText(jwf, GREY);
    expect(unknown).toEqual([]);
    const s = f.solid!;
    expect(s.enabled).toBe(true);
    expect(s.ao).toEqual({ enabled: false, intensity: 0.3, searchRadius: 5, blurRadius: 2.5, radiusSamples: 4, azimuthSamples: 9, falloff: 0.7, affectDiffuse: 0.2 });
    expect(s.shadows).toEqual({ type: 'SMOOTH', smoothRadius: 1.5, mapSize: 1024, bias: 0.02 });
    expect(s.materials.length).toBe(2);
    expect(s.materials[0]).toMatchObject({ diffuse: 0.4, ambient: 0.6, phong: 0.0794, phongSize: 12, phongColor: [1, 0.5, 0.25], diffFunc: 'COSA_HALVE', reflMapping: 'BLINN_NEWELL' });
    expect(s.materials[1]).toMatchObject({ diffuse: 0.1, ambient: 0.8, phong: 0.6, phongSize: 15, diffFunc: 'COSA', reflMapping: 'SPHERICAL' });
    expect(s.lights.length).toBe(2);
    expect(s.lights[0]).toMatchObject({ altitude: -123.8096, azimuth: -85.7142, intensity: 0.9, color: [1, 0.9, 0.8], castShadows: true, shadowIntensity: 0.7 });
    expect(s.lights[1]).toMatchObject({ altitude: 64, azimuth: 55, intensity: 0.6, castShadows: false });
    expect(f.layers[0].xforms[0].material).toBe(0.3);
    expect(f.layers[0].xforms[1].material).toBeUndefined();
  });

  it('enabled without counts falls back to JWildfire\'s default lights/material; absent = off', () => {
    const { flame: f } = importFlameText('<flame name="s" size="10 10" scale="5" sld_render_enabled="1"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY);
    expect(f.solid?.enabled).toBe(true);
    expect(f.solid?.lights.map((l) => [l.altitude, l.azimuth, l.intensity, l.castShadows])).toEqual([[55, -22, 0.8, true], [64, 55, 0.6, false]]);
    expect(f.solid?.materials.length).toBe(1);
    expect(f.solid?.materials[0]).toMatchObject({ ambient: 0.5, diffuse: 0.7, phong: 0.6, phongSize: 24 });
    expect(f.solid?.ao.enabled).toBe(true);
    expect(f.solid?.shadows.type).toBe('OFF');
    const { flame: g } = importFlameText('<flame name="s" size="10 10" scale="5"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY);
    expect(g.solid).toBeUndefined();
  });

  it('round-trips through our writer and JSON', () => {
    const { flame: f } = importFlameText(jwf, GREY);
    const { flame: back, unknown } = roundTrip(f);
    expect(unknown).toEqual([]);
    expect(back.solid).toEqual(f.solid);
    expect(back.layers[0].xforms[0].material).toBe(0.3);
    expect(normalizeFlame(JSON.parse(flameToJSON(f)), GREY).solid).toEqual(f.solid);
    // solid off → nothing written
    const off = { ...f, solid: { ...f.solid!, enabled: false } };
    expect(flameToXML(off)).not.toContain('sld_render');
    expect(roundTrip(off).flame.solid).toBeUndefined();
  });
});

describe('zero-amount variations (JWildfire applies them)', () => {
  it('keeps pre/post/hide/dc/state variations with amount 0 and drops plain sums', () => {
    const xml = '<flame name="z" size="64 64" scale="10"><xform weight="1" linear="1" spherical="0.0" pre_stabilize="0.0" pre_stabilize_n="10" post_mirror_wf="0.0" post_mirror_wf_zaxis="1" coefs="1 0 0 1 0 0"/></flame>';
    const { flame } = importFlameText(xml, GREY);
    const names = flame.layers[0].xforms[0].variations.map((v) => `${v.name}:${v.weight}`);
    expect(names).toContain('pre_stabilize:0');
    expect(names).toContain('post_mirror_wf:0');
    expect(names).not.toContain('spherical:0');
    expect(names).toContain('linear:1');
  });
  it('imports brightness beyond the slider range (JWildfire files carry 50, 150)', () => {
    const { flame } = importFlameText('<flame name="b" size="64 64" scale="10" brightness="150"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY);
    expect(flame.brightness).toBe(150);
  });
  it('keeps gamma 0 (old JWildfire files: the flat tonemap, exponent 0) but rejects a negative gamma', () => {
    const xml = (g: string) => `<flame name="g" size="64 64" scale="10" gamma="${g}"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>`;
    expect(importFlameText(xml('0.0'), GREY).flame.gamma).toBe(0);
    expect(importFlameText(xml('-1'), GREY).flame.gamma).toBe(4);
    expect(normalizeFlame(JSON.parse(flameToJSON(importFlameText(xml('0.0'), GREY).flame)), GREY).gamma).toBe(0);
  });
});

describe('JWildfire background gradients', () => {
  it('imports GRADIENT_2X2_C corners + centre, treats an all-equal gradient as a single colour, round-trips', () => {
    const g = '<flame name="g" size="64 64" scale="10" background_type="GRADIENT_2X2_C" background_ul="0.0 0.2 0.4" background_ur="0.0 0.0 0.4" background_ll="0.6 0.0 0.0" background_lr="0.6 0.0 0.0" background_cc="0.0 0.0 0.0"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>';
    const { flame } = importFlameText(g, GREY);
    expect(flame.bgGradient).toEqual({ type: 'GRADIENT_2X2_C', ul: [0, 0.2, 0.4], ur: [0, 0, 0.4], ll: [0.6, 0, 0], lr: [0.6, 0, 0], cc: [0, 0, 0] });
    const back = roundTrip(flame).flame;
    expect(back.bgGradient).toEqual(flame.bgGradient);
    const flat = importFlameText('<flame name="f" size="64 64" scale="10" background_type="GRADIENT_2X2_C" background_ul="0.1 0.1 0.1" background_ur="0.1 0.1 0.1" background_ll="0.1 0.1 0.1" background_lr="0.1 0.1 0.1" background_cc="0.1 0.1 0.1"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY).flame;
    expect(flat.bgGradient).toBeUndefined();
    expect(flat.background[0]).toBeCloseTo(0.1, 6);
  });
});

describe('pack repairs (JWildfire\'s regex reader accepts what XML forbids)', () => {
  const one = (n: string) => `<flame name="${n}" size="64 64" scale="10"><xform weight="1" color="0.3" linear="1" coefs="1 0 0 1 0 0"/></flame>`;
  it('concatenated root <flame> elements without a <flames> wrapper', () => {
    expect(importFlameText(one('a') + '\n' + one('b') + '\n' + one('c'), GREY).count).toBe(3);
    expect(importFlameText('<?xml version="1.0"?>\n' + one('a') + one('b'), GREY).count).toBe(2);
  });
  it('a <flames> pack that was never closed, or cut off inside a flame, keeps the complete flames', () => {
    expect(importFlameText('<flames>\n' + one('a') + '\n' + one('b') + '\n', GREY).count).toBe(2);
    const cut = '<flames>\n' + one('a') + '\n' + one('b').slice(0, 40);
    const r = importFlameText(cut, GREY);
    expect(r.count).toBe(1); expect(r.flame.name).toBe('a');
  });
  it('repeated variation instances written as name#1# and attribute names with spaces', () => {
    const xml = '<flames><flame name="d" size="64 64" scale="10"><xform weight="1" color="0.3" linear="1" linear#1#="0.5" coefs="1 0 0 1 0 0"/></flame></flames>';
    const f = importFlameText(xml, GREY).flame;
    expect(f.layers[0].xforms[0].variations.map((v) => v.name + "=" + v.weight)).toEqual(["linear=1", "linear=0.5"]);
  });
});

describe('JWildfire colour types CYCLIC / DISTANCE', () => {
  it('imports, keeps the symmetry, exports, and the kernel carries the colour rules', async () => {
    const xml = '<flame name="c" size="64 64" scale="10"><xform weight="1" color="0.3" symmetry="-0.83" color_type="DISTANCE" linear="1" coefs="1 0 0 1 0 0"/><xform weight="1" color="0.5" symmetry="0.25" color_type="CYCLIC" spherical="1" coefs="1 0 0 1 0 0"/><xform weight="1" color="0.5" color_type="NONE" linear="1" coefs="1 0 0 1 0 0"/></flame>';
    const { flame } = importFlameText(xml, GREY);
    const [a, b, c] = flame.layers[0].xforms;
    expect(a.colorType).toBe('DISTANCE'); expect(a.colorSpeed).toBeCloseTo((1 + 0.83) / 2, 6);
    expect(b.colorType).toBe('CYCLIC'); expect(b.colorSpeed).toBeCloseTo((1 - 0.25) / 2, 6);
    expect(c.colorType).toBe('NONE'); expect(c.colorSpeed).toBe(0);
    const back = roundTrip(flame).flame;
    expect(back.layers[0].xforms.map((x) => x.colorType)).toEqual(['DISTANCE', 'CYCLIC', 'NONE']);
    const { compileFlame } = await import('../src/gpu/codegen');
    const compiled = compileFlame(flame, 1024);
    const w = compiled.wgsl;
    expect(w).toContain('c = fract(c + (1.0 - 2.0 * xd[');
    expect(w).toContain('length(np - p) * (2.0 - 2.0 * xd[');
    expect(w).toContain('rgbo.w > 0.25');
    // DISTANCE / NONE normal xforms: the point's RGB is carried across iterations (JWildfire XYZPoint.redColor)
    expect(compiled.usesCarry).toBe(true);
    expect(w).toContain('var<storage, read_write> crgb');
    expect(w).toContain('var rgbo = crgbv;');
    expect(w).toContain('crgb[idx] = crgbv;');
  });

  it('a pure DIFFUSION flame carries nothing; finals default to NONE and a recolouring final exports as DIFFUSION', async () => {
    const xml = '<flame name="d" size="64 64" scale="10"><xform weight="1" color="0.3" linear="1" coefs="1 0 0 1 0 0"/><finalxform color="0.5" symmetry="0.5" linear="1" coefs="1 0 0 1 0 0"/></flame>';
    const { flame } = importFlameText(xml, GREY);
    expect(flame.layers[0].final?.colorType).toBe('NONE');
    const { compileFlame } = await import('../src/gpu/codegen');
    const c1 = compileFlame(flame, 1024);
    expect(c1.usesCarry).toBe(false);
    expect(c1.wgsl).not.toContain('crgb');
    expect(c1.wgsl).toContain('var rgbo = vec4f(0.0);');
    // a legacy JSON final (colorSpeed 0, no colour type) normalises to NONE; speed > 0 means DIFFUSION
    const { normalizeFlame } = await import('../src/core/flame');
    const legacy = JSON.parse(JSON.stringify(flame)); delete legacy.layers[0].final.colorType;
    expect(normalizeFlame(legacy, GREY).layers[0].final?.colorType).toBe('NONE');
    legacy.layers[0].final.colorSpeed = 0.4;
    const n2 = normalizeFlame(legacy, GREY);
    expect(n2.layers[0].final?.colorType).toBeUndefined();
    expect(flameToXML(n2, { curves: [] })).toMatch(/<finalxform [^>]*color_type="DIFFUSION"/);
    expect(flameToXML(flame, { curves: [] })).toMatch(/<finalxform [^>]*color_type="NONE"/);
  });
});

describe('layer colouring options: smooth_gradient and gradient_map', () => {
  it('round-trip on a flat file and on <layer> elements; the kernel gets the smooth lookup', async () => {
    const xml = '<flame name="s" size="64 64" scale="10" smooth_gradient="1" gradient_map="C:\\\\maps\\\\tex.png" gradient_map_hoffset="0.1" gradient_map_hscale="2" gradient_map_voffset="-0.2" gradient_map_vscale="3" gradient_map_lcolor_add="0.5" gradient_map_lcolor_scale="0.25"><xform weight="1" color="0.3" linear="1" coefs="1 0 0 1 0 0"/></flame>';
    const f = importFlameText(xml, GREY).flame;
    const ly = f.layers[0];
    expect(ly.smoothGradient).toBe(true);
    expect(ly.gradientMap).toEqual({ file: 'tex.png', hOffset: 0.1, hScale: 2, vOffset: -0.2, vScale: 3, lcolorAdd: 0.5, lcolorScale: 0.25 });
    const back = roundTrip(f).flame;
    expect(back.layers[0].smoothGradient).toBe(true);
    expect(back.layers[0].gradientMap?.file).toBe('tex.png');
    expect(back.layers[0].gradientMap?.vScale).toBe(3);
    const layered = '<flame name="l" size="64 64" scale="10"><layer weight="1" smooth_gradient="1"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></layer><layer weight="1"><xform weight="1" spherical="1" coefs="1 0 0 1 0 0"/></layer></flame>';
    const g = importFlameText(layered, GREY).flame;
    expect(g.layers.map((l) => !!l.smoothGradient)).toEqual([true, false]);
    expect(roundTrip(g).flame.layers.map((l) => !!l.smoothGradient)).toEqual([true, false]);
    const { compileFlame } = await import('../src/gpu/codegen');
    expect(compileFlame(g, 1024).wgsl).toContain('mix(lc.xyz, rc.xyz, ci - floor(ci))');
    // the gradient map: a texture binding, the sampling function with the layer's constants, the step after the transform
    const cm = compileFlame(f, 1024);
    expect(cm.usesGmap).toBe(true);
    expect(cm.wgsl).toContain('@binding(15) var gmTex');
    expect(cm.wgsl).toContain('fn gmap0(p: vec3f, c: f32)');
    expect(cm.wgsl).toContain('* 2.0 + 0.1;'); // hscale, hoffset baked in
    expect(cm.wgsl).toContain('rgbo = vec4f(gmap0(np, c), 0.8);');
    expect(compileFlame(g, 1024).usesGmap).toBe(false);
    expect(normalizeFlame(JSON.parse(JSON.stringify(f)), GREY).layers[0].gradientMap?.lcolorAdd).toBe(0.5);
  });
});

describe('motion blur', () => {
  it('round-trips the attributes and lays out JWildfire\'s sub-frames and weights', async () => {
    const xml = '<flame name="m" size="64 64" scale="10" motion_blur_length="4" motion_blur_timestep="0.1" motion_blur_decay="0.5" fps="25"><xform weight="1" color="0.3" linear="1" coefs="1 0 0 1 0 0"/></flame>';
    const f = importFlameText(xml, GREY).flame;
    expect(f.motionBlur).toEqual({ length: 4, timeStep: 0.1, decay: 0.5 });
    expect(roundTrip(f).flame.motionBlur).toEqual({ length: 4, timeStep: 0.1, decay: 0.5 });
    expect(normalizeFlame(JSON.parse(JSON.stringify(f)), GREY).motionBlur?.length).toBe(4);
    const { motionBlurFrames } = await import('../src/ui/motionBlur');
    const times: number[] = [];
    const frames = motionBlurFrames(f, { evalAt: (t) => { times.push(+t.toFixed(4)); return f; }, t: 2, fps: 25 })!;
    expect(frames).toHaveLength(5);
    expect(frames[0].weight).toBe(1);
    // currTime = 50 + 4·0.1/2 = 50.2 frames, then −0.1 per packet → 50.1, 50.0, 49.9, 49.8 frames (÷25 s)
    expect(times).toEqual([2.004, 2, 1.996, 1.992]);
    expect(frames.slice(1).map((x) => +x.weight.toFixed(4))).toEqual([0.9912, 0.965, 0.9213, 0.86]); // 1 − p²·0.5·0.07/4 (0.99125 rounds down in binary)
    expect(frames[1].flame.lowDensityBrightness).toBe(0);
  });
});
