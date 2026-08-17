import { describe, it, expect } from 'vitest';
import { normalizeFlame, flameToJSON, flameSignature, expandStops, cloneFlame } from '../src/core/flame';
import { interpFlame } from '../src/core/animate';
import { PRESETS } from '../src/core/presets';
import { GREY, preset } from './helpers';

describe('normalizeFlame', () => {
  it('round-trips every preset through JSON unchanged', () => {
    for (const p of PRESETS) {
      const f = p.make();
      const back = normalizeFlame(JSON.parse(flameToJSON(f)), GREY);
      expect(back).toEqual(f);
    }
  });
  it('fills defaults for old files (no 3D / DOF / tonemap fields)', () => {
    const f = normalizeFlame({ layers: [{ xforms: [{ variations: [{ name: 'linear', weight: 1 }] }] }] }, GREY);
    expect(f.brightness).toBe(4);
    expect(f.gamma).toBe(4);
    expect(f.gammaThreshold).toBe(0.01);
    expect(f.deRadius).toBe(1);
    expect(f.whiteLevel).toBe(220);
    expect(f.camDOF).toBe(0);
    expect(f.layers[0].xforms[0].yz).toBeUndefined();
  });
  it('drops identity 3D affines and keeps real ones', () => {
    const f = normalizeFlame({ layers: [{ xforms: [{ yz: [1, 0, 0, 0, 1, 0], zx: [1, 0, 0.1, 0, 1, 0] }] }] }, GREY);
    expect(f.layers[0].xforms[0].yz).toBeUndefined();
    expect(f.layers[0].xforms[0].zx).toEqual([1, 0, 0.1, 0, 1, 0]);
  });
  it('accepts paletteStops', () => {
    const f = normalizeFlame({ xforms: [{}], paletteStops: [[0, 1, 0, 0], [1, 0, 0, 1]] }, GREY);
    expect(f.layers[0].palette[0]).toEqual([1, 0, 0]);
    expect(f.layers[0].palette[255][2]).toBeCloseTo(1);
  });
});

describe('flameSignature', () => {
  it('changes only on structural edits', () => {
    const f = preset('Clockwork');
    const s0 = flameSignature(f);
    f.layers[0].xforms[0].weight *= 2; f.brightness = 1;
    expect(flameSignature(f)).toBe(s0);
    f.layers[0].xforms[0].variations.push({ name: 'julian', weight: 1, params: { power: 3, dist: 1 } });
    expect(flameSignature(f)).not.toBe(s0);
  });
});

describe('expandStops', () => {
  it('interpolates 256 entries between stops', () => {
    const p = expandStops([[0, 0, 0, 0], [1, 1, 1, 1]]);
    expect(p).toHaveLength(256);
    expect(p[128][0]).toBeCloseTo(128 / 255, 2);
  });
});

describe('interpFlame', () => {
  it('lerps tone, camera and 3D affines; keeps structure', () => {
    const a = preset('Clockwork'), b = cloneFlame(a);
    b.brightness = a.brightness + 2; b.camPitch = 40; b.layers[0].xforms[0].yz = [1, 0, 0.4, 0, 1, 0];
    const m = interpFlame(a, b, 0.5);
    expect(m.brightness).toBeCloseTo(a.brightness + 1);
    expect(m.camPitch).toBeCloseTo(20);
    expect(m.layers[0].xforms[0].yz?.[2]).toBeCloseTo(0.2);
    expect(flameSignature(m)).toBe(flameSignature(a));
  });
});
