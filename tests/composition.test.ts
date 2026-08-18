import { describe, it, expect } from 'vitest';
import { blendPixel, normalizeComposition, wrapFlame, isSingleFlame, compositionToJSON, BLEND_MODES, BLEND_WGSL } from '../src/core/composition';
import { PRESETS } from '../src/core/presets';
import { GREY } from './helpers';

const px = (r: number, g: number, b: number, a = 1): [number, number, number, number] => [r, g, b, a];

describe('composition model + blend maths', () => {
  it('blendPixel follows the W3C separable formulas', () => {
    expect(blendPixel('normal', px(0.2, 0.2, 0.2), px(0.8, 0.4, 0.1))).toEqual([0.8, 0.4, 0.1, 1]);
    expect(blendPixel('multiply', px(0.5, 0.5, 0.5), px(0.5, 0.5, 0.5))[0]).toBeCloseTo(0.25, 9);
    expect(blendPixel('screen', px(0.5, 0.5, 0.5), px(0.5, 0.5, 0.5))[0]).toBeCloseTo(0.75, 9);
    expect(blendPixel('difference', px(0.9, 0.1, 0.5), px(0.4, 0.4, 0.5))).toEqual([expect.closeTo(0.5, 9), expect.closeTo(0.3, 9), expect.closeTo(0, 9), 1]);
    expect(blendPixel('add', px(0.7, 0, 0), px(0.7, 0, 0))[0]).toBe(1);
    // opacity halves the source contribution over an opaque backdrop
    expect(blendPixel('normal', px(0, 0, 0), px(1, 1, 1), 0.5)[0]).toBeCloseTo(0.5, 9);
    // a transparent source leaves the backdrop; a transparent backdrop takes the source
    expect(blendPixel('multiply', px(0.3, 0.6, 0.9), px(1, 1, 1, 0))).toEqual([0.3, 0.6, 0.9, 1]);
    const r = blendPixel('multiply', px(0, 0, 0, 0), px(0.3, 0.6, 0.9, 0.5));
    expect(r[3]).toBeCloseTo(0.5, 9); expect(r[0]).toBeCloseTo(0.3, 9);
    // every mode is defined and stays in range
    for (const m of BLEND_MODES) for (const v of [0, 0.25, 0.5, 0.75, 1]) { const o = blendPixel(m, px(v, 1 - v, 0.5), px(1 - v, v, 0.5)); for (const c of o) { expect(c).toBeGreaterThanOrEqual(0); expect(c).toBeLessThanOrEqual(1); } }
    // the WGSL twin lists the same modes in the same order (ids)
    for (let i = 0; i < BLEND_MODES.length; i++) expect(BLEND_WGSL).toContain(`case ${i}u:`);
  });

  it('a bare flame normalises to a one-layer composition; compositions round-trip through JSON', () => {
    const f = PRESETS[0].make();
    const c1 = normalizeComposition(JSON.parse(JSON.stringify(f)), GREY);
    expect(isSingleFlame(c1)).toBe(true);
    expect(c1.layers[0].flame.name).toBe(f.name);
    const c2 = wrapFlame(f);
    c2.layers.push({ ...c2.layers[0], id: 'x', name: 'top', blend: 'screen', opacity: 0.4, ownBackground: false, clip: true, flame: PRESETS[1].make() });
    const back = normalizeComposition(JSON.parse(compositionToJSON(c2)), GREY);
    expect(back.layers.length).toBe(2);
    expect(back.layers[1]).toMatchObject({ id: 'x', name: 'top', blend: 'screen', opacity: 0.4, ownBackground: false, clip: true });
    expect(isSingleFlame(back)).toBe(false);
    // unknown blend / kinds are tolerated
    const c3 = normalizeComposition({ layers: [{ kind: 'flame', flame: f, blend: 'weird' }, { kind: 'escape', foo: 1 }] }, GREY);
    expect(c3.layers.length).toBe(1);
    expect(c3.layers[0].blend).toBe('normal');
  });
});
