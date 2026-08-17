import { describe, it, expect } from 'vitest';
import { applyEdits, flameSummary, flameJSONFor, paletteStops, variationCatalogue, estimateTokens } from '../src/ai/context';
import { normalizeFlame } from '../src/core/flame';
import { GREY, preset } from './helpers';

describe('applyEdits', () => {
  it('applies set / addvar / delvar / palette / name and reports bad lines', () => {
    const f = preset('Clockwork');
    const r = applyEdits(f, `
      set brightness 5
      set T1.weight 1.3
      set T1.variations.0.weight 0.9
      addvar T1 julian 0.5 power=3 dist=1
      delvar T2 nothing
      palette [[0,0.1,0,0.2],[0.5,1,0.4,0],[1,1,1,0.8]]
      name AI test
      frobnicate T1
    `, 0);
    expect(r.applied).toBe(6);
    expect(r.errors).toHaveLength(2);
    expect(r.flame.brightness).toBe(5);
    expect(r.flame.layers[0].xforms[0].weight).toBe(1.3);
    expect(r.flame.layers[0].xforms[0].variations.map((v) => v.name)).toContain('julian');
    expect(r.flame.layers[0].xforms[0].variations.find((v) => v.name === 'julian')!.params.power).toBe(3);
    expect(r.flame.name).toBe('AI test');
    expect(r.flame.layers[0].palette[0]).toEqual([0.1, 0, 0.2]);
    expect(f.brightness).not.toBe(5); // base untouched
  });
  it('addxform / delxform / final paths, refuses deleting the last transform', () => {
    const f = preset('Clockwork');
    const n = f.layers[0].xforms.length;
    const r = applyEdits(f, `addxform 0.4\nset T${n + 1}.color 0.7\ndelxform T1`, 0);
    expect(r.errors).toEqual([]);
    expect(r.flame.layers[0].xforms.length).toBe(n);
    expect(r.flame.layers[0].xforms[n - 1].color).toBe(0.7);
    const one = normalizeFlame({ xforms: [{}] }, GREY);
    expect(applyEdits(one, 'delxform T1', 0).errors[0]).toMatch(/last transform/);
  });
  it('the result renormalizes cleanly', () => {
    const f = preset('Clockwork');
    const r = applyEdits(f, 'set T2.opacity 0.5', 0);
    expect(() => normalizeFlame(JSON.parse(JSON.stringify(r.flame)), GREY)).not.toThrow();
  });
});

describe('context shaping', () => {
  it('summary lists every transform with its path; JSON variants shrink the palette', () => {
    const f = preset('Clockwork');
    const s = flameSummary(f, 'stops');
    for (let i = 0; i < f.layers[0].xforms.length; i++) expect(s).toContain(`T${i + 1} (layers.0.xforms.${i})`);
    expect(s).toContain('palette stops');
    const full = flameJSONFor(f, 'full'), stops = flameJSONFor(f, 'stops'), none = flameJSONFor(f, 'none');
    expect(full.length).toBeGreaterThan(stops.length * 3);
    expect(stops).toContain('paletteStops');
    expect(none).not.toContain('palette');
    expect(normalizeFlame(JSON.parse(stops), GREY).layers[0].palette).toHaveLength(256);
  });
  it('palette stops sample the ends exactly', () => {
    const st = paletteStops(GREY, 8);
    expect(st[0]).toEqual([0, 0, 0, 0]);
    expect(st[7][0]).toBe(1); expect(st[7][1]).toBe(1);
  });
  it('variation catalogue modes', () => {
    const f = preset('Clockwork');
    expect(variationCatalogue(f, 'none')).toBe('');
    const used = variationCatalogue(f, 'used');
    expect(used).toContain('ngon');
    expect(used.length).toBeLessThan(variationCatalogue(f, 'all').length / 5);
    expect(estimateTokens(4000, 1)).toBe(1800);
  });
});
