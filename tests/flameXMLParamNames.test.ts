// Variations whose parameter names are not XML names ("Density Pixels", "Red Fac.", "color mode 1-4"):
// the exporter writes them JWildfire's way (verbatim) and the importer's lenient pre-parser takes them back.
import { describe, it, expect } from 'vitest';
import { VARIATIONS } from '../src/core/variations';
import { defaultFlame } from '../src/core/flame';
import { flameToXML, importFlameText } from '../src/core/flameXML';

const affected = Object.entries(VARIATIONS as Record<string, { params?: { name: string; def: number }[] }>)
  .filter(([, v]) => v.params?.some((p) => /[^\w.-]/.test(p.name))).map(([n]) => n);
const pal = Array.from({ length: 256 }, (_, i) => [i / 255, 0.5, 1 - i / 255] as [number, number, number]);

describe('parameter names that are not XML names', () => {
  it('exist in the registry (glsl_*, crop_trapezoid, mobius_strip, flame_bulb…)', () => {
    expect(affected.length).toBeGreaterThan(10);
    expect(affected).toContain('crop_trapezoid');
  });
  it('are written verbatim, as JWildfire does, and every value comes back through the importer', () => {
    for (const n of affected) {
      const f = defaultFlame(pal);
      const params = Object.fromEntries((VARIATIONS[n] as { params?: { name: string; def: number }[] }).params!.map((p, i) => [p.name, p.def + (i + 1) * 0.5]));
      f.layers[0].xforms[0].variations = [{ name: n, weight: 0.5, params }];
      const xml = flameToXML(f);
      for (const p of Object.keys(params)) expect(xml, `${n}: attribute ${n}_${p}`).toContain(` ${n}_${p}="`);
      const back = importFlameText(xml, pal).flames[0].layers[0].xforms[0].variations.find((v) => v.name === n);
      expect(back, `${n} after import`).toBeDefined();
      for (const [k, v] of Object.entries(params)) expect(back!.params[k], `${n}.${k}`).toBeCloseTo(v, 5);
    }
  });
  it('a pack mixing several of them on one xform survives too', () => {
    const f = defaultFlame(pal);
    const pick = ['glsl_kaleidocomplex', 'crop_trapezoid', 'mobius_strip', 'flame_bulb'].filter((n) => VARIATIONS[n]);
    f.layers[0].xforms[0].variations = pick.map((n) => ({ name: n, weight: 0.25, params: Object.fromEntries((VARIATIONS[n] as { params?: { name: string; def: number }[] }).params!.map((p) => [p.name, p.def])) }));
    const back = importFlameText(flameToXML(f), pal).flames[0].layers[0].xforms[0].variations.map((v) => v.name);
    expect(back).toEqual(pick);
  });
});
