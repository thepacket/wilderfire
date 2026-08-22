import { describe, it, expect } from 'vitest';
import { RANDOM_STYLES, randomFlameInStyle } from '../src/core/randomStyles';
import { VARIATIONS } from '../src/core/variations';
import { flameToXML, importFlameText } from '../src/core/flameXML';

const GREY = Array.from({ length: 256 }, (_, i) => [i / 255, i / 255, i / 255] as [number, number, number]);

describe('random flame styles', () => {
  it('lists 18 styles with distinct ids', () => {
    expect(RANDOM_STYLES).toHaveLength(18);
    expect(new Set(RANDOM_STYLES.map((s) => s.id)).size).toBe(18);
  });
  it('every style makes valid, renderable flames — repeatedly (the generators are random)', () => {
    for (const { id, name } of RANDOM_STYLES) {
      for (let k = 0; k < 12; k++) {
        const f = randomFlameInStyle(id, GREY);
        expect(f.name.startsWith(name + ' - ')).toBe(true);
        expect(f.layers).toHaveLength(1);
        expect(f.layers[0].xforms.length).toBeGreaterThan(0);
        for (const x of [...f.layers[0].xforms, f.layers[0].final, ...f.layers[0].moreFinals].filter(Boolean)) {
          for (const c of [...x!.affine, ...x!.post]) expect(Number.isFinite(c)).toBe(true);
          expect(x!.weight).toBeGreaterThan(0);
          expect(x!.variations.length).toBeGreaterThan(0);
          for (const v of x!.variations) {
            expect(VARIATIONS[v.name], `${id}: unknown variation ${v.name}`).toBeDefined();
            expect(Number.isFinite(v.weight)).toBe(true);
          }
          if (x!.xaos) expect(x!.xaos).toHaveLength(f.layers[0].xforms.length);
        }
        expect(Number.isFinite(f.zoom) && f.zoom > 0).toBe(true);
        // survives the .flame round trip
        const back = importFlameText(flameToXML(f), GREY).flame;
        expect(back.layers[0].xforms).toHaveLength(f.layers[0].xforms.length);
      }
    }
  });
  it('"any" picks a style and an unknown id falls back to a random one', () => {
    const names = new Set(RANDOM_STYLES.map((s) => s.name));
    for (let k = 0; k < 20; k++) {
      const f = randomFlameInStyle('any', GREY);
      expect(names.has(f.name.split(' - ')[0])).toBe(true);
    }
    expect(names.has(randomFlameInStyle('no-such-style', GREY).name.split(' - ')[0])).toBe(true);
  });
  it('ports JWildfire\'s coefficient convention: Xenomorph\'s fixed affines land as a d b e c f', () => {
    const f = randomFlameInStyle('xenomorph', GREY);
    const x3 = f.layers[0].xforms[2]; // the bipolar transform has no random moves
    expect(x3.affine.map((v) => +v.toFixed(6))).toEqual([0.687247, -0.726424, -2.148126, 0.726424, 0.687247, 2.399942]);
    expect(x3.post.map((v) => +v.toFixed(6))).toEqual([0.606464, 0.795111, -1.061351, -0.795111, 0.606464, -0.636951]);
  });
});
