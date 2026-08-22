import { describe, it, expect } from 'vitest';
import { flameSignature, similarity, rankSimilar } from '../src/core/similarity';

const pal = (hue: number) => Array.from({ length: 256 }, () => {
  // a saturated colour of one hue (HSL → RGB at s=1, l=0.5)
  const h = hue * 6, c = 1, x = c * (1 - Math.abs((h % 2) - 1));
  const [r, g, b] = h < 1 ? [c, x, 0] : h < 2 ? [x, c, 0] : h < 3 ? [0, c, x] : h < 4 ? [0, x, c] : h < 5 ? [x, 0, c] : [c, 0, x];
  return [r, g, b];
});
const flame = (vars: [string, number][], hue = 0, extra: any = {}) => ({
  layers: [{ xforms: vars.map(([name, weight]) => ({ weight: 1, variations: [{ name, weight }] })), final: null, palette: pal(hue) }],
  ...extra,
});

describe('similarity', () => {
  it('a flame is identical to itself and far from a different one', () => {
    const a = flameSignature(flame([['linear', 1], ['spherical', 0.5]], 0.0));
    const b = flameSignature(flame([['julian', 1], ['blur', 0.3], ['curl', 0.2]], 0.5, { camPitch: 30 }));
    expect(similarity(a, a)).toBeCloseTo(1, 6);
    expect(similarity(a, b)).toBeLessThan(0.35);
  });
  it('shared variations score higher than a palette match alone', () => {
    const base = flameSignature(flame([['linear', 1], ['spherical', 0.5]], 0.0));
    const sameVars = flameSignature(flame([['linear', 1], ['spherical', 0.5]], 0.5));
    const sameHue = flameSignature(flame([['julian', 1], ['blur', 0.3]], 0.0));
    expect(similarity(base, sameVars)).toBeGreaterThan(similarity(base, sameHue));
  });
  it('structure matters: a final, 3D and solid lower the score when they differ', () => {
    const a = flameSignature(flame([['linear', 1]], 0.2));
    const withFinal = flameSignature({ ...flame([['linear', 1]], 0.2), layers: [{ ...flame([['linear', 1]], 0.2).layers[0], final: { weight: 1, variations: [{ name: 'linear', weight: 1 }] } }] });
    const solid = flameSignature(flame([['linear', 1]], 0.2, { solid: { enabled: true } }));
    expect(similarity(a, withFinal)).toBeLessThan(1);
    expect(similarity(a, solid)).toBeLessThan(similarity(a, withFinal));
  });
  it('rankSimilar returns the best first and respects the limit', () => {
    const target = flameSignature(flame([['linear', 1], ['spherical', 0.5]], 0.0));
    const items = [
      { item: 'far', sig: flameSignature(flame([['julian', 1]], 0.5, { camPitch: 20 })) },
      { item: 'near', sig: flameSignature(flame([['linear', 1], ['spherical', 0.4]], 0.05)) },
      { item: 'mid', sig: flameSignature(flame([['linear', 1], ['blur', 0.5]], 0.0)) },
      { item: 'self', sig: target },
    ];
    const r = rankSimilar(target, items, 3, (x) => x === 'self');
    expect(r.map((x) => x.item)).toEqual(['near', 'mid', 'far']);
    expect(r[0].score).toBeGreaterThan(r[1].score);
    expect(rankSimilar(target, items, 2)).toHaveLength(2);
    // a large pool keeps exactly the top-N in order (the insertion path)
    const pool = Array.from({ length: 500 }, (_, i) => ({ item: i, sig: flameSignature(flame([['linear', 1], ['spherical', (i % 50) / 50]], (i % 12) / 12)) }));
    const top = rankSimilar(target, pool, 5);
    expect(top).toHaveLength(5);
    for (let i = 1; i < top.length; i++) expect(top[i - 1].score).toBeGreaterThanOrEqual(top[i].score);
    const brute = pool.map((p) => similarity(target, p.sig)).sort((a, b) => b - a).slice(0, 5);
    expect(top.map((t) => +t.score.toFixed(9))).toEqual(brute.map((v) => +v.toFixed(9)));
  });
  it('tolerates odd input', () => {
    expect(() => flameSignature({})).not.toThrow();
    expect(() => flameSignature({ layers: [{ xforms: [{}], palette: [[1, 2], null] }] })).not.toThrow();
  });
});
