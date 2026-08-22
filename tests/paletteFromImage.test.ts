import { describe, it, expect } from 'vitest';
import { paletteFromPixels } from '../src/core/paletteFromImage';

const img = (pixels: [number, number, number][]) => { const a = new Uint8ClampedArray(pixels.length * 4); pixels.forEach(([r, g, b], i) => { a[i * 4] = r; a[i * 4 + 1] = g; a[i * 4 + 2] = b; a[i * 4 + 3] = 255; }); return a; };
const hex = (c: [number, number, number]) => c.map((v) => Math.round(v * 255));

describe('gradient from an image', () => {
  it('three flat colours come out as those colours, sorted by hue band, the rest black', () => {
    const px = img([...Array(50).fill([0, 0, 200]), ...Array(30).fill([200, 20, 20]), ...Array(20).fill([10, 180, 10])]);
    const pal = paletteFromPixels(px);
    expect(pal).toHaveLength(256);
    const nonBlack = pal.filter((c) => c.some((v) => v > 0)).map(hex);
    expect(nonBlack).toHaveLength(3);
    // histogram cells are 8/4/8 wide, so the colours sit at cell centres; hue order red(0) → green → blue
    expect(nonBlack[0][0]).toBeGreaterThan(190); expect(nonBlack[0][1]).toBeLessThan(30);
    expect(nonBlack[1][1]).toBeGreaterThan(170);
    expect(nonBlack[2][2]).toBeGreaterThan(190);
    // JWildfire sorts the black padding too: band 0, brightness 0 → it comes first; the brightest blue is last
    expect(pal[0]).toEqual([0, 0, 0]);
    expect(hex(pal[255])[2]).toBeGreaterThan(190);
  });
  it('a smooth ramp yields many distinct colours in brightness order within a hue band', () => {
    const px = img(Array.from({ length: 4096 }, (_, i) => [Math.round(255 * (i / 4095)), Math.round(40 * (i / 4095)), 0] as [number, number, number]));
    const pal = paletteFromPixels(px);
    // 5-bit red cells: at most ~32 × a few green cells can be occupied, so the cut yields a few dozen colours
    const distinct = new Set(pal.map((c) => hex(c).join(','))).size;
    expect(distinct).toBeGreaterThan(30);
    // the sort contract: (hue band, brightness) never decreases along the gradient
    const key = ([r, g, b]: [number, number, number]) => {
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
      let h = 0;
      if (d > 0) { h = mx === r ? (g - b) / d : mx === g ? 2 + (b - r) / d : 4 + (r - g) / d; h /= 6; if (h < 0) h += 1; }
      return [Math.floor(3.5 * h + 0.5), mx];
    };
    const keys = pal.map(key);
    for (let i = 1; i < keys.length; i++) expect(keys[i][0] > keys[i - 1][0] || (keys[i][0] === keys[i - 1][0] && keys[i][1] >= keys[i - 1][1] - 1e-9)).toBe(true);
  });
  it('handles an empty image', () => {
    const pal = paletteFromPixels(new Uint8ClampedArray(0));
    expect(pal).toHaveLength(256);
    expect(pal.every((c) => c.every((v) => v === 0))).toBe(true);
  });
});
