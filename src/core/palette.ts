import type { RGB } from './flame';
import { expandStops } from './flame';

export type Stops = [number, number, number, number][]; // [pos, r, g, b]

export const PALETTE_PRESETS: Record<string, Stops> = {
  Ember: [
    [0.0, 0.01, 0.0, 0.02],
    [0.25, 0.45, 0.05, 0.02],
    [0.5, 0.95, 0.35, 0.05],
    [0.75, 1.0, 0.75, 0.25],
    [1.0, 1.0, 0.97, 0.82],
  ],
  Ocean: [
    [0.0, 0.0, 0.02, 0.1],
    [0.3, 0.0, 0.25, 0.5],
    [0.6, 0.05, 0.6, 0.75],
    [0.85, 0.55, 0.9, 0.9],
    [1.0, 0.95, 1.0, 0.98],
  ],
  Aurora: [
    [0.0, 0.02, 0.0, 0.08],
    [0.25, 0.05, 0.35, 0.35],
    [0.5, 0.1, 0.85, 0.45],
    [0.75, 0.65, 0.95, 0.55],
    [1.0, 0.95, 0.85, 1.0],
  ],
  Violet: [
    [0.0, 0.03, 0.0, 0.06],
    [0.3, 0.3, 0.05, 0.5],
    [0.6, 0.75, 0.25, 0.85],
    [0.85, 0.95, 0.65, 0.85],
    [1.0, 1.0, 0.95, 0.95],
  ],
  Solar: [
    [0.0, 0.1, 0.02, 0.0],
    [0.3, 0.7, 0.2, 0.0],
    [0.55, 1.0, 0.6, 0.0],
    [0.8, 1.0, 0.9, 0.4],
    [1.0, 0.85, 1.0, 0.95],
  ],
  Ice: [
    [0.0, 0.01, 0.02, 0.08],
    [0.35, 0.15, 0.3, 0.6],
    [0.65, 0.45, 0.65, 0.9],
    [1.0, 0.95, 0.98, 1.0],
  ],
  Rainbow: [
    [0.0, 0.9, 0.1, 0.15],
    [0.2, 0.95, 0.6, 0.1],
    [0.4, 0.85, 0.9, 0.15],
    [0.6, 0.15, 0.8, 0.4],
    [0.8, 0.15, 0.4, 0.9],
    [1.0, 0.7, 0.2, 0.85],
  ],
  Mono: [
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.55, 0.55, 0.6],
    [1.0, 1.0, 1.0, 1.0],
  ],
};

export function paletteFromPreset(name: string): RGB[] {
  return expandStops(PALETTE_PRESETS[name] ?? PALETTE_PRESETS.Ember);
}

/** IQ-style cosine palette: c(t) = a + b*cos(2π(ct + d)), clamped. */
export function cosinePalette(
  a: RGB, b: RGB, c: RGB, d: RGB,
): RGB[] {
  const out: RGB[] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    out.push([0, 1, 2].map((k) =>
      Math.min(1, Math.max(0, a[k] + b[k] * Math.cos(2 * Math.PI * (c[k] * t + d[k]))))
    ) as RGB);
  }
  return out;
}

export function randomPalette(rng: () => number = Math.random): RGB[] {
  const rr = (lo: number, hi: number) => lo + rng() * (hi - lo);
  const a: RGB = [rr(0.3, 0.6), rr(0.3, 0.6), rr(0.3, 0.6)];
  const b: RGB = [rr(0.3, 0.55), rr(0.3, 0.55), rr(0.3, 0.55)];
  const c: RGB = [rr(0.5, 1.4), rr(0.5, 1.4), rr(0.5, 1.4)];
  const d: RGB = [rng(), rng(), rng()];
  return cosinePalette(a, b, c, d);
}

export function rotatePalette(pal: RGB[], amount01: number): RGB[] {
  const shift = Math.round(amount01 * 256) % 256;
  return pal.map((_, i) => pal[(i + shift + 256) % 256]);
}

export function drawPalette(canvas: HTMLCanvasElement, pal: RGB[]) {
  const ctx = canvas.getContext('2d')!;
  const w = canvas.width, h = canvas.height;
  const img = ctx.createImageData(w, h);
  for (let x = 0; x < w; x++) {
    const c = pal[Math.min(255, Math.floor((x / w) * 256))];
    for (let y = 0; y < h; y++) {
      const o = (y * w + x) * 4;
      img.data[o] = c[0] * 255;
      img.data[o + 1] = c[1] * 255;
      img.data[o + 2] = c[2] * 255;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}
