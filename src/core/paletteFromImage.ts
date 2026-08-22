// Gradient from an image — JWildfire's MedianCutQuantizer (the libjpeg jquant2 median cut on a
// 5/6/5-bit RGB histogram: split the most populous box until half the colours exist, then the
// largest box; each box's colour is its pixel-weighted mean) followed by RGBPalette.sort
// (hue in four bands, then brightness). 256 colours, padded with black like JWildfire.
import type { RGB } from './flame';

const C0_BITS = 5, C1_BITS = 6, C2_BITS = 5;
const C0_SHIFT = 8 - C0_BITS, C1_SHIFT = 8 - C1_BITS, C2_SHIFT = 8 - C2_BITS;
const N0 = 1 << C0_BITS, N1 = 1 << C1_BITS, N2 = 1 << C2_BITS;
const C0_SCALE = 2, C1_SCALE = 3, C2_SCALE = 1;
const FTOI = (v: number) => (v >= 0 ? Math.floor(v + 0.5) : -Math.floor(-v + 0.5));

interface Box { c0min: number; c0max: number; c1min: number; c1max: number; c2min: number; c2max: number; volume: number; colorcount: number }
const hidx = (c0: number, c1: number, c2: number) => (c0 * N1 + c1) * N2 + c2;

function updateBox(h: Uint32Array, b: Box) {
  let { c0min, c0max, c1min, c1max, c2min, c2max } = b;
  const any = (c0a: number, c0b: number, c1a: number, c1b: number, c2a: number, c2b: number) => {
    for (let c0 = c0a; c0 <= c0b; c0++) for (let c1 = c1a; c1 <= c1b; c1++) for (let c2 = c2a; c2 <= c2b; c2++) if (h[hidx(c0, c1, c2)] !== 0) return true;
    return false;
  };
  if (c0max > c0min) for (let c0 = c0min; c0 <= c0max; c0++) if (any(c0, c0, c1min, c1max, c2min, c2max)) { b.c0min = c0min = c0; break; }
  if (c0max > c0min) for (let c0 = c0max; c0 >= c0min; c0--) if (any(c0, c0, c1min, c1max, c2min, c2max)) { b.c0max = c0max = c0; break; }
  if (c1max > c1min) for (let c1 = c1min; c1 <= c1max; c1++) if (any(c0min, c0max, c1, c1, c2min, c2max)) { b.c1min = c1min = c1; break; }
  if (c1max > c1min) for (let c1 = c1max; c1 >= c1min; c1--) if (any(c0min, c0max, c1, c1, c2min, c2max)) { b.c1max = c1max = c1; break; }
  if (c2max > c2min) for (let c2 = c2min; c2 <= c2max; c2++) if (any(c0min, c0max, c1min, c1max, c2, c2)) { b.c2min = c2min = c2; break; }
  if (c2max > c2min) for (let c2 = c2max; c2 >= c2min; c2--) if (any(c0min, c0max, c1min, c1max, c2, c2)) { b.c2max = c2max = c2; break; }
  const d0 = ((c0max - c0min) << C0_SHIFT) * C0_SCALE;
  const d1 = ((c1max - c1min) << C1_SHIFT) * C1_SCALE;
  const d2 = ((c2max - c2min) << C2_SHIFT) * C2_SCALE;
  b.volume = d0 * d0 + d1 * d1 + d2 * d2;
  let n = 0;
  for (let c0 = c0min; c0 <= c0max; c0++) for (let c1 = c1min; c1 <= c1max; c1++) for (let c2 = c2min; c2 <= c2max; c2++) if (h[hidx(c0, c1, c2)] !== 0) n++;
  b.colorcount = n;
}

function medianCut(h: Uint32Array, boxes: Box[], desired: number): number {
  let numboxes = 1;
  while (numboxes < desired) {
    // by population for the first half, then by volume
    let b1: Box | null = null;
    if (numboxes * 2 <= desired) { let maxc = 0; for (let i = 0; i < numboxes; i++) { const b = boxes[i]; if (b.colorcount > maxc && b.volume > 0) { b1 = b; maxc = b.colorcount; } } }
    else { let maxv = 0; for (let i = 0; i < numboxes; i++) { const b = boxes[i]; if (b.volume > maxv) { b1 = b; maxv = b.volume; } } }
    if (!b1) break;
    const b2 = boxes[numboxes];
    b2.c0max = b1.c0max; b2.c1max = b1.c1max; b2.c2max = b1.c2max; b2.c0min = b1.c0min; b2.c1min = b1.c1min; b2.c2min = b1.c2min;
    // longest scaled axis; ties favour green, then red, blue last
    const c0 = ((b1.c0max - b1.c0min) << C0_SHIFT) * C0_SCALE;
    const c1 = ((b1.c1max - b1.c1min) << C1_SHIFT) * C1_SCALE;
    const c2 = ((b1.c2max - b1.c2min) << C2_SHIFT) * C2_SCALE;
    let cmax = c1, n = 1;
    if (c0 > cmax) { cmax = c0; n = 0; }
    if (c2 > cmax) n = 2;
    if (n === 0) { const lb = (b1.c0max + b1.c0min) >> 1; b1.c0max = lb; b2.c0min = lb + 1; }
    else if (n === 1) { const lb = (b1.c1max + b1.c1min) >> 1; b1.c1max = lb; b2.c1min = lb + 1; }
    else { const lb = (b1.c2max + b1.c2min) >> 1; b1.c2max = lb; b2.c2min = lb + 1; }
    updateBox(h, b1); updateBox(h, b2);
    numboxes++;
  }
  return numboxes;
}

function boxColor(h: Uint32Array, b: Box): [number, number, number] {
  let total = 0, t0 = 0, t1 = 0, t2 = 0;
  for (let c0 = b.c0min; c0 <= b.c0max; c0++) for (let c1 = b.c1min; c1 <= b.c1max; c1++) for (let c2 = b.c2min; c2 <= b.c2max; c2++) {
    const n = h[hidx(c0, c1, c2)];
    if (n) { total += n; t0 += ((c0 << C0_SHIFT) + ((1 << C0_SHIFT) >> 1)) * n; t1 += ((c1 << C1_SHIFT) + ((1 << C1_SHIFT) >> 1)) * n; t2 += ((c2 << C2_SHIFT) + ((1 << C2_SHIFT) >> 1)) * n; }
  }
  if (!total) return [0, 0, 0];
  const lim = (v: number) => Math.min(255, Math.max(0, Math.floor((v + (total >> 1)) / total)));
  return [lim(t0), lim(t1), lim(t2)];
}

/** RGBColor.compareToRGBColor: hue in four bands (FTOI(3.5·h), h in 0..1), then value. */
function hsvKey(r: number, g: number, b: number): [number, number] {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const v = mx / 255;
  const d = mx - mn;
  let h = 0;
  if (d > 0) {
    if (mx === r) h = (g - b) / d; else if (mx === g) h = 2 + (b - r) / d; else h = 4 + (r - g) / d;
    h /= 6; if (h < 0) h += 1; if (h >= 1) h -= 1;
  }
  return [FTOI(3.5 * h), v];
}

/** MedianCutQuantizer.createPalette on RGBA pixels (alpha ignored): 256 colours in JWildfire's order, 0..1. */
export function paletteFromPixels(rgba: Uint8ClampedArray | Uint8Array, count = 256): RGB[] {
  const h = new Uint32Array(N0 * N1 * N2);
  for (let i = 0; i + 2 < rgba.length; i += 4) h[hidx(rgba[i] >> C0_SHIFT, rgba[i + 1] >> C1_SHIFT, rgba[i + 2] >> C2_SHIFT)]++;
  const boxes: Box[] = Array.from({ length: count }, () => ({ c0min: 0, c0max: 255 >> C0_SHIFT, c1min: 0, c1max: 255 >> C1_SHIFT, c2min: 0, c2max: 255 >> C2_SHIFT, volume: 0, colorcount: 0 }));
  updateBox(h, boxes[0]);
  const n = medianCut(h, boxes, count);
  const colors: [number, number, number][] = [];
  for (let i = 0; i < n; i++) colors.push(boxColor(h, boxes[i]));
  while (colors.length < count) colors.push([0, 0, 0]);
  colors.sort((a, b) => { const ka = hsvKey(...a), kb = hsvKey(...b); return ka[0] - kb[0] || ka[1] - kb[1]; });
  return colors.map(([r, g, b]) => [r / 255, g / 255, b / 255] as RGB);
}

/** Browser side: decode an image file (any format the browser shows), downsample to ≤ 256 px, quantize. */
export async function paletteFromImageFile(file: Blob): Promise<RGB[]> {
  const bmp = await createImageBitmap(file);
  try {
    const scale = Math.min(1, 256 / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale)), hgt = Math.max(1, Math.round(bmp.height * scale));
    const c = document.createElement('canvas');
    c.width = w; c.height = hgt;
    const g = c.getContext('2d', { willReadFrequently: true })!;
    g.drawImage(bmp, 0, 0, w, hgt);
    return paletteFromPixels(g.getImageData(0, 0, w, hgt).data);
  } finally {
    bmp.close();
  }
}
