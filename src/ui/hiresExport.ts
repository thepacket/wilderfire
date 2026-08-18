// Hi-res export: renders a flame offscreen, tiled (with padding so the density-estimation
// filter does not seam), into a PNG blob. Shared by the Render tab's "Hi-res PNG" button and
// the batch export queue.
import type { Flame } from '../core/flame';

/** What the tile loop needs: a FlameRenderer (one flame) or the Composer (the whole layer stack). */
export interface RegionSource {
  setFlame(f: Flame): void;
  renderRegion(o: { fullW: number; fullH: number; tileX: number; tileY: number; tileW: number; tileH: number; spp: number; transparent?: boolean }): Promise<Uint8ClampedArray<ArrayBuffer>>;
}

/** Output-size choices: a multiple of the canvas or a fixed 16:9 frame (value `WxH`). */
export const SIZE_OPTIONS: { label: string; value: string }[] = [
  { label: '2×', value: '2' }, { label: '3×', value: '3' }, { label: '4×', value: '4' },
  { label: '1080p', value: '1920x1080' }, { label: '1440p', value: '2560x1440' }, { label: '4K', value: '3840x2160' },
];
export const QUALITY_OPTIONS: { label: string; value: string }[] = [
  { label: 'Fast', value: '250' }, { label: 'Good', value: '700' }, { label: 'Ultra', value: '1500' },
];

/** Resolve a SIZE_OPTIONS value against the live canvas size (even dimensions keep encoders happy). */
export function resolveSize(value: string, canvasW: number, canvasH: number): { w: number; h: number } {
  const fixed = /^(\d+)x(\d+)$/.exec(value);
  if (fixed) return { w: Number(fixed[1]), h: Number(fixed[2]) };
  const scale = parseInt(value) || 2;
  return { w: (canvasW * scale) & ~1, h: (canvasH * scale) & ~1 };
}

export const safeFileName = (name: string) => (name || 'wilderfire').replace(/[\\/:*?"<>|]+/g, '_').trim() || 'wilderfire';

export interface HiResOpts {
  w: number; h: number;
  /** plotted samples per pixel */
  spp: number;
  transparent?: boolean;
  /** called after every tile: done/total tiles */
  onTile?: (done: number, total: number) => void;
  /** set `signal.aborted` to stop after the current tile (an AbortError is thrown) */
  signal?: AbortSignal;
}

/**
 * Render `flame` (or, with `flame` null, whatever the source currently holds — the composer's layer stack)
 * at w×h and return the PNG. The renderer must already be in export mode (`renderer.exporting = true`,
 * restored by the caller together with `setFlame(app.flame)`); this sets the flame on the renderer, so it
 * compiles the kernel for it.
 */
export async function renderHiRes(renderer: RegionSource, flame: Flame | null, o: HiResOpts): Promise<Blob> {
  const TILE = 1024, PAD = 8;
  const { w: fullW, h: fullH } = o;
  if (flame) renderer.setFlame(flame);
  const out = document.createElement('canvas');
  out.width = fullW;
  out.height = fullH;
  const ctx = out.getContext('2d')!;
  const tilesX = Math.ceil(fullW / TILE);
  const tilesY = Math.ceil(fullH / TILE);
  let n = 0;
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      if (o.signal?.aborted) throw new DOMException('Export cancelled', 'AbortError');
      const x0 = tx * TILE, y0 = ty * TILE;
      const tw = Math.min(TILE, fullW - x0);
      const th = Math.min(TILE, fullH - y0);
      // Render with padding so the DE filter doesn't seam at tile edges.
      const pw = tw + 2 * PAD, ph = th + 2 * PAD;
      const px = await renderer.renderRegion({
        fullW, fullH, tileX: x0 - PAD, tileY: y0 - PAD,
        tileW: pw, tileH: ph, spp: o.spp, transparent: o.transparent,
      });
      const img = new ImageData(tw, th);
      for (let y = 0; y < th; y++) {
        const srcOff = ((y + PAD) * pw + PAD) * 4;
        img.data.set(px.subarray(srcOff, srcOff + tw * 4), y * tw * 4);
      }
      ctx.putImageData(img, x0, y0);
      n++;
      o.onTile?.(n, tilesX * tilesY);
    }
  }
  const blob = await new Promise<Blob | null>((res) => out.toBlob(res, 'image/png'));
  if (!blob) throw new Error('PNG encoding failed');
  return blob;
}
