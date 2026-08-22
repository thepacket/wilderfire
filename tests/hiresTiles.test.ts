import { describe, it, expect } from 'vitest';
import { planTiles } from '../src/ui/hiresExport';

describe('hi-res export tiling', () => {
  const lim1GB = { maxCells: Math.floor((1 << 30) / 16), maxSide: 16384 }; // 67 Mpx of 16-byte histogram cells
  it('renders 4K and 8K in one tile on a 1 GB budget', () => {
    expect(planTiles(3840, 2160, lim1GB, 8)).toEqual({ tile: 3840, tilesX: 1, tilesY: 1 });
    expect(planTiles(7680, 4320, lim1GB, 8)).toEqual({ tile: 7680, tilesX: 1, tilesY: 1 });
  });
  it('splits into the largest square tiles that fit, apron included, when the image is too big', () => {
    const p = planTiles(11520, 7200, lim1GB, 8); // 83 Mpx
    expect(p.tilesX * p.tilesY).toBeGreaterThan(1);
    expect((p.tile + 16) ** 2).toBeLessThanOrEqual(lim1GB.maxCells);
    expect(p.tile + 16).toBeLessThanOrEqual(lim1GB.maxSide);
    expect(p.tilesX * p.tile).toBeGreaterThanOrEqual(11520);
    expect(p.tilesY * p.tile).toBeGreaterThanOrEqual(7200);
  });
  it('the texture side limit forces tiles even when the cells would fit', () => {
    const p = planTiles(9000, 1000, { maxCells: 100_000_000, maxSide: 8192 }, 8);
    expect(p.tilesX).toBe(2); expect(p.tilesY).toBe(1); expect(p.tile + 16).toBeLessThanOrEqual(8192);
  });
  it('the old 128 MB budget (8.4 Mpx) still fits 4K in one tile but splits a 4096² image', () => {
    const lim = { maxCells: (128 << 20) / 16, maxSide: 8192 };
    expect(planTiles(3840, 2160, lim, 8).tilesX * planTiles(3840, 2160, lim, 8).tilesY).toBe(1);
    const p = planTiles(4096, 4096, lim, 8);
    expect(p.tilesX * p.tilesY).toBeGreaterThan(1);
    expect((p.tile + 16) ** 2).toBeLessThanOrEqual(lim.maxCells);
  });
});
