import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseGradientBundle, searchGradients } from '../src/core/gradientLibrary';

describe('gradient library bundle', () => {
  const buf = readFileSync('public/gradients/jwildfire.bin');
  const lib = parseGradientBundle(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  it('holds the five JWildfire packs, de-duplicated', () => {
    expect(lib.packs).toEqual(['carr', 'floral', 'universe', 'sky', 'star']);
    expect(lib.entries.length).toBe(899);
    expect(new Set(lib.entries.map((e) => e.name + '|' + e.pack)).size).toBeGreaterThan(850);
  });
  it('gives 256 colours in 0..1 and a CSS preview', () => {
    const p = lib.palette(0);
    expect(p.length).toBe(256);
    expect(p.every((c) => c.length === 3 && c.every((v) => v >= 0 && v <= 1))).toBe(true);
    expect(lib.css(0)).toMatch(/^linear-gradient\(to right, rgb\(\d+,\d+,\d+\) 0\.0%, .* 100%\)$/);
  });
  it('searches names and packs', () => {
    expect(searchGradients(lib, 'sky').every((e) => e.pack === 'sky' || /sky/i.test(e.name))).toBe(true);
    expect(searchGradients(lib, '', 'star').length).toBe(15);
    expect(searchGradients(lib, 'zzzz-nothing').length).toBe(0);
  });
});
