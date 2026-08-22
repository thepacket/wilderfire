import { describe, it, expect } from 'vitest';
import { addPngText, readPngText, flameXmlFromPng, toLatin1Xml, FLAME_PNG_KEY } from '../src/core/pngMeta';
import { importFlameText } from '../src/core/flameXML';

/** A minimal PNG: signature, IHDR, one IDAT (junk), IEND — CRCs are not checked by the reader. */
function tinyPng(): Uint8Array {
  const chunk = (type: string, data: number[]) => {
    const out = new Uint8Array(12 + data.length);
    new DataView(out.buffer).setUint32(0, data.length);
    out.set([...type].map((c) => c.charCodeAt(0)), 4);
    out.set(data, 8);
    return out;
  };
  const parts = [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', [0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0]), chunk('IDAT', [1, 2, 3]), chunk('IEND', [])];
  const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
  return all;
}
const GREY = Array.from({ length: 256 }, (_, i) => [i / 255, i / 255, i / 255] as [number, number, number]);

describe('flame in the PNG', () => {
  it('adds a tEXt chunk after IHDR that reads back, and replaces it on a second add', () => {
    const png = tinyPng();
    const a = addPngText(png, FLAME_PNG_KEY, '<flame name="a"/>');
    expect(a.length).toBe(png.length + 12 + FLAME_PNG_KEY.length + 1 + '<flame name="a"/>'.length);
    expect(String.fromCharCode(...a.subarray(8 + 25 + 4, 8 + 25 + 8))).toBe('tEXt'); // IHDR is 12 + 13 = 25 bytes
    expect(readPngText(a)[FLAME_PNG_KEY]).toBe('<flame name="a"/>');
    const b = addPngText(a, FLAME_PNG_KEY, '<flame name="b"/>');
    expect(readPngText(b)[FLAME_PNG_KEY]).toBe('<flame name="b"/>');
    expect(b.length).toBe(a.length);
    expect(flameXmlFromPng(png)).toBeNull();
  });
  it('keeps non-Latin-1 names as XML numeric references the importer understands', () => {
    const xml = '<flame name="Étoile – 星" size="100 100" scale="25"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>';
    expect(toLatin1Xml(xml)).toContain('&#x2013;');
    expect(toLatin1Xml(xml)).toContain('É'); // Latin-1 stays
    const back = flameXmlFromPng(addPngText(tinyPng(), FLAME_PNG_KEY, xml))!;
    expect(importFlameText(back, GREY).flame.name).toBe('Étoile – 星');
  });
  it('reads uncompressed iTXt too and ignores non-PNG input', () => {
    const key = [...'flam3_genome'].map((c) => c.charCodeAt(0));
    const body = new TextEncoder().encode('<flame name="i"/>');
    const data = [...key, 0, 0, 0, 0, 0, ...body]; // key NUL, compression flag 0, method 0, lang NUL, translated NUL
    const png = tinyPng();
    const chunk = new Uint8Array(12 + data.length);
    new DataView(chunk.buffer).setUint32(0, data.length);
    chunk.set([0x69, 0x54, 0x58, 0x74], 4); chunk.set(data, 8);
    const withI = new Uint8Array(png.length + chunk.length);
    withI.set(png.subarray(0, 33)); withI.set(chunk, 33); withI.set(png.subarray(33), 33 + chunk.length);
    expect(readPngText(withI)[FLAME_PNG_KEY]).toBe('<flame name="i"/>');
    expect(readPngText(new TextEncoder().encode('<flame/>'))).toEqual({});
    expect(() => addPngText(new TextEncoder().encode('nope'), 'k', 'v')).toThrow(/not a PNG/);
  });
});
