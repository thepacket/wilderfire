import { describe, it, expect } from 'vitest';
import { encodeFlameHash, decodeFlameHash, isFlameHash } from '../src/core/shareLink';
import { importFlameText } from '../src/core/flameXML';

const GREY = Array.from({ length: 256 }, (_, i) => [i / 255, i / 255, i / 255] as [number, number, number]);
const sample = '<flame name="Shared – 星" size="800 600" scale="120" brightness="4.2"><xform weight="0.6" color="0.2" linear="1" julian="0.4" julian_power="3" julian_dist="1" coefs="0.7 0.1 -0.1 0.7 0.2 0.1"/><xform weight="0.4" color="0.8" spherical="1" coefs="0.5 0 0 0.5 -0.3 0.1"/></flame>';

describe('share links', () => {
  it('round-trips a flame through the hash, compressed and URL-safe', async () => {
    if (typeof CompressionStream === 'undefined') return;
    const { flame, curves } = importFlameText(sample, GREY);
    const hash = await encodeFlameHash(flame, curves);
    expect(isFlameHash(hash)).toBe(true);
    expect(hash).toMatch(/^#f=[A-Za-z0-9_-]+$/);
    const xml = await decodeFlameHash(hash);
    expect(xml).not.toBeNull();
    const back = importFlameText(xml!, GREY).flame;
    expect(back.name).toBe('Shared – 星');
    expect(back.layers[0].xforms).toHaveLength(2);
    expect(back.layers[0].xforms[0].variations.map((v) => v.name)).toEqual(['linear', 'julian']);
    expect(back.brightness).toBeCloseTo(4.2, 6);
    // a real saving: the hash is well under the XML it carries (the palette alone is 1.5 KB of hex)
    expect(hash.length).toBeLessThan(xml!.length);
  });
  it('rejects hashes that are not flame links or are corrupt', async () => {
    expect(isFlameHash('#x=abc')).toBe(false);
    expect(isFlameHash('#f=')).toBe(false);
    expect(await decodeFlameHash('#other')).toBeNull();
    expect(await decodeFlameHash('#f=not-really-deflate-data!!')).toBeNull();
  });
});
