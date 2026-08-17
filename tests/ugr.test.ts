import { describe, it, expect } from 'vitest';
import { parseUGRAll, parseUGR, parseMAP, toUGR, toMAP } from '../src/ui/palettePanel';
import { GREY } from './helpers';

const pack = `Fire {
gradient:
  title="Fire" smooth=yes
  index=0 color=0
  index=199 color=255
  index=399 color=16777215
}

Sea {
gradient:
  title="Sea" smooth=yes
  index=0 color=16711680
  index=399 color=65280
}
`;

describe('gradient files', () => {
  it('parses every gradient of a .ugr pack with its title and packed colours (R + G·256 + B·65536)', () => {
    const g = parseUGRAll(pack);
    expect(g.map((x) => x.name)).toEqual(['Fire', 'Sea']);
    expect(g[0].palette[0]).toEqual([0, 0, 0]);
    expect(g[0].palette[255]).toEqual([1, 1, 1]);
    expect(g[1].palette[0]).toEqual([0, 0, 1]);      // 16711680 = blue
    expect(g[1].palette[255]).toEqual([0, 1, 0]);    // 65280 = green
    expect(parseUGR(pack)).toEqual(g[0].palette);
  });
  it('round-trips a palette through .ugr and .map', () => {
    const pal = GREY.map((c, i) => [c[0], (i * 7 % 256) / 255, 1 - c[2]] as [number, number, number]);
    const back = parseUGR(toUGR(pal, 'Test "quoted"'))!;
    expect(back).toHaveLength(256);
    for (const i of [0, 8, 128, 248, 255]) for (let k = 0; k < 3; k++) expect(Math.abs(back[i][k] - pal[i][k])).toBeLessThan(0.02);
    const map = parseMAP(toMAP(pal))!;
    for (const i of [0, 77, 255]) for (let k = 0; k < 3; k++) expect(Math.abs(map[i][k] - pal[i][k])).toBeLessThan(0.003);
  });
});
