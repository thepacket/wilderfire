// The gradient library: the packs JWildfire ships (the classic Apophysis / UltraFractal packs carr, floral,
// universe, sky, star — 899 gradients after de-duplication), bundled by scripts/jwf-port/ugr2bin.mjs into
// public/gradients/jwildfire.bin and fetched the first time the library opens. See NOTICE.md.
import type { RGB } from './flame';

export interface GradientEntry { id: number; name: string; pack: string }
export interface GradientLibrary {
  packs: string[];
  entries: GradientEntry[];
  /** the 256 colours of one gradient, 0..1 */
  palette(id: number): RGB[];
  /** `rgb(...)` stops for a CSS linear-gradient preview (every `step`-th entry) */
  css(id: number, step?: number): string;
}

/** Parse the bundle: u32 LE header length, JSON header { packs, names, pack[] }, then 768 bytes per gradient. */
export function parseGradientBundle(buf: ArrayBuffer): GradientLibrary {
  const dv = new DataView(buf);
  const hlen = dv.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen))) as { packs: string[]; names: string[]; pack: number[] };
  const data = new Uint8Array(buf, 4 + hlen);
  if (data.length < header.names.length * 768) throw new Error('gradient bundle is truncated');
  const entries = header.names.map((name, id) => ({ id, name, pack: header.packs[header.pack[id]] ?? '' }));
  const palette = (id: number): RGB[] => {
    const o = id * 768;
    return Array.from({ length: 256 }, (_, i) => [data[o + i * 3] / 255, data[o + i * 3 + 1] / 255, data[o + i * 3 + 2] / 255] as RGB);
  };
  const css = (id: number, step = 8): string => {
    const o = id * 768;
    const stops: string[] = [];
    for (let i = 0; i < 256; i += step) stops.push(`rgb(${data[o + i * 3]},${data[o + i * 3 + 1]},${data[o + i * 3 + 2]}) ${(i / 255 * 100).toFixed(1)}%`);
    stops.push(`rgb(${data[o + 765]},${data[o + 766]},${data[o + 767]}) 100%`);
    return `linear-gradient(to right, ${stops.join(', ')})`;
  };
  return { packs: header.packs, entries, palette, css };
}

let loading: Promise<GradientLibrary> | null = null;
/** The bundled library (fetched once). */
export function loadGradientLibrary(): Promise<GradientLibrary> {
  if (!loading) {
    loading = fetch('/gradients/jwildfire.bin').then(async (r) => {
      if (!r.ok) throw new Error(`gradient library: HTTP ${r.status}`);
      return parseGradientBundle(await r.arrayBuffer());
    }).catch((e) => { loading = null; throw e; });
  }
  return loading;
}

/** Case-insensitive substring search over names (and pack names), optionally within one pack. */
export function searchGradients(lib: GradientLibrary, query: string, pack?: string): GradientEntry[] {
  const q = query.trim().toLowerCase();
  return lib.entries.filter((e) => (!pack || e.pack === pack) && (!q || e.name.toLowerCase().includes(q) || e.pack.toLowerCase().includes(q)));
}
