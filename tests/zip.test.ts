import { describe, it, expect } from 'vitest';
import { readZip, isZipName } from '../src/core/zip';

/** Build a zip in memory: stored (method 0) entries, optionally one deflated via CompressionStream. */
async function makeZip(files: { name: string; data: string; deflate?: boolean }[]): Promise<ArrayBuffer> {
  const enc = new TextEncoder();
  const parts: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  const u16 = (v: number) => [v & 255, (v >> 8) & 255];
  const u32 = (v: number) => [v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255];
  for (const f of files) {
    const raw = enc.encode(f.data);
    let payload = raw, method = 0;
    if (f.deflate) {
      const cs = new CompressionStream('deflate-raw');
      payload = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(cs)).arrayBuffer());
      method = 8;
    }
    const name = enc.encode(f.name);
    const local = new Uint8Array([...u32(0x04034b50), ...u16(20), ...u16(0x800), ...u16(method), ...u16(0), ...u16(0), ...u32(0), ...u32(payload.length), ...u32(raw.length), ...u16(name.length), ...u16(0), ...name, ...payload]);
    central.push(new Uint8Array([...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0x800), ...u16(method), ...u16(0), ...u16(0), ...u32(0), ...u32(payload.length), ...u32(raw.length), ...u16(name.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(0), ...u32(offset), ...name]));
    parts.push(local);
    offset += local.length;
  }
  const cdirOff = offset;
  const cdir = central.reduce((n, c) => n + c.length, 0);
  const eocd = new Uint8Array([...u32(0x06054b50), ...u16(0), ...u16(0), ...u16(files.length), ...u16(files.length), ...u32(cdir), ...u32(cdirOff), ...u16(0)]);
  const all = new Uint8Array(offset + cdir + eocd.length);
  let p = 0;
  for (const b of [...parts, ...central, eocd]) { all.set(b, p); p += b.length; }
  return all.buffer;
}

describe('zip reader', () => {
  it('lists entries, skips directories, decodes stored entries', async () => {
    const zip = await makeZip([{ name: 'pack/', data: '' }, { name: 'pack/a.flame', data: '<flame name="a"/>' }, { name: 'pack/readme.txt', data: 'hi' }]);
    const entries = readZip(zip);
    expect(entries.map((e) => e.name)).toEqual(['pack/a.flame', 'pack/readme.txt']);
    expect(await entries[0].text()).toBe('<flame name="a"/>');
    expect(entries[0].size).toBe('<flame name="a"/>'.length);
  });
  it('inflates deflated entries through DecompressionStream', async () => {
    if (typeof CompressionStream === 'undefined') return; // runtime without the Streams API
    const body = '<flame name="deflated">' + 'x'.repeat(5000) + '</flame>';
    const entries = readZip(await makeZip([{ name: 'b.flame', data: body, deflate: true }]));
    expect(await entries[0].text()).toBe(body);
  });
  it('rejects non-zip data with a clear message', () => {
    expect(() => readZip(new TextEncoder().encode('<flame/>').buffer as ArrayBuffer)).toThrow(/not a zip/);
  });
  it('recognises zip names', () => {
    expect(isZipName('Pack.ZIP')).toBe(true);
    expect(isZipName('pack.flame')).toBe(false);
  });
});
