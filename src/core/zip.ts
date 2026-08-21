// Minimal ZIP reader for flame packs dropped on the canvas (no dependency: the browser's
// DecompressionStream handles DEFLATE, which is what every archiver writes). Reads the central
// directory, then each entry's local header. Stored (0) and deflated (8) entries only; ZIP64 and
// encrypted archives are reported, not guessed at.

export interface ZipEntry {
  name: string;
  size: number;
  text: () => Promise<string>;
}

const SIG_EOCD = 0x06054b50;
const SIG_CDIR = 0x02014b50;
const SIG_LOCAL = 0x04034b50;

async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') throw new Error('this browser cannot unzip (no DecompressionStream)');
  const ds = new DecompressionStream('deflate-raw');
  const out = new Response(new Blob([data as BlobPart]).stream().pipeThrough(ds));
  return new Uint8Array(await out.arrayBuffer());
}

/** Entries of a .zip, lazily decoded. Directory entries are skipped. */
export function readZip(buf: ArrayBuffer): ZipEntry[] {
  const bytes = new Uint8Array(buf);
  const dv = new DataView(buf);
  // End-of-central-directory record: scan back past the (≤ 64 KB) comment
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 65535); i--) {
    if (dv.getUint32(i, true) === SIG_EOCD) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('not a zip file (no end-of-central-directory record)');
  const count = dv.getUint16(eocd + 10, true);
  const cdirOff = dv.getUint32(eocd + 16, true);
  if (count === 0xffff || cdirOff === 0xffffffff) throw new Error('ZIP64 archives are not supported');
  const utf8 = new TextDecoder('utf-8');
  const latin1 = new TextDecoder('latin1');
  const entries: ZipEntry[] = [];
  let p = cdirOff;
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(p, true) !== SIG_CDIR) throw new Error('corrupt zip (central directory)');
    const flags = dv.getUint16(p + 8, true);
    const method = dv.getUint16(p + 10, true);
    const csize = dv.getUint32(p + 20, true);
    const usize = dv.getUint32(p + 24, true);
    const nameLen = dv.getUint16(p + 28, true);
    const extraLen = dv.getUint16(p + 30, true);
    const commentLen = dv.getUint16(p + 32, true);
    const localOff = dv.getUint32(p + 42, true);
    const nameBytes = bytes.subarray(p + 46, p + 46 + nameLen);
    const name = (flags & 0x800 ? utf8 : latin1).decode(nameBytes);
    p += 46 + nameLen + extraLen + commentLen;
    if (name.endsWith('/')) continue; // directory
    const encrypted = (flags & 1) !== 0;
    entries.push({
      name,
      size: usize,
      text: async () => {
        if (encrypted) throw new Error(`${name}: encrypted entries are not supported`);
        if (method !== 0 && method !== 8) throw new Error(`${name}: unsupported compression method ${method}`);
        if (dv.getUint32(localOff, true) !== SIG_LOCAL) throw new Error(`${name}: corrupt local header`);
        const ln = dv.getUint16(localOff + 26, true);
        const le = dv.getUint16(localOff + 28, true);
        const start = localOff + 30 + ln + le;
        const raw = bytes.subarray(start, start + csize);
        const data = method === 0 ? raw : await inflateRaw(raw);
        return utf8.decode(data);
      },
    });
  }
  return entries;
}

export const isZipName = (name: string) => /\.zip$/i.test(name);
