// Share links: the flame's .flame XML, deflated and base64url-encoded into the URL hash
// (`#f=…`). The hash never leaves the browser — nothing is sent to any server — and opening
// the link in WilderFire loads the flame, motion curves included. Needs CompressionStream /
// DecompressionStream, which every WebGPU-capable browser has.
import type { Flame } from './flame';
import type { MotionCurve } from './motion';

const PREFIX = '#f=';

const b64url = (bytes: Uint8Array): string => {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const fromB64url = (s: string): Uint8Array => {
  const b = atob(s.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (s.length % 4)) % 4));
  const out = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b.charCodeAt(i);
  return out;
};

async function deflate(text: string): Promise<Uint8Array> {
  const cs = new CompressionStream('deflate-raw');
  const buf = await new Response(new Blob([new TextEncoder().encode(text) as BlobPart]).stream().pipeThrough(cs)).arrayBuffer();
  return new Uint8Array(buf);
}
async function inflate(bytes: Uint8Array): Promise<string> {
  const ds = new DecompressionStream('deflate-raw');
  return new Response(new Blob([bytes as BlobPart]).stream().pipeThrough(ds)).text();
}

/** The hash fragment (`#f=…`) for a flame; prepend the app's URL to make it a link. */
export async function encodeFlameHash(flame: Flame, curves: MotionCurve[] = []): Promise<string> {
  const { flameToXML } = await import('./flameXML');
  return PREFIX + b64url(await deflate(flameToXML(flame, { curves })));
}

/** A full share link for the current page. */
export async function encodeFlameLink(flame: Flame, curves: MotionCurve[] = [], base = location.origin + location.pathname): Promise<string> {
  return base + (await encodeFlameHash(flame, curves));
}

export const isFlameHash = (hash: string) => hash.startsWith(PREFIX) && hash.length > PREFIX.length;

/** The .flame XML carried by a `#f=…` hash, or null when the hash is not one (or is corrupt). */
export async function decodeFlameHash(hash: string): Promise<string | null> {
  if (!isFlameHash(hash)) return null;
  try {
    const xml = await inflate(fromB64url(hash.slice(PREFIX.length)));
    return xml.trimStart().startsWith('<') ? xml : null;
  } catch {
    return null;
  }
}
