// Image store helpers for image layers: a picture is stored once under its content hash (IndexedDB), a layer
// refers to the key; composition files embed the pictures as data URLs (`assets`) so they travel.
import { imagePut, imageGet } from './libraryStore';
import type { Composition } from './composition';
import { forgetImage } from '../gpu/imageRenderer';

async function hashBlob(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const h = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(h).slice(0, 16), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Store a picture; returns its key and natural size. */
export async function storeImage(blob: Blob): Promise<{ key: string; w: number; h: number }> {
  const key = await hashBlob(blob);
  const bmp = await createImageBitmap(blob);
  const w = bmp.width, h = bmp.height; bmp.close();
  await imagePut(key, blob);
  forgetImage(key);
  return { key, w, h };
}

const blobToDataURL = (b: Blob) => new Promise<string>((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.onerror = () => rej(r.error); r.readAsDataURL(b); });
async function dataURLToBlob(u: string): Promise<Blob> { return (await fetch(u)).blob(); }

/** JSON of a composition with its pictures embedded (`assets: { key: dataURL }`). */
export async function compositionFileJSON(c: Composition): Promise<string> {
  const assets: Record<string, string> = {};
  for (const l of c.layers) if (l.kind === 'image' && !assets[l.image.key]) { const b = await imageGet(l.image.key); if (b) assets[l.image.key] = await blobToDataURL(b); }
  return JSON.stringify(Object.keys(assets).length ? { ...c, assets } : c);
}

/** Put a composition file's embedded pictures into the store (call before normalizing/using it). */
export async function importCompositionAssets(obj: any): Promise<void> {
  const assets = obj?.assets;
  if (!assets || typeof assets !== 'object') return;
  for (const [key, url] of Object.entries(assets)) {
    if (typeof url !== 'string' || !url.startsWith('data:')) continue;
    try { await imagePut(key, await dataURLToBlob(url)); forgetImage(key); } catch (e) { console.warn('composition asset:', e); }
  }
}
