// Reflection maps for solid rendering (JWildfire MaterialSettings.reflMapFilename): an image per material, named
// by its file name and kept in the browser's image store (src/core/libraryStore.ts), resampled to REFL_SIZE²
// and uploaded as one layer of the renderer's reflection texture array. A .flame names the file; the other side
// needs the same image in its store (like obj_mesh_wf's meshes) — without it the material simply has no map,
// which is what JWildfire does when the path does not resolve.
import { imagePut, imageGet, imageNames, imageDelete } from './libraryStore';
import type { Flame } from './flame';

export const REFL_SIZE = 512;

const version = new Map<string, number>();
const cache = new Map<string, Promise<ImageData | null>>();

export const reflMapNames = imageNames;
export const reflMapKey = (name: string): string => `${name}@${version.get(name) ?? 0}`;

/** Store an image file under its name; flames naming it pick it up on their next uniform write. */
export async function storeReflMap(name: string, file: Blob): Promise<void> {
  await imagePut(name, file);
  version.set(name, (version.get(name) ?? 0) + 1);
}
export async function removeReflMap(name: string): Promise<void> {
  await imageDelete(name);
  version.set(name, (version.get(name) ?? 0) + 1);
}

/** The image's pixels, stretched to REFL_SIZE × REFL_SIZE (null when the store has no such image). Cached per version. */
export function loadReflMap(name: string): Promise<ImageData | null> {
  const key = reflMapKey(name);
  let p = cache.get(key);
  if (!p) {
    p = (async () => {
      const blob = await imageGet(name);
      if (!blob) { console.warn(`reflection map "${name}" is not in this browser's image store (Render → Solid → material → ⬆ image); the material renders without it`); return null; }
      const bmp = await createImageBitmap(blob);
      try {
        const c = document.createElement('canvas');
        c.width = c.height = REFL_SIZE;
        const g = c.getContext('2d', { willReadFrequently: true })!;
        g.drawImage(bmp, 0, 0, REFL_SIZE, REFL_SIZE);
        return g.getImageData(0, 0, REFL_SIZE, REFL_SIZE);
      } finally { bmp.close(); }
    })().catch((e) => { console.warn(`reflection map "${name}": ${(e as Error).message}`); return null; });
    cache.set(key, p);
  }
  return p;
}

/** The reflection-map file names a flame's solid materials use (intensity > 0), in material order, unique. */
export function flameReflMaps(flame: Pick<Flame, 'solid'>): string[] {
  const out: string[] = [];
  if (!flame.solid?.enabled) return out;
  for (const m of flame.solid.materials) if (m.reflMapFilename && m.reflMapIntensity > 1e-9 && !out.includes(m.reflMapFilename)) out.push(m.reflMapFilename);
  return out;
}
