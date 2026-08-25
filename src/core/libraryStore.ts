// Flame library storage on IndexedDB (no practical size cap, unlike the ~5 MB localStorage
// budget it replaces). Entries carry the flame JSON and a JPEG thumbnail — a Blob, which IndexedDB
// stores as binary and hands back without copying, so loading a few thousand entries is a fraction
// of what the old base64 data URLs cost in time and space. Entries written by earlier builds (and
// library JSON exports, which must stay text) carry data URLs; they are converted on the way in.
// The old localStorage library (`wilderfire.library`, ≤48 entries) is migrated on first use.
// The same database keeps the user's meshes for obj_mesh_wf (store `meshes`: file name → mesh binary).

/** JPEG Blob; a data URL in legacy entries and JSON exports */
export type Thumb = Blob | string;

export interface LibEntry {
  id: string;
  name: string;
  date: number;
  flame: unknown;
  thumb: Thumb;
  /** where it came from: the dropped file, `zip name › entry path`, or a folder path (set at import) */
  source?: string;
  /** JWildfire meta_info_author, when the file carried one */
  author?: string;
  /** ★ */
  fav?: boolean;
  /** free-form labels; a tag doubles as a collection in the library dialog */
  tags?: string[];
}

/** The pack a library entry came from: the zip / pack file / top folder of its source. */
export const packOf = (e: LibEntry): string | undefined => e.source ? e.source.split(' › ')[0].split('/')[0] : undefined;

const DB = 'wilderfire';
const STORE = 'library';
const MESHES = 'meshes';
const IMAGES = 'images';
const STHUMBS = 'sampleThumbs'; // built-in samples: entry id → thumbnail Blob (the flames themselves are code/files; only their pictures are worth caching)
const LS_LEGACY = 'wilderfire.library';

let dbp: Promise<IDBDatabase> | null = null;
function db(): Promise<IDBDatabase> {
  if (dbp) return dbp;
  dbp = new Promise((res, rej) => {
    const req = indexedDB.open(DB, 4);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains(STORE)) d.createObjectStore(STORE, { keyPath: 'id' }).createIndex('date', 'date');
      if (!d.objectStoreNames.contains(MESHES)) d.createObjectStore(MESHES);
      if (!d.objectStoreNames.contains(IMAGES)) d.createObjectStore(IMAGES); // reflection maps (file name → image Blob)
      if (!d.objectStoreNames.contains(STHUMBS)) d.createObjectStore(STHUMBS);
    };
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error ?? new Error('IndexedDB unavailable'));
  });
  return dbp;
}
const done = (tx: IDBTransaction) => new Promise<void>((res, rej) => { tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error); });

// ---- thumbnails ----
/** A data URL as a Blob (null when the string is not one). */
export function dataUrlToBlob(url: string): Blob | null {
  const m = /^data:([^;,]*)(;base64)?,([\s\S]*)$/.exec(url);
  if (!m) return null;
  const type = m[1] || 'image/jpeg';
  if (!m[2]) return new Blob([decodeURIComponent(m[3])], { type });
  const bin = atob(m[3]);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Blob([u8], { type });
}
const blobToDataUrl = (b: Blob) => new Promise<string>((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result as string); r.onerror = () => rej(r.error); r.readAsDataURL(b); });
/** Text form of a thumbnail (library JSON exports). */
export const thumbDataUrl = (t: Thumb): Promise<string> => (typeof t === 'string' ? Promise.resolve(t) : blobToDataUrl(t));
/** Something an <img> can show. Object URLs are cached per Blob (the grid re-creates cards while
 *  scrolling; one URL per picture) and released by releaseThumbSrcs(). */
const srcs = new Map<Blob, string>();
export function thumbSrc(t: Thumb): string {
  if (typeof t === 'string') return t;
  let u = srcs.get(t);
  if (!u) { u = URL.createObjectURL(t); srcs.set(t, u); }
  return u;
}
export function releaseThumbSrcs(): void {
  for (const u of srcs.values()) URL.revokeObjectURL(u);
  srcs.clear();
}
/** Legacy data-URL thumbnails become Blobs (in place); returns the entries that changed. */
function blobifyThumbs(entries: LibEntry[]): LibEntry[] {
  const changed: LibEntry[] = [];
  for (const e of entries) {
    if (typeof e.thumb === 'string' && e.thumb.startsWith('data:')) {
      const b = dataUrlToBlob(e.thumb);
      if (b) { e.thumb = b; changed.push(e); }
    }
  }
  return changed;
}

/** All entries, newest first (migrates the legacy localStorage library the first time; entries still
 *  carrying data-URL thumbnails are converted in memory and written back in the background). */
export async function libAll(): Promise<LibEntry[]> {
  const d = await db();
  await migrateLegacy(d);
  const tx = d.transaction(STORE, 'readonly');
  const all = await new Promise<LibEntry[]>((res, rej) => { const r = tx.objectStore(STORE).getAll(); r.onsuccess = () => res(r.result as LibEntry[]); r.onerror = () => rej(r.error); });
  const changed = blobifyThumbs(all);
  // entries saved before provenance existed: the author still sits in the flame (meta_info_author) — lift it up
  for (const e of all) {
    const a = (e.flame as { author?: unknown } | null)?.author;
    if (!e.author && typeof a === 'string' && a.trim()) { e.author = a.trim(); if (!changed.includes(e)) changed.push(e); }
  }
  if (changed.length && !migratingThumbs) {
    migratingThumbs = true;
    void (async () => {
      try {
        for (let i = 0; i < changed.length; i += 500) { // one transaction per batch, one pass over the library
          const tx2 = d.transaction(STORE, 'readwrite');
          for (const e of changed.slice(i, i + 500)) tx2.objectStore(STORE).put(e);
          await done(tx2);
        }
        console.info(`Library: ${changed.length} entr${changed.length === 1 ? 'y' : 'ies'} updated (binary thumbnails / author lifted from the flame).`);
      } catch (e) { console.warn('Library thumbnail conversion failed: ' + (e as Error).message); }
      finally { migratingThumbs = false; }
    })();
  }
  return all.sort((a, b) => b.date - a.date);
}
let migratingThumbs = false;

export async function libPut(entries: LibEntry | LibEntry[]): Promise<void> {
  const list = Array.isArray(entries) ? entries : [entries];
  blobifyThumbs(list); // JSON imports and legacy callers hand us data URLs
  const d = await db();
  const tx = d.transaction(STORE, 'readwrite');
  for (const e of list) tx.objectStore(STORE).put(e);
  await done(tx);
}

/** Remove every entry (the flame library only — user meshes stay). */
export async function libClear(): Promise<void> {
  const d = await db();
  const tx = d.transaction(STORE, 'readwrite');
  tx.objectStore(STORE).clear();
  await done(tx);
}

/** Remove several entries in one transaction. */
export async function libDeleteMany(ids: string[]): Promise<void> {
  if (!ids.length) return;
  const d = await db();
  const tx = d.transaction(STORE, 'readwrite');
  const st = tx.objectStore(STORE);
  for (const id of ids) st.delete(id);
  await done(tx);
}

export async function libDelete(id: string): Promise<void> {
  const d = await db();
  const tx = d.transaction(STORE, 'readwrite');
  tx.objectStore(STORE).delete(id);
  await done(tx);
}

let migrated = false;
async function migrateLegacy(d: IDBDatabase): Promise<void> {
  if (migrated) return;
  migrated = true;
  let legacy: LibEntry[] = [];
  try { const raw = localStorage.getItem(LS_LEGACY); if (raw) legacy = JSON.parse(raw); } catch { /* ignore */ }
  if (!Array.isArray(legacy) || !legacy.length) return;
  const tx = d.transaction(STORE, 'readwrite');
  for (const e of legacy) if (e && typeof e.id === 'string') tx.objectStore(STORE).put(e);
  await done(tx);
  localStorage.removeItem(LS_LEGACY);
}

// ---- thumbnails of the built-in sample flames (the library's "Samples" collection) ----
// Kept out of the flame store so the samples never mix into the user's library — they cannot be
// deleted, tagged, exported or counted with it; only their pictures live here, rendered once.
export async function sampleThumbAll(): Promise<Map<string, Blob>> {
  const d = await db();
  const tx = d.transaction(STHUMBS, 'readonly');
  const st = tx.objectStore(STHUMBS);
  const [keys, vals] = await Promise.all([
    new Promise<IDBValidKey[]>((res, rej) => { const r = st.getAllKeys(); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); }),
    new Promise<Blob[]>((res, rej) => { const r = st.getAll(); r.onsuccess = () => res(r.result as Blob[]); r.onerror = () => rej(r.error); }),
  ]);
  return new Map(keys.map((k, i) => [String(k), vals[i]]));
}
/** Drop cached pictures whose flame is no longer offered (the built-in list changed). */
export async function sampleThumbPrune(keep: string[]): Promise<void> {
  const d = await db();
  const tx = d.transaction(STHUMBS, 'readwrite');
  const st = tx.objectStore(STHUMBS);
  const keys = await new Promise<IDBValidKey[]>((res, rej) => { const r = st.getAllKeys(); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  const wanted = new Set(keep);
  for (const k of keys) if (!wanted.has(String(k))) st.delete(k);
  await done(tx);
}
export async function sampleThumbPut(id: string, thumb: Blob): Promise<void> {
  const d = await db();
  const tx = d.transaction(STHUMBS, 'readwrite');
  tx.objectStore(STHUMBS).put(thumb, id);
  await done(tx);
}

// ---- user meshes (obj_mesh_wf) ----
export async function meshPut(name: string, bin: ArrayBuffer): Promise<void> {
  const d = await db();
  const tx = d.transaction(MESHES, 'readwrite');
  tx.objectStore(MESHES).put(bin, name);
  await done(tx);
}
export async function meshGet(name: string): Promise<ArrayBuffer | undefined> {
  const d = await db();
  const tx = d.transaction(MESHES, 'readonly');
  return new Promise((res, rej) => { const r = tx.objectStore(MESHES).get(name); r.onsuccess = () => res(r.result as ArrayBuffer | undefined); r.onerror = () => rej(r.error); });
}
export async function meshNames(): Promise<string[]> {
  const d = await db();
  const tx = d.transaction(MESHES, 'readonly');
  return new Promise((res, rej) => { const r = tx.objectStore(MESHES).getAllKeys(); r.onsuccess = () => res((r.result as IDBValidKey[]).map(String).sort()); r.onerror = () => rej(r.error); });
}
export async function meshDelete(name: string): Promise<void> {
  const d = await db();
  const tx = d.transaction(MESHES, 'readwrite');
  tx.objectStore(MESHES).delete(name);
  await done(tx);
}

// ---- user images (solid-rendering reflection maps) ----
export async function imagePut(name: string, blob: Blob): Promise<void> {
  const d = await db();
  const tx = d.transaction(IMAGES, 'readwrite');
  tx.objectStore(IMAGES).put(blob, name);
  await done(tx);
}
export async function imageGet(name: string): Promise<Blob | undefined> {
  const d = await db();
  const tx = d.transaction(IMAGES, 'readonly');
  return new Promise((res, rej) => { const r = tx.objectStore(IMAGES).get(name); r.onsuccess = () => res(r.result as Blob | undefined); r.onerror = () => rej(r.error); });
}
export async function imageNames(): Promise<string[]> {
  const d = await db();
  const tx = d.transaction(IMAGES, 'readonly');
  return new Promise((res, rej) => { const r = tx.objectStore(IMAGES).getAllKeys(); r.onsuccess = () => res((r.result as IDBValidKey[]).map(String).sort()); r.onerror = () => rej(r.error); });
}
export async function imageDelete(name: string): Promise<void> {
  const d = await db();
  const tx = d.transaction(IMAGES, 'readwrite');
  tx.objectStore(IMAGES).delete(name);
  await done(tx);
}
