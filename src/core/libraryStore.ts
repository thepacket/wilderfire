// Flame library storage on IndexedDB (no practical size cap, unlike the ~5 MB localStorage
// budget it replaces). Entries carry the flame JSON and a JPEG thumbnail. The old
// localStorage library (`wilderfire.library`, ≤48 entries) is migrated on first use.
// The same database keeps the user's meshes for obj_mesh_wf (store `meshes`: file name → mesh binary).

export interface LibEntry {
  id: string;
  name: string;
  date: number;
  flame: unknown;
  thumb: string; // jpeg data URL
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
const LS_LEGACY = 'wilderfire.library';

let dbp: Promise<IDBDatabase> | null = null;
function db(): Promise<IDBDatabase> {
  if (dbp) return dbp;
  dbp = new Promise((res, rej) => {
    const req = indexedDB.open(DB, 2);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains(STORE)) d.createObjectStore(STORE, { keyPath: 'id' }).createIndex('date', 'date');
      if (!d.objectStoreNames.contains(MESHES)) d.createObjectStore(MESHES);
    };
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error ?? new Error('IndexedDB unavailable'));
  });
  return dbp;
}
const done = (tx: IDBTransaction) => new Promise<void>((res, rej) => { tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error); });

/** All entries, newest first (migrates the legacy localStorage library the first time). */
export async function libAll(): Promise<LibEntry[]> {
  const d = await db();
  await migrateLegacy(d);
  const tx = d.transaction(STORE, 'readonly');
  const all = await new Promise<LibEntry[]>((res, rej) => { const r = tx.objectStore(STORE).getAll(); r.onsuccess = () => res(r.result as LibEntry[]); r.onerror = () => rej(r.error); });
  return all.sort((a, b) => b.date - a.date);
}

export async function libPut(entries: LibEntry | LibEntry[]): Promise<void> {
  const d = await db();
  const tx = d.transaction(STORE, 'readwrite');
  for (const e of Array.isArray(entries) ? entries : [entries]) tx.objectStore(STORE).put(e);
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
