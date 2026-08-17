// Flame library storage on IndexedDB (no practical size cap, unlike the ~5 MB localStorage
// budget it replaces). Entries carry the flame JSON and a JPEG thumbnail. The old
// localStorage library (`wilderfire.library`, ≤48 entries) is migrated on first use.

export interface LibEntry {
  id: string;
  name: string;
  date: number;
  flame: unknown;
  thumb: string; // jpeg data URL
}

const DB = 'wilderfire';
const STORE = 'library';
const LS_LEGACY = 'wilderfire.library';

let dbp: Promise<IDBDatabase> | null = null;
function db(): Promise<IDBDatabase> {
  if (dbp) return dbp;
  dbp = new Promise((res, rej) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains(STORE)) d.createObjectStore(STORE, { keyPath: 'id' }).createIndex('date', 'date');
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
