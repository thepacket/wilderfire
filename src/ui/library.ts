// Flame library (IndexedDB, see ../core/libraryStore.ts) + session autosave (localStorage).
import { App, el, openModal } from './common';
import { normalizeFlame, type Flame } from '../core/flame';
import { libAll, libPut, libDelete, type LibEntry } from '../core/libraryStore';
import { saveText } from './saveFile';
import type { AnimAPI } from './animPanel';

export type { LibEntry };
const LS_AUTOSAVE = 'wilderfire.autosave';

/** The saved flames, newest first (for the batch export queue). */
export const listLibrary = (): Promise<LibEntry[]> => libAll();

export function buildLibrary(app: App, anim: AnimAPI) {
  function thumbnail(size = 144): string {
    return app.renderer.captureSync((cv) => {
      const c = document.createElement('canvas');
      c.width = size;
      c.height = size;
      const g = c.getContext('2d')!;
      // Cover-crop the (usually non-square) render into a square thumb.
      const s = Math.min(cv.width, cv.height);
      g.drawImage(cv, (cv.width - s) / 2, (cv.height - s) / 2, s, s, 0, 0, size, size);
      return c.toDataURL('image/jpeg', 0.72);
    });
  }

  function save() {
    const entry: LibEntry = {
      id: Math.random().toString(36).slice(2),
      name: app.flame.name || 'untitled',
      date: Date.now(),
      flame: JSON.parse(JSON.stringify(app.flame)),
      thumb: thumbnail(),
    };
    libPut(entry).catch((e) => alert('Could not save to the library: ' + (e as Error).message));
  }

  async function open() {
    const { body, close } = openModal('Flame library');
    let entries: LibEntry[] = [];
    try { entries = await libAll(); } catch (e) { body.append(el('div', 'hint', '⚠ Library unavailable: ' + (e as Error).message)); return; }
    const tools = el('div', 'btn-row');
    const expBtn = el('button', '', '⬇ Export library');
    expBtn.title = 'Save every entry (flames + thumbnails) as one JSON file — a backup, or to move the library to another browser';
    expBtn.onclick = () => saveText(JSON.stringify({ wilderfireLibrary: 1, entries }), { suggestedName: 'wilderfire-library.json', description: 'WilderFire library', mime: 'application/json', ext: '.json' });
    const impBtn = el('button', '', '⬆ Import library');
    impBtn.title = 'Merge entries from an exported library JSON (existing entries are kept)';
    const impFile = el('input') as HTMLInputElement;
    impFile.type = 'file'; impFile.accept = '.json,application/json'; impFile.style.display = 'none';
    impBtn.onclick = () => impFile.click();
    impFile.onchange = async () => {
      const f = impFile.files?.[0]; impFile.value = '';
      if (!f) return;
      try {
        const j = JSON.parse(await f.text());
        const list: LibEntry[] = Array.isArray(j?.entries) ? j.entries : Array.isArray(j) ? j : [];
        const ok = list.filter((e) => e && typeof e.id === 'string' && e.flame);
        if (!ok.length) throw new Error('no library entries in this file');
        await libPut(ok);
        close(); open();
      } catch (e) { alert('Import failed: ' + (e as Error).message); }
    };
    tools.append(expBtn, impBtn, impFile);
    if (!entries.length) {
      body.append(el('div', 'hint', 'Empty — use 💾 Save to keep the current flame here. Stored in your browser (IndexedDB).'), tools);
      return;
    }
    const grid = el('div', 'lib-grid');
    for (const e of entries) {
      const item = el('div', 'lib-item');
      const img = el('img') as HTMLImageElement;
      img.src = e.thumb;
      const meta = el('div', 'lib-meta');
      meta.append(
        el('div', 'lib-name', e.name),
        el('div', 'lib-date', new Date(e.date).toLocaleString()),
      );
      const del = el('button', 'lib-del danger', '✕');
      del.onclick = (ev) => {
        ev.stopPropagation();
        libDelete(e.id).then(() => item.remove()).catch((err) => alert('Delete failed: ' + (err as Error).message));
      };
      item.append(img, meta, del);
      item.onclick = () => {
        app.setFlame(normalizeFlame(e.flame, app.activeLayer.palette));
        close();
      };
      grid.append(item);
    }
    body.append(el('div', 'hint', `${entries.length} flame${entries.length > 1 ? 's' : ''}`), grid, tools);
  }

  // ---------- Autosave ----------
  let timer = 0;
  const autosave = () => {
    clearTimeout(timer);
    timer = window.setTimeout(() => {
      try {
        localStorage.setItem(LS_AUTOSAVE, JSON.stringify({
          flame: app.flame,
          anim: anim.getState(),
        }));
      } catch { /* storage full — non-fatal */ }
    }, 900);
  };
  app.on('flame', autosave);
  app.on('tone', autosave);
  app.on('history', autosave);

  return { save, open };
}

/** Read the autosaved session, if any. Call before boot picks a preset. */
export function loadAutosave(): { flame: unknown; anim?: unknown } | null {
  try {
    const raw = localStorage.getItem(LS_AUTOSAVE);
    if (!raw) return null;
    const obj = JSON.parse(raw);
    return obj && obj.flame ? obj : null;
  } catch { return null; }
}

export function restoreFlame(saved: { flame: unknown }, fallback: Flame): Flame {
  try {
    return normalizeFlame(saved.flame, fallback.layers[0].palette);
  } catch { return fallback; }
}
