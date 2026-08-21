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

/** Render a square JPEG thumbnail for a flame the renderer isn't currently showing (pack import). */
async function offscreenThumb(app: App, flame: Flame, size = 144, spp = 150): Promise<string> {
  app.renderer.setFlame(flame);
  const px = await app.renderer.renderRegion({ fullW: size, fullH: size, tileX: 0, tileY: 0, tileW: size, tileH: size, spp });
  const c = document.createElement('canvas');
  c.width = c.height = size;
  c.getContext('2d')!.putImageData(new ImageData(px, size, size), 0, 0);
  return c.toDataURL('image/jpeg', 0.72);
}

/** Add every flame of an imported pack to the library, rendering a thumbnail for each.
 *  `onProgress` may return false to stop early; the flames done so far are kept. */
export async function addFlamesToLibrary(
  app: App,
  flames: Flame[],
  onProgress?: (done: number, total: number, name: string) => boolean | void,
): Promise<number> {
  const entries: LibEntry[] = [];
  const now = Date.now();
  try {
    for (const [i, f] of flames.entries()) {
      let thumb = '';
      try { thumb = await offscreenThumb(app, f); } catch { /* keep the flame, skip its picture */ }
      entries.push({
        id: Math.random().toString(36).slice(2),
        name: f.name || `pack ${i + 1}`,
        date: now - i, // the grid is newest-first, so descending stamps keep the pack's own order
        flame: JSON.parse(JSON.stringify(f)),
        thumb,
      });
      if (onProgress?.(i + 1, flames.length, f.name) === false) break;
    }
  } finally {
    app.resumeRender();
  }
  if (entries.length) await libPut(entries);
  return entries.length;
}

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
      body.append(tools, el('div', 'hint', 'Empty — use 💾 Save to keep the current flame here, or drop .flame / .zip files on the canvas. Stored in your browser (IndexedDB).'));
      return;
    }
    const grid = el('div', 'lib-grid');
    const items: HTMLElement[] = [];
    let sel = -1;
    const select = (i: number, scroll = true) => {
      if (!items.length) return;
      i = Math.max(0, Math.min(items.length - 1, i));
      items[sel]?.classList.remove('sel');
      sel = i;
      items[sel].classList.add('sel');
      if (scroll) items[sel].scrollIntoView({ block: 'nearest' });
    };
    const load = (i: number) => {
      const e = entries[i];
      if (!e) return;
      app.setFlame(normalizeFlame(e.flame, app.activeLayer.palette));
      close();
    };
    const remove = (i: number) => {
      const e = entries[i];
      if (!e) return;
      libDelete(e.id).then(() => {
        items[i].remove(); items.splice(i, 1); entries.splice(i, 1);
        hint.textContent = `${entries.length} flame${entries.length === 1 ? '' : 's'}`;
        if (items.length) select(Math.min(i, items.length - 1)); else sel = -1;
      }).catch((err) => alert('Delete failed: ' + (err as Error).message));
    };
    entries.forEach((e, i) => {
      const item = el('div', 'lib-item');
      const img = el('img') as HTMLImageElement;
      if (e.thumb) img.src = e.thumb;
      else img.alt = e.name; // pack import whose thumbnail render failed
      const meta = el('div', 'lib-meta');
      meta.append(
        el('div', 'lib-name', e.name),
        el('div', 'lib-date', new Date(e.date).toLocaleString()),
      );
      meta.title = e.name; // the full name, where the grid has to truncate it
      const del = el('button', 'lib-del danger', '✕');
      del.onclick = (ev) => { ev.stopPropagation(); remove(items.indexOf(item)); };
      item.append(img, meta, del);
      item.onclick = () => load(items.indexOf(item));
      item.onmouseenter = () => select(items.indexOf(item), false);
      items.push(item);
      grid.append(item);
    });
    const hint = el('div', 'hint', `${entries.length} flame${entries.length > 1 ? 's' : ''} — arrow keys / Page Up / Page Down move, Enter loads, Delete removes, Esc closes`);
    body.append(tools, hint, grid);
    // Keyboard navigation: the grid is a wrapping flex of fixed-width cards, so a row is
    // however many fit across the body; Page Up/Down jump by the rows the viewport shows.
    body.tabIndex = 0;
    const columns = () => {
      if (items.length < 2) return 1;
      const y0 = items[0].offsetTop;
      let n = 1;
      while (n < items.length && items[n].offsetTop === y0) n++;
      return n;
    };
    const pageRows = () => Math.max(1, Math.floor(body.clientHeight / (items[0].offsetHeight + 10)));
    body.addEventListener('keydown', (ev) => {
      const cols = columns();
      const cur = sel < 0 ? 0 : sel;
      switch (ev.key) {
        case 'ArrowRight': select(cur + 1); break;
        case 'ArrowLeft': select(cur - 1); break;
        case 'ArrowDown': select(sel < 0 ? 0 : cur + cols); break;
        case 'ArrowUp': select(cur - cols); break;
        case 'PageDown': select(cur + cols * pageRows()); break;
        case 'PageUp': select(cur - cols * pageRows()); break;
        case 'Home': select(0); break;
        case 'End': select(items.length - 1); break;
        case 'Enter': if (sel >= 0) load(sel); break;
        case 'Delete': case 'Backspace': if (sel >= 0) remove(sel); break;
        case 'Escape': close(); break;
        default: return;
      }
      ev.preventDefault();
    });
    body.focus();
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
