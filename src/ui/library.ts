// Flame library + session autosave, both in localStorage.
import { App, el, openModal } from './common';
import { normalizeFlame, type Flame } from '../core/flame';
import type { AnimAPI } from './animPanel';

const LS_LIB = 'wilderfire.library';
const LS_AUTOSAVE = 'wilderfire.autosave';
const MAX_ENTRIES = 48;

interface LibEntry {
  id: string;
  name: string;
  date: number;
  flame: unknown;
  thumb: string; // jpeg data URL
}

function loadLib(): LibEntry[] {
  try {
    const raw = localStorage.getItem(LS_LIB);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}

function storeLib(entries: LibEntry[]) {
  try {
    localStorage.setItem(LS_LIB, JSON.stringify(entries));
  } catch {
    alert('Library storage is full — delete some entries.');
  }
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
    const entries = loadLib();
    entries.unshift({
      id: Math.random().toString(36).slice(2),
      name: app.flame.name || 'untitled',
      date: Date.now(),
      flame: JSON.parse(JSON.stringify(app.flame)),
      thumb: thumbnail(),
    });
    while (entries.length > MAX_ENTRIES) entries.pop();
    storeLib(entries);
  }

  function open() {
    const { body, close } = openModal('Flame library');
    const entries = loadLib();
    if (!entries.length) {
      body.append(el('div', 'hint', 'Empty — use 💾 Save to keep the current flame here. Stored in your browser (localStorage).'));
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
        storeLib(loadLib().filter((x) => x.id !== e.id));
        item.remove();
      };
      item.append(img, meta, del);
      item.onclick = () => {
        app.setFlame(normalizeFlame(e.flame, app.activeLayer.palette));
        close();
      };
      grid.append(item);
    }
    body.append(grid);
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
