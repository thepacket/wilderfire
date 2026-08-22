// Flame library (IndexedDB, see ../core/libraryStore.ts) + session autosave (localStorage).
import { App, el, openModal } from './common';
import { normalizeFlame, type Flame } from '../core/flame';
import { libAll, libPut, libDelete, libDeleteMany, libClear, type LibEntry } from '../core/libraryStore';
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
  sources?: (string | undefined)[],
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
        ...(sources?.[i] ? { source: sources[i] } : {}),
        ...(f.author ? { author: f.author } : {}),
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
      ...(app.flameSource ? { source: app.flameSource } : {}),
      ...(app.flame.author ? { author: app.flame.author } : {}),
    };
    libPut(entry).catch((e) => alert('Could not save to the library: ' + (e as Error).message));
  }

  async function open() {
    const { body, close } = openModal('Flame library');
    let entries: LibEntry[] = [];
    // The store returns newest-first (batch export wants that); the dialog shows names in order —
    // natural numeric order so "Flame 2" sorts before "Flame 10", ties oldest-first.
    const byName = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
    try { entries = (await libAll()).sort((a, b) => byName.compare(a.name, b.name) || a.date - b.date); } catch (e) { body.append(el('div', 'hint', '⚠ Library unavailable: ' + (e as Error).message)); return; }
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
    const clearBtn = el('button', 'danger', '🗑 Empty library');
    clearBtn.title = 'Remove every flame from the library (asks first; export a backup if in doubt)';
    clearBtn.onclick = async () => {
      const n = entries.length;
      if (!n) return;
      if (!confirm(`Remove all ${n} flame${n === 1 ? '' : 's'} from the library?\n\nThis cannot be undone — ⬇ Export library first if you want a backup.`)) return;
      try { await libClear(); close(); open(); } catch (e) { alert('Could not empty the library: ' + (e as Error).message); }
    };
    const dedupBtn = el('button', '', '⧉ Remove duplicates');
    dedupBtn.title = 'Remove entries whose flame is identical to an earlier one (same parameters, the name may differ) — the first instance added is kept';
    dedupBtn.onclick = async () => {
      // identity = the flame's parameters; the name, date and thumbnail are not part of it
      const key = (e: LibEntry) => { const f = { ...(e.flame as Record<string, unknown>) }; delete f.name; return JSON.stringify(f); };
      const seen = new Set<string>();
      const dupes: LibEntry[] = [];
      for (const e of [...entries].sort((a, b) => a.date - b.date)) { const k = key(e); if (seen.has(k)) dupes.push(e); else seen.add(k); }
      if (!dupes.length) { alert('No duplicates — every flame in the library is different.'); return; }
      if (!confirm(`Remove ${dupes.length} duplicate${dupes.length === 1 ? '' : 's'}, keeping the first instance of each flame?\n\n${dupes.slice(0, 6).map((e) => '• ' + e.name).join('\n')}${dupes.length > 6 ? '\n…' : ''}`)) return;
      try { await libDeleteMany(dupes.map((e) => e.id)); close(); open(); }
      catch (e) { alert('Could not remove duplicates: ' + (e as Error).message); }
    };
    const search = el('input', 'lib-search') as HTMLInputElement;
    search.type = 'search';
    search.placeholder = 'Search names, authors, sources…';
    search.title = 'Show only flames whose name, author or source contains this text (press / to get here, Esc to clear)';
    search.spellcheck = false;
    const galBtn = el('button', 'primary', '▶ Gallery');
    galBtn.title = 'Fullscreen slideshow through these flames (the search result, in this order) — ← → browse, Space auto-advances, Esc leaves with the shown flame loaded';
    galBtn.onclick = async () => {
      const list = vis.map((card) => entries[entryOf(card)]);
      if (!list.length) return;
      const startAt = sel >= 0 ? sel : 0;
      const { openGallery } = await import('./gallery');
      close();
      openGallery(app, list, startAt);
    };
    tools.append(search, galBtn, expBtn, impBtn, dedupBtn, clearBtn, impFile);
    if (!entries.length) {
      const empty = el('div', 'hint', 'Empty — use 💾 Save to keep the current flame here, or drop .flame / .zip files on the canvas. Stored in your browser (IndexedDB). ');
      const link = el('a', '', 'Flame packs to get started: jwfsanctuary.club/downloads/flamepacks') as HTMLAnchorElement;
      link.href = 'https://www.jwfsanctuary.club/downloads/flamepacks/'; link.target = '_blank'; link.rel = 'noopener';
      empty.append(link);
      body.append(tools, empty);
      return;
    }
    const grid = el('div', 'lib-grid');
    const items: HTMLElement[] = [];
    let vis: HTMLElement[] = items; // the cards the search leaves visible; navigation runs over these
    let sel = -1; // index into vis
    const select = (i: number, scroll = true) => {
      if (!vis.length) { sel = -1; return; }
      i = Math.max(0, Math.min(vis.length - 1, i));
      vis[sel]?.classList.remove('sel');
      sel = i;
      vis[sel].classList.add('sel');
      if (scroll) vis[sel].scrollIntoView({ block: 'nearest' });
    };
    const entryOf = (card: HTMLElement) => items.indexOf(card); // only on click/Enter/Delete — O(N) per user action, not per render
    const load = (card: HTMLElement) => {
      const e = entries[entryOf(card)];
      if (!e) return;
      app.flameSource = e.source ?? `library: ${e.name}`;
      app.setFlame(normalizeFlame(e.flame, app.activeLayer.palette));
      close();
    };
    const remove = (card: HTMLElement) => {
      const i = entryOf(card);
      const e = entries[i];
      if (!e) return;
      libDelete(e.id).then(() => {
        const vi = vis.indexOf(card);
        card.remove(); items.splice(i, 1); entries.splice(i, 1);
        applyFilter();
        if (vis.length) select(Math.min(Math.max(vi, 0), vis.length - 1)); else sel = -1;
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
      const prov = [e.author ? 'by ' + e.author : '', e.source ?? ''].filter(Boolean).join(' · ');
      if (prov) meta.append(el('div', 'lib-prov', prov));
      meta.title = [e.name, e.author ? 'Author: ' + e.author : '', e.source ? 'Source: ' + e.source : ''].filter(Boolean).join('\n');
      const del = el('button', 'lib-del danger', '✕');
      del.onclick = (ev) => { ev.stopPropagation(); remove(item); };
      item.append(img, meta, del);
      item.onclick = () => load(item);
      item.onmouseenter = () => select(vis.indexOf(item), false);
      items.push(item);
      grid.append(item);
    });
    const hint = el('div', 'hint');
    const applyFilter = () => {
      const q = search.value.trim().toLowerCase();
      vis[sel]?.classList.remove('sel');
      sel = -1;
      vis = [];
      for (let i = 0; i < items.length; i++) {
        const en = entries[i];
        const show = !q || en.name.toLowerCase().includes(q) || (en.author ?? '').toLowerCase().includes(q) || (en.source ?? '').toLowerCase().includes(q);
        items[i].style.display = show ? '' : 'none';
        if (show) vis.push(items[i]);
      }
      const n = entries.length;
      hint.textContent = (q ? `${vis.length} of ${n} flame${n === 1 ? '' : 's'} match` : `${n} flame${n === 1 ? '' : 's'}`) +
        ' — arrow keys / Page Up / Page Down move, Enter loads, Delete removes, Esc closes';
    };
    search.addEventListener('input', applyFilter);
    search.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') { ev.preventDefault(); if (search.value) { search.value = ''; applyFilter(); } else close(); }
      if (ev.key === 'Enter' || ev.key === 'ArrowDown') { ev.preventDefault(); body.focus(); select(0); }
      ev.stopPropagation(); // typing must not drive the grid
    });
    applyFilter();
    body.append(tools, hint, grid);
    // Keyboard navigation: the grid is a wrapping flex of fixed-width cards, so a row is
    // however many fit across the body; Page Up/Down jump by the rows the viewport shows.
    body.tabIndex = 0;
    const columns = () => {
      if (vis.length < 2) return 1;
      const y0 = vis[0].offsetTop;
      let n = 1;
      while (n < vis.length && vis[n].offsetTop === y0) n++;
      return n;
    };
    const pageRows = () => Math.max(1, Math.floor(body.clientHeight / ((vis[0]?.offsetHeight ?? 160) + 10)));
    body.addEventListener('keydown', (ev) => {
      if (ev.target === search) return;
      const cols = columns();
      const cur = sel < 0 ? 0 : sel;
      switch (ev.key) {
        case 'ArrowRight': select(cur + 1); break;
        case 'ArrowLeft': select(cur - 1); break;
        case 'ArrowDown': select(sel < 0 ? 0 : cur + cols); break;
        case 'ArrowUp': if (cur - cols < 0 && sel >= 0) { search.focus(); sel >= 0 && vis[sel].classList.remove('sel'); sel = -1; } else select(cur - cols); break;
        case 'PageDown': select(cur + cols * pageRows()); break;
        case 'PageUp': select(cur - cols * pageRows()); break;
        case 'Home': select(0); break;
        case 'End': select(vis.length - 1); break;
        case 'Enter': if (sel >= 0) load(vis[sel]); break;
        case 'Delete': case 'Backspace': if (sel >= 0) remove(vis[sel]); break;
        case 'Escape': close(); break;
        case '/': search.focus(); search.select(); break;
        default:
          if (ev.key.length === 1 && !ev.ctrlKey && !ev.metaKey && !ev.altKey) { search.focus(); return; } // start typing = search
          return;
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
