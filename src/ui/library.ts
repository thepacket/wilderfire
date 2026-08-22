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
      const list = vis.slice();
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
    // ---- Virtualised grid: only the rows in view exist as DOM; everything else is arithmetic ----
    // Cards are absolutely positioned in a spacer whose height is rows × rowH, so the scrollbar
    // behaves as if every card were there. ~40–60 cards live at any time, whatever the library size.
    const GAP = 10, MIN_W = 140;
    const grid = el('div', 'lib-grid lib-virtual');
    let vis: LibEntry[] = entries;   // the search result, in display order
    let sel = -1;                    // index into vis
    let cols = 1, cardW = MIN_W, rowH = MIN_W + 66;
    const cards = new Map<string, HTMLElement>(); // entry id → rendered card
    const idx = new Map<string, number>();        // entry id → index in vis
    const hint = el('div', 'hint');
    const gridTop = () => grid.getBoundingClientRect().top - body.getBoundingClientRect().top + body.scrollTop;
    const rowOf = (i: number) => Math.floor(i / cols);
    const rows = () => Math.ceil(vis.length / cols);
    const layout = () => {
      const w = grid.clientWidth || body.clientWidth - 28;
      cols = Math.max(1, Math.floor((w + GAP) / (MIN_W + GAP)));
      cardW = (w - GAP * (cols - 1)) / cols;
      const probe = cards.values().next().value as HTMLElement | undefined;
      rowH = probe ? probe.offsetHeight : Math.round(cardW) + 66;
      grid.style.height = `${Math.max(0, rows() * (rowH + GAP) - GAP)}px`;
    };
    const makeCard = (e: LibEntry, i: number): HTMLElement => {
      const item = el('div', 'lib-item');
      item.dataset.id = e.id;
      const img = el('img') as HTMLImageElement;
      if (e.thumb) img.src = e.thumb; else img.alt = e.name;
      const meta = el('div', 'lib-meta');
      const prov = [e.author ? 'by ' + e.author : '', e.source ?? ''].filter(Boolean).join(' · ');
      meta.append(el('div', 'lib-name', e.name), el('div', 'lib-date', new Date(e.date).toLocaleString()), el('div', 'lib-prov', prov || '\u00a0'));
      meta.title = [e.name, e.author ? 'Author: ' + e.author : '', e.source ? 'Source: ' + e.source : ''].filter(Boolean).join('\n');
      const del = el('button', 'lib-del danger', '✕');
      del.onclick = (ev) => { ev.stopPropagation(); remove(e); };
      item.append(img, meta, del);
      item.onclick = () => load(e);
      item.onmouseenter = () => select(idx.get(e.id) ?? i, false);
      return item;
    };
    const place = (card: HTMLElement, i: number) => {
      card.style.left = `${(i % cols) * (cardW + GAP)}px`;
      card.style.top = `${rowOf(i) * (rowH + GAP)}px`;
      card.style.width = `${cardW}px`;
      card.classList.toggle('sel', i === sel);
    };
    const render = () => {
      if (!vis.length) { for (const c of cards.values()) c.remove(); cards.clear(); return; }
      const top = body.scrollTop - gridTop();
      const r0 = Math.max(0, Math.floor(top / (rowH + GAP)) - 1);
      const r1 = Math.min(rows() - 1, Math.ceil((top + body.clientHeight) / (rowH + GAP)) + 1);
      const i0 = r0 * cols, i1 = Math.min(vis.length, (r1 + 1) * cols);
      const keep = new Set<string>();
      for (let i = i0; i < i1; i++) {
        const e = vis[i];
        keep.add(e.id);
        let card = cards.get(e.id);
        if (!card) { card = makeCard(e, i); cards.set(e.id, card); grid.append(card); }
        place(card, i);
      }
      for (const [id, card] of cards) if (!keep.has(id)) { card.remove(); cards.delete(id); }
      // the first real card tells the true row height; re-layout once if the guess was off
      const probe = cards.values().next().value as HTMLElement | undefined;
      if (probe && Math.abs(probe.offsetHeight - rowH) > 1) { layout(); for (const [id, card] of cards) place(card, idx.get(id)!); }
    };
    let renderQueued = false;
    const scheduleRender = () => { if (renderQueued) return; renderQueued = true; setTimeout(() => { renderQueued = false; render(); }, 0); };
    const select = (i: number, scroll = true) => {
      if (!vis.length) { sel = -1; return; }
      i = Math.max(0, Math.min(vis.length - 1, i));
      const prev = sel; sel = i;
      if (prev >= 0) cards.get(vis[prev]?.id ?? '')?.classList.remove('sel');
      if (scroll) {
        const rowTop = gridTop() + rowOf(i) * (rowH + GAP);
        if (rowTop < body.scrollTop) body.scrollTop = rowTop;
        else if (rowTop + rowH > body.scrollTop + body.clientHeight) body.scrollTop = rowTop + rowH - body.clientHeight;
      }
      render();
      cards.get(vis[i].id)?.classList.add('sel');
    };
    const load = (e: LibEntry) => {
      app.flameSource = e.source ?? `library: ${e.name}`;
      app.setFlame(normalizeFlame(e.flame, app.activeLayer.palette));
      close();
    };
    const remove = (e: LibEntry) => {
      libDelete(e.id).then(() => {
        const vi = idx.get(e.id) ?? -1;
        const ei = entries.findIndex((x) => x.id === e.id);
        if (ei >= 0) entries.splice(ei, 1);
        cards.get(e.id)?.remove(); cards.delete(e.id);
        applyFilter();
        if (vis.length) select(Math.min(Math.max(vi, 0), vis.length - 1)); else sel = -1;
      }).catch((err) => alert('Delete failed: ' + (err as Error).message));
    };
    const applyFilter = () => {
      const q = search.value.trim().toLowerCase();
      sel = -1;
      vis = q ? entries.filter((en) => en.name.toLowerCase().includes(q) || (en.author ?? '').toLowerCase().includes(q) || (en.source ?? '').toLowerCase().includes(q)) : entries;
      idx.clear();
      vis.forEach((en, i) => idx.set(en.id, i));
      for (const [id, card] of cards) if (!idx.has(id)) { card.remove(); cards.delete(id); }
      const n = entries.length;
      hint.textContent = (q ? `${vis.length} of ${n} flame${n === 1 ? '' : 's'} match` : `${n} flame${n === 1 ? '' : 's'}`) +
        ' — arrow keys / Page Up / Page Down move, Enter loads, Delete removes, Esc closes';
      layout();
      render();
    };
    search.addEventListener('input', applyFilter);
    search.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') { ev.preventDefault(); if (search.value) { search.value = ''; applyFilter(); } else close(); }
      if (ev.key === 'Enter' || ev.key === 'ArrowDown') { ev.preventDefault(); body.focus(); select(0); }
      ev.stopPropagation(); // typing must not drive the grid
    });
    body.append(tools, hint, grid);
    body.addEventListener('scroll', scheduleRender, { passive: true });
    new ResizeObserver(() => { layout(); render(); }).observe(grid);
    applyFilter();
    // Keyboard navigation over the visible (filtered) list; Page Up/Down jump by the rows the viewport shows.
    body.tabIndex = 0;
    const pageRows = () => Math.max(1, Math.floor(body.clientHeight / (rowH + GAP)));
    body.addEventListener('keydown', (ev) => {
      if (ev.target === search) return;
      const cur = sel < 0 ? 0 : sel;
      switch (ev.key) {
        case 'ArrowRight': select(cur + 1); break;
        case 'ArrowLeft': select(cur - 1); break;
        case 'ArrowDown': select(sel < 0 ? 0 : cur + cols); break;
        case 'ArrowUp': if (cur - cols < 0 && sel >= 0) { search.focus(); cards.get(vis[sel]?.id ?? '')?.classList.remove('sel'); sel = -1; } else select(cur - cols); break;
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
