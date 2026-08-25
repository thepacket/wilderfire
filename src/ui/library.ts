// Flame library (IndexedDB, see ../core/libraryStore.ts) + session autosave (localStorage).
import { App, el, openModal } from './common';
import { normalizeFlame, type Flame } from '../core/flame';
import { libAll, libPut, libDelete, libDeleteMany, libClear, packOf, sampleThumbPut, type LibEntry, thumbSrc, releaseThumbSrcs, thumbDataUrl, type Thumb } from '../core/libraryStore';
import { sampleEntries, isSample, SAMPLE_COUNT } from './sampleLib';
import { flameSignature, rankSimilar, type FlameSig } from '../core/similarity';

/** Signatures cached per entry id (a library entry's flame never changes; a re-save makes a new id). */
const sigCache = new Map<string, FlameSig>();
export const sigOf = (e: LibEntry): FlameSig => { let s = sigCache.get(e.id); if (!s) { s = flameSignature(e.flame); sigCache.set(e.id, s); } return s; };
import { saveText } from './saveFile';
import type { AnimAPI } from './animPanel';

export type { LibEntry };
const LS_AUTOSAVE = 'wilderfire.autosave';

/** The saved flames, newest first (for the batch export queue). */
export const listLibrary = (): Promise<LibEntry[]> => libAll();

/** Render a square JPEG thumbnail for a flame the renderer isn't currently showing (pack import). */
async function offscreenThumb(app: App, flame: Flame, size = 144, spp = 150): Promise<Blob> {
  app.renderer.setFlame(flame);
  const px = await app.renderer.renderRegion({ fullW: size, fullH: size, tileX: 0, tileY: 0, tileW: size, tileH: size, spp });
  const c = document.createElement('canvas');
  c.width = c.height = size;
  c.getContext('2d')!.putImageData(new ImageData(px, size, size), 0, 0);
  return jpegBlob(c);
}
/** A canvas as a JPEG Blob (what the library stores). */
export const jpegBlob = (c: HTMLCanvasElement): Promise<Blob> =>
  new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('could not encode the thumbnail'))), 'image/jpeg', 0.72));

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
      let thumb: Thumb = '';
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

/** Re-render the thumbnails of library entries with the current engine (entries saved before a renderer fix, or
 *  whose picture failed). Writes are batched; `onProgress` may return false to stop — the ones done so far are kept.
 *  Returns the number of thumbnails replaced. */
export async function renderThumbnails(
  app: App,
  entries: LibEntry[],
  onProgress?: (done: number, total: number, entry: LibEntry) => boolean | void,
): Promise<number> {
  let done = 0;
  let pending: LibEntry[] = [];
  const flush = async () => { if (pending.length) { await libPut(pending); done += pending.length; pending = []; } };
  try {
    for (const [i, e] of entries.entries()) {
      try {
        e.thumb = await offscreenThumb(app, normalizeFlame(e.flame, app.activeLayer.palette));
        pending.push(e);
      } catch { /* keep the old picture */ }
      if (pending.length >= 25) await flush();
      if (onProgress?.(i + 1, entries.length, e) === false) break;
    }
  } finally {
    app.resumeRender();
  }
  await flush();
  return done;
}

export function buildLibrary(app: App, anim: AnimAPI) {
  function thumbnail(size = 144): Promise<Blob> {
    // the copy out of the WebGPU canvas must happen inside captureSync (synchronously, after the
    // present); only the JPEG encoding is asynchronous
    return app.renderer.captureSync((cv) => {
      const c = document.createElement('canvas');
      c.width = size;
      c.height = size;
      const g = c.getContext('2d')!;
      // Cover-crop the (usually non-square) render into a square thumb.
      const s = Math.min(cv.width, cv.height);
      g.drawImage(cv, (cv.width - s) / 2, (cv.height - s) / 2, s, s, 0, 0, size, size);
      return jpegBlob(c);
    });
  }

  async function save() {
    try {
      const entry: LibEntry = {
        id: Math.random().toString(36).slice(2),
        name: app.flame.name || 'untitled',
        date: Date.now(),
        flame: JSON.parse(JSON.stringify(app.flame)),
        thumb: await thumbnail(),
        ...(app.flameSource ? { source: app.flameSource } : {}),
        ...(app.flame.author ? { author: app.flame.author } : {}),
      };
      await libPut(entry);
    } catch (e) { alert('Could not save to the library: ' + (e as Error).message); }
  }

  // Where the dialog was when it was last closed, so reopening lands on the same page: the scroll
  // offset, the search text and the collection (a "≈ similar" view is transient — its target is gone,
  // so it comes back as "All flames"). Kept for the session, not stored.
  const view = { scrollTop: 0, search: '', collection: 'all' };

  async function open() {
    releaseThumbSrcs(); // object URLs of the previous dialog's pictures
    const { body, close } = openModal('Flame library');
    let entries: LibEntry[] = [];
    // The store returns newest-first (batch export wants that); the dialog shows names in order —
    // natural numeric order so "Flame 2" sorts before "Flame 10", ties oldest-first.
    const byName = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
    try { entries = (await libAll()).sort((a, b) => byName.compare(a.name, b.name) || a.date - b.date); } catch (e) { body.append(el('div', 'hint', '⚠ Library unavailable: ' + (e as Error).message)); return; }
    const tools = el('div', 'btn-row');
    const expBtn = el('button', '', '⬇ Export library');
    expBtn.title = 'Save every entry (flames + thumbnails) as one JSON file — a backup, or to move the library to another browser';
    expBtn.onclick = async () => {
      // JSON is text: thumbnails go out as data URLs (and come back in as Blobs through libPut)
      const out: LibEntry[] = [];
      for (let i = 0; i < entries.length; i += 200) out.push(...await Promise.all(entries.slice(i, i + 200).map(async (e) => ({ ...e, thumb: await thumbDataUrl(e.thumb) }))));
      await saveText(JSON.stringify({ wilderfireLibrary: 1, entries: out }), { suggestedName: 'wilderfire-library.json', description: 'WilderFire library', mime: 'application/json', ext: '.json' });
    };
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
    search.value = view.search;
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
    // Collections: ★ favourites, every tag, every source pack — one pass over the entries for the counts
    const collSel = el('select', 'lib-coll') as HTMLSelectElement;
    collSel.title = 'Show one collection: your favourites, a tag, or the pack the flames came from';
    let collection = view.collection; // refreshCollections falls back to "all" if that collection is gone
    const refreshCollections = () => {
      let favs = 0;
      const tags = new Map<string, number>();
      const packs = new Map<string, number>();
      for (const e of entries) {
        if (e.fav) favs++;
        for (const t of e.tags ?? []) tags.set(t, (tags.get(t) ?? 0) + 1);
        const pk = packOf(e);
        if (pk) packs.set(pk, (packs.get(pk) ?? 0) + 1);
      }
      const byName = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
      collSel.textContent = '';
      const add = (value: string, label: string, group?: HTMLOptGroupElement) => { const o = el('option', '', label) as HTMLOptionElement; o.value = value; (group ?? collSel).append(o); };
      add('all', `All flames (${entries.length})`);
      add('fav', `★ Favourites (${favs})`);
      add('samples', `◆ Samples (${SAMPLE_COUNT})`); // the built-in flames, see ./sampleLib.ts
      add('similar', similarTo ? `≈ Similar to: ${similarTo.name}` : '≈ Similar to the current flame');
      if (tags.size) { const g = el('optgroup') as HTMLOptGroupElement; g.label = 'Tags'; for (const [t, n] of [...tags].sort((a, b) => byName.compare(a[0], b[0]))) add('tag:' + t, `${t} (${n})`, g); collSel.append(g); }
      if (packs.size) { const g = el('optgroup') as HTMLOptGroupElement; g.label = 'Packs'; for (const [k, n] of [...packs].sort((a, b) => byName.compare(a[0], b[0]))) add('pack:' + k, `${k} (${n})`, g); collSel.append(g); }
      collSel.value = [...collSel.options].some((o) => o.value === collection) ? collection : (collection = 'all');
    };
    // The built-in flames: loaded (and their pictures rendered) the first time the collection is shown.
    let samples: LibEntry[] = [];
    let samplesLoading = false;
    const ensureSamples = async () => {
      if (samples.length || samplesLoading) return;
      samplesLoading = true;
      hint.textContent = 'Loading the built-in flames…';
      try { samples = await sampleEntries(); }
      catch (e) { hint.textContent = 'Could not load the built-in flames: ' + (e as Error).message; return; }
      finally { samplesLoading = false; }
      applyFilter();
      void renderSampleThumbs(samples.filter((e) => !e.thumb));
    };
    // One render per missing picture, cached in IndexedDB — so this costs something once and nothing
    // afterwards. It never touches the user's own entries.
    let thumbing = false;
    const renderSampleThumbs = async (list: LibEntry[]) => {
      if (thumbing || !list.length) return;
      thumbing = true;
      try {
        for (const [i, e] of list.entries()) {
          if (!body.isConnected) break; // dialog closed: stop, keep what is cached
          try {
            const thumb = await offscreenThumb(app, normalizeFlame(e.flame, app.activeLayer.palette));
            e.thumb = thumb;
            await sampleThumbPut(e.id, thumb);
          } catch { /* leave this one without a picture */ }
          hint.textContent = `Rendering the sample pictures… ${i + 1}/${list.length} — ${e.name}`;
          refreshCard(e);
        }
      } finally { thumbing = false; app.resumeRender(); }
      if (body.isConnected) applyFilter();
    };

    // "More like this": rank the library by similarity to a target — the current flame (the
    // standing option) or a selected card (the ≈ button); the 60 best show in score order.
    let similarTo: { name: string; sig: FlameSig; id?: string } | null = null;
    const similarScores = new Map<string, number>();
    const setSimilarTarget = (t: { name: string; sig: FlameSig; id?: string }) => { similarTo = t; collection = 'similar'; refreshCollections(); applyFilter(); };
    collSel.onchange = () => {
      collection = collSel.value;
      if (collection === 'similar' && !similarTo) similarTo = { name: app.flame.name || 'the current flame', sig: flameSignature(app.flame) };
      if (collection === 'samples') void ensureSamples();
      applyFilter(); body.focus();
    };
    const simBtn = el('button', '', '≈ Similar');
    simBtn.title = 'More like this: rank the library by similarity to the selected card — or, with nothing selected, to the flame in the editor (variations, palette, structure)';
    simBtn.onclick = () => {
      const e = sel >= 0 ? vis[sel] : null;
      setSimilarTarget(e ? { name: e.name, sig: sigOf(e), id: e.id } : { name: app.flame.name || 'the current flame', sig: flameSignature(app.flame) });
      body.focus();
    };
    const inCollection = (e: LibEntry) => collection === 'all' || (collection === 'fav' ? !!e.fav : collection.startsWith('tag:') ? (e.tags ?? []).includes(collection.slice(4)) : collection.startsWith('pack:') ? packOf(e) === collection.slice(5) : true);
    const tagAllBtn = el('button', '', '🏷 Tag all shown…');
    tagAllBtn.title = 'Add a tag to every flame in the current result (search + collection) — one write for the lot';
    tagAllBtn.onclick = async () => {
      if (!vis.length) return;
      const t = (prompt(`Tag to add to the ${vis.length} flame${vis.length === 1 ? '' : 's'} shown:`) ?? '').trim();
      if (!t) return;
      const changed = vis.filter((e) => !(e.tags ?? []).includes(t));
      for (const e of changed) e.tags = [...(e.tags ?? []), t];
      try { await libPut(changed); } catch (e) { alert('Could not save tags: ' + (e as Error).message); return; }
      refreshCollections(); applyFilter();
    };
    const renderBtn = el('button', '', '🎨 Render thumbnails') as HTMLButtonElement;
    renderBtn.title = 'Re-render the thumbnails of the flames shown (search + collection) with the current engine — for entries saved before a renderer fix, or whose picture is missing. Stop keeps the ones done so far';
    renderBtn.onclick = async () => {
      const list = vis.slice();
      if (!list.length) return;
      if (!confirm(`Re-render the thumbnails of the ${list.length} flame${list.length === 1 ? '' : 's'} shown?`)) return;
      if (collection === 'samples') { await renderSampleThumbs(list); return; } // cached apart from the library
      renderBtn.disabled = true;
      let stop = false;
      const stopBtn = el('button', 'danger', 'Stop');
      stopBtn.onclick = () => { stop = true; };
      renderBtn.after(stopBtn);
      const t0 = performance.now();
      try {
        const n = await renderThumbnails(app, list, (done, total, e) => {
          hint.textContent = `Rendering thumbnails… ${done}/${total} — ${e.name}`;
          refreshCard(e);
          return !stop && body.isConnected; // the dialog may have been closed meanwhile
        });
        hint.textContent = `${n} thumbnail${n === 1 ? '' : 's'} re-rendered in ${Math.round((performance.now() - t0) / 1000)} s.`;
      } catch (e) { hint.textContent = 'Could not render thumbnails: ' + (e as Error).message; }
      stopBtn.remove();
      renderBtn.disabled = false;
    };
    const tools2 = el('div', 'btn-row');
    tools.append(search, collSel, simBtn, galBtn);
    tools2.append(expBtn, impBtn, tagAllBtn, dedupBtn, renderBtn, clearBtn, impFile);
    // An empty library is not an empty dialog any more: the built-in flames are there to start from, so
    // the note explains how to fill it and the grid opens on ◆ Samples.
    let emptyNote: HTMLElement | null = null;
    if (!entries.length) {
      emptyNote = el('div', 'hint', 'Your library is empty — use 💾 Save to keep the current flame here, or drop .flame / .zip files on the canvas. Stored in your browser (IndexedDB). ');
      const link = el('a', '', 'Flame packs to get started: jwfsanctuary.club/downloads/flamepacks') as HTMLAnchorElement;
      link.href = 'https://www.jwfsanctuary.club/downloads/flamepacks/'; link.target = '_blank'; link.rel = 'noopener';
      emptyNote.append(link);
      if (collection === 'all') collection = 'samples';
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
      const smp = isSample(e);
      const item = el('div', 'lib-item');
      item.dataset.id = e.id;
      const img = el('img') as HTMLImageElement;
      if (e.thumb) img.src = thumbSrc(e.thumb); else img.alt = e.name;
      const meta = el('div', 'lib-meta');
      // a sample has no date of its own; its origin goes in that line instead
      const prov = [e.author ? 'by ' + e.author : '', smp ? '' : e.source ?? ''].filter(Boolean).join(' · ');
      const tagsEl = el('div', 'lib-tags', (e.tags ?? []).join(' · ') || '\u00a0');
      const score = similarScores.get(e.id);
      meta.append(el('div', 'lib-name', e.name), el('div', 'lib-date', score !== undefined ? `≈ ${Math.round(score * 100)} % similar` : smp ? e.source ?? 'built-in' : new Date(e.date).toLocaleString()), el('div', 'lib-prov', prov || '\u00a0'), tagsEl);
      meta.title = [e.name, e.author ? 'Author: ' + e.author : '', e.source ? 'Source: ' + e.source : '', e.tags?.length ? 'Tags: ' + e.tags.join(', ') : ''].filter(Boolean).join('\n');
      const fav = el('button', 'lib-fav' + (e.fav ? ' on' : ''), e.fav ? '★' : '☆');
      fav.title = 'Favourite (Space on the selected card)';
      fav.onclick = (ev) => { ev.stopPropagation(); toggleFav(e); };
      const tagBtn = el('button', 'lib-tag', '🏷');
      tagBtn.title = 'Edit this flame\'s tags';
      tagBtn.onclick = (ev) => { ev.stopPropagation(); editTags(e); };
      const del = el('button', 'lib-del danger', '✕');
      del.onclick = (ev) => { ev.stopPropagation(); remove(e); };
      // ★ / 🏷 / ✕ all write to the flame store, which a sample never enters
      if (smp) item.append(img, meta); else item.append(img, meta, fav, tagBtn, del);
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
    // remember the page for the next opening (every close path scrolls or filters through here)
    const saveView = () => {
      view.scrollTop = body.scrollTop;
      view.search = search.value;
      view.collection = collection === 'similar' ? 'all' : collection;
    };
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
    const refreshCard = (e: LibEntry) => {
      const old = cards.get(e.id);
      if (!old) return;
      const fresh = makeCard(e, idx.get(e.id) ?? 0);
      old.replaceWith(fresh);
      cards.set(e.id, fresh);
      place(fresh, idx.get(e.id) ?? 0);
    };
    const toggleFav = (e: LibEntry) => {
      e.fav = !e.fav;
      libPut(e).catch((err) => alert('Could not save: ' + (err as Error).message));
      refreshCollections();
      if (collection === 'fav' && !e.fav) applyFilter(); else refreshCard(e);
    };
    const editTags = (e: LibEntry) => {
      const cur = (e.tags ?? []).join(', ');
      const v = prompt(`Tags for "${e.name}" (comma separated):`, cur);
      if (v === null) return;
      const tags = [...new Set(v.split(',').map((t) => t.trim()).filter(Boolean))];
      if (tags.join('\u0000') === (e.tags ?? []).join('\u0000')) return;
      e.tags = tags;
      libPut(e).catch((err) => alert('Could not save tags: ' + (err as Error).message));
      refreshCollections();
      if (collection.startsWith('tag:') && !tags.includes(collection.slice(4))) applyFilter(); else refreshCard(e);
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
      const textHit = (en: LibEntry) => !q || en.name.toLowerCase().includes(q) || (en.author ?? '').toLowerCase().includes(q) || (en.source ?? '').toLowerCase().includes(q) || (en.tags ?? []).some((t) => t.toLowerCase().includes(q));
      similarScores.clear();
      if (collection === 'similar' && similarTo) {
        const target = similarTo;
        const pool = entries.filter((en) => en.id !== target.id && textHit(en)).map((en) => ({ item: en, sig: sigOf(en) }));
        const ranked = rankSimilar(target.sig, pool, 60);
        vis = ranked.map((r) => { similarScores.set(r.item.id, r.score); return r.item; });
      } else if (collection === 'samples') {
        vis = q ? samples.filter(textHit) : samples;
      } else {
        vis = collection === 'all' && !q ? entries : entries.filter((en) => inCollection(en) && textHit(en));
      }
      idx.clear();
      vis.forEach((en, i) => idx.set(en.id, i));
      for (const [id, card] of cards) if (!idx.has(id)) { card.remove(); cards.delete(id); }
      const n = entries.length;
      const where = collection === 'all' ? '' : ` in ${collSel.selectedOptions[0]?.label.replace(/ \(\d+\)$/, '') ?? collection}`;
      hint.textContent = (collection === 'similar' && similarTo
        ? `${vis.length} most similar to "${similarTo.name}"${q ? ` among names matching "${q}"` : ''}, best first`
        : collection === 'samples'
          ? (samplesLoading ? 'Loading the built-in flames…' : `${vis.length} built-in flame${vis.length === 1 ? '' : 's'}${q ? ' match' : ''} — WilderFire's own presets; they are not part of your library`)
          : q || collection !== 'all' ? `${vis.length} of ${n} flame${n === 1 ? '' : 's'}${where}${q ? ' match' : ''}` : `${n} flame${n === 1 ? '' : 's'}`) +
        (collection === 'samples' ? ' — arrows move, Enter loads, Esc closes' : ' — arrows move, Enter loads, Space ★, Delete removes, Esc closes');
      layout();
      render();
      saveView();
    };
    search.addEventListener('input', applyFilter);
    search.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') { ev.preventDefault(); if (search.value) { search.value = ''; applyFilter(); } else close(); }
      if (ev.key === 'Enter' || ev.key === 'ArrowDown') { ev.preventDefault(); body.focus(); select(0); }
      ev.stopPropagation(); // typing must not drive the grid
    });
    body.append(tools, tools2, ...(emptyNote ? [emptyNote] : []), hint, grid);
    body.addEventListener('scroll', () => { saveView(); scheduleRender(); }, { passive: true });
    new ResizeObserver(() => { layout(); render(); }).observe(grid);
    refreshCollections();
    if (collection === 'samples') void ensureSamples();
    const wanted = view.scrollTop; // applyFilter saves the (still zero) scroll, so keep it first
    applyFilter();
    // back to the page the dialog was closed on (twice: the row height is only exact once real cards
    // exist, so the grid — and with it the scrollable range — settles one frame later)
    if (wanted > 0) {
      const restore = () => { body.scrollTop = wanted; render(); };
      restore();
      requestAnimationFrame(restore);
    }
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
        case ' ': if (sel >= 0 && !isSample(vis[sel])) toggleFav(vis[sel]); break;
        case 'Delete': case 'Backspace': if (sel >= 0 && !isSample(vis[sel])) remove(vis[sel]); break;
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
