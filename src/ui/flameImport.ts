// Loading flames from a file — the Render tab's ⬆ Load button and drag-and-drop onto the canvas.
// One flame is loaded into the editor; a pack (a JWildfire .flame file holding many flames) opens
// a chooser that can add every flame to the library (see ./library.ts).
import { App, el, openModal } from './common';
import { importFlameText } from '../core/flameXML';
import type { Flame } from '../core/flame';
import type { MotionCurve } from '../core/motion';

/** Report the variations an imported file used that this renderer has no port for. */
async function noteUnknown(unknown: string[]) {
  if (!unknown.length) return;
  // deliberately unported JWildfire variations carry a reason (src/core/variations.unportable.ts)
  const { UNPORTABLE } = await import('../core/variations.unportable');
  const lines = unknown.map((n) => (UNPORTABLE[n] ? `${n} — ${UNPORTABLE[n]}` : n));
  console.warn(`Unsupported variations skipped: ${unknown.join(', ')}`);
  alert(`Loaded, but ${unknown.length} variation${unknown.length > 1 ? 's are' : ' is'} not supported and ${unknown.length > 1 ? 'were' : 'was'} skipped:\n${lines.join('\n')}`);
}

export interface FlameItem { flame: Flame; source?: string }

/** JWildfire ships flame packs as one file holding many flames: pick one, or add them all to the library. */
export function openPackChooser(app: App, items: FlameItem[], unknown: string[], autoAdd = false) {
  const flames = items.map((it) => it.flame);
  const { body, close } = openModal(`Flame pack — ${flames.length} flames`);
  const tools = el('div', 'btn-row');
  const addBtn = el('button', 'primary', `＋ Add all ${flames.length} to the library`) as HTMLButtonElement;
  const status = el('div', 'hint', 'Click a name to load one flame.');
  let stop = false;
  addBtn.onclick = async () => {
    const { addFlamesToLibrary } = await import('./library');
    addBtn.disabled = true;
    const cancel = el('button', 'danger', 'Stop');
    cancel.onclick = () => { stop = true; };
    tools.append(cancel);
    try {
      const n = await addFlamesToLibrary(app, flames, (done, total, name) => {
        status.textContent = `Rendering thumbnails… ${done}/${total} — ${name}`;
        return !stop;
      }, items.map((it) => it.source));
      status.textContent = `Added ${n} flame${n === 1 ? '' : 's'} to the library.`;
    } catch (e) {
      status.textContent = 'Could not add to the library: ' + (e as Error).message;
    }
    cancel.remove();
    addBtn.disabled = false;
  };
  tools.append(addBtn);
  const grid = el('div', 'ugr-grid');
  flames.forEach((fl, i) => {
    const item = el('div', 'ugr-item');
    item.append(el('span', 'ugr-name', `${i + 1}. ${fl.name || 'untitled'}`));
    item.onclick = () => { app.flameSource = items[i].source; app.setFlame(fl); close(); void noteUnknown(unknown); };
    grid.append(item);
  });
  body.append(tools, status, grid);
  if (autoAdd) addBtn.click(); // dropping a pack goes straight to the library
}

/** Every flame inside a .zip (flame packs are distributed zipped): each .flame/.flames entry is
 *  parsed, unnamed flames are named after their entry, and the lot is returned as one pack. */
async function flamesFromZip(app: App, file: File): Promise<{ items: FlameItem[]; unknown: string[]; skipped: number }> {
  const { readZip } = await import('../core/zip');
  const entries = readZip(await file.arrayBuffer()).filter((e) => /\.flames?$/i.test(e.name));
  if (!entries.length) throw new Error('no .flame files inside this zip');
  const items: FlameItem[] = [];
  const unknown = new Set<string>();
  let skipped = 0;
  for (const e of entries) {
    try {
      const r = importFlameText(await e.text(), app.activeLayer.palette);
      const base = e.name.replace(/^.*\//, '').replace(/\.[^.]+$/, '');
      r.flames.forEach((fl, i) => { if (!fl.name || fl.name === 'imported') fl.name = r.flames.length > 1 ? `${base} ${i + 1}` : base; });
      const source = `${file.name} › ${e.name}`;
      items.push(...r.flames.map((flame) => ({ flame, source })));
      for (const u of r.unknown) unknown.add(u);
    } catch (err) {
      skipped++;
      console.warn(`zip entry ${e.name}: ${(err as Error).message}`);
    }
  }
  if (!items.length) throw new Error(`none of the ${entries.length} .flame entries could be read`);
  return { items, unknown: [...unknown], skipped };
}

/** Folder drops hand us Files without their path; filesFromDrop records each entry's full path here. */
const pathOf = new WeakMap<File, string>();

/** Signals "this picture became the gradient" up to importFlameFiles (not an error for the user). */
class GradientTaken extends Error { constructor(public file: string) { super(`gradient taken from ${file}`); } }

/** The flames inside one file (.flame / .flames / .json / .zip). A whole-library export JSON is
 *  merged into the library on the spot and contributes no flames. */
async function flamesFromFile(app: App, file: File): Promise<{ items: FlameItem[]; unknown: string[]; curves: MotionCurve[] }> {
  if (/\.zip$/i.test(file.name) || file.type === 'application/zip') {
    const { items, unknown, skipped } = await flamesFromZip(app, file);
    if (skipped) console.info(`${skipped} entr${skipped === 1 ? 'y' : 'ies'} in ${file.name} could not be read (see warnings above).`);
    return { items, unknown, curves: [] };
  }
  if (/\.(rar|7z)$/i.test(file.name)) throw new Error(`${file.name}: only .zip archives can be opened in the browser — unpack it first`);
  let text: string;
  const isImage = /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(file.name) || /^image\//.test(file.type);
  if (isImage) {
    // a PNG saved by WilderFire (or flam3) carries its flame in a flam3_genome text chunk;
    // any other picture becomes the active layer's gradient (JWildfire's grab-palette-from-image)
    const isPng = /\.png$/i.test(file.name) || file.type === 'image/png';
    const xml = isPng ? (await import('../core/pngMeta')).flameXmlFromPng(new Uint8Array(await file.arrayBuffer())) : null;
    if (!xml) {
      const { paletteFromImageFile } = await import('../core/paletteFromImage');
      app.applyPalette(await paletteFromImageFile(file));
      throw new GradientTaken(file.name);
    }
    text = xml;
  } else {
    text = await file.text();
  }
  if (/^\s*\{/.test(text) && text.includes('"wilderfireLibrary"')) {
    const { libPut } = await import('../core/libraryStore');
    const entries = JSON.parse(text).entries;
    if (!Array.isArray(entries) || !entries.length) throw new Error('no library entries in this file');
    await libPut(entries);
    console.info(`Merged ${entries.length} library entries from ${file.name}.`);
    return { items: [], unknown: [], curves: [] };
  }
  const { flames, count, unknown, curves } = importFlameText(text, app.activeLayer.palette);
  // Packs often carry unnamed flames (the importer's placeholder) — name those after the file.
  const base = file.name.replace(/\.[^.]+$/, '');
  flames.forEach((fl, i) => { if (!fl.name || fl.name === 'imported') fl.name = count > 1 ? `${base} ${i + 1}` : base; });
  const source = pathOf.get(file) ?? file.name;
  return { items: flames.map((flame) => ({ flame, source })), unknown, curves };
}

/** Load any number of files: one flame in total goes into the editor (with its motion curves),
 *  more open the pack chooser. `autoAdd` starts the library import right away (drag-and-drop). */
export async function importFlameFiles(app: App, files: File[], autoAdd = false): Promise<void> {
  const all: FlameItem[] = [];
  const unknown = new Set<string>();
  let curves: MotionCurve[] = [];
  const failed: string[] = [];
  for (const f of files) {
    const picture = /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(f.name) || /^image\//.test(f.type);
    if (!/\.(flame|flames|json|xml|zip|rar|7z)$/i.test(f.name) && f.type !== 'application/zip' && !picture) continue; // readmes…
    if (picture && files.length > 1 && !/\.png$/i.test(f.name) && f.type !== 'image/png') { console.info(`${f.name}: skipped (a picture among several files — drop it alone to make a gradient from it)`); continue; }
    try {
      const r = await flamesFromFile(app, f);
      all.push(...r.items);
      for (const u of r.unknown) unknown.add(u);
      if (r.curves.length && files.length === 1) curves = r.curves;
    } catch (e) {
      if (e instanceof GradientTaken) {
        if (files.length === 1) { console.info(`Gradient taken from ${f.name} (Gradient tab).`); return; }
        console.info(`${f.name}: a pack's preview image — skipped`); // several files: pictures are not gradients
        continue;
      }
      failed.push(`${f.name}: ${(e as Error).message}`);
    }
  }
  if (failed.length) {
    console.warn('Not imported:\n' + failed.join('\n'));
    if (!all.length) { alert('Could not import:\n' + failed.slice(0, 8).join('\n')); return; }
    if (failed.length === files.length) return;
  }
  if (!all.length) return; // e.g. only a library export — already merged
  if (all.length === 1) {
    app.flameSource = all[0].source;
    app.setFlame(all[0].flame);
    if (curves.length) {
      app.setCurves(curves);
      console.info(`Loaded ${curves.length} motion curve${curves.length > 1 ? 's' : ''} from the file (Anim tab).`);
    }
    await noteUnknown([...unknown]);
    return;
  }
  openPackChooser(app, all, [...unknown], autoAdd);
}

/** One file — the ⬆ Load button's single-file path. */
export const importFlameFile = (app: App, file: File, autoAdd = false) => importFlameFiles(app, [file], autoAdd);

/** Every file under a dropped item, recursing into folders (Chrome/Safari/Firefox expose
 *  dropped directories through the FileSystem Entry API; plain files come straight through). */
async function filesFromDrop(dt: DataTransfer): Promise<File[]> {
  const items = Array.from(dt.items ?? []);
  const entries = items.map((i) => (i.kind === 'file' && 'webkitGetAsEntry' in i ? i.webkitGetAsEntry() : null));
  if (!entries.some((e) => e)) return Array.from(dt.files ?? []);
  const out: File[] = [];
  const readDir = (dir: FileSystemDirectoryEntry): Promise<FileSystemEntry[]> => new Promise((res, rej) => {
    const reader = dir.createReader();
    const acc: FileSystemEntry[] = [];
    const step = () => reader.readEntries((batch) => { if (!batch.length) res(acc); else { acc.push(...batch); step(); } }, rej); // readEntries returns ≤ 100 per call
    step();
  });
  const walk = async (e: FileSystemEntry | null): Promise<void> => {
    if (!e) return;
    if (e.isFile) {
      const f = await new Promise<File>((res, rej) => (e as FileSystemFileEntry).file(res, rej));
      pathOf.set(f, e.fullPath.replace(/^\//, ''));
      out.push(f);
    } else if (e.isDirectory) {
      for (const child of await readDir(e as FileSystemDirectoryEntry)) await walk(child);
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

/** Drag .flame / .json / .zip files — or whole folders — onto the canvas; a pack goes into the library. */
export function enableFlameDrop(app: App, target: HTMLElement) {
  const dragged = (e: DragEvent) => Array.from(e.dataTransfer?.items ?? []).some((i) => i.kind === 'file');
  let depth = 0; // dragenter/leave also fire for children (the overlay canvas, the status bar)
  target.addEventListener('dragenter', (e) => {
    if (!dragged(e)) return;
    if (++depth === 1) target.classList.add('drop-over');
  });
  target.addEventListener('dragover', (e) => {
    if (!dragged(e)) return;
    e.preventDefault(); // without this the browser navigates to the file
    e.dataTransfer!.dropEffect = 'copy';
  });
  target.addEventListener('dragleave', () => {
    if (depth > 0 && --depth === 0) target.classList.remove('drop-over');
  });
  target.addEventListener('drop', (e) => {
    depth = 0;
    target.classList.remove('drop-over');
    if (!e.dataTransfer) return;
    e.preventDefault();
    const dt = e.dataTransfer;
    void filesFromDrop(dt).then((files) => { if (files.length) return importFlameFiles(app, files, true); });
  });
}
