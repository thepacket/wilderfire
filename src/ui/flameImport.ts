// Loading flames from a file — the Render tab's ⬆ Load button and drag-and-drop onto the canvas.
// One flame is loaded into the editor; a pack (a JWildfire .flame file holding many flames) opens
// a chooser that can add every flame to the library (see ./library.ts).
import { App, el, openModal } from './common';
import { importFlameText } from '../core/flameXML';
import type { Flame } from '../core/flame';

/** Report the variations an imported file used that this renderer has no port for. */
async function noteUnknown(unknown: string[]) {
  if (!unknown.length) return;
  // deliberately unported JWildfire variations carry a reason (src/core/variations.unportable.ts)
  const { UNPORTABLE } = await import('../core/variations.unportable');
  const lines = unknown.map((n) => (UNPORTABLE[n] ? `${n} — ${UNPORTABLE[n]}` : n));
  console.warn(`Unsupported variations skipped: ${unknown.join(', ')}`);
  alert(`Loaded, but ${unknown.length} variation${unknown.length > 1 ? 's are' : ' is'} not supported and ${unknown.length > 1 ? 'were' : 'was'} skipped:\n${lines.join('\n')}`);
}

/** JWildfire ships flame packs as one file holding many flames: pick one, or add them all to the library. */
export function openPackChooser(app: App, flames: Flame[], unknown: string[], autoAdd = false) {
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
      });
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
    item.onclick = () => { app.setFlame(fl); close(); void noteUnknown(unknown); };
    grid.append(item);
  });
  body.append(tools, status, grid);
  if (autoAdd) addBtn.click(); // dropping a pack goes straight to the library
}

/** Every flame inside a .zip (flame packs are distributed zipped): each .flame/.flames entry is
 *  parsed, unnamed flames are named after their entry, and the lot is returned as one pack. */
async function flamesFromZip(app: App, file: File): Promise<{ flames: Flame[]; unknown: string[]; skipped: number }> {
  const { readZip } = await import('../core/zip');
  const entries = readZip(await file.arrayBuffer()).filter((e) => /\.flames?$/i.test(e.name));
  if (!entries.length) throw new Error('no .flame files inside this zip');
  const flames: Flame[] = [];
  const unknown = new Set<string>();
  let skipped = 0;
  for (const e of entries) {
    try {
      const r = importFlameText(await e.text(), app.activeLayer.palette);
      const base = e.name.replace(/^.*\//, '').replace(/\.[^.]+$/, '');
      r.flames.forEach((fl, i) => { if (!fl.name || fl.name === 'imported') fl.name = r.flames.length > 1 ? `${base} ${i + 1}` : base; });
      flames.push(...r.flames);
      for (const u of r.unknown) unknown.add(u);
    } catch (err) {
      skipped++;
      console.warn(`zip entry ${e.name}: ${(err as Error).message}`);
    }
  }
  if (!flames.length) throw new Error(`none of the ${entries.length} .flame entries could be read`);
  return { flames, unknown: [...unknown], skipped };
}

/** Load a .flame / .json / .zip file: one flame into the editor, a pack into the chooser.
 *  `autoAdd` starts a pack's library import right away (drag-and-drop). */
export async function importFlameFile(app: App, file: File, autoAdd = false): Promise<void> {
  try {
    if (/\.zip$/i.test(file.name) || file.type === 'application/zip') {
      const { flames, unknown, skipped } = await flamesFromZip(app, file);
      if (skipped) console.info(`${skipped} entr${skipped === 1 ? 'y' : 'ies'} in ${file.name} could not be read (see warnings above).`);
      if (flames.length === 1) { app.setFlame(flames[0]); await noteUnknown(unknown); return; }
      openPackChooser(app, flames, unknown, autoAdd);
      return;
    }
    if (/\.(rar|7z)$/i.test(file.name)) throw new Error(`${file.name}: only .zip archives can be opened in the browser — unpack it first`);
    const text = await file.text();
    // A whole-library export dropped by mistake — merge it instead of failing on the outer object.
    if (/^\s*\{/.test(text) && text.includes('"wilderfireLibrary"')) {
      const { libPut } = await import('../core/libraryStore');
      const entries = JSON.parse(text).entries;
      if (!Array.isArray(entries) || !entries.length) throw new Error('no library entries in this file');
      await libPut(entries);
      console.info(`Merged ${entries.length} library entries from ${file.name}.`);
      return;
    }
    const { flame, flames, count, unknown, curves } = importFlameText(text, app.activeLayer.palette);
    if (count > 1) {
      // Packs often carry unnamed flames (the importer's placeholder) — name those after the file.
      const base = file.name.replace(/\.[^.]+$/, '');
      flames.forEach((fl, i) => { if (!fl.name || fl.name === 'imported') fl.name = `${base} ${i + 1}`; });
      openPackChooser(app, flames, unknown, autoAdd);
      return;
    }
    app.setFlame(flame);
    if (curves.length) {
      app.setCurves(curves);
      console.info(`Loaded ${curves.length} motion curve${curves.length > 1 ? 's' : ''} from the file (Anim tab).`);
    }
    await noteUnknown(unknown);
  } catch (e) {
    alert('Could not import flame: ' + (e as Error).message);
  }
}

/** Drag a .flame / .json onto the canvas to load it; a pack goes into the library. */
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
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    e.preventDefault();
    void importFlameFile(app, file, true);
  });
}
