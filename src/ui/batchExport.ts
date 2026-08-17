// Batch export: render several flames (the current one and/or library entries) at several
// output sizes in one go, offscreen and tiled, into a folder (File System Access API) or as
// a series of downloads. Jobs run one after another on the single renderer; the queue can
// be cancelled between tiles.
import { App, el, openModal } from './common';
import { normalizeFlame, type Flame } from '../core/flame';
import { listLibrary } from './library';
import { pickDirectory, hasDirDialog, type DirTarget } from './saveFile';
import { renderHiRes, resolveSize, safeFileName, SIZE_OPTIONS, QUALITY_OPTIONS } from './hiresExport';

interface Job { name: string; file: string; flame: Flame; size: string; w: number; h: number; status: HTMLElement; done: boolean }

export function openBatchExport(app: App) {
  const { body, close } = openModal('Batch export');
  body.classList.add('batch-body');
  const lib = listLibrary();

  // ---- flames ----
  const flamesSec = el('div', 'batch-sec');
  flamesSec.append(el('h4', '', 'Flames'));
  const flameList = el('div', 'batch-list');
  const flameChecks: { chk: HTMLInputElement; get: () => Flame; name: () => string }[] = [];
  const row = (label: string, thumb: string | null, get: () => Flame, name: () => string, checked: boolean) => {
    const lab = el('label', 'batch-item');
    const chk = el('input') as HTMLInputElement;
    chk.type = 'checkbox';
    chk.checked = checked;
    lab.append(chk);
    if (thumb) { const img = el('img') as HTMLImageElement; img.src = thumb; lab.append(img); }
    else lab.append(el('span', 'batch-thumb-cur', '●'));
    lab.append(el('span', 'batch-name', label));
    flameList.append(lab);
    flameChecks.push({ chk, get, name });
    chk.onchange = refresh;
  };
  row(`Current flame (${app.flame.name || 'untitled'})`, null, () => app.flame, () => app.flame.name || 'untitled', true);
  for (const e of lib) row(e.name, e.thumb, () => normalizeFlame(e.flame, app.activeLayer.palette), () => e.name, false);
  const allRow = el('div', 'btn-row');
  const allBtn = el('button', '', 'Select all');
  const noneBtn = el('button', '', 'None');
  allBtn.onclick = () => { flameChecks.forEach((f) => { f.chk.checked = true; }); refresh(); };
  noneBtn.onclick = () => { flameChecks.forEach((f) => { f.chk.checked = false; }); refresh(); };
  allRow.append(allBtn, noneBtn);
  if (!lib.length) flamesSec.append(el('div', 'hint', 'Save flames to the library (💾) to export several at once.'));
  flamesSec.append(flameList, allRow);

  // ---- sizes / quality ----
  const sizeSec = el('div', 'batch-sec');
  sizeSec.append(el('h4', '', 'Sizes'));
  const sizeRow = el('div', 'batch-sizes');
  const sizeChecks: { chk: HTMLInputElement; value: string }[] = [];
  const r = app.renderer;
  for (const s of SIZE_OPTIONS) {
    const lab = el('label', 'batch-size');
    const chk = el('input') as HTMLInputElement;
    chk.type = 'checkbox';
    chk.checked = s.value === '2';
    const { w, h } = resolveSize(s.value, r.width, r.height);
    lab.append(chk, el('span', '', ` ${s.label}`), el('span', 'hint', ` ${w}×${h}`));
    lab.title = /x/.test(s.value) ? 'Fixed 16:9 frame (the flame is scaled to the width)' : `${s.value}× the canvas`;
    sizeRow.append(lab);
    sizeChecks.push({ chk, value: s.value });
    chk.onchange = refresh;
  }
  const optRow = el('div', 'btn-row');
  const qSel = el('select') as HTMLSelectElement;
  for (const q of QUALITY_OPTIONS) { const o = el('option', '', q.label) as HTMLOptionElement; o.value = q.value; if (q.value === '700') o.selected = true; qSel.append(o); }
  qSel.title = 'Samples per pixel';
  const alphaChk = el('input') as HTMLInputElement;
  alphaChk.type = 'checkbox';
  const alphaLab = el('label', '', ' alpha');
  alphaLab.prepend(alphaChk);
  alphaLab.title = 'Transparent background';
  optRow.append(el('span', 'hint', 'Quality '), qSel, alphaLab);
  sizeSec.append(sizeRow, optRow);

  // ---- queue ----
  const queueSec = el('div', 'batch-sec');
  queueSec.append(el('h4', '', 'Queue'));
  const queue = el('div', 'batch-queue');
  const summary = el('div', 'hint');
  const foot = el('div', 'btn-row');
  const startBtn = el('button', 'primary', hasDirDialog() ? '⬇ Choose folder & start' : '⬇ Start (downloads)');
  const cancelBtn = el('button', '', 'Cancel');
  cancelBtn.disabled = true;
  foot.append(startBtn, cancelBtn);
  queueSec.append(queue, summary, foot);
  body.append(flamesSec, sizeSec, queueSec);

  let jobs: Job[] = [];
  let running = false;
  let abort: AbortController | null = null;
  const buildJobs = (): Job[] => {
    const out: Job[] = [];
    const flames = flameChecks.filter((f) => f.chk.checked);
    const sizes = sizeChecks.filter((s) => s.chk.checked).map((s) => s.value);
    for (const f of flames) {
      let flame: Flame | null = null;
      for (const size of sizes) {
        const { w, h } = resolveSize(size, r.width, r.height);
        out.push({ name: f.name(), file: '', flame: (flame ??= f.get()), size, w, h, status: el('span', 'batch-status', 'queued'), done: false });
      }
    }
    // file names: <name>-<WxH>.png, numbered when two selected flames share a name
    const seen = new Map<string, number>();
    for (const j of out) {
      const base = `${safeFileName(j.name)}-${j.w}x${j.h}`;
      const n = (seen.get(base) ?? 0) + 1;
      seen.set(base, n);
      j.file = `${base}${n > 1 ? `-${n}` : ''}.png`;
    }
    return out;
  };
  const fileNameOf = (j: Job) => j.file;
  function refresh() {
    if (running) return;
    jobs = buildJobs();
    queue.textContent = '';
    for (const j of jobs) {
      const line = el('div', 'batch-job');
      line.append(el('span', 'batch-file', fileNameOf(j)), j.status);
      queue.append(line);
    }
    const mp = jobs.reduce((a, j) => a + j.w * j.h, 0) / 1e6;
    summary.textContent = jobs.length ? `${jobs.length} image${jobs.length > 1 ? 's' : ''}, ${mp.toFixed(0)} Mpixel in total.` : 'Pick at least one flame and one size.';
    startBtn.disabled = !jobs.length;
  }
  refresh();

  startBtn.onclick = async () => {
    if (running || !jobs.length) return;
    // the folder dialog needs the click's user gesture: ask first
    const target: DirTarget | null = await pickDirectory();
    if (!target) return;
    running = true;
    abort = new AbortController();
    startBtn.disabled = true;
    cancelBtn.disabled = false;
    for (const c of [...flameChecks.map((f) => f.chk), ...sizeChecks.map((s) => s.chk), qSel, alphaChk, allBtn, noneBtn]) c.disabled = true;
    const spp = parseInt(qSel.value);
    const transparent = alphaChk.checked;
    if (app.solo) app.setSolo(false); // exports always render the whole flame
    r.exporting = true;
    let ok = 0, failed = 0;
    const t0 = performance.now();
    try {
      for (const j of jobs) {
        if (abort.signal.aborted) { j.status.textContent = 'cancelled'; continue; }
        j.status.textContent = 'rendering…';
        j.status.classList.add('busy');
        try {
          const blob = await renderHiRes(r, j.flame, {
            w: j.w, h: j.h, spp, transparent, signal: abort.signal,
            onTile: (d, n) => { j.status.textContent = n > 1 ? `tile ${d}/${n}` : 'rendering…'; },
          });
          j.status.textContent = 'saving…';
          await target.write(fileNameOf(j), blob);
          j.status.textContent = `✓ ${(blob.size / 1e6).toFixed(1)} MB`;
          j.done = true;
          ok++;
        } catch (e) {
          if ((e as DOMException)?.name === 'AbortError') { j.status.textContent = 'cancelled'; }
          else { j.status.textContent = '⚠ ' + (e as Error).message; failed++; }
        }
        j.status.classList.remove('busy');
        summary.textContent = `${ok}/${jobs.length} saved${failed ? `, ${failed} failed` : ''}${target.kind === 'dir' ? ` → ${target.name}/` : ''} · ${((performance.now() - t0) / 1000).toFixed(0)} s`;
      }
    } finally {
      r.exporting = false;
      r.setFlame(app.flame);
      running = false;
      abort = null;
      cancelBtn.disabled = true;
      for (const c of [...flameChecks.map((f) => f.chk), ...sizeChecks.map((s) => s.chk), qSel, alphaChk, allBtn, noneBtn]) c.disabled = false;
      startBtn.disabled = false;
      startBtn.textContent = '⬇ Run again';
    }
  };
  cancelBtn.onclick = () => { abort?.abort(); cancelBtn.disabled = true; };
  // closing the dialog (✕ or backdrop) cancels a running batch after the current tile
  const box = body.parentElement!, backdrop = box.parentElement!;
  box.querySelector('.modal-head button')?.addEventListener('click', () => abort?.abort());
  backdrop.addEventListener('pointerdown', (e) => { if (e.target === backdrop) abort?.abort(); });
  return { close };
}
