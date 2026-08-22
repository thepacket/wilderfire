// Batch export: render several flames (the current one and/or library entries) at several
// output sizes in one go, offscreen and tiled, into a folder (File System Access API) or as
// a series of downloads. Jobs run one after another on the single renderer; the queue can
// be cancelled between tiles.
import { thumbSrc, type Thumb } from '../core/libraryStore';
import { App, el, openModal } from './common';
import { normalizeFlame, type Flame } from '../core/flame';
import { listLibrary } from './library';
import { pickDirectory, hasDirDialog, type DirTarget } from './saveFile';
import { renderHiRes, resolveSize, safeFileName, SIZE_OPTIONS, QUALITY_OPTIONS } from './hiresExport';
import { renderVideo, videoFileExt, VIDEO_SIZE_OPTIONS, VIDEO_QUALITY_OPTIONS, type VideoFormat } from './videoExport';

interface Job {
  name: string; file: string; status: HTMLElement; done: boolean;
  /** still: flame + size */
  flame?: Flame; size?: string; w: number; h: number;
  /** animation job (the current timeline) */
  video?: { format: VideoFormat; fps: number; passes: number; size?: { w: number; h: number } };
}

export async function openBatchExport(app: App) {
  const { body, close } = openModal('Batch export');
  body.classList.add('batch-body');
  let lib: Awaited<ReturnType<typeof listLibrary>> = [];
  try { lib = await listLibrary(); } catch { /* no library: current flame only */ }

  // ---- flames ----
  const flamesSec = el('div', 'batch-sec');
  flamesSec.append(el('h4', '', 'Flames'));
  const flameList = el('div', 'batch-list');
  const flameChecks: { chk: HTMLInputElement; get: () => Flame; name: () => string }[] = [];
  const row = (label: string, thumb: Thumb | null, get: () => Flame, name: () => string, checked: boolean) => {
    const lab = el('label', 'batch-item');
    const chk = el('input') as HTMLInputElement;
    chk.type = 'checkbox';
    chk.checked = checked;
    lab.append(chk);
    if (thumb) { const img = el('img') as HTMLImageElement; img.src = thumbSrc(thumb); lab.append(img); }
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

  // ---- animation (the Anim tab's timeline, when there is one) ----
  const tl = app.timeline();
  const animSec = el('div', 'batch-sec');
  const animChk = el('input') as HTMLInputElement;
  animChk.type = 'checkbox';
  const animSizeChecks: { chk: HTMLInputElement; value: string }[] = [];
  const animFmt = el('select') as HTMLSelectElement;
  const animFps = el('select') as HTMLSelectElement;
  const animQ = el('select') as HTMLSelectElement;
  if (tl) {
    const h = el('h4', '');
    const lab = el('label', 'batch-item');
    lab.append(animChk, el('span', 'batch-name', `Animation (${tl.total.toFixed(1)} s timeline of the current flame)`));
    h.append('Animation');
    animSec.append(h, lab);
    const sizes = el('div', 'batch-sizes');
    for (const so of VIDEO_SIZE_OPTIONS) {
      const l = el('label', 'batch-size');
      const chk = el('input') as HTMLInputElement;
      chk.type = 'checkbox';
      chk.checked = so.value === '';
      const dim = so.value ? so.value.replace('x', '×') : `${r.width & ~1}×${r.height & ~1}`;
      l.append(chk, el('span', '', ` ${so.label}`), el('span', 'hint', ` ${dim}`));
      sizes.append(l);
      animSizeChecks.push({ chk, value: so.value });
      chk.onchange = refresh;
    }
    const opt = el('div', 'btn-row');
    for (const [label, v] of [['WebM (VP9)', 'webm'], ['MP4 (H.264)', 'mp4']] as const) { const o = el('option', '', label) as HTMLOptionElement; o.value = v; animFmt.append(o); }
    for (const v of ['24', '30', '60']) { const o = el('option', '', v + ' fps') as HTMLOptionElement; o.value = v; if (v === '30') o.selected = true; animFps.append(o); }
    for (const q of VIDEO_QUALITY_OPTIONS) { const o = el('option', '', q.label) as HTMLOptionElement; o.value = q.value; if (q.value === '72') o.selected = true; animQ.append(o); }
    animQ.title = 'Accumulation passes per frame';
    opt.append(animFmt, animFps, el('span', 'hint', 'Quality '), animQ);
    animSec.append(sizes, opt);
    for (const c of [animChk, animFmt, animFps, animQ]) c.addEventListener('change', refresh);
  }

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
  body.append(flamesSec, sizeSec, ...(tl ? [animSec] : []), queueSec);

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
    // animation jobs: the current timeline at each chosen video size
    if (tl && animChk.checked) {
      const format = animFmt.value as VideoFormat, fps = parseInt(animFps.value), passes = parseInt(animQ.value);
      for (const sc of animSizeChecks.filter((x) => x.chk.checked)) {
        const fixed = /^(\d+)x(\d+)$/.exec(sc.value);
        const size = fixed ? { w: Number(fixed[1]), h: Number(fixed[2]) } : undefined;
        out.push({ name: `${app.flame.name || 'untitled'}-anim`, file: '', w: size?.w ?? (r.width & ~1), h: size?.h ?? (r.height & ~1), video: { format, fps, passes, size }, status: el('span', 'batch-status', 'queued'), done: false });
      }
    }
    // file names: <name>-<WxH>.png / .webm|.mp4, numbered when two selected flames share a name
    const seen = new Map<string, number>();
    for (const j of out) {
      const base = `${safeFileName(j.name)}-${j.w}x${j.h}`;
      const n = (seen.get(base) ?? 0) + 1;
      seen.set(base, n);
      j.file = `${base}${n > 1 ? `-${n}` : ''}${j.video ? videoFileExt(j.video.format) : '.png'}`;
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
    const stills = jobs.filter((j) => !j.video), vids = jobs.filter((j) => j.video);
    const mp = stills.reduce((a, j) => a + j.w * j.h, 0) / 1e6;
    const frames = tl ? Math.round(tl.total * parseInt(animFps.value || '30')) + 1 : 0;
    summary.textContent = jobs.length
      ? [stills.length ? `${stills.length} image${stills.length > 1 ? 's' : ''} (${mp.toFixed(0)} Mpixel)` : '', vids.length ? `${vids.length} video${vids.length > 1 ? 's' : ''} (${frames} frames each)` : ''].filter(Boolean).join(', ') + '.'
      : 'Pick at least one flame and one size (or the animation).';
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
    for (const c of [...flameChecks.map((f) => f.chk), ...sizeChecks.map((s) => s.chk), ...animSizeChecks.map((s) => s.chk), qSel, alphaChk, allBtn, noneBtn, animChk, animFmt, animFps, animQ]) c.disabled = true;
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
          const blob = j.video
            ? await renderVideo(r, tl!, {
              ...j.video, signal: abort.signal,
              onFrame: (i, n) => { j.status.textContent = `frame ${i}/${n}`; },
              onStatus: (t) => { j.status.textContent = t; },
            })
            : await renderHiRes(r, j.flame!, {
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
      for (const c of [...flameChecks.map((f) => f.chk), ...sizeChecks.map((s) => s.chk), ...animSizeChecks.map((s) => s.chk), qSel, alphaChk, allBtn, noneBtn, animChk, animFmt, animFps, animQ]) c.disabled = false;
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
