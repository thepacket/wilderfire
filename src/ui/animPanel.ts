// Right panel — Anim tab: keyframe timeline, morphing playback/scrub, and
// offline WebM export via WebCodecs (VideoEncoder) + webm-muxer. Everything
// stays client-side.
import { App, el, slider } from './common';
import { renderVideo, videoFileExt, videoMime, VIDEO_SIZE_OPTIONS, VIDEO_QUALITY_OPTIONS, type VideoFormat } from './videoExport';
import { cloneFlame, normalizeFlame } from '../core/flame';
import { flameAt, sortKeys, type Keyframe, type Easing } from '../core/animate';
import { pickSave, saveText, type SaveTarget } from './saveFile';
import { applyCurves, applyCompCurves, compParamPaths, getCompParam, isCompPath, curvesEnd, evalCurve, getParam, paramPaths, setPoint, INTERPS, type MotionCurve, type CurveInterp, type CurvePoint } from '../core/motion';

interface OverlayHandle { setVisible(v: boolean): void; readonly visible: boolean; }

export interface AnimState {
  keys: { time: number; flame: unknown; ease?: Easing }[];
  easing: Easing;
  /** per-parameter motion curves (see core/motion.ts) */
  curves?: MotionCurve[];
}

export interface AnimAPI {
  addKey(): void;
  play(): void;
  stop(): void;
  exportWebM(opts?: { fps?: number; passes?: number; download?: boolean }): Promise<Blob>;
  getState(): AnimState;
  setState(state: AnimState | null | undefined): void;
  keys: Keyframe[];
}

export function buildAnimPanel(app: App, root: HTMLElement, overlay: OverlayHandle): AnimAPI {
  let keys: Keyframe[] = [];
  let curves: MotionCurve[] = [];
  let playing = false;
  let raf = 0;
  let wasTriangles = true;
  let exporting = false;

  // ---------- Keyframes ----------
  const kfSec = el('div', 'section');
  kfSec.append(el('h3', '', 'Keyframes'));
  const list = el('div', 'xform-list');
  kfSec.append(list);
  const kfBtns = el('div', 'btn-row');
  const capBtn = el('button', 'primary', '+ Capture keyframe');
  const loopBtn = el('button', '', 'Close loop');
  loopBtn.title = 'Append a copy of the first keyframe at the end for a seamless loop';
  const saveAnimBtn = el('button', 'icon', '⬇');
  saveAnimBtn.title = 'Save animation JSON';
  const loadAnimBtn = el('button', 'icon', '⬆');
  loadAnimBtn.title = 'Load animation JSON';
  const animFile = el('input') as HTMLInputElement;
  animFile.type = 'file';
  animFile.accept = '.json,application/json';
  animFile.style.display = 'none';
  kfBtns.append(capBtn, loopBtn, saveAnimBtn, loadAnimBtn, animFile);
  kfSec.append(kfBtns);
  kfSec.append(el('div', 'hint', 'Set up a flame, capture it, tweak (or load another preset), capture again — playback morphs between keyframes. The timeline autosaves with your session.'));

  // ---------- Playback ----------
  const pbSec = el('div', 'section');
  pbSec.append(el('h3', '', 'Playback'));
  const playBtn = el('button', 'primary', '▶ Play');
  const loopChk = el('input') as HTMLInputElement;
  loopChk.type = 'checkbox';
  loopChk.checked = true;
  const loopLab = el('label', '', ' loop');
  loopLab.prepend(loopChk);
  loopLab.style.color = 'var(--fg-dim)';
  const fpsSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['10 fps (crisp)', '10'], ['15 fps', '15'], ['20 fps (smooth)', '20']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    if (v === '15') o.selected = true;
    fpsSel.append(o);
  }
  const easeSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['Linear', 'linear'], ['Smooth', 'smooth']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    easeSel.append(o);
  }
  easeSel.title = 'Segment easing';
  const easing = (): Easing => easeSel.value as Easing;
  const pbRow = el('div', 'btn-row');
  pbRow.append(playBtn, fpsSel, easeSel, loopLab);
  pbSec.append(pbRow);
  const scrub = slider({
    label: 'Scrub', min: 0, max: 1, step: 0.002, value: 0,
    fmt: (v) => (v * 100).toFixed(0) + '%',
    onInput: (v) => {
      if (playing || exporting || !isReady()) return;
      const [t0, t1] = timeRange();
      curT = t0 + v * (t1 - t0);
      app.applyPreview(evalAt(curT));
      curveSec.querySelector('.mc-now')!.textContent = `t = ${curT.toFixed(2)} s`;
    },
  });
  pbSec.append(scrub.root);

  const onChange = () => app.emit('history'); // nudge autosave listeners

  // ---------- Timeline evaluation (keyframe morph + motion curves) ----------
  let curT = 0; // scrub position in seconds
  const isReady = () => keys.length >= 2 || curves.some((c) => c.enabled !== false && c.points.length > 0);
  /** [start, end] of the timeline in seconds: keyframes ∪ curve points. */
  const timeRange = (): [number, number] => {
    const ks = sortKeys(keys);
    const t0 = ks.length ? ks[0].time : 0;
    const t1 = Math.max(ks.length ? ks[ks.length - 1].time : 0, curvesEnd(curves), t0 + 0.01);
    return [t0, t1];
  };
  /** Flame at time t: keyframe morph (or the editor flame when < 2 keys) with curves applied. */
  const evalAt = (t: number) => {
    const ks = sortKeys(keys);
    const base = ks.length >= 2 ? flameAt(ks, t, easing()) : ks.length === 1 ? cloneFlame(ks[0].flame) : app.flame;
    applyCompCurves(app.comp, curves, t); // composition-layer curves (escape/image params, opacity, effects) act on the document in place
    return applyCurves(base, curves, t);
  };
  /** a parameter's current value: flame path or composition-layer path */
  const readParam = (path: string) => (isCompPath(path) ? getCompParam(app.comp, path) : getParam(app.flame, path));
  const allParams = () => [...paramPaths(app.flame), ...compParamPaths(app.comp)];

  // ---------- Motion curves ----------
  const curveSec = el('div', 'section');
  curveSec.append(el('h3', '', 'Motion curves'));
  const mcTop = el('div', 'btn-row');
  const paramSel = el('select') as HTMLSelectElement;
  paramSel.style.maxWidth = '190px';
  const addCurveBtn = el('button', '', '+ Curve');
  addCurveBtn.title = 'Animate this parameter with its own curve (independent of keyframes)';
  mcTop.append(paramSel, addCurveBtn, el('span', 'hint mc-now', 't = 0.00 s'));
  const curveList = el('div', 'mc-list');
  curveSec.append(mcTop, curveList);
  curveSec.append(el('div', 'hint', 'A curve drives one parameter over time. Scrub to a time, adjust the parameter in the editor, then “+ key” to record its value there — or drag points on the graph (double-click adds, Alt-click removes). Curves layer on top of keyframe morphs.'));

  function rebuildParamSel() {
    const cur = paramSel.value;
    paramSel.textContent = '';
    let grp: HTMLOptGroupElement | null = null;
    for (const p of allParams()) {
      if (!grp || grp.label !== p.group) { grp = el('optgroup') as HTMLOptGroupElement; grp.label = p.group; paramSel.append(grp); }
      const o = el('option', '', p.label) as HTMLOptionElement;
      o.value = p.path;
      grp.append(o);
    }
    if (cur && [...paramSel.options].some((o) => o.value === cur)) paramSel.value = cur;
  }
  addCurveBtn.onclick = () => {
    const path = paramSel.value;
    if (!path) return;
    let c = curves.find((k) => k.path === path);
    if (!c) { c = { path, points: [], interp: 'spline' }; curves.push(c); }
    const v = readParam(path);
    if (v !== undefined) setPoint(c, curT, v);
    rebuildCurves(); rebuildList(); onChange();
  };
  // Value range shown for a curve: its points ± 10 % (frozen while dragging so
  // a point can be pulled outside the initial range without the axis jumping).
  const curveRange = (c: MotionCurve): [number, number] => {
    let lo = Infinity, hi = -Infinity;
    for (const p of c.points) { lo = Math.min(lo, p.v); hi = Math.max(hi, p.v); }
    if (!c.points.length) { lo = 0; hi = 1; }
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.1;
    return [lo - pad, hi + pad];
  };
  const frozenRange = new WeakMap<MotionCurve, [number, number]>();
  function curveMap(cv: HTMLCanvasElement, c: MotionCurve) {
    const W = cv.width, H = cv.height;
    const [t0, t1] = timeRange();
    const [lo, hi] = frozenRange.get(c) ?? curveRange(c);
    const X = (t: number) => 4 + (t - t0) / (t1 - t0) * (W - 8);
    const Y = (v: number) => H - 4 - (v - lo) / (hi - lo) * (H - 8);
    const invX = (x: number) => t0 + ((x - 4) / (W - 8)) * (t1 - t0);
    const invY = (y: number) => lo + ((H - 4 - y) / (H - 8)) * (hi - lo);
    return { W, H, t0, t1, lo, hi, X, Y, invX, invY };
  }
  function drawCurve(cv: HTMLCanvasElement, c: MotionCurve, hot?: CurvePoint) {
    const ctx = cv.getContext('2d')!;
    const { W, H, t0, t1, lo, X, Y } = curveMap(cv, c);
    ctx.clearRect(0, 0, W, H);
    if (!c.points.length) return;
    const css = getComputedStyle(cv);
    const accent = css.getPropertyValue('--accent').trim() || '#ff7a3c';
    // zero line if in range
    if (Y(0) > 0 && Y(0) < H) {
      ctx.strokeStyle = 'rgba(128,128,128,0.25)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke();
    }
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= W; i += 2) {
      const t = t0 + (i / W) * (t1 - t0);
      const v = evalCurve(c, t) ?? lo;
      i === 0 ? ctx.moveTo(X(t), Y(v)) : ctx.lineTo(X(t), Y(v));
    }
    ctx.stroke();
    for (const p of c.points) {
      ctx.fillStyle = p === hot ? accent : '#fff';
      ctx.beginPath(); ctx.arc(X(p.t), Y(p.v), p === hot ? 4 : 3, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(curT), 0); ctx.lineTo(X(curT), H); ctx.stroke();
  }
  /** Mouse editing on the sparkline: drag points, double-click to add, Alt/right-click to delete. */
  function attachCurveEditing(cv: HTMLCanvasElement, c: MotionCurve, refresh: () => void) {
    const toCanvas = (e: PointerEvent | MouseEvent) => {
      const r = cv.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * cv.width, y: ((e.clientY - r.top) / r.height) * cv.height };
    };
    const hit = (x: number, y: number): CurvePoint | null => {
      const { X, Y } = curveMap(cv, c);
      let best: CurvePoint | null = null, bd = 8;
      for (const p of c.points) { const d = Math.hypot(X(p.t) - x, Y(p.v) - y); if (d < bd) { bd = d; best = p; } }
      return best;
    };
    let drag: CurvePoint | null = null;
    let moved = false;
    cv.style.cursor = 'crosshair';
    cv.onpointerdown = (e) => {
      if (e.button !== 0) return;
      const { x, y } = toCanvas(e);
      const p = hit(x, y);
      if (!p) return;
      if (e.altKey) { // delete
        c.points.splice(c.points.indexOf(p), 1);
        refresh(); onChange();
        return;
      }
      drag = p; moved = false;
      frozenRange.set(c, curveRange(c));
      cv.setPointerCapture(e.pointerId);
      cv.style.cursor = 'grabbing';
      drawCurve(cv, c, p);
      e.preventDefault();
    };
    cv.onpointermove = (e) => {
      const { x, y } = toCanvas(e);
      if (!drag) { cv.style.cursor = hit(x, y) ? 'grab' : 'crosshair'; return; }
      const { invX, invY } = curveMap(cv, c);
      // Shift = value only, Ctrl/Cmd = time only
      if (!e.ctrlKey && !e.metaKey) drag.v = Math.round(invY(y) * 1e4) / 1e4;
      if (!e.shiftKey) drag.t = Math.max(0, Math.round(invX(x) * 100) / 100);
      moved = true;
      c.points.sort((a, b) => a.t - b.t);
      drawCurve(cv, c, drag);
      // live preview at the scrub time
      app.applyPreview(evalAt(curT));
    };
    const end = () => {
      if (!drag) return;
      drag = null;
      frozenRange.delete(c);
      cv.style.cursor = 'crosshair';
      if (moved) { refresh(); onChange(); } else drawCurve(cv, c);
    };
    cv.onpointerup = end;
    cv.onpointercancel = end;
    cv.ondblclick = (e) => {
      const { x, y } = toCanvas(e);
      if (hit(x, y)) return;
      const { invX, invY } = curveMap(cv, c);
      setPoint(c, Math.max(0, Math.round(invX(x) * 100) / 100), Math.round(invY(y) * 1e4) / 1e4);
      refresh(); onChange();
    };
    cv.oncontextmenu = (e) => {
      const { x, y } = toCanvas(e);
      const p = hit(x, y);
      if (!p) return;
      e.preventDefault();
      c.points.splice(c.points.indexOf(p), 1);
      refresh(); onChange();
    };
  }
  function rebuildCurves() {
    rebuildParamSel();
    curveList.textContent = '';
    const labels = new Map(allParams().map((p) => [p.path, p.label]));
    for (const c of curves) {
      const row = el('div', 'mc-row');
      const head = el('div', 'mc-head');
      const en = el('input') as HTMLInputElement;
      en.type = 'checkbox'; en.checked = c.enabled !== false; en.title = 'enabled';
      en.onchange = () => { c.enabled = en.checked; onChange(); };
      const nm = el('span', 'vname', labels.get(c.path) ?? c.path);
      nm.title = c.path;
      const interp = el('select') as HTMLSelectElement;
      for (const it of INTERPS) { const o = el('option', '', it) as HTMLOptionElement; o.value = it; interp.append(o); }
      interp.value = c.interp;
      interp.onchange = () => { c.interp = interp.value as CurveInterp; drawCurve(cv, c); onChange(); };
      const keyBtn = el('button', 'icon', '+ key');
      keyBtn.title = 'Record the parameter’s current editor value at the scrub time';
      keyBtn.onclick = () => { const v = readParam(c.path); if (v !== undefined) { setPoint(c, curT, v); rebuildCurves(); rebuildList(); onChange(); } };
      const rm = el('button', 'icon danger', '✕');
      rm.onclick = () => { curves = curves.filter((k) => k !== c); rebuildCurves(); rebuildList(); onChange(); };
      head.append(en, nm, interp, keyBtn, rm);
      const cv = el('canvas') as HTMLCanvasElement;
      cv.className = 'mc-canvas'; cv.width = 300; cv.height = 64;
      cv.title = 'Drag points · double-click to add · Alt/right-click to delete · Shift: value only · Ctrl/⌘: time only';
      drawCurve(cv, c);
      attachCurveEditing(cv, c, () => { rebuildCurves(); rebuildList(); });
      // point table
      const pts = el('div', 'mc-points');
      c.points.forEach((p, i) => {
        const pr = el('span', 'vp');
        const ti = el('input') as HTMLInputElement; ti.type = 'number'; ti.step = '0.1'; ti.value = String(Math.round(p.t * 1000) / 1000); ti.title = 'time (s)';
        const vi = el('input') as HTMLInputElement; vi.type = 'number'; vi.step = '0.01'; vi.value = String(Math.round(p.v * 10000) / 10000); vi.title = 'value';
        ti.onchange = () => { p.t = Math.max(0, parseFloat(ti.value) || 0); c.points.sort((a, b) => a.t - b.t); rebuildCurves(); rebuildList(); onChange(); };
        vi.onchange = () => { p.v = parseFloat(vi.value) || 0; drawCurve(cv, c); onChange(); };
        const del = el('button', 'icon', '✕'); del.title = 'remove point';
        del.onclick = () => { c.points.splice(i, 1); rebuildCurves(); rebuildList(); onChange(); };
        pr.append(ti, vi, del);
        pts.append(pr);
      });
      row.append(head, cv, pts);
      curveList.append(row);
    }
  }

  // ---------- Export ----------
  const exSec = el('div', 'section');
  exSec.append(el('h3', '', 'Export video'));
  const exFpsSel = el('select') as HTMLSelectElement;
  for (const v of ['24', '30', '60']) {
    const o = el('option', '', v + ' fps') as HTMLOptionElement;
    o.value = v;
    if (v === '30') o.selected = true;
    exFpsSel.append(o);
  }
  const qSel = el('select') as HTMLSelectElement;
  for (const q of VIDEO_QUALITY_OPTIONS) {
    const o = el('option', '', q.label) as HTMLOptionElement;
    o.value = q.value;
    if (q.value === '72') o.selected = true;
    qSel.append(o);
  }
  qSel.title = 'Accumulation passes per frame (fixed sizes render the same samples per pixel)';
  const sizeSel = el('select') as HTMLSelectElement;
  for (const so of VIDEO_SIZE_OPTIONS) {
    const o = el('option', '', so.label) as HTMLOptionElement;
    o.value = so.value;
    sizeSel.append(o);
  }
  sizeSel.title = 'Video size: the canvas as shown, or a fixed 16:9 frame rendered offscreen (slower)';
  const fmtSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['WebM (VP9)', 'webm'], ['MP4 (H.264)', 'mp4']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    fmtSel.append(o);
  }
  const exBtn = el('button', 'primary', '⬇ Export video');
  const exRow = el('div', 'btn-row');
  exRow.append(exBtn, fmtSel, sizeSel, exFpsSel, qSel);
  exSec.append(exRow);
  const exStatus = el('div', 'hint', 'Rendered fully in your browser via WebCodecs.');
  exSec.append(exStatus);

  root.append(kfSec, curveSec, pbSec, exSec);

  function rebuildList() {
    list.textContent = '';
    sortKeys(keys).forEach((k) => {
      const idx = keys.indexOf(k);
      const item = el('div', 'xform-item');
      const sw = el('span', 'xform-swatch');
      const midPal = k.flame.layers[0]?.palette[128] ?? [0.5, 0.5, 0.5];
      sw.style.background = `rgb(${midPal.map((v: number) => Math.round(v * 255)).join(',')})`;
      const name = el('span', 'xname', `K${idx + 1}`);
      const tIn = el('input') as HTMLInputElement;
      tIn.type = 'number';
      tIn.step = '0.5';
      tIn.value = String(k.time);
      tIn.title = 'time (s)';
      tIn.style.width = '56px';
      tIn.addEventListener('change', () => {
        const v = parseFloat(tIn.value);
        if (isFinite(v)) { k.time = Math.max(0, v); rebuildList(); }
      });
      const goBtn = el('button', 'icon', '⌖');
      goBtn.title = 'Load this keyframe into the editor';
      goBtn.onclick = () => app.setFlame(cloneFlame(k.flame));
      const setBtn = el('button', 'icon', '↧');
      setBtn.title = 'Overwrite with the current flame';
      setBtn.onclick = () => { k.flame = cloneFlame(app.flame); rebuildList(); };
      const easeIn = el('select') as HTMLSelectElement;
      easeIn.title = 'Easing of the segment after this keyframe';
      for (const [label, v] of [['·', ''], ['lin', 'linear'], ['smt', 'smooth'], ['in', 'in'], ['out', 'out']] as const) {
        const o = el('option', '', label) as HTMLOptionElement;
        o.value = v;
        easeIn.append(o);
      }
      easeIn.value = k.ease ?? '';
      easeIn.addEventListener('click', (e) => e.stopPropagation());
      easeIn.addEventListener('change', () => {
        if (easeIn.value) k.ease = easeIn.value as Easing;
        else delete k.ease;
      });
      const rmBtn = el('button', 'icon danger', '✕');
      rmBtn.onclick = () => { keys.splice(idx, 1); rebuildList(); };
      item.append(sw, name, tIn, easeIn, goBtn, setBtn, rmBtn);
      list.append(item);
    });
    const ready = isReady();
    playBtn.disabled = !ready || exporting;
    exBtn.disabled = !ready || exporting;
    loopBtn.disabled = keys.length < 1;
  }

  capBtn.onclick = () => {
    const t = keys.length ? Math.max(...keys.map((k) => k.time)) + 2 : 0;
    keys.push({ time: t, flame: cloneFlame(app.flame) });
    rebuildList();
  };
  loopBtn.onclick = () => {
    if (!keys.length) return;
    const ks = sortKeys(keys);
    keys.push({ time: ks[ks.length - 1].time + 2, flame: cloneFlame(ks[0].flame) });
    rebuildList();
  };

  // ---------- Playback engine ----------
  function stop() {
    if (!playing) return;
    playing = false;
    cancelAnimationFrame(raf);
    playBtn.textContent = '▶ Play';
    overlay.setVisible(wasTriangles);
    app.setFlame(cloneFlame(app.flame)); // sync panels + history with final frame
  }

  function play() {
    if (playing) { stop(); return; }
    if (!isReady() || exporting) return;
    const [t0, t1] = timeRange();
    const total = Math.max(t1 - t0, 0.01);
    const stepFps = parseInt(fpsSel.value);
    playing = true;
    playBtn.textContent = '⏹ Stop';
    wasTriangles = overlay.visible;
    overlay.setVisible(false);
    const wallStart = performance.now() / 1000;
    let lastStep = -1;

    const tick = () => {
      if (!playing) return;
      raf = requestAnimationFrame(tick);
      let elapsed = performance.now() / 1000 - wallStart;
      if (loopChk.checked) {
        elapsed = elapsed % total;
      } else if (elapsed >= total) {
        app.applyPreview(evalAt(t1));
        stop();
        return;
      }
      // Advance flame only at the chosen animation fps so accumulation
      // gets several rAF frames per animation step.
      const step = Math.floor(elapsed * stepFps);
      if (step !== lastStep) {
        lastStep = step;
        const t = t0 + elapsed;
        curT = t;
        app.applyPreview(evalAt(t));
        scrub.set(elapsed / total);
      }
    };
    raf = requestAnimationFrame(tick);
  }

  playBtn.onclick = play;

  // ---------- Video export (src/ui/videoExport.ts) ----------
  async function exportWebM(opts?: { fps?: number; passes?: number; download?: boolean; format?: VideoFormat }): Promise<Blob> {
    if (!isReady()) throw new Error('Need at least 2 keyframes or a motion curve.');
    const fps = opts?.fps ?? parseInt(exFpsSel.value);
    const passes = opts?.passes ?? parseInt(qSel.value);
    const download = opts?.download ?? true;
    const format = opts?.format ?? (fmtSel.value as VideoFormat);
    const fixed = /^(\d+)x(\d+)$/.exec(sizeSel.value);
    const size = fixed ? { w: Number(fixed[1]), h: Number(fixed[2]) } : undefined;
    // Pick the destination now (needs the click's user gesture); write after encoding.
    let target: SaveTarget | null = null;
    if (download) {
      const name = (app.flame.name || 'wilderfire').replace(/[\\/:*?"<>|]+/g, '_');
      target = await pickSave({
        suggestedName: `${name}-anim${size ? `-${size.w}x${size.h}` : ''}${videoFileExt(format)}`, description: format === 'mp4' ? 'MP4 video' : 'WebM video',
        mime: videoMime(format), ext: videoFileExt(format),
      });
      if (!target) throw new Error('Export cancelled.');
    }
    const renderer = app.renderer;
    const savedFlame = app.flame;
    if (playing) stop();
    exporting = true;
    renderer.exporting = true;
    exBtn.disabled = true;
    playBtn.disabled = true;
    try {
      const blob = await renderVideo(renderer, timeline()!, {
        fps, passes, format, size,
        onFrame: (i, n) => { exStatus.textContent = `Rendering frame ${i}/${n}…`; },
        onStatus: (t) => { exStatus.textContent = t; },
      });
      exStatus.textContent = `Done — ${(blob.size / 1e6).toFixed(1)} MB${size ? `, ${size.w}×${size.h}` : ''} @ ${fps} fps.`;
      if (target) await target.write(blob);
      return blob;
    } finally {
      exporting = false;
      renderer.exporting = false;
      app.flame = savedFlame;
      app.pushRender();
      rebuildList();
    }
  }
  /** The timeline for offscreen renderers (batch export), or null when there is nothing to animate. */
  const timeline = () => {
    if (!isReady()) return null;
    const [t0, t1] = timeRange();
    return { t0, total: Math.max(t1 - t0, 0.01), evalAt };
  };
  app.timeline = timeline;

  exBtn.onclick = () => {
    exportWebM().catch((e) => {
      exStatus.textContent = '⚠ ' + (e as Error).message;
    });
  };

  // ---------- Persistence ----------
  const getState = (): AnimState => ({
    keys: keys.map((k) => ({ time: k.time, flame: k.flame, ...(k.ease ? { ease: k.ease } : {}) })),
    easing: easing(),
    curves: curves.map((c) => ({ path: c.path, interp: c.interp, points: c.points.map((p) => ({ t: p.t, v: p.v })), ...(c.enabled === false ? { enabled: false } : {}) })),
  });
  const setState = (state: AnimState | null | undefined) => {
    if (!state || !Array.isArray(state.keys)) return;
    keys.length = 0;
    for (const k of state.keys.slice(0, 64)) {
      if (typeof k?.time !== 'number') continue;
      const kf: Keyframe = { time: Math.max(0, k.time), flame: normalizeFlame(k.flame, app.activeLayer.palette) };
      if (k.ease === 'smooth' || k.ease === 'in' || k.ease === 'out' || k.ease === 'linear') kf.ease = k.ease;
      keys.push(kf);
    }
    if (state.easing === 'smooth' || state.easing === 'linear') easeSel.value = state.easing;
    curves = Array.isArray(state.curves)
      ? state.curves.filter((c) => c && typeof c.path === 'string' && Array.isArray(c.points)).map((c) => ({
          path: c.path,
          interp: INTERPS.includes(c.interp) ? c.interp : 'spline',
          points: c.points.filter((p) => typeof p?.t === 'number' && typeof p?.v === 'number').map((p) => ({ t: p.t, v: p.v })).sort((a, b) => a.t - b.t),
          ...(c.enabled === false ? { enabled: false } : {}),
        }))
      : [];
    rebuildCurves();
    rebuildList();
  };

  saveAnimBtn.onclick = () => {
    saveText(JSON.stringify(getState(), null, 1), {
      suggestedName: `${app.flame.name || 'wilderfire'}-anim.json`, description: 'WilderFire animation (JSON)', mime: 'application/json', ext: '.json',
    });
  };
  loadAnimBtn.onclick = () => animFile.click();
  animFile.onchange = async () => {
    const f = animFile.files?.[0];
    if (!f) return;
    try {
      setState(JSON.parse(await f.text()));
      onChange();
    } catch (e) {
      alert('Could not parse animation JSON: ' + (e as Error).message);
    }
    animFile.value = '';
  };

  rebuildCurves();
  rebuildList();
  app.on('flame', (src) => { if (src !== 'preview' && !playing) rebuildCurves(); });
  // Bridge for .flame XML export/import (Render panel) — curves live here.
  app.getCurves = () => getState().curves ?? [];
  app.setCurves = (cs) => { setState({ keys: getState().keys, easing: easing(), curves: cs }); onChange(); };

  return {
    addKey: () => capBtn.click(),
    play,
    stop,
    exportWebM,
    getState,
    setState,
    keys,
  };
}
