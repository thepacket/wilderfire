// Right panel — Anim tab: keyframe timeline, morphing playback/scrub, and
// offline WebM export via WebCodecs (VideoEncoder) + webm-muxer. Everything
// stays client-side.
import { Muxer as WebMMuxer, ArrayBufferTarget as WebMTarget } from 'webm-muxer';
import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4Target } from 'mp4-muxer';
import { App, el, slider } from './common';
import { cloneFlame, normalizeFlame } from '../core/flame';
import { flameAt, sortKeys, type Keyframe, type Easing } from '../core/animate';

interface OverlayHandle { setVisible(v: boolean): void; readonly visible: boolean; }

export interface AnimState {
  keys: { time: number; flame: unknown; ease?: Easing }[];
  easing: Easing;
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
      if (playing || exporting || keys.length < 2) return;
      const ks = sortKeys(keys);
      const t = ks[0].time + v * (ks[ks.length - 1].time - ks[0].time);
      app.applyPreview(flameAt(ks, t, easing()));
    },
  });
  pbSec.append(scrub.root);

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
  for (const [label, v] of [['Draft', '24'], ['Good', '72'], ['High', '160']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    if (v === '72') o.selected = true;
    qSel.append(o);
  }
  qSel.title = 'Accumulation passes per frame';
  const fmtSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['WebM (VP9)', 'webm'], ['MP4 (H.264)', 'mp4']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    fmtSel.append(o);
  }
  const exBtn = el('button', 'primary', '⬇ Export video');
  const exRow = el('div', 'btn-row');
  exRow.append(exBtn, fmtSel, exFpsSel, qSel);
  exSec.append(exRow);
  const exStatus = el('div', 'hint', 'Rendered fully in your browser via WebCodecs.');
  exSec.append(exStatus);

  root.append(kfSec, pbSec, exSec);

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
    const ready = keys.length >= 2;
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
    if (keys.length < 2 || exporting) return;
    const ks = sortKeys(keys);
    const t0 = ks[0].time;
    const t1 = ks[ks.length - 1].time;
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
        app.applyPreview(flameAt(ks, t1, easing()));
        stop();
        return;
      }
      // Advance flame only at the chosen animation fps so accumulation
      // gets several rAF frames per animation step.
      const step = Math.floor(elapsed * stepFps);
      if (step !== lastStep) {
        lastStep = step;
        const t = t0 + elapsed;
        app.applyPreview(flameAt(ks, t, easing()));
        scrub.set(elapsed / total);
      }
    };
    raf = requestAnimationFrame(tick);
  }

  playBtn.onclick = play;

  // ---------- WebM export ----------
  async function exportWebM(opts?: { fps?: number; passes?: number; download?: boolean; format?: 'webm' | 'mp4' }): Promise<Blob> {
    if (keys.length < 2) throw new Error('Need at least 2 keyframes.');
    if (!('VideoEncoder' in window)) {
      throw new Error('WebCodecs (VideoEncoder) is not available in this browser.');
    }
    const fps = opts?.fps ?? parseInt(exFpsSel.value);
    const passes = opts?.passes ?? parseInt(qSel.value);
    const download = opts?.download ?? true;
    const format = opts?.format ?? (fmtSel.value as 'webm' | 'mp4');
    const ks = sortKeys(keys);
    const t0 = ks[0].time;
    const total = Math.max(ks[ks.length - 1].time - t0, 0.01);
    const nFrames = Math.max(2, Math.round(total * fps) + 1);

    const renderer = app.renderer;
    const width = renderer.width & ~1;
    const height = renderer.height & ~1;

    const savedFlame = app.flame;
    if (playing) stop();
    exporting = true;
    renderer.exporting = true;
    exBtn.disabled = true;
    playBtn.disabled = true;

    try {
      let addChunk: (chunk: EncodedVideoChunk, meta?: EncodedVideoChunkMetadata) => void;
      let finalize: () => ArrayBuffer;
      let mime: string;
      const encCfg: VideoEncoderConfig = { codec: '', width, height, bitrate: 12_000_000, framerate: fps };

      if (format === 'mp4') {
        let codec = '';
        for (const c of ['avc1.640028', 'avc1.4D0028', 'avc1.420028']) {
          const s = await VideoEncoder.isConfigSupported({ ...encCfg, codec: c });
          if (s.supported) { codec = c; break; }
        }
        if (!codec) throw new Error('H.264 encoding not supported here — use WebM.');
        encCfg.codec = codec;
        (encCfg as VideoEncoderConfig & { avc?: { format: string } }).avc = { format: 'avc' };
        const muxer = new Mp4Muxer({
          target: new Mp4Target(),
          video: { codec: 'avc', width, height, frameRate: fps },
          fastStart: 'in-memory',
        });
        addChunk = (c, m) => muxer.addVideoChunk(c, m);
        finalize = () => { muxer.finalize(); return muxer.target.buffer; };
        mime = 'video/mp4';
      } else {
        let codec = 'vp09.00.10.08';
        let muxCodec = 'V_VP9';
        const support = await VideoEncoder.isConfigSupported({ ...encCfg, codec });
        if (!support.supported) { codec = 'vp8'; muxCodec = 'V_VP8'; }
        encCfg.codec = codec;
        const muxer = new WebMMuxer({
          target: new WebMTarget(),
          video: { codec: muxCodec, width, height, frameRate: fps },
        });
        addChunk = (c, m) => muxer.addVideoChunk(c, m!);
        finalize = () => { muxer.finalize(); return muxer.target.buffer; };
        mime = 'video/webm';
      }

      const encoder = new VideoEncoder({
        output: (chunk, meta) => addChunk(chunk, meta),
        error: (e) => console.error('VideoEncoder:', e),
      });
      encoder.configure(encCfg);

      for (let i = 0; i < nFrames; i++) {
        const t = t0 + (i / fps);
        renderer.setFlame(flameAt(ks, Math.min(t, t0 + total), easing()));
        await renderer.stepExport(passes);
        // Capture must stay in the same task as the tonemap submit.
        const frame = renderer.captureSync((cv) => new VideoFrame(cv, {
          timestamp: Math.round((i * 1e6) / fps),
          duration: Math.round(1e6 / fps),
        }));
        encoder.encode(frame, { keyFrame: i % (fps * 2) === 0 });
        frame.close();
        while (encoder.encodeQueueSize > 4) {
          await new Promise((r) => setTimeout(r, 4));
        }
        exStatus.textContent = `Rendering frame ${i + 1}/${nFrames}…`;
      }
      exStatus.textContent = 'Encoding…';
      await encoder.flush();
      const blob = new Blob([finalize()], { type: mime });
      exStatus.textContent = `Done — ${(blob.size / 1e6).toFixed(1)} MB, ${nFrames} frames @ ${fps} fps.`;
      if (download) {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${savedFlame.name || 'wilderfire'}-anim.${format}`;
        a.click();
        URL.revokeObjectURL(a.href);
      }
      return blob;
    } finally {
      exporting = false;
      renderer.exporting = false;
      renderer.setFlame(savedFlame);
      app.flame = savedFlame;
      rebuildList();
    }
  }

  exBtn.onclick = () => {
    exportWebM().catch((e) => {
      exStatus.textContent = '⚠ ' + (e as Error).message;
    });
  };

  // ---------- Persistence ----------
  const getState = (): AnimState => ({
    keys: keys.map((k) => ({ time: k.time, flame: k.flame, ...(k.ease ? { ease: k.ease } : {}) })),
    easing: easing(),
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
    rebuildList();
  };
  const onChange = () => app.emit('history'); // nudge autosave listeners

  saveAnimBtn.onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(getState(), null, 1)], { type: 'application/json' }));
    a.download = `${app.flame.name || 'wilderfire'}-anim.json`;
    a.click();
    URL.revokeObjectURL(a.href);
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

  rebuildList();

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
