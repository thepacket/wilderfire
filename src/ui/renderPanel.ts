// Right panel — Render tab: camera, tonemap, quality, export.
import { App, el, slider } from './common';
import { flameToJSON } from '../core/flame';
import { flameToXML, importFlameText } from '../core/flameXML';
import { pickSave, saveBlob, saveText } from './saveFile';

const SRC = 'render';

export function buildRenderPanel(app: App, root: HTMLElement) {
  const cam = el('div', 'section');
  cam.append(el('h3', '', 'Camera'));
  const zoomS = slider({
    label: 'Zoom', min: -2, max: 3, step: 0.01, value: Math.log2(app.flame.zoom),
    fmt: (v) => (2 ** v).toFixed(2) + '×',
    onInput: (v) => { app.flame.zoom = 2 ** v; app.commit(SRC); },
  });
  const rotS = slider({
    label: 'Rotation', min: -180, max: 180, step: 1, value: (app.flame.rotation * 180) / Math.PI,
    fmt: (v) => v.toFixed(0) + '°',
    onInput: (v) => { app.flame.rotation = (v * Math.PI) / 180; app.commit(SRC); },
  });
  const pitchS = slider({
    label: 'Pitch', min: -180, max: 180, step: 1, value: app.flame.camPitch,
    fmt: (v) => v.toFixed(0) + '°',
    onInput: (v) => { app.flame.camPitch = v; app.commit(SRC); },
  });
  const yawS = slider({
    label: 'Yaw', min: -180, max: 180, step: 1, value: app.flame.camYaw,
    fmt: (v) => v.toFixed(0) + '°',
    onInput: (v) => { app.flame.camYaw = v; app.commit(SRC); },
  });
  const perspS = slider({
    label: 'Perspective', min: 0, max: 1, step: 0.01, value: app.flame.camPersp,
    onInput: (v) => { app.flame.camPersp = v; app.commit(SRC); },
  });
  const camZS = slider({
    label: 'Cam Z', min: -2, max: 2, step: 0.01, value: app.flame.camPosZ,
    onInput: (v) => { app.flame.camPosZ = v; app.commit(SRC); },
  });
  const pzRow = el('div', 'row');
  const pzChk = el('input') as HTMLInputElement;
  pzChk.type = 'checkbox';
  pzChk.checked = app.flame.preserveZ;
  const pzLab = el('label', 'check', ' Preserve Z');
  pzLab.prepend(pzChk);
  pzLab.title = "2D variations keep the point's depth instead of flattening it (preserve_z)";
  pzChk.onchange = () => { app.flame.preserveZ = pzChk.checked; app.commit(SRC); };
  pzRow.append(pzLab);
  cam.append(zoomS.root, rotS.root, pitchS.root, yawS.root, perspS.root, camZS.root, pzRow);
  cam.append(el('div', 'hint', 'Drag the canvas to pan · scroll to zoom · drag triangle handles to edit transforms. Pitch/yaw/perspective view the flame in 3D.'));

  // Depth of field + depth fade
  const dof = el('div', 'section');
  dof.append(el('h3', '', 'Depth of field'));
  const dofS = slider({
    label: 'Amount', min: 0, max: 2, step: 0.01, value: app.flame.camDOF ?? 0,
    onInput: (v) => { app.flame.camDOF = v; app.commit(SRC); },
  });
  const dofAreaS = slider({
    label: 'Focus area', min: 0, max: 2, step: 0.01, value: app.flame.camDOFArea ?? 0.5,
    onInput: (v) => { app.flame.camDOFArea = v; app.commit(SRC); },
  });
  const focusZS = slider({
    label: 'Focus Z', min: -2, max: 2, step: 0.01, value: app.flame.focusZ ?? 0,
    onInput: (v) => { app.flame.focusZ = v; app.flame.newDOF = true; app.commit(SRC); },
  });
  const fadeS = slider({
    label: 'Fade', min: 0, max: 1, step: 0.01, value: app.flame.camDOFFade ?? 1,
    onInput: (v) => { app.flame.camDOFFade = v; app.commit(SRC); },
  });
  const dimS = slider({
    label: 'Dimish Z', min: 0, max: 5, step: 0.05, value: app.flame.dimishZ ?? 0,
    onInput: (v) => { app.flame.dimishZ = v; app.commit(SRC); },
  });
  const dimDistS = slider({
    label: 'Dim distance', min: -2, max: 2, step: 0.01, value: app.flame.dimZDist ?? 0,
    onInput: (v) => { app.flame.dimZDist = v; app.commit(SRC); },
  });
  const dimColRow = el('div', 'row');
  dimColRow.append(el('label', '', 'Dim colour'));
  const dimColInp = el('input') as HTMLInputElement;
  dimColInp.type = 'color';
  const toHex3 = (c: number[]) => '#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
  dimColInp.value = toHex3(app.flame.dimZColor ?? [0, 0, 0]);
  dimColInp.addEventListener('input', () => {
    const h = dimColInp.value;
    app.flame.dimZColor = [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
    app.commit(SRC);
  });
  dimColRow.append(dimColInp);
  dof.append(dofS.root, dofAreaS.root, focusZS.root, fadeS.root, dimS.root, dimDistS.root, dimColRow);
  dof.append(el('div', 'hint', 'Points away from the focus blur into discs (amount × distance); Dimish Z fades points beyond the dim distance toward the dim colour. Both need depth: use 3D variations, pitch/yaw, or a 3D affine.'));

  const tone = el('div', 'section');
  tone.append(el('h3', '', 'Tone'));
  const brS = slider({
    label: 'Brightness', min: 0.1, max: 6, step: 0.05, value: app.flame.brightness,
    onInput: (v) => { app.flame.brightness = v; app.commitTone(SRC); },
  });
  const gaS = slider({
    label: 'Gamma', min: 1, max: 6, step: 0.05, value: app.flame.gamma,
    onInput: (v) => { app.flame.gamma = v; app.commitTone(SRC); },
  });
  const gtS = slider({
    label: 'Gamma thresh', min: 0, max: 0.2, step: 0.005, value: app.flame.gammaThreshold ?? 0.04,
    fmt: (v) => v.toFixed(3),
    onInput: (v) => { app.flame.gammaThreshold = v; app.commitTone(SRC); },
  });
  const viS = slider({
    label: 'Vibrancy', min: 0, max: 1, step: 0.01, value: app.flame.vibrancy,
    onInput: (v) => { app.flame.vibrancy = v; app.commitTone(SRC); },
  });
  const ctS = slider({
    label: 'Contrast', min: 0.1, max: 3, step: 0.05, value: app.flame.contrast ?? 1,
    onInput: (v) => { app.flame.contrast = v; app.commitTone(SRC); },
  });
  const wlS = slider({
    label: 'White level', min: 100, max: 255, step: 1, value: app.flame.whiteLevel ?? 220,
    fmt: (v) => v.toFixed(0),
    onInput: (v) => { app.flame.whiteLevel = v; app.commitTone(SRC); },
  });
  const sfS = slider({
    label: 'Filter', min: 0, max: 2, step: 0.05, value: app.flame.filterRadius ?? 0,
    onInput: (v) => { app.flame.filterRadius = v; app.commitTone(SRC); },
  });
  sfS.root.title = 'Spatial filter radius (Mitchell kernel over the log-scaled image); 0 = off';
  const bgRow = el('div', 'row');
  bgRow.append(el('label', '', 'Background'));
  const bgInp = el('input') as HTMLInputElement;
  bgInp.type = 'color';
  const toHex = (c: number[]) => '#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
  bgInp.value = toHex(app.flame.background);
  bgInp.addEventListener('input', () => {
    const h = bgInp.value;
    app.flame.background = [
      parseInt(h.slice(1, 3), 16) / 255,
      parseInt(h.slice(3, 5), 16) / 255,
      parseInt(h.slice(5, 7), 16) / 255,
    ];
    app.commitTone(SRC);
  });
  bgRow.append(bgInp);
  tone.append(brS.root, gaS.root, gtS.root, viS.root, ctS.root, wlS.root, sfS.root, bgRow);

  const perf = el('div', 'section');
  perf.append(el('h3', '', 'Engine'));
  const speedRow = el('div', 'row');
  speedRow.append(el('label', '', 'Speed'));
  const speedSel = el('select') as HTMLSelectElement;
  for (const [label, val] of [['Eco', '1'], ['Balanced', '2'], ['Fast', '4'], ['Furnace', '8']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = val;
    if (val === '2') o.selected = true;
    speedSel.append(o);
  }
  speedSel.onchange = () => {
    app.renderer.passesPerFrame = parseInt(speedSel.value);
    app.renderer.resetAccumulation();
  };
  speedRow.append(speedSel);

  // Density estimation lives on the flame (JWildfire de_radius / de_curve); the
  // live preview caps the estimator radius for speed, exports use it in full.
  const deS = slider({
    label: 'DE radius', min: 0, max: 2, step: 0.05, value: app.flame.deRadius ?? 1,
    onInput: (v) => { app.flame.deRadius = v; app.commitTone(SRC); },
  });
  deS.root.title = 'Density estimation: sparse regions gather from up to radius×9 px of similar density (soft glow), dense structure stays sharp. 0 = off';
  const deCurveS = slider({
    label: 'DE curve', min: 0.05, max: 1, step: 0.05, value: app.flame.deCurve ?? 0.8,
    onInput: (v) => { app.flame.deCurve = v; app.commitTone(SRC); },
  });
  const deLiveRow = el('div', 'row');
  deLiveRow.append(el('label', '', 'DE preview'));
  const deLiveSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['Fast (r ≤ 4)', '4'], ['Balanced (r ≤ 6)', '6'], ['Full', '-1']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    if (v === '6') o.selected = true;
    deLiveSel.append(o);
  }
  deLiveSel.title = 'Cap on the DE radius while previewing (exports always use the full radius)';
  deLiveSel.onchange = () => { app.renderer.deLiveCap = parseInt(deLiveSel.value); };
  deLiveRow.append(deLiveSel);

  const osRow = el('div', 'row');
  osRow.append(el('label', '', 'Oversample'));
  const osSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['1× (fast)', '1'], ['2× (AA)', '2']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    osSel.append(o);
  }
  osSel.title = 'Render the histogram at 2× resolution and box-downsample — smoother edges, ~4× slower convergence';
  osSel.onchange = () => {
    const applied = app.renderer.setOversample(parseInt(osSel.value));
    if (applied !== parseInt(osSel.value)) {
      osSel.value = String(applied);
      alert('Not enough GPU buffer headroom for 2× oversampling at this canvas size.');
    }
  };
  osRow.append(osSel);

  const qRow = el('div', 'row');
  qRow.append(el('label', '', 'Quality cap'));
  const qSel = el('select') as HTMLSelectElement;
  for (const q of [500, 1000, 2000, 4000, 10000]) {
    const o = el('option', '', String(q)) as HTMLOptionElement;
    o.value = String(q);
    if (q === 4000) o.selected = true;
    qSel.append(o);
  }
  qSel.onchange = () => { app.renderer.targetQuality = parseInt(qSel.value); };
  qRow.append(qSel);

  const pauseBtn = el('button', '', '⏸ Pause');
  pauseBtn.onclick = () => {
    const p = !app.renderer.isPaused();
    app.renderer.setPaused(p);
    pauseBtn.textContent = p ? '▶ Resume' : '⏸ Pause';
  };
  const restartBtn = el('button', '', '↻ Re-render');
  restartBtn.onclick = () => app.renderer.resetAccumulation();
  const pRow = el('div', 'btn-row');
  pRow.append(pauseBtn, restartBtn);
  perf.append(speedRow, deS.root, deCurveS.root, deLiveRow, osRow, qRow, pRow);

  const io = el('div', 'section');
  io.append(el('h3', '', 'Export / Import'));
  const ioRow = el('div', 'btn-row');
  const pngBtn = el('button', 'primary', '⬇ Save PNG');
  const baseName = () => (app.flame.name || 'wilderfire').replace(/[\\/:*?"<>|]+/g, '_');
  pngBtn.onclick = async () => {
    // Ask for the destination first: the dialog needs the click's user gesture.
    const target = await pickSave({ suggestedName: `${baseName()}.png`, description: 'PNG image', mime: 'image/png', ext: '.png' });
    if (!target) return;
    const blob = await app.renderer.exportPNG();
    if (blob) await target.write(blob);
  };
  const jsonBtn = el('button', '', '⬇ JSON');
  jsonBtn.onclick = () =>
    saveText(flameToJSON(app.flame), { suggestedName: `${baseName()}.json`, description: 'WilderFire flame (JSON)', mime: 'application/json', ext: '.json' });
  const xmlBtn = el('button', '', '⬇ .flame');
  xmlBtn.title = 'Export as .flame XML (flam3 / Apophysis compatible)';
  xmlBtn.onclick = () =>
    saveText(flameToXML(app.flame, { curves: app.getCurves() }), { suggestedName: `${baseName()}.flame`, description: 'Flame XML', mime: 'application/xml', ext: '.flame' });
  const loadBtn = el('button', '', '⬆ Load');
  loadBtn.title = 'Load a WilderFire JSON or a .flame XML (flam3 / Apophysis compatible)';
  const fileInp = el('input') as HTMLInputElement;
  fileInp.type = 'file';
  fileInp.accept = '.json,.flame,.xml,application/json,application/xml,text/xml';
  fileInp.style.display = 'none';
  fileInp.onchange = async () => {
    const f = fileInp.files?.[0];
    if (!f) return;
    try {
      const { flame, count, unknown, curves } = importFlameText(await f.text(), app.activeLayer.palette);
      app.setFlame(flame);
      if (curves.length) {
        app.setCurves(curves);
        console.info(`Loaded ${curves.length} motion curve${curves.length > 1 ? 's' : ''} from the file (Anim tab).`);
      }
      if (count > 1) {
        console.info(`File contained ${count} flames — loaded the first ("${flame.name}").`);
      }
      if (unknown.length) {
        console.warn(`Unsupported variations skipped: ${unknown.join(', ')}`);
        alert(`Loaded, but ${unknown.length} variation${unknown.length > 1 ? 's are' : ' is'} not supported and ${unknown.length > 1 ? 'were' : 'was'} skipped:\n${unknown.join(', ')}`);
      }
    } catch (e) {
      alert('Could not import flame: ' + (e as Error).message);
    }
    fileInp.value = '';
  };
  loadBtn.onclick = () => fileInp.click();
  ioRow.append(pngBtn, jsonBtn, xmlBtn, loadBtn, fileInp);
  io.append(ioRow);

  // Hi-res tiled export
  const hiRow = el('div', 'btn-row');
  const hiScale = el('select') as HTMLSelectElement;
  for (const s of ['2', '3', '4']) {
    const o = el('option', '', s + '×') as HTMLOptionElement;
    o.value = s;
    hiScale.append(o);
  }
  const hiQ = el('select') as HTMLSelectElement;
  for (const [label, v] of [['Fast', '250'], ['Good', '700'], ['Ultra', '1500']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    if (v === '700') o.selected = true;
    hiQ.append(o);
  }
  const alphaChk = el('input') as HTMLInputElement;
  alphaChk.type = 'checkbox';
  const alphaLab = el('label', '', ' alpha');
  alphaLab.prepend(alphaChk);
  alphaLab.title = 'Transparent background';
  alphaLab.style.color = 'var(--fg-dim)';
  const hiBtn = el('button', '', '⬇ Hi-res PNG');
  hiRow.append(hiBtn, hiScale, hiQ, alphaLab);
  io.append(hiRow);
  const hiStatus = el('div', 'hint',
    'Load accepts WilderFire JSON and .flame XML (flam3 / Apophysis compatible). Hi-res renders tiled at up to 4× screen resolution.');
  io.append(hiStatus);

  hiBtn.onclick = async () => {
    const r = app.renderer;
    const scale = parseInt(hiScale.value);
    const spp = parseInt(hiQ.value);
    const transparent = alphaChk.checked;
    const fullW = (r.width * scale) & ~1;
    const fullH = (r.height * scale) & ~1;
    const TILE = 1024, PAD = 8;
    const target = await pickSave({ suggestedName: `${baseName()}-${fullW}x${fullH}.png`, description: 'PNG image', mime: 'image/png', ext: '.png' });
    if (!target) return;
    hiBtn.disabled = true;
    r.exporting = true;
    try {
      const out = document.createElement('canvas');
      out.width = fullW;
      out.height = fullH;
      const ctx = out.getContext('2d')!;
      const tilesX = Math.ceil(fullW / TILE);
      const tilesY = Math.ceil(fullH / TILE);
      let n = 0;
      for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
          const x0 = tx * TILE, y0 = ty * TILE;
          const tw = Math.min(TILE, fullW - x0);
          const th = Math.min(TILE, fullH - y0);
          // Render with padding so the DE filter doesn't seam at tile edges.
          const pw = tw + 2 * PAD, ph = th + 2 * PAD;
          const px = await r.renderRegion({
            fullW, fullH, tileX: x0 - PAD, tileY: y0 - PAD,
            tileW: pw, tileH: ph, spp, transparent,
          });
          const img = new ImageData(tw, th);
          for (let y = 0; y < th; y++) {
            const srcOff = ((y + PAD) * pw + PAD) * 4;
            img.data.set(px.subarray(srcOff, srcOff + tw * 4), y * tw * 4);
          }
          ctx.putImageData(img, x0, y0);
          n++;
          hiStatus.textContent = `Hi-res: tile ${n}/${tilesX * tilesY} (${fullW}×${fullH})…`;
        }
      }
      hiStatus.textContent = 'Encoding PNG…';
      const blob = await new Promise<Blob | null>((res) => out.toBlob(res, 'image/png'));
      if (blob) {
        await target.write(blob);
        hiStatus.textContent = `Saved ${fullW}×${fullH} PNG (${(blob.size / 1e6).toFixed(1)} MB).`;
      }
    } catch (e) {
      hiStatus.textContent = '⚠ ' + (e as Error).message;
    } finally {
      r.exporting = false;
      r.setFlame(app.flame);
      hiBtn.disabled = false;
    }
  };

  root.append(cam, dof, tone, perf, io);

  app.on('flame', (src) => {
    if (src === SRC) return;
    zoomS.set(Math.log2(app.flame.zoom));
    rotS.set((app.flame.rotation * 180) / Math.PI);
    pitchS.set(app.flame.camPitch); yawS.set(app.flame.camYaw); perspS.set(app.flame.camPersp); camZS.set(app.flame.camPosZ);
    pzChk.checked = app.flame.preserveZ;
    dofS.set(app.flame.camDOF ?? 0); dofAreaS.set(app.flame.camDOFArea ?? 0.5); focusZS.set(app.flame.focusZ ?? 0); fadeS.set(app.flame.camDOFFade ?? 1);
    dimS.set(app.flame.dimishZ ?? 0); dimDistS.set(app.flame.dimZDist ?? 0); dimColInp.value = toHex3(app.flame.dimZColor ?? [0, 0, 0]);
    brS.set(app.flame.brightness);
    gaS.set(app.flame.gamma);
    gtS.set(app.flame.gammaThreshold ?? 0.04);
    viS.set(app.flame.vibrancy);
    ctS.set(app.flame.contrast ?? 1); wlS.set(app.flame.whiteLevel ?? 220); sfS.set(app.flame.filterRadius ?? 0);
    deS.set(app.flame.deRadius ?? 1); deCurveS.set(app.flame.deCurve ?? 0.8);
    bgInp.value = toHex(app.flame.background);
  });
}
