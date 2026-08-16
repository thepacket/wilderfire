// Right panel — Render tab: camera, tonemap, quality, export.
import { App, el, slider } from './common';
import { flameToJSON } from '../core/flame';
import { flameToXML, importFlameText } from '../core/flameXML';

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
  cam.append(zoomS.root, rotS.root);
  cam.append(el('div', 'hint', 'Drag the canvas to pan · scroll to zoom · drag triangle handles to edit transforms.'));

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
  tone.append(brS.root, gaS.root, gtS.root, viS.root, bgRow);

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

  const deRow = el('div', 'row');
  deRow.append(el('label', '', 'DE filter'));
  const deSel = el('select') as HTMLSelectElement;
  const DE_MODES: [string, number, number][] = [
    ['Off', 0, 0.4], ['Subtle', 1, 0.4], ['Medium', 2, 0.35], ['Strong', 4, 0.3], ['Max', 7, 0.28],
  ];
  DE_MODES.forEach(([label], i) => {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = String(i);
    if (label === 'Subtle') o.selected = true;
    deSel.append(o);
  });
  deSel.title = 'Density-estimation filter: smooths sparse regions, keeps dense structure crisp';
  deSel.onchange = () => {
    const [, r, a] = DE_MODES[parseInt(deSel.value)];
    app.renderer.deMaxRadius = r;
    app.renderer.deAlpha = a;
  };
  deRow.append(deSel);

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
  perf.append(speedRow, deRow, osRow, qRow, pRow);

  const io = el('div', 'section');
  io.append(el('h3', '', 'Export / Import'));
  const ioRow = el('div', 'btn-row');
  const pngBtn = el('button', 'primary', '⬇ Save PNG');
  pngBtn.onclick = async () => {
    const blob = await app.renderer.exportPNG();
    if (!blob) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${app.flame.name || 'wilderfire'}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const download = (content: string, filename: string, type: string) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([content], { type }));
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const jsonBtn = el('button', '', '⬇ JSON');
  jsonBtn.onclick = () =>
    download(flameToJSON(app.flame), `${app.flame.name || 'wilderfire'}.json`, 'application/json');
  const xmlBtn = el('button', '', '⬇ .flame');
  xmlBtn.title = 'Export as flam3 / Apophysis / JWildfire XML';
  xmlBtn.onclick = () =>
    download(flameToXML(app.flame), `${app.flame.name || 'wilderfire'}.flame`, 'application/xml');
  const loadBtn = el('button', '', '⬆ Load');
  loadBtn.title = 'Load a WilderFire JSON or a .flame XML (flam3 / Apophysis / JWildfire)';
  const fileInp = el('input') as HTMLInputElement;
  fileInp.type = 'file';
  fileInp.accept = '.json,.flame,.xml,application/json,application/xml,text/xml';
  fileInp.style.display = 'none';
  fileInp.onchange = async () => {
    const f = fileInp.files?.[0];
    if (!f) return;
    try {
      const { flame, count, unknown } = importFlameText(await f.text(), app.activeLayer.palette);
      app.setFlame(flame);
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
    'Load accepts WilderFire JSON and .flame XML from flam3 / Apophysis / JWildfire. Hi-res renders tiled at up to 4× screen resolution.');
  io.append(hiStatus);

  hiBtn.onclick = async () => {
    const r = app.renderer;
    const scale = parseInt(hiScale.value);
    const spp = parseInt(hiQ.value);
    const transparent = alphaChk.checked;
    const fullW = (r.width * scale) & ~1;
    const fullH = (r.height * scale) & ~1;
    const TILE = 1024, PAD = 8;
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
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `${app.flame.name || 'wilderfire'}-${fullW}x${fullH}.png`;
        a.click();
        URL.revokeObjectURL(a.href);
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

  root.append(cam, tone, perf, io);

  app.on('flame', (src) => {
    if (src === SRC) return;
    zoomS.set(Math.log2(app.flame.zoom));
    rotS.set((app.flame.rotation * 180) / Math.PI);
    brS.set(app.flame.brightness);
    gaS.set(app.flame.gamma);
    gtS.set(app.flame.gammaThreshold ?? 0.04);
    viS.set(app.flame.vibrancy);
    bgInp.value = toHex(app.flame.background);
  });
}
