// Right panel — Gradient tab: presets, stop editor, .ugr/.map import.
import { App, el, slider } from './common';
import type { RGB } from '../core/flame';
import { expandStops } from '../core/flame';
import { PALETTE_PRESETS, paletteFromPreset, randomPalette, rotatePalette, drawPalette } from '../core/palette';

interface Stop { pos: number; rgb: RGB; }

const toHex = (c: RGB) => '#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
const fromHex = (h: string): RGB => [
  parseInt(h.slice(1, 3), 16) / 255,
  parseInt(h.slice(3, 5), 16) / 255,
  parseInt(h.slice(5, 7), 16) / 255,
];

/** Parse an UltraFractal/Apophysis .ugr gradient file (first gradient). */
export function parseUGR(text: string): RGB[] | null {
  const seg = text.split(/gradient:/i)[1];
  if (!seg) return null;
  const stops: [number, number, number, number][] = [];
  const re = /index=(\d+)\s+color=(\d+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(seg))) {
    const idx = parseInt(m[1]);
    const c = parseInt(m[2]); // UltraFractal packs colors as R + G·256 + B·65536
    stops.push([Math.min(idx / 399, 1), (c & 255) / 255, ((c >> 8) & 255) / 255, ((c >> 16) & 255) / 255]);
  }
  return stops.length >= 2 ? expandStops(stops) : null;
}

/** Parse a fractint-style .map (lines of "R G B", 0-255). */
export function parseMAP(text: string): RGB[] | null {
  const colors: RGB[] = [];
  for (const line of text.split('\n')) {
    const v = line.trim().split(/\s+/).map(Number);
    if (v.length >= 3 && v.slice(0, 3).every((n) => isFinite(n))) {
      colors.push([v[0] / 255, v[1] / 255, v[2] / 255]);
    }
  }
  if (colors.length < 2) return null;
  return expandStops(colors.map((c, i) => [i / (colors.length - 1), c[0], c[1], c[2]]));
}

const SRC = 'palette';

export function buildPalettePanel(app: App, root: HTMLElement) {
  const sec = el('div', 'section');
  sec.append(el('h3', '', 'Gradient'));

  const strip = el('canvas', 'grad-strip') as HTMLCanvasElement;
  strip.width = 512;
  strip.height = 26;
  sec.append(strip);

  let base: RGB[] = app.activeLayer.palette.map((c) => [...c] as RGB);
  let shift = 0;

  const apply = () => {
    app.activeLayer.palette = rotatePalette(base, shift);
    drawPalette(strip, app.activeLayer.palette);
    app.commit(SRC);
  };

  const presetRow = el('div', 'row');
  presetRow.append(el('label', '', 'Preset'));
  const sel = el('select') as HTMLSelectElement;
  const custom = el('option', '', '(current)') as HTMLOptionElement;
  custom.value = '';
  sel.append(custom);
  for (const name of Object.keys(PALETTE_PRESETS)) {
    const o = el('option', '', name) as HTMLOptionElement;
    o.value = name;
    sel.append(o);
  }
  sel.onchange = () => {
    if (!sel.value) return;
    base = paletteFromPreset(sel.value);
    shift = 0;
    shiftS.set(0);
    apply();
  };
  presetRow.append(sel);
  sec.append(presetRow);

  const shiftS = slider({
    label: 'Rotate', min: 0, max: 1, step: 0.004, value: 0,
    fmt: (v) => Math.round(v * 360) + '°',
    onInput: (v) => { shift = v; apply(); },
  });
  sec.append(shiftS.root);

  const btnRow = el('div', 'btn-row');
  const randBtn = el('button', '', '🎲 Random gradient');
  randBtn.onclick = () => {
    base = randomPalette();
    shift = 0;
    shiftS.set(0);
    sel.value = '';
    apply();
    initStops();
  };
  const impBtn = el('button', '', '⬆ .ugr/.map');
  impBtn.title = 'Import an UltraFractal .ugr or fractint .map gradient';
  const impFile = el('input') as HTMLInputElement;
  impFile.type = 'file';
  impFile.accept = '.ugr,.map,.txt';
  impFile.style.display = 'none';
  impBtn.onclick = () => impFile.click();
  impFile.onchange = async () => {
    const f = impFile.files?.[0];
    if (!f) return;
    const text = await f.text();
    const pal = f.name.toLowerCase().endsWith('.map') ? parseMAP(text) : (parseUGR(text) ?? parseMAP(text));
    if (!pal) {
      alert('Could not parse a gradient from this file.');
    } else {
      base = pal;
      shift = 0;
      shiftS.set(0);
      sel.value = '';
      apply();
      initStops();
    }
    impFile.value = '';
  };
  btnRow.append(randBtn, impBtn, impFile);
  sec.append(btnRow);
  sec.append(el('div', 'hint', 'Edits the ACTIVE layer’s gradient. Each transform picks its hue via its Color slider (position in this gradient).'));

  // ---------- Stop editor ----------
  const edSec = el('div', 'section');
  edSec.append(el('h3', '', 'Stop editor'));
  const track = el('div', 'stop-track');
  edSec.append(track);
  let stops: Stop[] = [];
  let selIdx = 0;

  const ctlRow = el('div', 'row');
  const colInp = el('input') as HTMLInputElement;
  colInp.type = 'color';
  const posInp = el('input') as HTMLInputElement;
  posInp.type = 'number';
  posInp.step = '0.01';
  posInp.min = '0';
  posInp.max = '1';
  posInp.style.width = '64px';
  const addStopBtn = el('button', 'icon', '+');
  addStopBtn.title = 'Add a stop';
  const delStopBtn = el('button', 'icon danger', '–');
  delStopBtn.title = 'Remove the selected stop';
  ctlRow.append(el('label', '', 'Stop'), colInp, posInp, addStopBtn, delStopBtn);
  edSec.append(ctlRow);
  edSec.append(el('div', 'hint', 'Drag stops along the strip; pick colors below. Changes apply live to the active layer.'));

  function initStops() {
    const pal = app.activeLayer.palette;
    stops = [0, 0.25, 0.5, 0.75, 1].map((p) => ({
      pos: p,
      rgb: [...(pal[Math.round(p * 255)] ?? [0, 0, 0])] as RGB,
    }));
    selIdx = 0;
    renderTrack();
  }

  function applyStops() {
    base = expandStops(stops.map((s) => [s.pos, s.rgb[0], s.rgb[1], s.rgb[2]]));
    shift = 0;
    shiftS.set(0);
    sel.value = '';
    apply();
  }

  function syncCtl() {
    const s = stops[selIdx];
    if (!s) return;
    colInp.value = toHex(s.rgb);
    posInp.value = String(Math.round(s.pos * 100) / 100);
  }

  function renderTrack() {
    track.textContent = '';
    stops.forEach((s, i) => {
      const m = el('div', 'stop-marker' + (i === selIdx ? ' selected' : ''));
      m.style.left = `${s.pos * 100}%`;
      m.style.background = toHex(s.rgb);
      m.addEventListener('pointerdown', (e) => {
        selIdx = i;
        renderTrack();
        m.setPointerCapture(e.pointerId);
        const rect = track.getBoundingClientRect();
        const move = (ev: PointerEvent) => {
          s.pos = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
          m.style.left = `${s.pos * 100}%`;
          syncCtl();
          applyStops();
        };
        const up = () => {
          m.removeEventListener('pointermove', move);
          m.removeEventListener('pointerup', up);
        };
        m.addEventListener('pointermove', move);
        m.addEventListener('pointerup', up);
      });
      track.append(m);
    });
    syncCtl();
  }

  colInp.addEventListener('input', () => {
    const s = stops[selIdx];
    if (!s) return;
    s.rgb = fromHex(colInp.value);
    renderTrack();
    applyStops();
  });
  posInp.addEventListener('change', () => {
    const s = stops[selIdx];
    const v = parseFloat(posInp.value);
    if (!s || !isFinite(v)) return;
    s.pos = Math.min(1, Math.max(0, v));
    renderTrack();
    applyStops();
  });
  addStopBtn.onclick = () => {
    const pal = app.activeLayer.palette;
    stops.push({ pos: 0.5, rgb: [...(pal[128] ?? [0.5, 0.5, 0.5])] as RGB });
    selIdx = stops.length - 1;
    renderTrack();
    applyStops();
  };
  delStopBtn.onclick = () => {
    if (stops.length <= 2) return;
    stops.splice(selIdx, 1);
    selIdx = Math.max(0, selIdx - 1);
    renderTrack();
    applyStops();
  };

  root.append(sec, edSec);
  drawPalette(strip, app.activeLayer.palette);
  initStops();

  const refresh = (src: string) => {
    if (src === SRC) return;
    base = app.activeLayer.palette.map((c) => [...c] as RGB);
    shift = 0;
    shiftS.set(0);
    sel.value = '';
    drawPalette(strip, app.activeLayer.palette);
    initStops();
  };
  app.on('flame', refresh);
  app.on('select', refresh);
}
