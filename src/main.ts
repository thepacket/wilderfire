// WilderFire — GPU-native fractal flame editor for the browser.
// Copyright © 2026 Andre Paquette. SPDX-License-Identifier: LGPL-2.1-or-later
// See LICENSE and NOTICE.md (third-party material, notably JWildfire-derived code).
import './style.css';
import { App, el } from './ui/common';
import { FlameRenderer } from './gpu/renderer';
import { PRESETS } from './core/presets';
import { randomFlame } from './core/random';
import { buildTransformsPanel } from './ui/transformsPanel';
import { buildRenderPanel } from './ui/renderPanel';
import { buildPalettePanel } from './ui/palettePanel';
import { buildAIPanel } from './ui/aiPanel';
import { buildAnimPanel, type AnimState } from './ui/animPanel';
import { buildLibrary, loadAutosave, restoreFlame } from './ui/library';
import { buildMutate } from './ui/mutate';
import { buildOverlay } from './ui/overlay';

const LS_THEME = 'wilderfire.theme';

function initTheme() {
  const saved = localStorage.getItem(LS_THEME);
  const prefersLight = window.matchMedia?.('(prefers-color-scheme: light)').matches;
  const theme = saved ?? (prefersLight ? 'light' : 'dark');
  document.documentElement.dataset.theme = theme;
  return theme;
}

function fatal(msg: string) {
  const div = el('div', 'fatal');
  const box = el('div', 'box');
  box.append(el('h2', '', 'WilderFire needs WebGPU'), el('p', '', msg));
  box.append(el('p', 'hint', 'Use a recent Chrome, Edge, or Safari 18+ (Firefox: enable dom.webgpu.enabled).'));
  div.append(box);
  document.body.append(div);
}

async function boot() {
  initTheme();
  const app = new App();
  const root = document.getElementById('app')!;

  // ---------- Header ----------
  const header = el('div', 'header');
  const logo = el('div', 'logo');
  const w1 = el('span', 'wf-wilder', 'Wilder');
  const w2 = el('span', 'wf-fire', 'Fire');
  logo.append(w1, w2);

  const presetSel = el('select') as HTMLSelectElement;
  {
    const o = el('option', '', 'Tests') as HTMLOptionElement;
    o.value = '';
    presetSel.append(o);
    presetSel.title = 'Test flames: the sample flames bundled with JWildfire';
  }
  // one list: the JWildfire sample flames (src/core/samples.ts)
  const { JWF_SAMPLES } = await import('./core/samples');
  JWF_SAMPLES.forEach((s, i) => {
    const o = el('option', '', s.name) as HTMLOptionElement;
    o.value = 'j:' + i;
    presetSel.append(o);
  });
  const loadSample = async (s: { file: string; name: string }) => {
    const { importFlameText } = await import('./core/flameXML');
    const text = await (await fetch('/flames/' + s.file)).text();
    const { flame } = importFlameText(text, app.activeLayer.palette);
    flame.name = s.name;
    app.setFlame(flame);
  };
  presetSel.onchange = async () => {
    const v = presetSel.value;
    if (v.startsWith('j:')) {
      const s = JWF_SAMPLES[parseInt(v.slice(2))];
      if (!s) return;
      // the selected test's name stays displayed (the select keeps its value)
      try { await loadSample(s); } catch (e) { console.error('Sample load failed:', e); presetSel.value = ''; }
    }
  };

  const randBtn = el('button', 'primary', '🎲 Randomize');
  randBtn.onclick = () => app.setFlame(randomFlame());

  const mutBtn = el('button', '', '🧬 Mutate');
  mutBtn.title = 'Explore mutations of the current flame';
  const saveBtn = el('button', 'icon', '💾');
  saveBtn.title = 'Save to library';
  const libBtn = el('button', '', '📚 Library');

  const undoBtn = el('button', 'icon', '↩');
  undoBtn.title = 'Undo (Ctrl/Cmd+Z)';
  undoBtn.onclick = () => app.undo();
  const redoBtn = el('button', 'icon', '↪');
  redoBtn.title = 'Redo (Ctrl/Cmd+Shift+Z)';
  redoBtn.onclick = () => app.redo();
  app.on('history', () => {
    undoBtn.disabled = !app.canUndo();
    redoBtn.disabled = !app.canRedo();
    undoBtn.style.opacity = app.canUndo() ? '1' : '0.4';
    redoBtn.style.opacity = app.canRedo() ? '1' : '0.4';
  });
  window.addEventListener('keydown', (e) => {
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) app.redo();
      else app.undo();
      return;
    }
    // Arrow keys nudge the selected transform's translation (Shift = coarse)
    const nudges: Record<string, [number, number]> = {
      ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1],
    };
    const n = nudges[e.key];
    if (n) {
      const ly = app.activeLayer;
      const x = app.selected === -1 ? ly.final : ly.xforms[app.selected];
      if (!x) return;
      e.preventDefault();
      const step = e.shiftKey ? 0.1 : 0.02;
      x.affine[2] += n[0] * step;
      x.affine[5] += n[1] * step;
      app.commit('nudge');
    }
  });

  const triBtn = el('button', '', '△ Triangles');
  const themeBtn = el('button', '', document.documentElement.dataset.theme === 'dark' ? '☀' : '🌙');
  themeBtn.title = 'Toggle theme';
  themeBtn.onclick = () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem(LS_THEME, next);
    themeBtn.textContent = next === 'dark' ? '☀' : '🌙';
  };

  const spacer = el('div', 'spacer');
  header.append(logo, presetSel, randBtn, mutBtn, undoBtn, redoBtn, spacer, saveBtn, libBtn, triBtn, themeBtn);

  // ---------- Layout ----------
  const main = el('div', 'main');
  const left = el('div', 'panel panel-left');
  const wrap = el('div', 'canvas-wrap');
  const right = el('div', 'panel panel-right');

  const canvas = el('canvas') as HTMLCanvasElement;
  canvas.id = 'render';
  const overlayCanvas = el('canvas') as HTMLCanvasElement;
  overlayCanvas.id = 'overlay';
  const status = el('div', 'statusbar', '—');
  wrap.append(canvas, overlayCanvas, status);

  // Collapse handles: a slim strip on the inner edge of each pane that stays
  // visible when the pane is hidden, so it can always be brought back.
  const makeHandle = (panel: HTMLElement, side: 'left' | 'right') => {
    const key = `wilderfire.pane.${side}`;
    const h = el('div', `pane-handle pane-handle-${side}`);
    const glyph = el('span', 'glyph');
    h.append(glyph);
    const apply = (collapsed: boolean) => {
      panel.classList.toggle('collapsed', collapsed);
      h.classList.toggle('collapsed', collapsed);
      const open = side === 'left' ? '‹' : '›';
      const closed = side === 'left' ? '›' : '‹';
      glyph.textContent = collapsed ? closed : open;
      h.title = `${collapsed ? 'Show' : 'Hide'} ${side} panel (${side === 'left' ? '[' : ']'})`;
      localStorage.setItem(key, collapsed ? '1' : '0');
    };
    const toggle = () => apply(!panel.classList.contains('collapsed'));
    h.onclick = toggle;
    apply(localStorage.getItem(key) === '1');
    return { el: h, toggle };
  };
  const leftHandle = makeHandle(left, 'left');
  const rightHandle = makeHandle(right, 'right');
  window.addEventListener('keydown', (e) => {
    const tag = (e.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === '[') { e.preventDefault(); leftHandle.toggle(); }
    else if (e.key === ']') { e.preventDefault(); rightHandle.toggle(); }
  });

  main.append(left, leftHandle.el, wrap, rightHandle.el, right);
  root.append(header, main);

  // ---------- Right panel tabs ----------
  const tabs = el('div', 'tabs');
  const tabDefs = ['Render', 'Gradient', 'Anim', 'AI'] as const;
  const bodies: HTMLElement[] = [];
  tabDefs.forEach((name, i) => {
    const b = el('button', i === 0 ? 'active' : '', name);
    b.onclick = () => {
      tabs.querySelectorAll('button').forEach((x, k) => x.classList.toggle('active', k === i));
      bodies.forEach((x, k) => x.classList.toggle('active', k === i));
    };
    tabs.append(b);
    const body = el('div', 'tab-body' + (i === 0 ? ' active' : ''));
    bodies.push(body);
  });
  right.append(tabs, ...bodies);

  // ---------- Renderer ----------
  const renderer = new FlameRenderer(canvas);
  app.renderer = renderer;
  const saved = loadAutosave();
  app.flame = saved ? restoreFlame(saved as { flame: unknown }, PRESETS[0].make()) : PRESETS[0].make();

  try {
    await renderer.init();
  } catch (e) {
    fatal((e as Error).message);
    return;
  }
  // first visit (no autosave): start on the first sample flame; the built-in fallback stays if the fetch fails
  if (!saved) loadSample(JWF_SAMPLES[0]).then(() => { presetSel.value = 'j:0'; }).catch((e) => console.warn('Sample load failed:', e));

  // Size canvas to container
  const fit = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = wrap.getBoundingClientRect();
    // Even dimensions keep video encoders happy.
    renderer.resize(Math.round(r.width * dpr) & ~1, Math.round(r.height * dpr) & ~1);
    overlay.sync();
  };

  renderer.onError = (msg) => { status.textContent = '⚠ ' + msg; };
  renderer.onFrame = (s) => {
    const mps = s.samplesPerSec / 1e6;
    status.textContent =
      `quality ${s.spp.toFixed(0)} spp` +
      (s.converged ? ' · done' : s.paused ? ' · paused' : ` · ${mps.toFixed(0)} M iters/s`) +
      ` · ${renderer.width}×${renderer.height}`;
  };

  // ---------- Panels ----------
  const overlay = buildOverlay(app, overlayCanvas, wrap);
  buildTransformsPanel(app, left);
  buildRenderPanel(app, bodies[0]);
  buildPalettePanel(app, bodies[1]);
  const anim = buildAnimPanel(app, bodies[2], overlay);
  buildAIPanel(app, bodies[3]);
  if (saved?.anim) anim.setState(saved.anim as AnimState);
  const library = buildLibrary(app, anim);
  saveBtn.onclick = () => {
    library.save();
    saveBtn.textContent = '✓';
    setTimeout(() => { saveBtn.textContent = '💾'; }, 900);
  };
  libBtn.onclick = () => library.open();
  const mutate = buildMutate(app);
  mutBtn.onclick = () => { mutate.open(); };

  triBtn.onclick = () => {
    overlay.setVisible(!overlay.visible);
    triBtn.classList.toggle('primary', overlay.visible);
  };
  triBtn.classList.add('primary');

  new ResizeObserver(fit).observe(wrap);
  fit();
  app.setFlame(app.flame);

  // Scripting / testing hook
  const { importFlameText, flameToXML } = await import('./core/flameXML');
  (window as any).wilderfire = {
    app,
    anim,
    importText: (text: string) => {
      const { flame, count, curves } = importFlameText(text, app.activeLayer.palette);
      app.setFlame(flame);
      if (curves.length) app.setCurves(curves);
      return count;
    },
    exportXML: () => flameToXML(app.flame, { curves: app.getCurves() }),
    // Variation oracle test (dev only; see scripts/jwf-port/README.md)
    varTest: async (opts?: { only?: string[]; verbose?: boolean; tol?: number }) => {
      const { runVarTest } = await import('./dev/varTest');
      return runVarTest(renderer.gpuDevice, opts);
    },
    varShader: async (name: string, source?: 'hand' | 'jwf') => (await import('./dev/varTest')).shaderFor(name, source),
    // Run varTest in batches (kinder to the GPU than one 870-variation sweep) and save the merged verdicts
    varTestAll: async (batch = 120) => {
      const { runVarTest, saveVerified, allVarNames } = await import('./dev/varTest');
      const names = await allVarNames();
      const results = [];
      for (let i = 0; i < names.length; i += batch) {
        results.push(...await runVarTest(renderer.gpuDevice, { only: names.slice(i, i + batch), save: false }));
        console.log(`varTestAll ${Math.min(i + batch, names.length)}/${names.length}`);
        await new Promise((r) => setTimeout(r, 200));
      }
      await saveVerified(results);
      return results;
    },
    // Render fixtures/samples/presets offscreen into compare-out/ for scripts/jwf-port/compare.py
    flameCompare: async (opts?: { only?: string[]; width?: number; quality?: number; sets?: ("fixtures" | "samples" | "presets")[]; files?: string[] }) => {
      const { runFlameCompare } = await import('./dev/flameCompare');
      return runFlameCompare(app, opts);
    },
    // Import + compile every JWildfire fixture flame (scripts/jwf-port/testflames)
    flameTest: async (opts?: { files?: string[]; verbose?: boolean }) => {
      const { runFlameTest } = await import('./dev/flameTest');
      return runFlameTest(renderer.gpuDevice, app.activeLayer.palette, opts);
    },
  };
}

boot();
