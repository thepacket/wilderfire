// WilderFire — GPU-native fractal flame editor for the browser.
// Copyright © 2026 Andre Paquette. SPDX-License-Identifier: LGPL-2.1-or-later
// See LICENSE and NOTICE.md (third-party material, notably JWildfire-derived code).
import './style.css';
import { App, el, openModal } from './ui/common';
import { FlameRenderer } from './gpu/renderer';
import { PRESETS } from './core/presets';

import { buildTransformsPanel } from './ui/transformsPanel';
import { buildRenderPanel } from './ui/renderPanel';
import { buildPalettePanel } from './ui/palettePanel';
import { buildAIPanel } from './ui/aiPanel';
import { buildAnimPanel, type AnimState } from './ui/animPanel';
import { buildLibrary, loadAutosave, restoreFlame } from './ui/library';
import { buildMutate } from './ui/mutate';
import { createRandomOptions } from './ui/randomOptions';
import { buildOverlay } from './ui/overlay';
import { loadJwfVariations } from './core/variations';

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
  // The JWildfire variation ports (~2 MB, their own cached chunk) download while the
  // shell is built and WebGPU initialises; the first render waits for both below.
  const variationsReady = loadJwfVariations();
  const app = new App();
  const root = document.getElementById('app')!;

  // ---------- Header ----------
  const header = el('div', 'header');
  const logo = el('div', 'logo');
  const w1 = el('span', 'wf-wilder', 'Wilder');
  const w2 = el('span', 'wf-fire', 'Fire');
  logo.append(w1, w2);

  // The built-in flames live in the Library dialog's ◆ Samples collection now — with pictures and
  // search — instead of a 59-line "Tests" list in the header. A first visit opens on PRESETS[0].

  // Random flame: the button, plus a caret opening the generator settings (style / symmetry / weighting
  // field) — see ui/randomOptions.ts for why they are not in the header itself.
  const randOpts = createRandomOptions();
  const randBtn = el('button', '', 'Randomize');
  randBtn.onclick = async () => {
    if (randBtn.disabled) return;
    randBtn.disabled = true;
    randBtn.classList.add('busy'); // amber while it draws; how many candidates it took is the sampler's business
    try {
      const { sampleRandomFlame } = await import('./ui/randomSampler');
      // JWildfire's LOW batch quality (8 candidates, coverage ≥ 0.32) within a few seconds — every candidate costs a kernel compile
      const f = await sampleRandomFlame(app, { style: randOpts.style, symmetry: randOpts.symmetry as never, wfield: randOpts.wfield as never, quality: 'low', budgetMs: 3000 });
      app.flameSource = undefined;
      app.setFlame(f);
    } finally { randBtn.disabled = false; randBtn.classList.remove('busy'); }
  };
  const randSplit = el('div', 'rand-split');
  randSplit.append(randBtn, randOpts.root);

  const mutBtn = el('button', '', 'Mutate');
  mutBtn.title = 'Explore mutations of the current flame';
  const shareBtn = el('button', 'icon', '🔗');
  shareBtn.title = 'Share: copy a link that opens this flame in WilderFire (the flame is in the link itself — nothing is uploaded)';
  const saveBtn = el('button', 'icon', '💾');
  saveBtn.title = 'Save to library';
  const libBtn = el('button', '', 'Library');

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
    // Arrow keys nudge the selected transform's translation (Shift = coarse) — unless a side pane has the focus
    // (a click inside it), where the arrows scroll the pane like anywhere else
    if ((e.target as HTMLElement)?.closest?.('.panel')) return;
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

  const triBtn = el('button', '', 'Triangles');
  const themeBtn = el('button', '', document.documentElement.dataset.theme === 'dark' ? '☀' : '🌙');
  themeBtn.title = 'Toggle theme';
  const setTheme = (next: 'dark' | 'light') => {
    document.documentElement.dataset.theme = next;
    localStorage.setItem(LS_THEME, next);
    themeBtn.textContent = next === 'dark' ? '☀' : '🌙';
  };
  app.theme = { get: () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'), set: setTheme };
  themeBtn.onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');

  const aboutBtn = el('button', 'icon', 'ⓘ');
  aboutBtn.title = 'About WilderFire';
  aboutBtn.onclick = () => {
    const { body } = openModal('About WilderFire');
    body.innerHTML = `
      <p><b>WilderFire</b> — a GPU-native fractal flame editor for the browser.<br>
      Copyright © 2026 Andre Paquette. Free software under the
      <a href="https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html" target="_blank" rel="noopener">GNU LGPL v2.1 or later</a>;
      source at <a href="https://github.com/thepacket/wilderfire" target="_blank" rel="noopener">github.com/thepacket/wilderfire</a>.</p>
      <p>WilderFire is an original work — the WebGPU renderer, the per-flame shader compiler, the editor,
      the animation system, the importer and the AI assistant were written for it from scratch.
      Specific, clearly delimited parts derive from
      <a href="https://github.com/thargor6/JWildfire" target="_blank" rel="noopener">JWildfire</a>
      (© Andreas Maschke and contributors, LGPL 2.1+): the mathematical formulas of the ported variations,
      the GPU helper library they use, the tone-mapping and solid-rendering (lighting) formulas, the random flame styles, and the sample flames shipped with the app.
      That is why the whole project carries the LGPL. The full list of third-party material is in
      <a href="https://github.com/thepacket/wilderfire/blob/main/NOTICE.md" target="_blank" rel="noopener">NOTICE.md</a>.</p>
    <p>Looking for flames? The JWildfire community shares hundreds of packs at
      <a href="https://www.jwfsanctuary.club/downloads/flamepacks/" target="_blank" rel="noopener">jwfsanctuary.club/downloads/flamepacks</a> —
      drop the downloaded .zip or .flame files on the canvas and they go into your library. The packs are their authors' work; check each pack's own terms.</p>`;
  };

  // The loaded flame's name, in full, where there is room for it; editing it renames the flame.
  const nameInp = el('input', 'flame-name') as HTMLInputElement;
  nameInp.type = 'text';
  nameInp.placeholder = 'untitled';
  nameInp.spellcheck = false;
  nameInp.title = 'Name of the loaded flame — click to rename';
  // Provenance next to the name: "by author" and/or where the flame came from (dropped file, zip entry, library source)
  const provEl = el('span', 'flame-prov');
  const showName = () => {
    nameInp.value = app.flame.name ?? '';
    const prov = [app.flame.author ? 'by ' + app.flame.author : '', app.flameSource ?? ''].filter(Boolean);
    provEl.textContent = prov.join(' · ');
    provEl.title = [app.flame.author ? 'Author: ' + app.flame.author : '', app.flameSource ? 'Source: ' + app.flameSource : '', app.flame.created ? 'Created: ' + app.flame.created : ''].filter(Boolean).join('\n');
    provEl.style.display = prov.length ? '' : 'none';
    nameInp.title = (app.flame.name || 'untitled') + (prov.length ? '\n' + provEl.title : '') + '\nclick to rename';
    document.title = (app.flame.name ? app.flame.name + ' — ' : '') + 'WilderFire';
  };
  nameInp.addEventListener('change', () => { app.flame.name = nameInp.value.trim(); app.commitTone('name'); showName(); });
  nameInp.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === 'Escape') { if (e.key === 'Escape') showName(); nameInp.blur(); } });
  header.append(logo, randSplit, mutBtn, undoBtn, redoBtn, nameInp, provEl, shareBtn, saveBtn, libBtn, triBtn, themeBtn, aboutBtn);

  // ---------- Layout ----------
  const main = el('div', 'main');
  const left = el('div', 'panel panel-left');
  left.tabIndex = 0; // keyboard scrolling (arrows, Page Up/Down, Home/End) once clicked
  const wrap = el('div', 'canvas-wrap');
  const right = el('div', 'panel panel-right');
  right.tabIndex = 0;

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
    else if (e.key === 's' || e.key === 'S') { e.preventDefault(); app.setSolo(!app.solo); }
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
    body.tabIndex = 0; // the scrolling element of the right pane: focusable so the keys scroll it
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
  try {
    status.textContent = 'loading variations…';
    await variationsReady;
    status.textContent = '—';
  } catch (e) {
    status.textContent = '⚠ variation registry failed to load — reload the page';
    console.error('Variation registry failed to load:', e);
  }
  // A share link (#f=…) carries a flame in the URL: it wins over the autosave and the first-visit sample.
  let linkedCurves: import('./core/motion').MotionCurve[] = [];
  let linked = false;
  try {
    const { decodeFlameHash } = await import('./core/shareLink');
    const xml = await decodeFlameHash(location.hash);
    if (xml) {
      const { importFlameText } = await import('./core/flameXML');
      const r = importFlameText(xml, app.flame.layers[0].palette);
      app.flame = r.flame;
      linkedCurves = r.curves;
      app.flameSource = 'shared link';
      linked = true;
      history.replaceState(null, '', location.pathname + location.search); // the editor owns the flame now; 🔗 makes a fresh link
    } else if (location.hash.startsWith('#f=')) {
      console.warn('The link carried a flame that could not be read.');
    }
  } catch (e) { console.warn('Share link:', e); }
  // first visit (no autosave): app.flame is already PRESETS[0] — name its origin like a library load does
  if (!saved && !linked) app.flameSource = 'WilderFire preset';

  // Size canvas to container
  const fit = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = wrap.getBoundingClientRect();
    // Even dimensions keep video encoders happy.
    renderer.resize(Math.round(r.width * dpr) & ~1, Math.round(r.height * dpr) & ~1);
    overlay.sync();
  };

  let swNote = ''; // sticky: a newer build is installed and takes over on the next load
  renderer.onError = (msg) => { status.textContent = '⚠ ' + msg; };
  renderer.onFrame = (s) => {
    const mps = s.samplesPerSec / 1e6;
    status.textContent = swNote +
      `quality ${s.spp.toFixed(0)} spp` +
      (s.converged ? (s.timedOut ? ` · stopped after ${Math.round(s.elapsedS)} s` : ' · done') : s.paused ? ' · paused' : ` · ${mps.toFixed(0)} M iters/s`) +
      (s.budgetScale < 1 && !s.converged && !s.paused ? ` · budget ${Math.round(s.budgetScale * 100)}%` : '') +
      ` · ${renderer.width}×${renderer.height}`;
  };

  // ---------- Panels ----------
  const overlay = buildOverlay(app, overlayCanvas, wrap);
  buildTransformsPanel(app, left);
  buildRenderPanel(app, bodies[0]);
  buildPalettePanel(app, bodies[1]);
  const anim = buildAnimPanel(app, bodies[2], overlay);
  app.anim = { addKey: () => anim.addKey(), play: () => anim.play(), stop: () => anim.stop(), keyCount: () => anim.keys.length };
  buildAIPanel(app, bodies[3]);
  if (saved?.anim) anim.setState(saved.anim as AnimState);
  const library = buildLibrary(app, anim);
  saveBtn.onclick = () => {
    library.save();
    saveBtn.textContent = '✓';
    setTimeout(() => { saveBtn.textContent = '💾'; }, 900);
  };
  libBtn.onclick = () => library.open();
  if (linkedCurves.length) app.setCurves(linkedCurves);
  // A flame link pasted into an already-open tab only changes the hash (no reload): load it from here.
  window.addEventListener('hashchange', async () => {
    try {
      const { decodeFlameHash } = await import('./core/shareLink');
      const xml = await decodeFlameHash(location.hash);
      if (!xml) return;
      const { importFlameText } = await import('./core/flameXML');
      const r = importFlameText(xml, app.activeLayer.palette);
      app.flameSource = 'shared link';
      app.setFlame(r.flame);
      if (r.curves.length) app.setCurves(r.curves);
      history.replaceState(null, '', location.pathname + location.search);
    } catch (e) { console.warn('Share link:', e); }
  });
  shareBtn.onclick = async () => {
    try {
      const { encodeFlameLink } = await import('./core/shareLink');
      const url = await encodeFlameLink(app.flame, app.getCurves());
      let copied = false;
      try { await navigator.clipboard.writeText(url); copied = true; } catch { /* no clipboard permission — show it instead */ }
      const kb = (url.length / 1024).toFixed(1);
      if (copied) {
        shareBtn.textContent = '✓';
        setTimeout(() => { shareBtn.textContent = '🔗'; }, 900);
        console.info(`Share link copied (${kb} KB): ${url}`);
        if (url.length > 32000) alert(`Link copied, but it is ${kb} KB — some chat apps and browsers truncate links this long. Saving a PNG (the flame is inside it) is the safer way to share this one.`);
      } else {
        prompt(`Copy this link (${kb} KB) — it opens the flame in WilderFire:`, url);
      }
    } catch (e) { alert('Could not make a link: ' + (e as Error).message); }
  };
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

  app.on('flame', showName);
  app.on('tone', (src) => { if (src !== 'name') showName(); });
  showName();

  // Drag a .flame / .json onto the canvas: one flame is loaded, a pack goes into the library.
  (await import('./ui/flameImport')).enableFlameDrop(app, wrap);

  // Offline: the build ships a service worker (vite.config.ts) that precaches this build's assets;
  // production only, so the dev server's modules are never cached. A newer build waiting to
  // take over is announced in the status bar (it takes effect on the next load).
  if (import.meta.env.PROD && 'serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').then((reg) => {
      reg.addEventListener('updatefound', () => {
        const nw = reg.installing;
        nw?.addEventListener('statechange', () => {
          if (nw.state === 'installed' && navigator.serviceWorker.controller) { swNote = '⬆ new version ready — reload · '; status.textContent = swNote + status.textContent; }
        });
      });
    }).catch((e) => console.warn('service worker:', e));
  }

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
    // obj_mesh_wf: put an OBJ file into the browser's mesh store from a script (the transform editor has the file picker)
    storeMesh: async (name: string, objText: string) => (await import('./core/meshes')).storeUserMesh(name, objText),
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
    flameCompare: async (opts?: { only?: string[]; width?: number; quality?: number; sets?: ("fixtures" | "samples" | "presets")[]; files?: string[]; prefix?: string }) => {
      const { runFlameCompare } = await import('./dev/flameCompare');
      return runFlameCompare(app, opts);
    },
    // Render-regression check against scripts/jwf-port/render-baseline.json ({ update: true } re-records it)
    renderCheck: async (opts?: { only?: string[]; update?: boolean; width?: number; quality?: number; sets?: ("fixtures" | "samples" | "presets")[]; verbose?: boolean }) => {
      const { runRenderCheck } = await import('./dev/renderCheck');
      return runRenderCheck(app, opts);
    },
    // Import + compile every JWildfire fixture flame (scripts/jwf-port/testflames)
    flameTest: async (opts?: { files?: string[]; verbose?: boolean }) => {
      const { runFlameTest } = await import('./dev/flameTest');
      return runFlameTest(renderer.gpuDevice, app.activeLayer.palette, opts);
    },
  };
}

boot();
