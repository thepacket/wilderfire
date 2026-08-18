// Composition panel (top of the left pane): the image stack — every layer is a flame rendered by its own
// renderer and blended into the picture (blend mode, opacity, own background, clip to what is below).
// The active layer is what the transforms/render/gradient panels and the overlay edit.
import { App, el, slider } from './common';
import { BLEND_MODES, MAX_COMP_LAYERS, defaultEscape, defaultImage } from '../core/composition';
import { cloneFlame } from '../core/flame';
import { randomFlame } from '../core/random';

const SRC = 'compose';

export function buildComposePanel(app: App, root: HTMLElement) {
  const sec = el('div', 'section');
  const head = el('h3', '', 'Composition');
  head.title = 'Image layers blended into the final picture — each one a flame with its own renderer';
  sec.append(head);
  const list = el('div', 'xform-list');
  sec.append(list);
  const btns = el('div', 'btn-row');
  const dupBtn = el('button', '', '+ Copy');
  dupBtn.title = 'New layer above this one: a copy of this flame';
  const randBtn = el('button', '', '+ Random');
  randBtn.title = 'New layer above this one: a random flame';
  const escBtn = el('button', '', '+ Escape');
  escBtn.title = 'New layer above this one: an escape-time fractal (Mandelbrot, Julia, Newton, custom formulas… — Escape tab)';
  const imgBtn = el('button', '', '+ Image');
  imgBtn.title = 'New layer above this one: a picture (PNG, JPEG, WebP…) kept in this browser\'s image store';
  const imgFile = el('input') as HTMLInputElement;
  imgFile.type = 'file'; imgFile.accept = 'image/*'; imgFile.style.display = 'none';
  imgBtn.onclick = () => imgFile.click();
  imgFile.onchange = async () => {
    const f = imgFile.files?.[0]; imgFile.value = '';
    if (!f) return;
    try {
      const { storeImage } = await import('../core/images');
      const info = await storeImage(f);
      app.addCompLayer({ image: defaultImage(info.key, info.w, info.h), name: f.name.replace(/\.[^.]+$/, '') });
    } catch (e) { alert('Image load failed: ' + (e as Error).message); }
  };
  const delBtn = el('button', 'danger', 'Delete');
  const upBtn = el('button', 'icon', '↑');
  const dnBtn = el('button', 'icon', '↓');
  upBtn.title = 'Move layer up (in front)';
  dnBtn.title = 'Move layer down (behind)';
  btns.append(dupBtn, randBtn, escBtn, imgBtn, delBtn, upBtn, dnBtn);
  sec.append(imgFile);
  sec.append(btns);
  const ctl = el('div');
  sec.append(ctl);
  root.append(sec);

  const rebuildList = () => {
    list.textContent = '';
    const layers = app.comp.layers;
    // top of the stack first, like every layer palette
    for (let i = layers.length - 1; i >= 0; i--) {
      const ly = layers[i];
      const item = el('div', 'xform-item' + (app.compIdx === i ? ' selected' : ''));
      const vis = el('input') as HTMLInputElement;
      vis.type = 'checkbox';
      vis.checked = ly.visible;
      vis.title = 'visible';
      vis.addEventListener('click', (e) => e.stopPropagation());
      vis.addEventListener('change', () => { ly.visible = vis.checked; app.commitComp(); });
      const sw = el('span', 'xform-swatch');
      const pal = ly.kind === 'flame' ? ly.flame.layers[0]?.palette : ly.kind === 'escape' ? ly.escape.palette : null;
      const mid = pal?.[128] ?? [0.5, 0.5, 0.5];
      sw.style.background = pal ? `rgb(${mid.map((v: number) => Math.round(v * 255)).join(',')})` : 'linear-gradient(135deg, #888 25%, #444 25%, #444 50%, #888 50%, #888 75%, #444 75%)';
      const name = el('span', 'xname', (ly.kind === 'escape' ? 'ƒ ' : ly.kind === 'image' ? '🖼 ' : '') + ly.name);
      name.title = ly.kind === 'escape' ? 'escape-time fractal layer (Escape tab)' : ly.kind === 'image' ? 'image layer' : 'flame layer';
      const info = el('span', 'xinfo', `${ly.blend}${ly.opacity < 1 ? ` · ${Math.round(ly.opacity * 100)}%` : ''}${ly.clip ? ' · clip' : ''}`);
      item.append(vis, sw, name, info);
      item.onclick = () => { if (app.compIdx !== i) app.selectCompLayer(i); };
      list.append(item);
    }
    const n = layers.length;
    // a document keeps at least one flame layer (the flame panels always have something to edit)
    delBtn.disabled = n <= 1 || (app.compLayer.kind === 'flame' && layers.filter((l) => l.kind === 'flame').length <= 1);
    dupBtn.disabled = n >= MAX_COMP_LAYERS || app.compLayer.kind !== 'flame';
    randBtn.disabled = n >= MAX_COMP_LAYERS;
    escBtn.disabled = n >= MAX_COMP_LAYERS;
    imgBtn.disabled = n >= MAX_COMP_LAYERS;
    upBtn.disabled = app.compIdx >= n - 1;
    dnBtn.disabled = app.compIdx <= 0;
  };

  const rebuildCtl = () => {
    ctl.textContent = '';
    const ly = app.compLayer;
    const nameRow = el('div', 'row');
    nameRow.append(el('label', '', 'Name'));
    const nameInp = el('input') as HTMLInputElement;
    nameInp.type = 'text';
    nameInp.value = ly.name;
    nameInp.addEventListener('change', () => { ly.name = nameInp.value.trim() || 'Layer'; app.commitComp(); });
    nameRow.append(nameInp);
    const blendRow = el('div', 'row');
    blendRow.append(el('label', '', 'Blend'));
    const blendSel = el('select') as HTMLSelectElement;
    for (const m of BLEND_MODES) { const o = el('option', '', m) as HTMLOptionElement; o.value = m; blendSel.append(o); }
    blendSel.value = ly.blend;
    blendSel.title = 'How this layer combines with what is below it (W3C / Photoshop blend modes)';
    blendSel.onchange = () => { ly.blend = blendSel.value as typeof ly.blend; app.commitComp(); };
    blendRow.append(blendSel);
    const opS = slider({ label: 'Opacity', min: 0, max: 1, step: 0.01, value: ly.opacity, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { ly.opacity = v; app.commitComp(SRC); } });
    const flags = el('div', 'row');
    const mk = (label: string, title: string, get: () => boolean, set: (v: boolean) => void) => {
      const lab = el('label');
      const c = el('input') as HTMLInputElement;
      c.type = 'checkbox'; c.checked = get(); c.title = title;
      c.onchange = () => { set(c.checked); app.commitComp(); };
      lab.append(c, document.createTextNode(' ' + label));
      lab.title = title;
      return lab;
    };
    if (ly.kind === 'flame') flags.append(mk('Own background', 'Draw this flame\'s background colour/gradient (opaque). Off: transparent where nothing is plotted, so the layers below show through', () => ly.ownBackground, (v) => { ly.ownBackground = v; }));
    flags.append(mk('Clip', 'Only draw where the layers below have already drawn (clipping mask)', () => ly.clip, (v) => { ly.clip = v; }));
    const bgRow = el('div', 'row');
    bgRow.append(el('label', '', 'Backdrop'));
    const bgInp = el('input') as HTMLInputElement;
    bgInp.type = 'color';
    bgInp.title = 'Composition background, visible where the whole stack is transparent';
    bgInp.value = '#' + app.comp.background.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
    bgInp.addEventListener('input', () => {
      const h = bgInp.value;
      app.comp.background = [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
      app.commitComp(SRC);
    });
    bgRow.append(bgInp);
    ctl.append(nameRow, blendRow, opS.root, flags);
    if (ly.kind === 'image') {
      const im = ly.image;
      const fitRow = el('div', 'row');
      fitRow.append(el('label', '', 'Fit'));
      const fitSel = el('select') as HTMLSelectElement;
      for (const [v, t] of [['contain', 'contain'], ['cover', 'cover'], ['stretch', 'stretch'], ['none', 'pixels']]) { const o = el('option', '', t) as HTMLOptionElement; o.value = v; fitSel.append(o); }
      fitSel.value = im.fit;
      fitSel.onchange = () => { im.fit = fitSel.value as typeof im.fit; app.commitComp(); };
      fitRow.append(fitSel);
      const tileChk = el('input') as HTMLInputElement; tileChk.type = 'checkbox'; tileChk.checked = im.tile; tileChk.title = 'repeat the picture beyond its edges';
      tileChk.onchange = () => { im.tile = tileChk.checked; app.commitComp(); };
      const tileLab = el('label'); tileLab.append(tileChk, document.createTextNode(' tile'));
      fitRow.append(tileLab);
      ctl.append(fitRow,
        slider({ label: 'Scale', min: -2, max: 2, step: 0.01, value: Math.log2(im.scale), fmt: (v) => Math.pow(2, v).toFixed(2) + '×', onInput: (v) => { im.scale = Math.pow(2, v); app.commitComp(SRC); } }).root,
        slider({ label: 'Offset X', min: -1, max: 1, step: 0.005, value: im.offsetX, onInput: (v) => { im.offsetX = v; app.commitComp(SRC); } }).root,
        slider({ label: 'Offset Y', min: -1, max: 1, step: 0.005, value: im.offsetY, onInput: (v) => { im.offsetY = v; app.commitComp(SRC); } }).root,
        slider({ label: 'Rotation', min: -180, max: 180, step: 1, value: im.rotation * 180 / Math.PI, fmt: (v) => `${v.toFixed(0)}°`, onInput: (v) => { im.rotation = v * Math.PI / 180; app.commitComp(SRC); } }).root,
        el('div', 'hint', `${im.w}×${im.h} px · stored in this browser; a saved composition file embeds it`));
    }
    ctl.append(bgRow);
  };

  const rebuild = () => { rebuildList(); rebuildCtl(); };

  dupBtn.onclick = () => { const f = cloneFlame(app.flame); f.name = (f.name || 'Flame') + ' copy'; app.addCompLayer(f); };
  randBtn.onclick = () => { app.addCompLayer(randomFlame()); };
  escBtn.onclick = () => { app.addCompLayer({ escape: defaultEscape(app.editPalette) }); };
  delBtn.onclick = () => {
    if (delBtn.disabled) return;
    app.comp.layers.splice(app.compIdx, 1);
    if (app.flameIdx > app.compIdx) app.flameIdx--;
    app.compIdx = Math.max(0, app.compIdx - 1);
    app.commitComp();
    app.emit('select');
  };
  const move = (d: number) => {
    const i = app.compIdx, j = i + d;
    if (j < 0 || j >= app.comp.layers.length) return;
    const ls = app.comp.layers;
    [ls[i], ls[j]] = [ls[j], ls[i]];
    if (app.flameIdx === i) app.flameIdx = j; else if (app.flameIdx === j) app.flameIdx = i;
    app.compIdx = j;
    app.commitComp();
  };
  upBtn.onclick = () => move(1);
  dnBtn.onclick = () => move(-1);

  app.on('comp', (src) => { if (src === SRC) rebuildList(); else rebuild(); });
  rebuild();
  return { rebuild };
}
