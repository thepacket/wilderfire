// Composition panel (top of the left pane): the image stack — every layer is a flame rendered by its own
// renderer and blended into the picture (blend mode, opacity, own background, clip to what is below).
// The active layer is what the transforms/render/gradient panels and the overlay edit.
import { App, el, slider } from './common';
import { BLEND_MODES, MAX_COMP_LAYERS, defaultEscape } from '../core/composition';
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
  const delBtn = el('button', 'danger', 'Delete');
  const upBtn = el('button', 'icon', '↑');
  const dnBtn = el('button', 'icon', '↓');
  upBtn.title = 'Move layer up (in front)';
  dnBtn.title = 'Move layer down (behind)';
  btns.append(dupBtn, randBtn, escBtn, delBtn, upBtn, dnBtn);
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
      const pal = ly.kind === 'flame' ? ly.flame.layers[0]?.palette : ly.escape.palette;
      const mid = pal?.[128] ?? [0.5, 0.5, 0.5];
      sw.style.background = `rgb(${mid.map((v: number) => Math.round(v * 255)).join(',')})`;
      const name = el('span', 'xname', (ly.kind === 'escape' ? '⌘ ' : '') + ly.name);
      name.title = ly.kind === 'escape' ? 'escape-time fractal layer (Escape tab)' : 'flame layer';
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
    flags.append(
      mk('Own background', 'Draw this flame\'s background colour/gradient (opaque). Off: transparent where nothing is plotted, so the layers below show through', () => ly.ownBackground, (v) => { ly.ownBackground = v; }),
      mk('Clip', 'Only draw where the layers below have already drawn (clipping mask)', () => ly.clip, (v) => { ly.clip = v; }),
    );
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
    ctl.append(nameRow, blendRow, opS.root, flags, bgRow);
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
