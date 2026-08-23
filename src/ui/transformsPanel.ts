// Left panel: layer stack + xform list + editor for the selected transform.
// All transform edits operate on the active layer.
import { App, el, slider, numberInput, formatNum, XFORM_COLORS } from './common';
import { compileFormula, formulaToWgsl } from '../core/formula';
import { PLOT_FAMILIES, plotFormulas, plotPreset } from '../core/plots';
import { cvarToWgsl, CVAR_DEFAULT_CODE } from '../core/cvar';
import { sattractorFormulas } from '../core/sattractor';
import { SATTRACTOR_PRESETS } from '../core/sattractorPresets';
import type { XForm } from '../core/flame';
import {
  defaultXForm, defaultLayer, cloneXForm, cloneLayer,
  rotateAffine, scaleAffine, IDENTITY, MAX_LAYERS, MAX_XFORMS, WFIELD_TYPES, defaultWeightingField,
} from '../core/flame';
import { randomPalette } from '../core/palette';
import { VARIATIONS, defaultParams } from '../core/variations';
import { storeUserMesh, userMeshNames } from '../core/meshes';
import { createVariationPicker } from './variationPicker';

const SRC = 'transforms';

export function buildTransformsPanel(app: App, root: HTMLElement) {
  let lastPicked = 'linear'; // picker selection survives editor rebuilds
  const layerSec = el('div', 'section');
  layerSec.append(el('h3', '', 'Layers'));
  const layerList = el('div', 'xform-list');
  layerSec.append(layerList);
  const layerBtns = el('div', 'btn-row');
  const addLayerBtn = el('button', '', '+ Layer');
  const dupLayerBtn = el('button', '', 'Duplicate');
  const delLayerBtn = el('button', 'danger', 'Delete');
  const upLayerBtn = el('button', 'icon', '↑');
  const dnLayerBtn = el('button', 'icon', '↓');
  upLayerBtn.title = 'Move layer up';
  dnLayerBtn.title = 'Move layer down';
  layerBtns.append(addLayerBtn, dupLayerBtn, delLayerBtn, upLayerBtn, dnLayerBtn);
  layerSec.append(layerBtns);
  const layerWeightWrap = el('div');
  layerSec.append(layerWeightWrap);

  const listSec = el('div', 'section');
  listSec.append(el('h3', '', 'Transforms'));
  const list = el('div', 'xform-list');
  listSec.append(list);

  const btnRow = el('div', 'btn-row');
  const addBtn = el('button', '', '+ Add');
  const dupBtn = el('button', '', 'Duplicate');
  const delBtn = el('button', 'danger', 'Delete');
  const finalBtn = el('button', '', '');
  // Solo: preview only the selected transform's points (dynamics unchanged; the others just do not plot)
  const soloBtn = el('button', '', 'Solo');
  soloBtn.title = 'Show only the points plotted by the selected transform (S)';
  const syncSolo = () => { soloBtn.textContent = app.solo ? 'Solo ✓' : 'Solo'; soloBtn.classList.toggle('active', app.solo); };
  soloBtn.onclick = () => { app.setSolo(!app.solo); };
  app.on('solo', syncSolo);
  btnRow.append(addBtn, dupBtn, delBtn, finalBtn, soloBtn);
  listSec.append(btnRow);

  const btnRow2 = el('div', 'btn-row');
  const upBtn = el('button', 'icon', '↑');
  const dnBtn = el('button', 'icon', '↓');
  upBtn.title = 'Move transform up (reorders xaos too)';
  dnBtn.title = 'Move transform down (reorders xaos too)';
  const copyBtn = el('button', 'icon', '⎘');
  copyBtn.title = 'Copy transform (paste into any layer or flame)';
  const pasteBtn = el('button', 'icon', '📋');
  pasteBtn.title = 'Paste copied transform';
  const symSel = el('select') as HTMLSelectElement;
  for (const n of [2, 3, 4, 5, 6, 8]) {
    const o = el('option', '', `×${n}`) as HTMLOptionElement;
    o.value = String(n);
    if (n === 3) o.selected = true;
    symSel.append(o);
  }
  symSel.title = 'Symmetry order';
  const symBtn = el('button', 'icon', '❋ Sym');
  symBtn.title = 'Add rotational symmetry transforms';
  const mirBtn = el('button', 'icon', '⇋ Mirror');
  mirBtn.title = 'Add a horizontal mirror transform';
  btnRow2.append(upBtn, dnBtn, copyBtn, pasteBtn, symSel, symBtn, mirBtn);
  listSec.append(btnRow2);

  const editorSec = el('div', 'section');
  root.append(layerSec, listSec, editorSec);

  const layer = () => app.activeLayer;

  const selectedXForm = (): XForm | null => {
    if (app.selected === -1) return layer().final;
    return layer().xforms[app.selected] ?? null;
  };

  // ---------- Layer actions ----------
  addLayerBtn.onclick = () => {
    if (app.flame.layers.length >= MAX_LAYERS) return;
    const ly = defaultLayer(randomPalette());
    ly.weight = 0.5;
    ly.xforms = [defaultXForm()];
    ly.xforms[0].affine = [0.5, 0, Math.random() - 0.5, 0, 0.5, Math.random() - 0.5];
    app.flame.layers.push(ly);
    app.layerIdx = app.flame.layers.length - 1;
    app.selected = 0;
    app.commit();
    app.emit('select');
    rebuild();
  };
  dupLayerBtn.onclick = () => {
    if (app.flame.layers.length >= MAX_LAYERS) return;
    app.flame.layers.push(cloneLayer(layer()));
    app.layerIdx = app.flame.layers.length - 1;
    app.selected = 0;
    app.commit();
    app.emit('select');
    rebuild();
  };
  delLayerBtn.onclick = () => {
    if (app.flame.layers.length <= 1) return;
    app.flame.layers.splice(app.layerIdx, 1);
    app.layerIdx = Math.max(0, app.layerIdx - 1);
    app.selected = 0;
    app.commit();
    app.emit('select');
    rebuild();
  };
  const moveLayer = (dir: number) => {
    const i = app.layerIdx, j = i + dir;
    const ls = app.flame.layers;
    if (j < 0 || j >= ls.length) return;
    [ls[i], ls[j]] = [ls[j], ls[i]];
    app.layerIdx = j;
    app.commit();
    app.emit('select');
    rebuild();
  };
  upLayerBtn.onclick = () => moveLayer(-1);
  dnLayerBtn.onclick = () => moveLayer(1);

  function rebuildLayers() {
    layerList.textContent = '';
    const multi = app.flame.layers.length > 1;
    layerSec.style.display = '';
    app.flame.layers.forEach((ly, li) => {
      const item = el('div', 'xform-item' + (app.layerIdx === li ? ' selected' : ''));
      const vis = el('input') as HTMLInputElement;
      vis.type = 'checkbox';
      vis.checked = ly.visible;
      vis.title = 'visible';
      vis.addEventListener('click', (e) => e.stopPropagation());
      vis.addEventListener('change', () => {
        ly.visible = vis.checked;
        app.commit();
      });
      const sw = el('span', 'xform-swatch');
      const mid = ly.palette[128] ?? [0.5, 0.5, 0.5];
      sw.style.background = `rgb(${mid.map((v) => Math.round(v * 255)).join(',')})`;
      const name = el('span', 'xname', `L${li + 1}`);
      const info = el('span', 'xinfo',
        `${ly.xforms.length} xform${ly.xforms.length === 1 ? '' : 's'} · w ${formatNum(ly.weight)}`);
      item.append(vis, sw, name, info);
      item.onclick = () => { app.selectLayer(li); rebuild(); };
      layerList.append(item);
    });
    delLayerBtn.disabled = !multi;

    layerWeightWrap.textContent = '';
    if (multi) {
      layerWeightWrap.append(slider({
        label: `L${app.layerIdx + 1} weight`, min: 0, max: 2, step: 0.02, value: layer().weight,
        onInput: (v) => { layer().weight = v; app.commit(SRC); rebuildLayersInfoOnly(); },
      }).root);
    }
  }

  function rebuildLayersInfoOnly() {
    // Light refresh of the weight text without rebuilding inputs mid-drag.
    const items = layerList.querySelectorAll('.xform-item .xinfo');
    app.flame.layers.forEach((ly, li) => {
      const n = items[li];
      if (n) n.textContent = `${ly.xforms.length} xform${ly.xforms.length === 1 ? '' : 's'} · w ${formatNum(ly.weight)}`;
    });
  }

  // ---------- XForm actions ----------
  addBtn.onclick = () => {
    if (layer().xforms.length >= MAX_XFORMS) return;
    const x = defaultXForm();
    x.color = Math.random();
    x.affine = [0.5, 0, Math.random() - 0.5, 0, 0.5, Math.random() - 0.5];
    layer().xforms.push(x);
    app.selected = layer().xforms.length - 1;
    app.commit();
    app.emit('select');
    rebuild();
  };
  dupBtn.onclick = () => {
    const x = selectedXForm();
    if (!x || app.selected === -1 || layer().xforms.length >= MAX_XFORMS) return;
    layer().xforms.push(cloneXForm(x));
    app.selected = layer().xforms.length - 1;
    app.commit();
    app.emit('select');
    rebuild();
  };
  delBtn.onclick = () => {
    if (app.selected === -1) {
      // deleting the Final promotes the next chained one (JWildfire imports), else none
      layer().final = layer().moreFinals.length ? layer().moreFinals.shift()! : null;
      if (!layer().final) app.selected = 0;
    } else {
      if (layer().xforms.length <= 1) return;
      layer().xforms.splice(app.selected, 1);
      app.selected = Math.max(0, app.selected - 1);
    }
    app.commit();
    app.emit('select');
    rebuild();
  };
  finalBtn.onclick = () => {
    if (layer().final) {
      app.selected = -1;
    } else {
      const fx = defaultXForm();
      fx.colorSpeed = 0; fx.colorType = 'NONE'; // like JWildfire: a final transform does not recolour unless asked to
      layer().final = fx;
      app.selected = -1;
      app.commit();
    }
    app.emit('select');
    rebuild();
  };

  /** Swap two transforms and keep every xaos row consistent. */
  const swapXForms = (i: number, j: number) => {
    const ly = layer();
    [ly.xforms[i], ly.xforms[j]] = [ly.xforms[j], ly.xforms[i]];
    for (const x of ly.xforms) {
      if (!x.xaos) continue;
      while (x.xaos.length < ly.xforms.length) x.xaos.push(1);
      [x.xaos[i], x.xaos[j]] = [x.xaos[j], x.xaos[i]];
    }
  };
  const moveXForm = (dir: number) => {
    if (app.selected === -1) return;
    const i = app.selected, j = i + dir;
    if (j < 0 || j >= layer().xforms.length) return;
    swapXForms(i, j);
    app.selected = j;
    app.commit();
    app.emit('select');
    rebuild();
  };
  upBtn.onclick = () => moveXForm(-1);
  dnBtn.onclick = () => moveXForm(1);

  copyBtn.onclick = () => {
    const x = selectedXForm();
    if (x) {
      app.xformClipboard = cloneXForm(x);
      copyBtn.textContent = '✓';
      setTimeout(() => { copyBtn.textContent = '⎘'; }, 700);
    }
  };
  pasteBtn.onclick = () => {
    if (!app.xformClipboard || layer().xforms.length >= MAX_XFORMS) return;
    layer().xforms.push(cloneXForm(app.xformClipboard));
    app.selected = layer().xforms.length - 1;
    app.commit();
    app.emit('select');
    rebuild();
  };

  symBtn.onclick = () => {
    const n = parseInt(symSel.value);
    const ly = layer();
    for (let k = 1; k < n && ly.xforms.length < MAX_XFORMS; k++) {
      const a = (2 * Math.PI * k) / n;
      const x = defaultXForm();
      x.affine = [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0];
      x.colorSpeed = 0; // symmetry transforms shouldn't shift color (flam3 convention)
      x.weight = 1;
      x.color = k / n;
      ly.xforms.push(x);
    }
    app.commit();
    app.emit('select');
    rebuild();
  };
  mirBtn.onclick = () => {
    const ly = layer();
    if (ly.xforms.length >= MAX_XFORMS) return;
    const x = defaultXForm();
    x.affine = [-1, 0, 0, 0, 1, 0];
    x.colorSpeed = 0;
    x.weight = 1;
    ly.xforms.push(x);
    app.commit();
    app.emit('select');
    rebuild();
  };

  function rebuildList() {
    list.textContent = '';
    layer().xforms.forEach((x, i) => {
      const item = el('div', 'xform-item' + (app.selected === i ? ' selected' : ''));
      const sw = el('span', 'xform-swatch');
      sw.style.background = XFORM_COLORS[i % XFORM_COLORS.length];
      const name = el('span', 'xname', `T${i + 1}`);
      const info = el('span', 'xinfo',
        `${x.variations.map((v) => v.name).join(', ') || '—'} · w ${formatNum(x.weight)}`);
      item.append(sw, name, info);
      item.onclick = () => { app.select(i); rebuild(); };
      list.append(item);
    });
    if (layer().final) {
      const item = el('div', 'xform-item' + (app.selected === -1 ? ' selected' : ''));
      const sw = el('span', 'xform-swatch');
      sw.style.background = '#ffffff';
      item.append(sw, el('span', 'xname', 'Final'),
        el('span', 'xinfo', layer().final!.variations.map((v) => v.name).join(', ')));
      item.onclick = () => { app.select(-1); rebuild(); };
      list.append(item);
    }
    // further final transforms (JWildfire files can chain several): rendered and exported, shown read-only here
    layer().moreFinals.forEach((mf, k) => {
      const item = el('div', 'xform-item');
      const sw = el('span', 'xform-swatch');
      sw.style.background = '#cccccc';
      item.title = 'Additional final transform from a JWildfire file — applied after the Final transform. Remove it, or make it the Final transform to edit it.';
      const rm = el('button', 'icon', '✕');
      rm.title = 'Remove this final transform';
      rm.onclick = (e) => { e.stopPropagation(); layer().moreFinals.splice(k, 1); app.commit(); rebuild(); };
      const up = el('button', 'icon', '↑');
      up.title = 'Swap with the Final transform';
      up.onclick = (e) => { e.stopPropagation(); const f = layer().final!; layer().final = mf; layer().moreFinals[k] = f; app.commit(); rebuild(); };
      item.append(sw, el('span', 'xname', `Final ${k + 2}`), el('span', 'xinfo', mf.variations.map((v) => v.name).join(', ')), up, rm);
      list.append(item);
    });
    finalBtn.textContent = layer().final ? 'Final ✓' : '+ Final';
  }

  function rebuildEditor() {
    editorSec.textContent = '';
    const x = selectedXForm();
    if (!x) return;
    const isFinal = app.selected === -1;
    editorSec.append(el('h3', '', isFinal ? 'Final Transform' : `Transform T${app.selected + 1}`));

    // Add-variation controls (picker + Variation / Pre / Post) sit at the top of the editor.
    const addRow = el('div', 'btn-row');
    const picker = createVariationPicker(lastPicked, (n) => { lastPicked = n; });
    const sel = picker.root;
    const mkVar = () => ({ name: picker.value, weight: 0.5, params: defaultParams(picker.value) });
    const addVar = el('button', '', '+ Variation');
    addVar.onclick = () => {
      x.variations.push(mkVar());
      app.commit();
      rebuildEditor();
    };
    const addPre = el('button', 'icon', '+ Pre');
    addPre.title = 'Add to the pre stage (transforms the affine result before the main variations; add linear 1 to keep a pass-through)';
    addPre.onclick = () => {
      (x.preVariations ??= []).push(mkVar());
      app.commit();
      rebuildEditor();
    };
    const addPost = el('button', 'icon', '+ Post');
    addPost.title = 'Add to the post stage (transforms the main variation output)';
    addPost.onclick = () => {
      (x.postVariations ??= []).push(mkVar());
      app.commit();
      rebuildEditor();
    };
    addRow.append(sel, addVar, addPre, addPost);
    editorSec.append(addRow);

    if (!isFinal) {
      editorSec.append(slider({
        label: 'Weight', min: 0.01, max: 4, step: 0.01, value: x.weight,
        onInput: (v) => { x.weight = v; app.commit(SRC); },
      }).root);
    }
    editorSec.append(slider({
      label: 'Color', min: 0, max: 1, step: 0.005, value: x.color,
      onInput: (v) => { x.color = v; app.commit(SRC); },
    }).root);
    editorSec.append(slider({
      label: 'Color speed', min: 0, max: 1, step: 0.01, value: x.colorSpeed,
      onInput: (v) => {
        x.colorSpeed = v;
        // NONE (no colour step) is how a final says "no recolouring"; moving its speed up asks for DIFFUSION, back to 0 for none
        if (v > 0 && x.colorType === 'NONE') delete x.colorType;
        else if (v === 0 && x.colorType === undefined && x === layer().final) x.colorType = 'NONE';
        app.commit(SRC);
      },
    }).root);
    editorSec.append(slider({
      label: 'Opacity', min: 0, max: 1, step: 0.01, value: x.opacity,
      onInput: (v) => { x.opacity = v; app.commit(SRC); },
    }).root);

    // Affine grid
    editorSec.append(el('h3', '', 'Affine  (a b c / d e f)'));
    const grid = el('div', 'affine-grid');
    x.affine.forEach((v, i) => {
      const inp = numberInput(v, 0.01, (nv) => { x.affine[i] = nv; app.commit(SRC); });
      grid.append(inp);
    });
    editorSec.append(grid);

    const aRow = el('div', 'btn-row');
    const mk = (label: string, fn: () => void) => {
      const b = el('button', 'icon', label);
      b.onclick = () => { fn(); app.commit(); rebuildEditor(); };
      return b;
    };
    aRow.append(
      mk('⟲ 15°', () => { x.affine = rotateAffine(x.affine, Math.PI / 12); }),
      mk('⟳ 15°', () => { x.affine = rotateAffine(x.affine, -Math.PI / 12); }),
      mk('× 1.1', () => { x.affine = scaleAffine(x.affine, 1.1); }),
      mk('÷ 1.1', () => { x.affine = scaleAffine(x.affine, 1 / 1.1); }),
      mk('Reset', () => { x.affine = [...IDENTITY] as typeof x.affine; }),
    );
    editorSec.append(aRow);

    // Post affine (collapsed unless non-identity)
    const postIsId = x.post.every((v, i) => Math.abs(v - IDENTITY[i]) < 1e-9);
    const postHead = el('h3', '', `Post affine ${postIsId ? '(identity)' : ''}`);
    postHead.style.cursor = 'pointer';
    editorSec.append(postHead);
    const postGrid = el('div', 'affine-grid');
    postGrid.style.display = postIsId ? 'none' : 'grid';
    postHead.onclick = () => {
      postGrid.style.display = postGrid.style.display === 'none' ? 'grid' : 'none';
    };
    x.post.forEach((v, i) => {
      postGrid.append(numberInput(v, 0.01, (nv) => { x.post[i] = nv; app.commit(SRC); }));
    });
    editorSec.append(postGrid);

    // 3D affines: yz / zx planes (+ post), collapsed unless any is set
    const planes = [['yz', 'YZ (y′ = a·y + b·z + c, z′ = d·y + e·z + f)'], ['zx', 'ZX (x′ = a·x + b·z + c, z′ = d·x + e·z + f)'], ['yzPost', 'YZ post'], ['zxPost', 'ZX post']] as const;
    const any3d = planes.some(([k]) => x[k]);
    const h3d = el('h3', '', `3D affines ${any3d ? '' : '(identity)'}`);
    h3d.style.cursor = 'pointer';
    h3d.title = 'Extra affine planes applied after the 2D affine (yz, then zx); post variants after the post affine.';
    editorSec.append(h3d);
    const box3d = el('div', '');
    box3d.style.display = any3d ? 'block' : 'none';
    h3d.onclick = () => { box3d.style.display = box3d.style.display === 'none' ? 'block' : 'none'; };
    for (const [key, label] of planes) {
      const cap = el('div', 'hint', label);
      cap.style.margin = '4px 0 0';
      const g = el('div', 'affine-grid');
      const cur = x[key] ?? IDENTITY;
      cur.forEach((v, i) => {
        g.append(numberInput(v, 0.01, (nv) => {
          const a = [...(x[key] ?? IDENTITY)] as typeof IDENTITY;
          a[i] = nv;
          if (a.every((q, k) => Math.abs(q - IDENTITY[k]) < 1e-12)) delete x[key]; else x[key] = a;
          app.commit(SRC);
        }));
      });
      box3d.append(cap, g);
    }
    editorSec.append(box3d);

    // Weighting field (JWildfire): noise that scales this transform's variation amounts / params / colour and jitters its output
    {
      const wfHead = el('h3', '', `Weighting field ${x.wfield ? '(' + x.wfield.type.toLowerCase().replace(/_/g, ' ') + ')' : '(none)'}`);
      wfHead.style.cursor = 'pointer';
      wfHead.title = 'JWildfire weighting field: a noise value at each point scales the variation amounts, chosen variation params, the colour and jitters the output';
      editorSec.append(wfHead);
      const wfBox = el('div');
      wfBox.style.display = x.wfield ? 'block' : 'none';
      wfHead.onclick = () => { wfBox.style.display = wfBox.style.display === 'none' ? 'block' : 'none'; };
      const selRow = (label: string, opts: string[], value: string, on: (v: string) => void) => {
        const row = el('div', 'row');
        row.append(el('label', '', label));
        const sel = el('select') as HTMLSelectElement;
        for (const o of opts) { const op = el('option', '', o.toLowerCase().replace(/_/g, ' ')) as HTMLOptionElement; op.value = o; sel.append(op); }
        sel.value = value;
        sel.onchange = () => on(sel.value);
        row.append(sel);
        return row;
      };
      wfBox.append(selRow('Type', ['NONE', ...WFIELD_TYPES], x.wfield?.type ?? 'NONE', (v) => {
        if (v === 'NONE') delete x.wfield; else x.wfield = { ...(x.wfield ?? defaultWeightingField(v)), type: v };
        app.commit(); rebuildEditor();
      }));
      const wf = x.wfield;
      if (wf) {
        const num = (label: string, key: keyof typeof wf, min: number, max: number, step: number) =>
          wfBox.append(slider({ label, min, max, step, value: wf[key] as number, onInput: (v) => { (wf as unknown as Record<string, number>)[key] = v; app.commit(SRC); } }).root);
        wfBox.append(selRow('Input', ['AFFINE', 'POSITION'], wf.input, (v) => { wf.input = v as typeof wf.input; app.commit(); }));
        num('Var amount', 'varAmount', -5, 5, 0.05);
        num('Colour', 'color', -10, 10, 0.1);
        num('Jitter', 'jitter', -10, 10, 0.1);
        num('Seed', 'seed', 0, 10000, 1);
        num('Frequency', 'frequency', 0, 10, 0.05);
        if (/FRACTAL/.test(wf.type)) {
          wfBox.append(selRow('Fractal', ['FBM', 'BILLOW', 'RIGID_MULTI'], wf.fractalType, (v) => { wf.fractalType = v as typeof wf.fractalType; app.commit(); }));
          num('Octaves', 'octaves', 1, 8, 1);
          num('Gain', 'gain', 0, 1, 0.01);
          num('Lacunarity', 'lacunarity', 0.5, 4, 0.05);
        }
        if (wf.type === 'CELLULAR_NOISE') {
          wfBox.append(selRow('Cell return', ['CELL_VALUE', 'DISTANCE', 'DISTANCE2', 'DISTANCE_ADD', 'DISTANCE_SUB', 'DISTANCE_MUL', 'DISTANCE_DIV'], wf.cellReturn, (v) => { wf.cellReturn = v as typeof wf.cellReturn; app.commit(); }));
          wfBox.append(selRow('Cell distance', ['EUCLIDIAN', 'MANHATTAN', 'NATURAL'], wf.cellDistance, (v) => { wf.cellDistance = v as typeof wf.cellDistance; app.commit(); }));
        }
        // up to three variation-param modulations
        const allVars = [...(x.preVariations ?? []), ...x.variations, ...(x.postVariations ?? [])].map((v) => v.name);
        wf.params.forEach((pp, k) => {
          const row = el('div', 'row');
          row.append(el('label', '', `Param ${k + 1}`));
          const vsel = el('select') as HTMLSelectElement;
          for (const n of [...new Set(allVars)]) { const op = el('option', '', n) as HTMLOptionElement; op.value = n; vsel.append(op); }
          vsel.value = pp.varName;
          const psel = el('select') as HTMLSelectElement;
          const fillP = () => { psel.textContent = ''; for (const n of ['amount', ...(VARIATIONS[vsel.value]?.params ?? []).map((d) => d.name)]) { const op = el('option', '', n) as HTMLOptionElement; op.value = n; psel.append(op); } psel.value = pp.paramName; if (!psel.value) psel.value = 'amount'; };
          fillP();
          vsel.onchange = () => { pp.varName = vsel.value; fillP(); pp.paramName = psel.value; app.commit(); };
          psel.onchange = () => { pp.paramName = psel.value; app.commit(); };
          const inp = numberInput(pp.intensity, 0.05, (nv) => { pp.intensity = nv; app.commit(SRC); });
          inp.title = 'intensity';
          const rm = el('button', 'icon danger', '✕');
          rm.onclick = () => { wf.params.splice(k, 1); app.commit(); rebuildEditor(); };
          row.append(vsel, psel, inp, rm);
          wfBox.append(row);
        });
        if (wf.params.length < 3 && allVars.length) {
          const add = el('button', 'small', '+ param modulation');
          add.onclick = () => { wf.params.push({ varName: allVars[0], paramName: 'amount', intensity: 1 }); app.commit(); rebuildEditor(); };
          wfBox.append(add);
        }
      }
      editorSec.append(wfBox);
    }

    // Variations (pre stage / main / post stage)
    const renderVarGroup = (title: string, arr: () => XForm['variations'], removable: boolean) => {
      const list = arr();
      if (!list.length && removable) return; // hide empty pre/post groups
      editorSec.append(el('h3', '', title));
      const wrap = el('div');
      list.forEach((vi, vidx) => {
        const item = el('div', 'var-item');
        const head = el('div', 'var-head');
        head.append(el('span', 'vname', vi.name));
        const wInp = numberInput(vi.weight, 0.05, (nv) => { vi.weight = nv; app.commit(SRC); });
        wInp.title = 'weight';
        const rm = el('button', 'icon danger', '✕');
        rm.onclick = () => {
          arr().splice(vidx, 1);
          if (!arr().length && title.startsWith('Pre')) delete x.preVariations;
          if (!arr().length && title.startsWith('Post')) delete x.postVariations;
          app.commit();
          rebuildEditor();
        };
        head.append(wInp, rm);
        item.append(head);
        const defs = VARIATIONS[vi.name]?.params ?? [];
        if (defs.length) {
          const pw = el('div', 'var-params');
          for (const pd of defs) {
            const vp = el('span', 'vp');
            vp.append(el('span', '', pd.name));
            vp.append(numberInput(vi.params[pd.name] ?? pd.def, 0.05, (nv) => {
              vi.params[pd.name] = nv;
              app.commit(SRC);
            }));
            pw.append(vp);
          }
          item.append(pw);
        }
        // obj_mesh_wf: the mesh file (a resource, not a number) — pick one from the browser's mesh store or load an .obj into it
        if (VARIATIONS[vi.name]?.res?.includes('obj_filename')) {
          const row = el('div', 'var-params');
          const vp = el('span', 'vp');
          vp.append(el('span', '', 'mesh file'));
          const sel = el('select') as HTMLSelectElement;
          sel.title = 'OBJ mesh this variation samples; "default cube" while none is chosen or the file is not in this browser\'s mesh store';
          const fill = async () => {
            const names = await userMeshNames().catch(() => [] as string[]);
            const cur = vi.res?.obj_filename ?? '';
            if (cur && !names.includes(cur)) names.push(cur);
            sel.textContent = '';
            const o0 = el('option', '', 'default cube') as HTMLOptionElement; o0.value = ''; sel.append(o0);
            for (const n of names) { const o = el('option', '', n) as HTMLOptionElement; o.value = n; sel.append(o); }
            sel.value = cur;
          };
          void fill();
          sel.onchange = () => {
            if (sel.value) (vi.res ??= {}).obj_filename = sel.value;
            else if (vi.res) { delete vi.res.obj_filename; if (!Object.keys(vi.res).length) delete vi.res; }
            app.commit();
          };
          const load = el('button', '', '⬆ .obj');
          load.title = 'Load a Wavefront OBJ file (v/f lines; quads are split) into the mesh store and use it here. Stored in your browser only — a shared .flame names the file, the other side needs the same file loaded.';
          const file = el('input') as HTMLInputElement;
          file.type = 'file'; file.accept = '.obj'; file.style.display = 'none';
          load.onclick = () => file.click();
          file.onchange = async () => {
            const f = file.files?.[0]; file.value = '';
            if (!f) return;
            try {
              const info = await storeUserMesh(f.name, await f.text());
              (vi.res ??= {}).obj_filename = f.name;
              await fill();
              app.commit();
              load.title = `${f.name}: ${info.vertices} vertices, ${info.faces} triangles`;
            } catch (e) { alert('Mesh load failed: ' + (e as Error).message); }
          };
          vp.append(sel, load, file);
          row.append(vp);
          item.append(row);
        }
        // subflame_wf: the nested flame (a resource) — load a .flame file, or reset to the built-in default
        if (VARIATIONS[vi.name]?.res?.includes('flame')) {
          const row = el('div', 'var-params');
          const vp = el('span', 'vp');
          const nm = () => { const x = vi.res?.flame; if (!x) return 'default sub-flame'; const m = /<flame[^>]*\sname="([^"]*)"/.exec(x); return m?.[1] || 'sub-flame'; };
          const lab = el('span', '', nm());
          const load = el('button', '', '⬆ .flame');
          load.title = 'Use a .flame file as the sub-flame (its first flame; embedded in this transform, JWildfire-compatible)';
          const file = el('input') as HTMLInputElement;
          file.type = 'file'; file.accept = '.flame,.xml'; file.style.display = 'none';
          load.onclick = () => file.click();
          file.onchange = async () => {
            const f = file.files?.[0]; file.value = '';
            if (!f) return;
            const text = (await f.text()).trim();
            if (!text.startsWith('<')) { alert('Not a .flame XML file'); return; }
            (vi.res ??= {}).flame = text;
            lab.textContent = nm();
            app.commit();
          };
          const reset = el('button', '', 'default');
          reset.title = 'Back to the built-in default sub-flame';
          reset.onclick = () => { if (vi.res) { delete vi.res.flame; if (!Object.keys(vi.res).length) delete vi.res; } lab.textContent = nm(); app.commit(); };
          vp.append(el('span', '', 'sub-flame'), lab, load, reset, file);
          row.append(vp);
          item.append(row);
        }
        // the plot family (yplot2d_wf … isosfplot3d_wf): the formula(s) the kernel compiles — the preset's while
        // preset_id ≥ 0, the instance's own text once edited (JWildfire's validatePresetId sets preset_id to −1 then);
        // "↺ preset" copies the preset's ranges and param_a…f as well, like refreshFormulaFromPreset
        if (PLOT_FAMILIES[vi.name]) {
          const fam = PLOT_FAMILIES[vi.name];
          const row = el('div', 'var-params sattr-formulas');
          const current = () => plotFormulas(vi.name, vi.params[fam.idParam] ?? -1, vi.res);
          const inputs: HTMLInputElement[] = [];
          for (const k of fam.formulas) {
            const vp = el('span', 'vp');
            const inp = el('input') as HTMLInputElement;
            inp.type = 'text'; inp.spellcheck = false; inp.className = 'sattr-formula';
            inp.value = current()[k];
            inp.title = `${k}: a formula of ${fam.vars.join(', ')}, param_a…param_${fam.letters[fam.letters.length - 1]} and pi (JWildfire syntax: + - * / ?: and MathLib functions). Editing it sets ${fam.idParam} to -1.`;
            inp.onchange = () => {
              const text = inp.value.trim() || fam.dflt.f[k];
              const vs: Record<string, string> = {}; for (const v of fam.vars) vs[v] = v; for (const c of fam.letters) vs['param_' + c] = 'p' + c;
              try { formulaToWgsl(text, vs); } catch (e) { alert(`${k}: ${(e as Error).message}`); inp.classList.add('bad'); return; }
              inp.classList.remove('bad');
              // the other formulas of the family keep their current (preset) text as the instance's own
              const cur = current();
              vi.res ??= {};
              for (const k2 of fam.formulas) vi.res[k2] = k2 === k ? text : cur[k2];
              vi.params[fam.idParam] = -1;
              app.commit();
            };
            inputs.push(inp);
            vp.append(el('span', '', k === 'formula' ? 'f' : k[0]), inp);
            row.append(vp);
          }
          const reset = el('button', '', '↺ preset');
          reset.title = `Take every value of the chosen ${fam.idParam} (formulas, ranges, param_a…) like JWildfire does when the preset changes`;
          reset.onclick = () => {
            const id = Math.round(vi.params[fam.idParam] ?? 0);
            const pr = plotPreset(vi.name, id);
            if (vi.res) { for (const k of fam.formulas) delete vi.res[k]; if (!Object.keys(vi.res).length) delete vi.res; }
            if (pr.id < 0) { vi.res ??= {}; for (const k of fam.formulas) vi.res[k] = pr.f[k]; vi.params[fam.idParam] = -1; }
            for (const k of fam.refresh) if (k in pr.p) vi.params[k] = pr.p[k];
            for (const c of fam.letters) vi.params['param_' + c] = pr.p['param_' + c] ?? 0;
            app.commit();
            rebuildEditor();
          };
          row.append(reset);
          item.append(row);
        }
        // c_var / pre_c_var / post_c_var: the complex function's Java method body (a resource; JWildfire's default until edited)
        else if (VARIATIONS[vi.name]?.res?.includes('code') && vi.name.endsWith('c_var')) {
          const row = el('div', 'var-params sattr-formulas');
          const vp = el('span', 'vp');
          const ta = el('textarea') as HTMLTextAreaElement;
          ta.className = 'sattr-formula cvar-code'; ta.spellcheck = false; ta.rows = 6;
          ta.value = vi.res?.code ?? CVAR_DEFAULT_CODE;
          ta.title = 'public vec2 f(vec2 z) { … } over the c_* complex helpers (c_add, c_mul, c_exp, c_ln, c_sqrt, c_pow, c_sin, c_asin, …), vec2 locals and new vec2(re, im); compiled into the kernel';
          ta.onchange = () => {
            const text = ta.value.trim();
            try { cvarToWgsl(text || CVAR_DEFAULT_CODE); } catch (e) { alert(`code: ${(e as Error).message}`); ta.classList.add('bad'); return; }
            ta.classList.remove('bad');
            if (text && text !== CVAR_DEFAULT_CODE) (vi.res ??= {}).code = text;
            else if (vi.res) { delete vi.res.code; if (!Object.keys(vi.res).length) delete vi.res; }
            app.commit();
          };
          vp.append(el('span', '', 'f(z)'), ta);
          row.append(vp);
          item.append(row);
        }
        // sattractor3D: the x/y/z formulas (resources) — empty = the preset's (shown greyed); "↺ preset" reloads every
        // preset value like JWildfire does when presetId changes
        else if (VARIATIONS[vi.name]?.res?.includes('xformula')) {
          const row = el('div', 'var-params sattr-formulas');
          const placeholders = () => sattractorFormulas(vi.params.presetId ?? 0);
          const inputs: Record<string, HTMLInputElement> = {};
          for (const k of ['x', 'y', 'z'] as const) {
            const vp = el('span', 'vp');
            const inp = el('input') as HTMLInputElement;
            inp.type = 'text'; inp.spellcheck = false; inp.className = 'sattr-formula';
            inp.value = vi.res?.[`${k}formula`] ?? '';
            inp.placeholder = placeholders()[k];
            inp.title = `d${k}/dt as a formula of x, y, z, param_a…param_h, pi (JWildfire syntax: + - * / ?: and MathLib functions). Empty = preset ${vi.params.presetId ?? 0}'s formula.`;
            inp.onchange = () => {
              const text = inp.value.trim();
              if (text) { try { compileFormula(text); } catch (e) { alert(`${k}formula: ${(e as Error).message}`); inp.classList.add('bad'); return; } }
              inp.classList.remove('bad');
              if (text) (vi.res ??= {})[`${k}formula`] = text;
              else if (vi.res) { delete vi.res[`${k}formula`]; if (!Object.keys(vi.res).length) delete vi.res; }
              app.commit();
            };
            inputs[k] = inp;
            vp.append(el('span', '', `${k}′`), inp);
            row.append(vp);
          }
          const reset = el('button', '', '↺ preset');
          reset.title = 'Take every value of the chosen presetId (formulas, start point, steps, radius, step time, param_a…h) like JWildfire does when the preset changes';
          reset.onclick = () => {
            const pr = SATTRACTOR_PRESETS[Math.round(vi.params.presetId ?? 0)];
            if (!pr) return;
            if (vi.res) { for (const k of ['xformula', 'yformula', 'zformula']) delete vi.res[k]; if (!Object.keys(vi.res).length) delete vi.res; }
            Object.assign(vi.params, { start_x: pr.start[0], start_y: pr.start[1], start_z: pr.start[2], steps: pr.steps, radius: pr.radius, stepTime: pr.stepTime });
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].forEach((c, i) => { vi.params[`param_${c}`] = pr.params[i]; });
            app.commit();
          };
          row.append(reset);
          item.append(row);
        }
        wrap.append(item);
      });
      editorSec.append(wrap);
    };
    renderVarGroup('Pre variations', () => x.preVariations ?? [], true);
    renderVarGroup('Variations', () => x.variations, false);
    renderVarGroup('Post variations', () => x.postVariations ?? [], true);

    // Xaos row (per-pair transition weights)
    if (!isFinal && layer().xforms.length > 1) {
      const xh = el('h3', '', 'Xaos → next transform');
      xh.title = 'Weight multiplier for choosing each transform right after this one (flam3 "chaos")';
      editorSec.append(xh);
      const xg = el('div', 'affine-grid');
      layer().xforms.forEach((_, j) => {
        const wrap2 = el('div');
        wrap2.style.display = 'flex';
        wrap2.style.flexDirection = 'column';
        wrap2.style.gap = '2px';
        const lab = el('span', '', `→ T${j + 1}`);
        lab.style.fontSize = '10px';
        lab.style.color = 'var(--fg-dim)';
        const inp = numberInput(x.xaos?.[j] ?? 1, 0.1, (nv) => {
          if (!x.xaos) x.xaos = layer().xforms.map(() => 1);
          while (x.xaos.length < layer().xforms.length) x.xaos.push(1);
          x.xaos[j] = Math.max(0, nv);
          app.commit(SRC);
        });
        wrap2.append(lab, inp);
        xg.append(wrap2);
      });
      editorSec.append(xg);
    }
  }

  function rebuild() {
    rebuildLayers();
    rebuildList();
    rebuildEditor();
  }

  app.on('flame', (src) => { if (src !== SRC && src !== 'overlay-view') rebuild(); });
  app.on('select', (src) => { if (src !== SRC) rebuild(); });
  rebuild();
}
