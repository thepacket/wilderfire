// Escape tab: the selected escape-time layer — formula, parameters, view, colouring. The gradient comes from
// the Gradient tab (it edits the selected escape layer's palette), pan/zoom from the canvas.
import { App, el, slider, numberInput } from './common';
import { ESCAPE_FORMULAS, OUTSIDE_COLORINGS, INSIDE_COLORINGS, TRAP_SHAPES, TRANSFERS, PRECISIONS, defaultEscape, escapeTier, perturbable, escapeCentreText, escapeSetCentre, type EscapeFormula, type EscapeLayerData, type Precision } from '../core/escape';
import { compileFormula, FORMULA_VARS } from '../core/formula';

const SRC = 'escape';

export function buildEscapePanel(app: App, root: HTMLElement) {
  const wrap = el('div');
  root.append(wrap);

  const toHex = (c: number[]) => '#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
  const fromHex = (h: string): [number, number, number] => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
  const sel = (opts: readonly string[] | { value: string; label: string }[], value: string, title: string, on: (v: string) => void) => {
    const s = el('select') as HTMLSelectElement;
    for (const o of opts) { const opt = el('option', '', typeof o === 'string' ? o : o.label) as HTMLOptionElement; opt.value = typeof o === 'string' ? o : o.value; s.append(opt); }
    s.value = value; s.title = title;
    s.onchange = () => on(s.value);
    return s;
  };
  const row = (label: string, ...ctl: HTMLElement[]) => { const r = el('div', 'row'); r.append(el('label', '', label), ...ctl); return r; };
  const num = (v: number, step: number, on: (v: number) => void, title = '') => { const i = numberInput(v, step, on); if (title) i.title = title; return i; };

  function build() {
    wrap.textContent = '';
    const layer = app.escapeLayer;
    if (!layer) {
      const sec = el('div', 'section');
      sec.append(el('h3', '', 'Escape-time layer'));
      sec.append(el('div', 'hint', 'No escape-time layer is selected. Select one in the Composition list (left), or add one:'));
      const add = el('button', 'primary', '+ Escape layer');
      add.onclick = () => { app.addCompLayer({ escape: defaultEscape(app.editPalette) }); };
      sec.append(add);
      sec.append(el('div', 'hint', 'An escape-time layer iterates a formula per pixel (Mandelbrot, Julia, Burning Ship, Newton, Phoenix, Magnet, Nova, or your own z = … expression) and colours the result through the gradient — blended with the flame layers below and above it.'));
      wrap.append(sec);
      return;
    }
    const e: EscapeLayerData = layer.escape;
    const commit = () => app.commit(SRC);
    const rebuild = () => { build(); };

    // ---- formula ----
    const fsec = el('div', 'section');
    fsec.append(el('h3', '', 'Formula'));
    const formulaSel = sel(Object.entries(ESCAPE_FORMULAS).map(([k, v]) => ({ value: k, label: v.label })), e.formula, 'The iteration z ← f(z, c)', (v) => { e.formula = v as EscapeFormula; commit(); rebuild(); });
    fsec.append(row('Formula', formulaSel));
    if (e.formula === 'custom') {
      const inp = el('input') as HTMLInputElement;
      inp.type = 'text'; inp.value = e.custom; inp.style.width = '100%';
      inp.title = `z, c, pixel, n, p1..p4, i, pi, e; + - * / ^ ( ); functions: sin cos tan sinh cosh tanh exp log sqrt abs arg re im conj recip sqr cube flip floor round norm pow(z,w)`;
      const err = el('div', 'hint');
      const check = () => { try { compileFormula(inp.value); err.textContent = ''; return true; } catch (ex) { err.textContent = '⚠ ' + (ex as Error).message; return false; } };
      inp.addEventListener('input', check);
      inp.addEventListener('change', () => { if (check()) { e.custom = inp.value; commit(); } });
      check();
      const zrow = el('div', 'row'); zrow.append(el('label', '', 'z ='), inp);
      fsec.append(zrow, err);
      fsec.append(el('div', 'hint', `Variables: ${FORMULA_VARS.join(', ')} · e.g. z^2 + c, z^3 + p1*z + c, sin(z) + c, (z^2 + c) / (z^2 - c)`));
      const prow = el('div', 'affine-grid');
      e.params.forEach((pv, i) => {
        const cell = el('div'); cell.style.display = 'flex'; cell.style.gap = '2px'; cell.style.alignItems = 'center';
        cell.append(el('span', '', `p${i + 1}`), num(pv[0], 0.01, (v) => { e.params[i][0] = v; commit(); }, 're'), num(pv[1], 0.01, (v) => { e.params[i][1] = v; commit(); }, 'im'));
        prow.append(cell);
      });
      fsec.append(prow);
    }
    const modeSel = sel([{ value: 'mandelbrot', label: 'Mandelbrot (c = pixel)' }, { value: 'julia', label: 'Julia (z₀ = pixel)' }], e.mode, 'What the pixel is', (v) => { e.mode = v as 'mandelbrot' | 'julia'; commit(); rebuild(); });
    fsec.append(row('Mode', modeSel));
    if (ESCAPE_FORMULAS[e.formula].power) fsec.append(row('Power p', num(e.power, 0.1, (v) => { e.power = v; commit(); }, 'exponent of the formula')));
    if (e.mode === 'mandelbrot') fsec.append(row('Seed z₀', num(e.seed[0], 0.01, (v) => { e.seed[0] = v; commit(); }, 're'), num(e.seed[1], 0.01, (v) => { e.seed[1] = v; commit(); }, 'im')));
    else fsec.append(row('Constant c', num(e.c[0], 0.001, (v) => { e.c[0] = v; commit(); }, 're'), num(e.c[1], 0.001, (v) => { e.c[1] = v; commit(); }, 'im')));
    fsec.append(slider({ label: 'Max iterations', min: 1, max: 4, step: 0.01, value: Math.log10(e.maxIter), fmt: (v) => String(Math.round(Math.pow(10, v))), onInput: (v) => { e.maxIter = Math.round(Math.pow(10, v)); commit(); } }).root);
    fsec.append(row('Bailout', num(e.bailout, 1, (v) => { e.bailout = Math.max(1e-6, v); commit(); }, '|z| that counts as escaped (large values smooth the colouring)')));
    fsec.append(row('Antialias', sel(['1', '2', '3'], String(e.antialias), 'samples per pixel edge (n² per pixel)', (v) => { e.antialias = parseInt(v); commit(); })));
    wrap.append(fsec);

    // ---- view ----
    const vsec = el('div', 'section');
    vsec.append(el('h3', '', 'View'));
    vsec.append(el('div', 'hint', 'Drag the canvas to pan, wheel to zoom.'));
    {
      // the centre as text: deep views carry more digits than a number input holds
      const [cx, cy] = escapeCentreText(e);
      const mk = (v: string, title: string) => { const i = el('input') as HTMLInputElement; i.type = 'text'; i.value = v; i.title = title; i.style.width = '100%'; i.style.fontFamily = 'monospace'; i.style.fontSize = '11px'; return i; };
      const ix = mk(cx, 're'), iy = mk(cy, 'im');
      const apply = () => { try { escapeSetCentre(e, ix.value, iy.value); commit(); } catch { ix.style.color = 'var(--danger, #f55)'; } };
      ix.addEventListener('change', apply); iy.addEventListener('change', apply);
      const r1 = el('div', 'row'); r1.append(el('label', '', 'Centre re'), ix);
      const r2 = el('div', 'row'); r2.append(el('label', '', 'Centre im'), iy);
      vsec.append(r1, r2);
    }
    {
      const zi = el('input') as HTMLInputElement; zi.type = 'text'; zi.value = e.zoom.toPrecision(6); zi.title = '1 = the ±2 frame; scientific notation welcome (1e30)';
      zi.addEventListener('change', () => { const v = parseFloat(zi.value); if (isFinite(v) && v > 0) { e.zoom = v; commit(); rebuild(); } });
      vsec.append(row('Zoom', zi));
    }
    {
      const tier = escapeTier(e);
      const prec = sel(PRECISIONS.map((p) => ({ value: p, label: p === 'auto' ? `auto (${tier})` : p })), e.precision, 'arithmetic: f32 (fast, sharp to ~3 000×), double-single (~1e10×), perturbation (reference orbit at the exact centre, any depth — z^p + c only)', (v) => { e.precision = v as Precision; commit(); rebuild(); });
      vsec.append(row('Precision', prec));
      const note = tier === 'perturb' ? `perturbation · reference orbit at the exact centre${e.centerHi ? ` (${e.centerHi[0].length} digits)` : ''}` : tier === 'ds' ? `double-single (~1e10× useful)${perturbable(e) ? '' : ' — this formula has no perturbation path'}` : 'single precision';
      vsec.append(el('div', 'hint', note));
    }
    vsec.append(slider({ label: 'Rotation', min: -180, max: 180, step: 1, value: e.rotation * 180 / Math.PI, fmt: (v) => `${v.toFixed(0)}°`, onInput: (v) => { e.rotation = v * Math.PI / 180; commit(); } }).root);
    const resetBtn = el('button', '', 'Reset view');
    resetBtn.onclick = () => { const d = defaultEscape(e.palette); e.centerX = d.centerX; e.centerY = d.centerY; e.zoom = 1; e.rotation = 0; commit(); rebuild(); };
    vsec.append(resetBtn);
    wrap.append(vsec);

    // ---- colouring ----
    const csec = el('div', 'section');
    csec.append(el('h3', '', 'Colouring'));
    const c = e.coloring;
    csec.append(row('Outside', sel(OUTSIDE_COLORINGS, c.outside, 'colouring of escaped points', (v) => { c.outside = v as typeof c.outside; commit(); rebuild(); })));
    csec.append(row('Inside', sel(INSIDE_COLORINGS, c.inside, 'colouring of points that never escape', (v) => { c.inside = v as typeof c.inside; commit(); rebuild(); })));
    csec.append(slider({ label: 'Density', min: -3, max: 2, step: 0.01, value: Math.log10(Math.max(1e-3, c.density)), fmt: (v) => Math.pow(10, v).toPrecision(3), onInput: (v) => { c.density = Math.pow(10, v); commit(); } }).root);
    csec.append(slider({ label: 'Offset', min: 0, max: 1, step: 0.005, value: c.offset, fmt: (v) => v.toFixed(3), onInput: (v) => { c.offset = v; commit(); } }).root);
    csec.append(row('Transfer', sel(TRANSFERS, c.transfer, 'applied to the colouring value before the gradient', (v) => { c.transfer = v as typeof c.transfer; commit(); })));
    const insideCol = el('input') as HTMLInputElement; insideCol.type = 'color'; insideCol.value = toHex(c.insideColor);
    insideCol.addEventListener('input', () => { c.insideColor = fromHex(insideCol.value); commit(); });
    csec.append(row('Inside colour', insideCol));
    csec.append(slider({ label: 'Inside alpha', min: 0, max: 1, step: 0.01, value: c.insideAlpha, onInput: (v) => { c.insideAlpha = v; commit(); } }).root);
    if (c.outside === 'solid') {
      const solidCol = el('input') as HTMLInputElement; solidCol.type = 'color'; solidCol.value = toHex(c.solidColor);
      solidCol.addEventListener('input', () => { c.solidColor = fromHex(solidCol.value); commit(); });
      csec.append(row('Outside colour', solidCol));
    }
    csec.append(slider({ label: 'Outside alpha', min: 0, max: 1, step: 0.01, value: c.outsideAlpha, onInput: (v) => { c.outsideAlpha = v; commit(); } }).root);
    if (c.outside === 'orbit-trap' || c.inside === 'orbit-trap') {
      csec.append(el('h3', '', 'Orbit trap'));
      csec.append(row('Shape', sel(TRAP_SHAPES, c.trap.shape, 'the shape whose distance the orbit is measured against', (v) => { c.trap.shape = v as typeof c.trap.shape; commit(); })));
      csec.append(row('Centre', num(c.trap.x, 0.01, (v) => { c.trap.x = v; commit(); }, 're'), num(c.trap.y, 0.01, (v) => { c.trap.y = v; commit(); }, 'im')));
      csec.append(row('Size', num(c.trap.size, 0.01, (v) => { c.trap.size = v; commit(); }, 'radius of circle / ring traps')));
      const minChk = el('input') as HTMLInputElement; minChk.type = 'checkbox'; minChk.checked = c.trap.min;
      minChk.onchange = () => { c.trap.min = minChk.checked; commit(); };
      const minLab = el('label'); minLab.append(minChk, document.createTextNode(' closest approach (else the last iteration)'));
      csec.append(minLab);
    }
    csec.append(el('div', 'hint', 'The gradient is edited in the Gradient tab (it follows the selected layer).'));
    wrap.append(csec);
  }

  app.on('comp', (src) => { if (src !== SRC) build(); });
  app.on('flame', (src) => { if (src === 'overlay-view' && app.escapeLayer) build(); }); // canvas pan/zoom → view fields
  build();
  return { rebuild: build };
}
