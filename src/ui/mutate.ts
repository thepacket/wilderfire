// MutaGen-style mutation grid: 3×3 modal, center = current flame,
// surrounding cells = random mutations rendered offscreen. Click to adopt.
import { App, el, openModal } from './common';
import { cloneFlame, visibleLayers, type Flame, type XForm } from '../core/flame';
import { SAFE_VARIATIONS } from '../core/random';
import { defaultParams } from '../core/variations';
import { rotatePalette, randomPalette } from '../core/palette';

const rr = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
const pick = <T,>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function mutateXForm(x: XForm) {
  const op = Math.floor(Math.random() * 6);
  switch (op) {
    case 0: // jitter affine
      for (let i = 0; i < 6; i++) x.affine[i] += rr(-0.12, 0.12);
      break;
    case 1:
      x.weight = Math.max(0.05, x.weight * rr(0.6, 1.6));
      break;
    case 2:
      x.color = clamp01(x.color + rr(-0.25, 0.25));
      x.colorSpeed = clamp01(x.colorSpeed + rr(-0.15, 0.15));
      break;
    case 3: { // jitter a variation weight
      const v = pick(x.variations);
      if (v) v.weight *= rr(0.55, 1.6);
      break;
    }
    case 4: { // jitter variation params
      const v = pick(x.variations);
      if (v) for (const k of Object.keys(v.params)) v.params[k] *= rr(0.7, 1.35);
      break;
    }
    case 5: { // swap a variation
      const i = Math.floor(Math.random() * x.variations.length);
      if (x.variations[i]) {
        const name = pick(SAFE_VARIATIONS);
        x.variations[i] = { name, weight: x.variations[i].weight, params: defaultParams(name) };
      }
      break;
    }
  }
}

export function mutateFlame(base: Flame): Flame {
  const f = cloneFlame(base);
  const layers = visibleLayers(f);
  const ly = pick(layers);
  const nOps = 1 + Math.floor(Math.random() * 3);
  for (let i = 0; i < nOps; i++) {
    const r = Math.random();
    if (r < 0.72) {
      mutateXForm(pick(ly.xforms));
    } else if (r < 0.86) {
      ly.palette = rotatePalette(ly.palette, Math.random());
    } else if (r < 0.94) {
      ly.palette = randomPalette();
    } else if (ly.xforms.length < 8) {
      const name = pick(SAFE_VARIATIONS);
      pick(ly.xforms).variations.push({ name, weight: rr(0.2, 0.5), params: defaultParams(name) });
    }
  }
  f.name = base.name.replace(/-m\d+$/, '') + '-m' + Math.floor(Math.random() * 100);
  return f;
}

export function buildMutate(app: App) {
  const CELL = 240;
  const SPP = 220;

  async function renderCell(flame: Flame, canvas: HTMLCanvasElement) {
    const r = app.renderer.layerRenderer; // the mutated flame alone, not the whole composition
    r.setFlame(flame);
    const px = await r.renderRegion({
      fullW: CELL, fullH: CELL, tileX: 0, tileY: 0, tileW: CELL, tileH: CELL,
      spp: SPP, transparent: false,
    });
    canvas.getContext('2d')!.putImageData(new ImageData(px, CELL, CELL), 0, 0);
  }

  async function open() {
    const base = cloneFlame(app.flame);
    const { body, close } = openModal('Mutations — click one to adopt it');
    const grid = el('div', 'mut-grid');
    body.append(grid);
    const refreshBtn = el('button', 'primary', '↻ New mutations');
    const hint = el('div', 'hint', 'Rendering…');
    const btnRow = el('div', 'btn-row');
    btnRow.style.marginTop = '10px';
    btnRow.append(refreshBtn, hint);
    body.append(btnRow);

    const doClose = close;

    async function fill() {
      grid.textContent = '';
      const cells: { flame: Flame; canvas: HTMLCanvasElement }[] = [];
      for (let i = 0; i < 9; i++) {
        const isCenter = i === 4;
        const item = el('div', 'mut-item' + (isCenter ? ' center' : ''));
        const cv = el('canvas') as HTMLCanvasElement;
        cv.width = CELL;
        cv.height = CELL;
        item.append(cv, el('span', 'mut-tag', isCenter ? 'current' : 'mutant'));
        const flame = isCenter ? base : mutateFlame(base);
        item.onclick = () => {
          doClose();
          app.renderer.exporting = false;
          app.setFlame(cloneFlame(flame));
        };
        grid.append(item);
        cells.push({ flame, canvas: cv });
      }
      const r = app.renderer;
      r.exporting = true;
      try {
        for (let i = 0; i < cells.length; i++) {
          if (!grid.isConnected) break; // modal was closed mid-render
          await renderCell(cells[i].flame, cells[i].canvas);
          hint.textContent = `Rendering ${i + 1}/9…`;
        }
        hint.textContent = 'Click a tile to adopt it, or mutate again.';
      } finally {
        r.exporting = false;
        app.renderer.restore();
      }
    }

    refreshBtn.onclick = fill;
    await fill();
  }

  return { open };
}
