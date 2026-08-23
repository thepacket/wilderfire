// MutaGen: a 3×3 grid around the current flame (the centre), each neighbour a mutated copy rendered offscreen —
// JWildfire's MutaGenController, with its typed mutations (core/mutations.ts), the "trend" (mutation type) and
// strength controls, and its probe-based acceptance: a candidate must cover ≥ 32 % of a small render and differ
// from the base on ≥ 18 % of it (up to 10 tries, else the best seen). Click a tile to adopt it.
import { App, el, openModal } from './common';
import { cloneFlame, type Flame } from '../core/flame';
import { applyMutation, coverage, diffCoverage, pixelize, MUTATION_TYPES, type MutationType } from '../core/mutations';

const LS_TREND = 'wilderfire.mutationTrend';
const LS_STRENGTH = 'wilderfire.mutationStrength';

/** One mutated copy of `base`: `n` mutations of `type` (MutaGenController.createMutationTypes for the inner ring:
 *  an edge cell gets one, a corner cell one horizontal and one vertical). */
export function mutateFlameWith(base: Flame, type: MutationType = 'all', strength = 1, n = 1): { flame: Flame; applied: MutationType[] } {
  const f = cloneFlame(base);
  const applied: MutationType[] = [];
  for (let i = 0; i < n; i++) applied.push(applyMutation(f, type, strength));
  f.name = base.name.replace(/-m\d+$/, '') + '-m' + Math.floor(Math.random() * 100);
  return { flame: f, applied };
}
export const mutateFlame = (base: Flame, type: MutationType = 'all', strength = 1, n = 1): Flame => mutateFlameWith(base, type, strength, n).flame;

export function buildMutate(app: App) {
  const CELL = 240;
  const SPP = 220;
  const PROBE_W = 80, PROBE_H = 60, PROBE_SPP = 40; // MutaGenController's probeSize (80×60) — a quick render for the acceptance test
  const MAX_ITER = 10, MIN_RENDER_COVERAGE = 0.32, MIN_DIFF_COVERAGE = 0.18;

  async function render(flame: Flame, w: number, h: number, spp: number): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const r = app.renderer;
    r.setFlame(flame);
    return r.renderRegion({ fullW: w, fullH: h, tileX: 0, tileY: 0, tileW: w, tileH: h, spp, transparent: false });
  }

  async function open() {
    const base = cloneFlame(app.flame);
    const { body, close } = openModal('Mutations — click one to adopt it');
    const tools = el('div', 'btn-row');
    const trendSel = el('select') as HTMLSelectElement;
    trendSel.title = 'Mutation trend — the kind of change every tile gets (JWildfire MutaGen\'s mutation types); All draws one at random per tile';
    for (const t of MUTATION_TYPES) { const o = el('option', '', t.name) as HTMLOptionElement; o.value = t.id; o.title = t.hint; trendSel.append(o); }
    trendSel.value = localStorage.getItem(LS_TREND) ?? 'all';
    if (!trendSel.value) trendSel.value = 'all';
    const strengthInp = el('input') as HTMLInputElement;
    strengthInp.type = 'range'; strengthInp.min = '0.1'; strengthInp.max = '3'; strengthInp.step = '0.1';
    strengthInp.value = localStorage.getItem(LS_STRENGTH) ?? '1';
    strengthInp.title = 'Mutation strength (1 = JWildfire\'s default; scales the size of the changes)';
    const strengthLbl = el('span', 'hint', `strength ${(+strengthInp.value).toFixed(1)}`);
    strengthInp.oninput = () => { strengthLbl.textContent = `strength ${(+strengthInp.value).toFixed(1)}`; localStorage.setItem(LS_STRENGTH, strengthInp.value); };
    trendSel.onchange = () => localStorage.setItem(LS_TREND, trendSel.value);
    const refreshBtn = el('button', 'primary', '↻ New mutations');
    tools.append(el('span', 'hint', 'Trend'), trendSel, strengthInp, strengthLbl, refreshBtn);
    const grid = el('div', 'mut-grid');
    const hint = el('div', 'hint', 'Rendering…');
    body.append(tools, grid, hint);

    const doClose = close;

    async function fill() {
      grid.textContent = '';
      const type = trendSel.value as MutationType, strength = +strengthInp.value || 1;
      const cells: { item: HTMLElement; canvas: HTMLCanvasElement; tag: HTMLElement; flame: Flame | null; n: number }[] = [];
      for (let i = 0; i < 9; i++) {
        const isCenter = i === 4;
        const dx = (i % 3) - 1, dy = Math.floor(i / 3) - 1;
        const item = el('div', 'mut-item' + (isCenter ? ' center' : ''));
        const cv = el('canvas') as HTMLCanvasElement;
        cv.width = CELL; cv.height = CELL;
        const tag = el('span', 'mut-tag', isCenter ? 'current' : '…');
        item.append(cv, tag);
        grid.append(item);
        cells.push({ item, canvas: cv, tag, flame: isCenter ? base : null, n: isCenter ? 0 : Math.abs(dx) + Math.abs(dy) });
      }
      const adopt = (flame: Flame) => { doClose(); app.renderer.exporting = false; app.setFlame(cloneFlame(flame)); };
      const r = app.renderer;
      r.exporting = true;
      try {
        // the base: its picture, and a pixelised probe the candidates are compared against
        const basePx = await render(base, CELL, CELL, SPP);
        cells[4].canvas.getContext('2d')!.putImageData(new ImageData(basePx, CELL, CELL), 0, 0);
        cells[4].item.onclick = () => adopt(base);
        const baseRef = pixelize(await render(base, PROBE_W, PROBE_H, PROBE_SPP), PROBE_W, PROBE_H);
        let done = 0;
        for (const cell of cells) {
          if (cell.n === 0) continue;
          if (!grid.isConnected) break; // modal was closed mid-render
          type Cand = { flame: Flame; applied: MutationType[] };
          let best: Cand | null = null, bestCov = -1, accepted: Cand | null = null;
          for (let iter = 0; iter < MAX_ITER && !accepted; iter++) {
            const cand = mutateFlameWith(base, type, strength, cell.n);
            let px: Uint8ClampedArray<ArrayBuffer>;
            try { px = await render(cand.flame, PROBE_W, PROBE_H, PROBE_SPP); } catch { continue; } // an invalid mutation does not count
            let cov = coverage(px, PROBE_W, PROBE_H, base.background, true);
            if (cov > MIN_RENDER_COVERAGE) cov = diffCoverage(px, baseRef, PROBE_W, PROBE_H);
            if (cov > MIN_DIFF_COVERAGE) accepted = cand;
            else if (cov > bestCov) { bestCov = cov; best = cand; }
          }
          const chosen = accepted ?? best;
          if (!chosen) continue;
          const flame = chosen.flame;
          cell.flame = flame;
          const px = await render(flame, CELL, CELL, SPP);
          cell.canvas.getContext('2d')!.putImageData(new ImageData(px, CELL, CELL), 0, 0);
          const names = chosen.applied.map((id) => MUTATION_TYPES.find((t) => t.id === id)?.name ?? id);
          cell.tag.textContent = names.join(' + ');
          cell.item.title = names.join(' + ');
          cell.item.onclick = () => adopt(flame);
          done++;
          hint.textContent = `Rendering ${done}/8…`;
        }
        hint.textContent = 'Click a tile to adopt it, or mutate again. Edge tiles carry one mutation, corners two.';
      } finally {
        r.exporting = false;
        r.setFlame(app.flame);
      }
    }

    refreshBtn.onclick = fill;
    await fill();
  }

  return { open };
}
