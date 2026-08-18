// Dev harness: render-regression check. Renders the fixture / sample / preset flames offscreen
// at a small size, reduces each image to a compact signature (mean luma, coverage, an 8×8 grid
// of block means, a 16-bin luma histogram) and compares it with the checked-in baseline
// scripts/jwf-port/render-baseline.json. A change that alters what WilderFire renders (engine
// semantics, tonemap, a variation port, the importer) shows up as a per-flame failure with the
// numbers; an intended change re-records the baseline.
//
//   await window.wilderfire.renderCheck()                 // compare against the baseline
//   await window.wilderfire.renderCheck({ update: true }) // re-record the baseline (dev server sink)
//   await window.wilderfire.renderCheck({ only: ['Gnarl_0', 'TINA0007'] })
//
// Renders are stochastic, so the tolerances (below) are loose enough for run-to-run noise at
// this size/quality and tight enough for the class of regression this is meant to catch (a wrong
// variation, a broken tonemap, a flipped axis, a dropped layer) — not 1 % shifts.
import { importFlameText } from '../core/flameXML';
import { collect } from './flameCompare';
import type { App } from '../ui/common';

export interface RenderSig { w: number; h: number; mean: number; cover: number; blocks: number[]; hist: number[] }
export interface RenderCheckOpts { only?: string[]; update?: boolean; width?: number; quality?: number; sets?: ('fixtures' | 'samples' | 'presets')[]; verbose?: boolean }
export interface RenderCheckResult { id: string; status: 'pass' | 'fail' | 'new' | 'error'; why?: string; sig?: RenderSig; ms: number }

const GRID = 8, BINS = 16;
/** tolerances (see header) */
const TOL = { meanRel: 0.05, meanAbs: 0.006, cover: 0.02, blockMae: 0.02, hist: 0.95 };

export function signature(px: Uint8ClampedArray, w: number, h: number): RenderSig {
  const blocks = new Array<number>(GRID * GRID).fill(0), counts = new Array<number>(GRID * GRID).fill(0);
  const hist = new Array<number>(BINS).fill(0);
  let sum = 0, cover = 0;
  for (let y = 0; y < h; y++) {
    const by = Math.min(GRID - 1, Math.floor((y * GRID) / h));
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const l = (0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]) / 255;
      sum += l;
      if (l > 2 / 255) cover++;
      const b = by * GRID + Math.min(GRID - 1, Math.floor((x * GRID) / w));
      blocks[b] += l; counts[b]++;
      hist[Math.min(BINS - 1, Math.floor(l * BINS))]++;
    }
  }
  const n = w * h;
  const r3 = (v: number) => Math.round(v * 1000) / 1000;
  return { w, h, mean: r3(sum / n), cover: r3(cover / n), blocks: blocks.map((v, i) => r3(v / Math.max(1, counts[i]))), hist: hist.map((v) => r3(v / n)) };
}

function compare(a: RenderSig, b: RenderSig): string | null {
  const why: string[] = [];
  if (a.w !== b.w || a.h !== b.h) return `size ${a.w}×${a.h} vs baseline ${b.w}×${b.h}`;
  const dm = Math.abs(a.mean - b.mean);
  if (dm > TOL.meanAbs && dm > TOL.meanRel * Math.max(a.mean, b.mean)) why.push(`mean ${a.mean} vs ${b.mean}`);
  if (Math.abs(a.cover - b.cover) > TOL.cover) why.push(`coverage ${a.cover} vs ${b.cover}`);
  const mae = a.blocks.reduce((s, v, i) => s + Math.abs(v - b.blocks[i]), 0) / a.blocks.length;
  if (mae > TOL.blockMae) why.push(`block MAE ${mae.toFixed(4)}`);
  const inter = a.hist.reduce((s, v, i) => s + Math.min(v, b.hist[i]), 0);
  if (inter < TOL.hist) why.push(`histogram ∩ ${inter.toFixed(3)}`);
  return why.length ? why.join(', ') : null;
}

export async function runRenderCheck(app: App, opts: RenderCheckOpts = {}): Promise<RenderCheckResult[]> {
  const W = opts.width ?? 256, quality = opts.quality ?? 200;
  const items = (await collect(app, opts.sets ?? ['fixtures', 'samples', 'presets'])).filter((it) => !opts.only || opts.only.includes(it.id));
  let baseline: Record<string, RenderSig> = {};
  try { const r = await fetch('/scripts/jwf-port/render-baseline.json'); if (r.ok) baseline = (await r.json()).flames ?? {}; } catch { /* none yet */ }
  const out: RenderCheckResult[] = [];
  const saved = app.flame;
  const sigs: Record<string, RenderSig> = { ...baseline };
  for (const it of items) {
    const t0 = performance.now();
    try {
      const sm = /size="(\d+)\s+(\d+)"/.exec(it.xml);
      const fw = sm ? Number(sm[1]) : 800, fh = sm ? Number(sm[2]) : 600;
      const H = Math.max(16, Math.round(W * fh / fw));
      const { flame } = importFlameText(it.xml, app.activeLayer.palette);
      app.setFlame(flame);
      app.renderer.setFlame(app.flame);
      const px = await app.renderer.renderRegion({ fullW: W, fullH: H, tileX: 0, tileY: 0, tileW: W, tileH: H, spp: quality });
      const sig = signature(px, W, H);
      sigs[it.id] = sig;
      const base = baseline[it.id];
      const why = base ? compare(sig, base) : null;
      out.push({ id: it.id, status: !base ? 'new' : why ? 'fail' : 'pass', why: why ?? undefined, sig, ms: performance.now() - t0 });
    } catch (err) {
      out.push({ id: it.id, status: 'error', why: String((err as Error).message ?? err), ms: performance.now() - t0 });
    }
    if (opts.verbose) console.log('renderCheck', out[out.length - 1].id, out[out.length - 1].status, out[out.length - 1].why ?? '');
  }
  app.setFlame(saved);
  app.renderer.setFlame(app.flame);
  const n = (s: string) => out.filter((o) => o.status === s).length;
  console.log(`renderCheck: ${n('pass')} pass, ${n('fail')} fail, ${n('new')} new, ${n('error')} error` + (n('fail') ? ' — ' + out.filter((o) => o.status === 'fail').map((o) => `${o.id}: ${o.why}`).join('; ') : ''));
  if (opts.update) {
    const body = { _note: 'Render-regression baseline written by window.wilderfire.renderCheck({ update: true }) (src/dev/renderCheck.ts): per flame, the signature of a 256 px offscreen render at 200 spp — mean luma, covered fraction, 8×8 block means, 16-bin luma histogram. renderCheck() compares the current build against it.', width: W, quality, date: new Date().toISOString().slice(0, 10), flames: Object.fromEntries(Object.entries(sigs).sort(([a], [b]) => a.localeCompare(b))) };
    const r = await fetch('/__jwf/baseline', { method: 'POST', body: JSON.stringify(body, null, 1) });
    console.log('render-baseline.json:', await r.text());
  }
  return out;
}
