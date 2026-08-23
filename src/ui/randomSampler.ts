// JWildfire's RandomFlameGeneratorSampler: a random flame is not the first draw but the first of up to N candidates
// whose small probe render covers enough of the picture (Sobel-filtered coverage ≥ 0.42 at NORMAL quality, and below the
// generator's maximum — 0.75 for the solid styles), else the best-covering candidate seen. The symmetry and weighting-field
// generators are applied to every candidate before its probe, as the sampler does.
import type { App } from './common';
import { cloneFlame, type Flame } from '../core/flame';
import { addRandomSymmetry, addRandomWeightingField, coverage, type SymmetryKind, type WFieldKind } from '../core/mutations';

export interface SamplerOpts {
  style: string;
  symmetry: SymmetryKind;
  wfield: WFieldKind;
  /** RandomBatchQuality: low (8 tries, ≥ 0.32), normal (16, ≥ 0.42) */
  quality?: 'low' | 'normal';
  /** stop trying after this many ms and keep the best candidate so far (each candidate costs a kernel compile) */
  budgetMs?: number;
  onProgress?: (tried: number, max: number) => void;
}

const PROBE_W = 80, PROBE_H = 60, PROBE_SPP = 40;

/** One random flame of `style` ('any' / 'wilderfire' / a style id), sampled the way JWildfire's random batch samples. */
export async function sampleRandomFlame(app: App, o: SamplerOpts): Promise<Flame> {
  const { randomFlameInStyle, RANDOM_STYLES } = await import('../core/randomStyles');
  const { randomFlame } = await import('../core/random');
  // "Any" settles on one generator for the whole sample, like AllRandomFlameGenerator.initState
  const style = o.style === 'any' ? RANDOM_STYLES[Math.floor(Math.random() * RANDOM_STYLES.length)].id : o.style;
  const make = (): Flame => {
    const f = style === 'wilderfire' ? randomFlame() : randomFlameInStyle(style);
    addRandomSymmetry(f, o.symmetry);
    addRandomWeightingField(f, o.wfield);
    return f;
  };
  const maxSamples = o.quality === 'low' ? 8 : 16, minCoverage = o.quality === 'low' ? 0.32 : 0.42;
  const r = app.renderer;
  const wasExporting = r.exporting;
  r.exporting = true;
  let best: Flame | null = null, bestCov = -1;
  const t0 = performance.now();
  try {
    for (let i = 0; i < maxSamples; i++) {
      if (i > 0 && o.budgetMs && performance.now() - t0 > o.budgetMs) break;
      const f = make();
      o.onProgress?.(i + 1, maxSamples);
      let cov: number;
      try {
        r.setFlame(f);
        const px = await r.renderRegion({ fullW: PROBE_W, fullH: PROBE_H, tileX: 0, tileY: 0, tileW: PROBE_W, tileH: PROBE_H, spp: PROBE_SPP, transparent: false });
        cov = coverage(px, PROBE_W, PROBE_H, f.background, f.filterRadius > 0);
      } catch { continue; } // a flame the kernel rejects does not count
      const maxCoverage = f.solid?.enabled ? 0.75 : 1;
      if (cov >= minCoverage && cov < maxCoverage) return cloneFlame(f);
      if (cov > bestCov) { bestCov = cov; best = f; }
    }
  } finally {
    r.exporting = wasExporting;
    r.setFlame(app.flame);
  }
  return best ?? make();
}
