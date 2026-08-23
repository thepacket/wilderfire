// JWildfire motion blur, approximated: FlameRenderer iterates `length` extra packets of the flame at times
// frame + length·step/2 − p·step (p = 1..length, in frames; the motion curves evaluated there) into one raster, each
// layer weight scaled by 1 − p²·decay·0.07/length (≥ 0.01) and the low-density brightness off. Here the sub-frames are
// rendered separately and their tonemapped pixels averaged with those weights — the same smear, though the averaging
// happens after the log-density curve rather than before it.
import type { Flame } from '../core/flame';
import type { FlameRenderer } from '../gpu/renderer';

export interface MotionBlurCtx { evalAt: (tSeconds: number) => Flame; t: number; fps: number }

/** The sub-frames of `flame` at time `ctx.t` with their weights (the flame itself first, weight 1), or null without motion blur. */
export function motionBlurFrames(flame: Flame, ctx: MotionBlurCtx): { flame: Flame; weight: number }[] | null {
  const mb = flame.motionBlur;
  if (!mb || mb.length <= 0) return null;
  const out: { flame: Flame; weight: number }[] = [{ flame: { ...flame, lowDensityBrightness: 0 }, weight: 1 }];
  let tf = ctx.t * ctx.fps + (mb.length * mb.timeStep) / 2;
  for (let p = 1; p <= mb.length; p++) {
    tf -= mb.timeStep;
    const w = Math.max(0.01, 1 - (p * p * mb.decay * 0.07) / mb.length);
    out.push({ flame: { ...ctx.evalAt(tf / ctx.fps), lowDensityBrightness: 0, motionBlur: undefined }, weight: w });
  }
  return out;
}

/** Render w×h with motion blur: the weighted average of the sub-frames' renders (plain render when the flame has none). */
export async function renderMotionBlurred(renderer: FlameRenderer, flame: Flame, ctx: MotionBlurCtx | null, w: number, h: number, spp: number, transparent = false): Promise<Uint8ClampedArray<ArrayBuffer>> {
  const region = (f: Flame) => { renderer.setFlame(f); return renderer.renderRegion({ fullW: w, fullH: h, tileX: 0, tileY: 0, tileW: w, tileH: h, spp, transparent }); };
  const frames = ctx ? motionBlurFrames(flame, ctx) : null;
  if (!frames) return region(flame);
  const acc = new Float32Array(w * h * 4);
  let wsum = 0;
  for (const fr of frames) {
    const px = await region(fr.flame);
    for (let i = 0; i < acc.length; i++) acc[i] += px[i] * fr.weight;
    wsum += fr.weight;
  }
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < acc.length; i++) out[i] = acc[i] / wsum;
  return out;
}
