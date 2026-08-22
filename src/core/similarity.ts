// "More like this": a compact signature per flame and a similarity score between two of them.
// Structural, not visual — it ranks by what a flame is made of (which variations, how many
// transforms, a final, 3D, solid) and what colours it uses (a hue histogram of the palette), so a
// query over thousands of library entries is one pass of a few dot products, no rendering.

export interface FlameSig {
  /** variation name → share of total weight (all layers, pre/main/post, finals included) */
  vars: Record<string, number>;
  /** 12-bin hue histogram of the palette, saturation-weighted, normalised to sum 1 */
  hue: number[];
  /** mean saturation and lightness of the palette */
  sat: number;
  light: number;
  /** log2 of the transform count across layers */
  logX: number;
  layers: number;
  hasFinal: boolean;
  is3D: boolean;
  solid: boolean;
}

const HUE_BINS = 12;

function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const l = (mx + mn) / 2;
  const d = mx - mn;
  if (d < 1e-6) return [0, 0, l];
  const s = l <= 0.5 ? d / (mx + mn) : d / (2 - mx - mn);
  let h: number;
  if (mx === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (mx === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return [h / 6, s, l];
}

/** Signature of a flame object (the stored JSON shape; tolerant of missing fields). */
export function flameSignature(f: any): FlameSig {
  const vars: Record<string, number> = {};
  let total = 0;
  let nX = 0;
  let hasFinal = false;
  const layers: any[] = Array.isArray(f?.layers) ? f.layers : [{ xforms: f?.xforms ?? [], final: f?.final ?? null, palette: f?.palette }];
  const addVars = (x: any, scale: number) => {
    for (const list of [x?.preVariations, x?.variations, x?.postVariations]) {
      if (!Array.isArray(list)) continue;
      for (const v of list) {
        if (!v?.name) continue;
        const w = Math.abs(Number(v.weight) || 0) * scale || 1e-3;
        vars[v.name] = (vars[v.name] ?? 0) + w;
        total += w;
      }
    }
  };
  for (const ly of layers) {
    const xs: any[] = Array.isArray(ly?.xforms) ? ly.xforms : [];
    nX += xs.length;
    for (const x of xs) addVars(x, Math.max(Number(x?.weight) || 0, 0.05));
    for (const fx of [ly?.final, ...(Array.isArray(ly?.moreFinals) ? ly.moreFinals : [])]) {
      if (fx) { hasFinal = true; addVars(fx, 0.5); }
    }
  }
  if (total > 0) for (const k of Object.keys(vars)) vars[k] /= total;
  // palette: the first layer's 256 colours (or stops)
  const hue = new Array(HUE_BINS).fill(0);
  let sat = 0, light = 0, n = 0, hsum = 0;
  const pal: any[] = Array.isArray(layers[0]?.palette) ? layers[0].palette : [];
  const step = Math.max(1, Math.floor(pal.length / 64));
  for (let i = 0; i < pal.length; i += step) {
    const c = pal[i];
    if (!Array.isArray(c) || c.length < 3) continue;
    const [h, s, l] = rgbToHsl(Number(c[0]) || 0, Number(c[1]) || 0, Number(c[2]) || 0);
    const w = s * (1 - Math.abs(l - 0.5) * 1.2); // greys and near-black/white say little about hue
    if (w > 0) { hue[Math.min(HUE_BINS - 1, Math.floor(h * HUE_BINS))] += w; hsum += w; }
    sat += s; light += l; n++;
  }
  if (hsum > 0) for (let i = 0; i < HUE_BINS; i++) hue[i] /= hsum;
  return {
    vars, hue,
    sat: n ? sat / n : 0, light: n ? light / n : 0,
    logX: Math.log2(Math.max(1, nX)),
    layers: layers.length,
    hasFinal,
    is3D: Math.abs(Number(f?.camPitch) || 0) > 1e-6 || Math.abs(Number(f?.camYaw) || 0) > 1e-6 || Math.abs(Number(f?.camPersp) || 0) > 1e-6,
    solid: !!f?.solid?.enabled,
  };
}

function cosineSparse(a: Record<string, number>, b: Record<string, number>): number {
  let dot = 0, na = 0, nb = 0;
  for (const k in a) { na += a[k] * a[k]; if (k in b) dot += a[k] * b[k]; }
  for (const k in b) nb += b[k] * b[k];
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}
function cosineDense(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

/** 0..1 — 1 for an identical signature. Variations 50 %, palette 30 %, structure 20 %. */
export function similarity(a: FlameSig, b: FlameSig): number {
  const v = cosineSparse(a.vars, b.vars);
  // the hue histogram is compared in its own right, and the overall saturation/lightness as a soft term
  const hueSim = cosineDense(a.hue, b.hue);
  const tone = 1 - Math.min(1, (Math.abs(a.sat - b.sat) + Math.abs(a.light - b.light)) / 1.2);
  const p = 0.75 * hueSim + 0.25 * tone;
  const dx = a.logX - b.logX;
  let s = Math.exp(-dx * dx);
  if (a.hasFinal !== b.hasFinal) s *= 0.75;
  if (a.is3D !== b.is3D) s *= 0.7;
  if (a.solid !== b.solid) s *= 0.5;
  if (a.layers !== b.layers) s *= 0.85;
  return 0.5 * v + 0.3 * p + 0.2 * s;
}

/** The `limit` most similar of `items` to `target`, best first, with scores. One pass, O(N log limit). */
export function rankSimilar<T>(target: FlameSig, items: { item: T; sig: FlameSig }[], limit = 60, skip?: (item: T) => boolean): { item: T; score: number }[] {
  const out: { item: T; score: number }[] = [];
  let worst = -1;
  for (const { item, sig } of items) {
    if (skip?.(item)) continue;
    const score = similarity(target, sig);
    if (out.length < limit) {
      out.push({ item, score });
      if (out.length === limit) { out.sort((x, y) => y.score - x.score); worst = out[limit - 1].score; }
    } else if (score > worst) {
      // keep the top-`limit` in order: insert, drop the tail
      let i = out.length - 1;
      out.pop();
      while (i > 0 && out[i - 1].score < score) i--;
      out.splice(i, 0, { item, score });
      worst = out[out.length - 1].score;
    }
  }
  if (out.length < limit) out.sort((x, y) => y.score - x.score);
  return out;
}
