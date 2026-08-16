// WGSL code generation: each flame compiles to a dedicated compute shader with
// its variation set inlined. Numeric parameters (affines, weights, variation
// params, layer densities) live in a storage buffer (`xd`) so slider tweaks
// never recompile — only structural edits (adding/removing layers, xforms or
// variations, toggling layer visibility) do.
//
// Layers: walker threads are partitioned across visible layers proportionally
// to layer weight (cutoffs in the xd header, so weight tweaks stay numeric).
// Each layer gets its own iteration function, CDF rows (xaos), xform blocks,
// and a 256-entry slice of the palette buffer.
//
// xd layout:
//   [0..7]              cumulative per-layer thread cutoffs (f32 thread index)
//   [8..15]             reserved
//   per visible layer:  CDF table ((n+1) rows × 16) then xform blocks
//   per-xform block:    6 affine, 6 post, color, colorSpeed, opacity, pad,
//                       then per variation: weight + params

import type { Flame, XForm, VarInstance } from '../core/flame';
import { visibleLayers } from '../core/flame';
import { VARIATIONS } from '../core/variations';

const CDF_ROW = 16;
const HEADER = 16;    // floats per xform block header
const XD_HEADER = 16; // floats reserved at the front of xd

export interface CompiledFlame {
  wgsl: string;
  dataSize: number; // float count for the xd buffer
  writeData(flame: Flame, out: Float32Array): void;
}

/** All variation lists of an xform in serialized order: pre, main, post. */
function varLists(x: XForm): [VarInstance[], VarInstance[], VarInstance[]] {
  return [x.preVariations ?? [], x.variations, x.postVariations ?? []];
}

function blockSize(x: XForm): number {
  let n = HEADER;
  for (const list of varLists(x)) {
    for (const v of list) {
      const def = VARIATIONS[v.name];
      if (def) n += 1 + (def.params?.length ?? 0);
    }
  }
  return n;
}

/** Module-scope WGSL (helper fns/consts) required by the variations of a flame, deduped by name. */
function collectFuncs(flame: Flame): string {
  const seen = new Set<string>();
  const parts: string[] = [];
  for (const ly of visibleLayers(flame)) {
    const xs = ly.final ? [...ly.xforms, ly.final] : ly.xforms;
    for (const x of xs) {
      for (const list of varLists(x)) {
        for (const vi of list) {
          const def = VARIATIONS[vi.name];
          if (!def?.funcs || !def.funcNames) continue;
          const key = def.funcNames.join('|');
          if (seen.has(key)) continue;
          // Item-level dedupe: two variations may share a helper (e.g. sqr, Complex struct)
          const items = splitItems(def.funcs);
          for (const [nm, text] of items) {
            if (seen.has('item:' + nm)) continue;
            seen.add('item:' + nm);
            parts.push(text);
          }
          seen.add(key);
        }
      }
    }
  }
  return parts.join('\n\n');
}

/** Splits a funcs blob into [name, text] items (struct/fn/const/var<private>). */
function splitItems(blob: string): [string, string][] {
  const out: [string, string][] = [];
  const chunks = blob.split(/\n\n(?=(?:struct|fn|const|var<private>) )/);
  for (const c of chunks) {
    const m = /^(?:struct|fn|const|var<private>) ([A-Za-z_]\w*)/.exec(c.trim());
    if (m) out.push([m[1], c.trim()]);
  }
  return out;
}

function genXformFn(name: string, x: XForm, B: number): string {
  const A = (i: number) => `xd[${B + i}u]`;
  const [pre, main, post] = varLists(x);
  let off = B + HEADER;

  // Emits one stage. The point (t), its polar precalcs and the accumulator (v)
  // are all mutable so JWildfire-style pre-priority variations can rewrite the
  // input and post-priority ones can rewrite the output; snippets run in
  // priority order (pre → normal → post), list order within a priority.
  const genStage = (list: VarInstance[], input: string, output: string, initV: string): string => {
    // xd layout follows list order (writeData mirrors it), so bind offsets first…
    const bound = list.map((vi) => {
      const def = VARIATIONS[vi.name];
      if (!def) return null;
      const w = `xd[${off}u]`;
      const p = (def.params ?? []).map((_, k) => `xd[${off + 1 + k}u]`);
      off += 1 + (def.params?.length ?? 0);
      return { def, w, p, prio: def.priority ?? 0 };
    }).filter((b): b is NonNullable<typeof b> => b !== null);
    // …then emit in priority order.
    let snips = '';
    for (const prio of [-2, -1, 0, 1, 2]) {
      for (const b of bound) if (b.prio === prio) snips += '    ' + b.def.code(b.w, b.p, A) + '\n';
    }
    return `  {
    var t = ${input};
    var r2 = max(dot(t, t), 1e-12);
    var r = sqrt(r2);
    var th = atan2(t.x, t.y);
    var ph = atan2(t.y, t.x);
    var v = ${initV};
${snips}    ${output} = v;
  }
`;
  };

  let body = `  var t0 = vec2f(${A(0)}*pin.x + ${A(1)}*pin.y + ${A(2)}, ${A(3)}*pin.x + ${A(4)}*pin.y + ${A(5)});\n`;
  if (pre.length) body += genStage(pre, 't0', 't0', 'vec2f(0.0, 0.0)');
  body += '  var vout = vec2f(0.0, 0.0);\n';
  body += genStage(main, 't0', 'vout', 'vec2f(0.0, 0.0)');
  if (post.length) body += genStage(post, 'vout', 'vout', 'vec2f(0.0, 0.0)');

  return `fn ${name}(pin: vec2f, cp: ptr<function, f32>, rs: ptr<function, u32>, hd: ptr<function, bool>) -> vec2f {
${body}  return vec2f(${A(6)}*vout.x + ${A(7)}*vout.y + ${A(8)}, ${A(9)}*vout.x + ${A(10)}*vout.y + ${A(11)});
}`;
}

export function compileFlame(flame: Flame, nPoints: number): CompiledFlame {
  const layers = visibleLayers(flame);
  const L = layers.length;

  // ---- Layout ----
  let off = XD_HEADER;
  const infos = layers.map((ly) => {
    const n = ly.xforms.length;
    const cdfBase = off;
    off += CDF_ROW * (n + 1);
    const bases = ly.xforms.map((x) => { const b = off; off += blockSize(x); return b; });
    let finalBase = -1;
    if (ly.final) { finalBase = off; off += blockSize(ly.final); }
    return { n, cdfBase, bases, finalBase };
  });
  const dataSize = Math.max(off, 64);

  // ---- Per-layer code ----
  let funcs = '';
  let iterFns = '';
  layers.forEach((ly, li) => {
    const info = infos[li];
    ly.xforms.forEach((x, i) => {
      funcs += genXformFn(`applyX${li}_${i}`, x, info.bases[i]) + '\n\n';
    });
    if (ly.final) funcs += genXformFn(`applyF${li}`, ly.final, info.finalBase) + '\n\n';

    let sel = `    let cb = ${info.cdfBase}u + (prev + 1u) * ${CDF_ROW}u;\n    var xi = 0u;\n`;
    for (let i = 0; i < info.n - 1; i++) {
      sel += `    if (rw > xd[cb + ${i}u]) { xi = ${i + 1}u; }\n`;
    }
    const cases = ly.xforms.map((_, i) => {
      const b = info.bases[i];
      // Color-speed blend runs first so direct-color variations get the last word.
      return `      case ${i}u: {
        let cs = xd[${b + 13}u];
        c = c * (1.0 - cs) + xd[${b + 12}u] * cs;
        op = xd[${b + 14}u];
        np = applyX${li}_${i}(p, &c, &rs, &hide);
      }`;
    }).join('\n');
    const finalBlock = ly.final ? `
      {
        let fcs = xd[${info.finalBase + 13}u];
        dc = dc * (1.0 - fcs) + xd[${info.finalBase + 12}u] * fcs;
      }
      dp = applyF${li}(p, &dc, &rs, &hide);` : '';

    iterFns += `
fn iterLayer${li}(idx: u32) {
  var pt = pts[idx];
  var rs = rngs[idx].x;
  var prev = min(rngs[idx].y, ${info.n - 1}u);
  var p = pt.xy;
  var c = pt.z;
  var fuse = pt.w;
  let ca = cos(P.rotation);
  let sa = sin(P.rotation);
  let offX = P.fullW * 0.5 - P.tileX;
  let offY = P.fullH * 0.5 - P.tileY;

  for (var it = 0u; it < P.iters; it = it + 1u) {
    let rw = rnd(&rs);
${sel}
    var np = p;
    var op = 1.0;
    var hide = false;
    switch xi {
${cases}
      default: {}
    }
    prev = xi;
    p = np;

    if (p.x != p.x || p.y != p.y || abs(p.x) > 1e10 || abs(p.y) > 1e10) {
      p = vec2f(rnd(&rs) * 2.0 - 1.0, rnd(&rs) * 2.0 - 1.0);
      c = rnd(&rs);
      fuse = 20.0;
      continue;
    }
    if (fuse > 0.0) {
      fuse = fuse - 1.0;
      continue;
    }

    var dp = p;
    var dc = c;${finalBlock}
    if (hide) { op = 0.0; }

    let ox = dp.x - P.centerX;
    let oy = dp.y - P.centerY;
    let rx = ox * ca - oy * sa;
    let ry = ox * sa + oy * ca;
    // flam3/JWildfire raster convention: +y grows downward on the image.
    let fx = rx * P.ppu + offX;
    let fy = ry * P.ppu + offY;
    if (fx >= 0.0 && fy >= 0.0 && fx < f32(P.width) && fy < f32(P.height)) {
      let hi = (u32(fy) * P.width + u32(fx)) * 4u;
      let col = pal[${li * 256}u + min(u32(clamp(dc, 0.0, 1.0) * 255.99), 255u)];
      atomicAdd(&hist[hi + 0u], u32(col.x * op * 255.0));
      atomicAdd(&hist[hi + 1u], u32(col.y * op * 255.0));
      atomicAdd(&hist[hi + 2u], u32(col.z * op * 255.0));
      atomicAdd(&hist[hi + 3u], u32(op * 256.0));
    }
  }

  pts[idx] = vec4f(p, c, fuse);
  rngs[idx] = vec2u(rs, prev);
}
`;
  });

  // ---- Layer dispatch by thread index ----
  let ldis = '  var layer = 0u;\n';
  for (let li = 0; li < L - 1; li++) {
    ldis += `  if (f32(idx) >= xd[${li}u]) { layer = ${li + 1}u; }\n`;
  }
  const lcases = layers.map((_, li) => `    case ${li}u: { iterLayer${li}(idx); }`).join('\n');

  const wgsl = `// Auto-generated WilderFire iteration kernel (${L} layer${L > 1 ? 's' : ''})
const PI: f32 = 3.14159265358979;

struct Params {
  width: u32,   // bounds of the target (tile) in pixels
  height: u32,
  iters: u32,
  _pad: u32,
  centerX: f32,
  centerY: f32,
  ppu: f32,
  rotation: f32,
  tileX: f32,   // tile origin within the full image (0 for on-screen render)
  tileY: f32,
  fullW: f32,   // full image dimensions the camera maps onto
  fullH: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> xd: array<f32>;
@group(0) @binding(2) var<storage, read_write> pts: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> rngs: array<vec2u>; // x: rng state, y: prev xform
@group(0) @binding(4) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> pal: array<vec4f>;

fn rnd(state: ptr<function, u32>) -> f32 {
  var x = *state;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  *state = x;
  return f32(x) * 2.3283064365386963e-10;
}

fn mmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }

${collectFuncs(flame)}

${funcs}${iterFns}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&rngs)) { return; }
${ldis}
  switch layer {
${lcases}
    default: {}
  }
}
`;

  // ---- Data writer (must mirror the layout above) ----
  const writeData = (fl: Flame, out: Float32Array) => {
    out.fill(0);
    const ls = visibleLayers(fl);

    // Layer thread cutoffs
    let lw = ls.map((l) => Math.max(l.weight, 0));
    let tot = lw.reduce((a, b) => a + b, 0);
    if (tot <= 1e-12) { lw = lw.map(() => 1); tot = lw.length; }
    let acc = 0;
    ls.forEach((_, i) => {
      acc += lw[i] / tot;
      out[i] = Math.round(acc * nPoints);
    });
    if (ls.length) out[ls.length - 1] = nPoints + 1;

    ls.forEach((ly, li) => {
      if (li >= infos.length) return;
      const info = infos[li];
      const m = Math.min(ly.xforms.length, CDF_ROW);
      const base = ly.xforms.map((x) => Math.max(x.weight, 1e-6));

      const writeRow = (rowBase: number, mult: (j: number) => number) => {
        const wj = base.map((bw, j) => bw * Math.max(mult(j), 0));
        let rtot = wj.reduce((a, b) => a + b, 0);
        if (rtot <= 1e-12) {
          for (let j = 0; j < m; j++) wj[j] = base[j];
          rtot = base.reduce((a, b) => a + b, 0);
        }
        let racc = 0;
        for (let j = 0; j < m; j++) {
          racc += wj[j] / rtot;
          out[rowBase + j] = racc;
        }
        out[rowBase + m - 1] = 1.0001;
      };
      writeRow(info.cdfBase, () => 1);
      for (let i = 0; i < m; i++) {
        writeRow(info.cdfBase + CDF_ROW * (i + 1), (j) => ly.xforms[i].xaos?.[j] ?? 1);
      }

      const writeBlock = (x: XForm, B: number) => {
        for (let i = 0; i < 6; i++) out[B + i] = x.affine[i];
        for (let i = 0; i < 6; i++) out[B + 6 + i] = x.post[i];
        out[B + 12] = x.color;
        out[B + 13] = x.colorSpeed;
        out[B + 14] = x.opacity;
        let o = B + HEADER;
        for (const list of varLists(x)) {
          for (const vi of list) {
            const def = VARIATIONS[vi.name];
            if (!def) continue;
            out[o++] = vi.weight;
            for (const pd of def.params ?? []) {
              out[o++] = vi.params[pd.name] ?? pd.def;
            }
          }
        }
      };
      ly.xforms.forEach((x, i) => { if (i < info.bases.length) writeBlock(x, info.bases[i]); });
      if (ly.final && info.finalBase >= 0) writeBlock(ly.final, info.finalBase);
    });
  };

  return { wgsl, dataSize, writeData };
}

export const TONEMAP_WGSL = `// WilderFire tonemap: density-estimation filter + log-density + gamma/vibrancy
struct TP {
  width: u32,
  height: u32,
  _a: u32,
  _b: u32,
  brightness: f32,
  gamma: f32,
  vibrancy: f32,
  spp: f32,
  de: vec4f, // x: max radius (0 = off), y: density falloff alpha, z: transparent bg flag, w: oversample factor
  bg: vec4f,
};

@group(0) @binding(0) var<uniform> T: TP;
@group(0) @binding(1) var<storage, read> hist: array<u32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(pos[vi], 0.0, 1.0);
}

// (r, g, b, opacity-weighted count) at a clamped supersampled cell
fn cell(cx: i32, cy: i32) -> vec4f {
  let os = max(u32(T.de.w + 0.5), 1u);
  let px = clamp(cx, 0, i32(T.width * os) - 1);
  let py = clamp(cy, 0, i32(T.height * os) - 1);
  let i = (u32(py) * T.width * os + u32(px)) * 4u;
  return vec4f(f32(hist[i]), f32(hist[i + 1u]), f32(hist[i + 2u]), f32(hist[i + 3u]) / 256.0);
}

// Box-filtered os×os block of supersampled cells for output pixel (bx, by)
fn blockAt(bx: i32, by: i32) -> vec4f {
  let os = max(i32(T.de.w + 0.5), 1);
  if (os == 1) { return cell(bx, by); }
  var acc = vec4f(0.0);
  for (var j = 0; j < os; j = j + 1) {
    for (var i = 0; i < os; i = i + 1) {
      acc += cell(bx * os + i, by * os + j);
    }
  }
  return acc / f32(os * os);
}

@fragment
fn fs(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let transparent = T.de.z > 0.5;
  let bgOut = select(vec4f(T.bg.rgb, 1.0), vec4f(0.0), transparent);
  let x = i32(fragPos.x);
  let y = i32(fragPos.y);
  if (x >= i32(T.width) || y >= i32(T.height) || T.spp <= 0.0) {
    return bgOut;
  }
  let c0 = blockAt(x, y);
  var rgb = c0.rgb;
  var cnt = c0.w;

  // Density-estimation filter: blur radius shrinks as local density grows,
  // smoothing sparse noise while leaving dense structure crisp (flam3-style).
  let maxR = i32(T.de.x + 0.5);
  if (maxR > 0) {
    let rad = min(i32(f32(maxR) / pow(c0.w + 1.0, T.de.y)), maxR);
    if (rad > 0) {
      var wsum = 0.0;
      var crgb = vec3f(0.0);
      var ccnt = 0.0;
      let sigma2 = 0.4 * f32(rad * rad) + 0.3;
      for (var dy = -rad; dy <= rad; dy = dy + 1) {
        for (var dx = -rad; dx <= rad; dx = dx + 1) {
          let cc = blockAt(x + dx, y + dy);
          let g = exp(-f32(dx * dx + dy * dy) / sigma2);
          crgb += cc.rgb * g;
          ccnt += cc.w * g;
          wsum += g;
        }
      }
      rgb = crgb / wsum;
      cnt = ccnt / wsum;
    }
  }

  if (cnt <= 0.0) {
    return bgOut;
  }

  // ---- flam3-style tone mapping ----
  // Log-density alpha: a = brightness * log10(1 + d), d = density relative to
  // the expected samples-per-cell (flam3's k1*log10(1 + count*k2)).
  let d = cnt / T.spp;
  let a = T.brightness * 0.43429448 * log(1.0 + d);
  let avg = rgb / (255.0 * cnt); // mean palette color, 0..1
  let crgb = avg * a;            // log-scaled color

  // Gamma with flam3's gamma_threshold: a linear ramp below the threshold so
  // gamma never amplifies single-sample speckle into visible noise.
  let g = 1.0 / max(T.gamma, 0.1);
  let thr = T.bg.w;
  var aG: f32;
  if (thr > 0.0 && a < thr) {
    let frac = a / thr;
    aG = (1.0 - frac) * a * (pow(thr, g) / thr) + frac * pow(a, g);
  } else {
    aG = pow(max(a, 0.0), g);
  }

  // flam3 vibrancy: scale channels by the gamma'd alpha ratio (saturation-
  // preserving) blended against per-channel gamma.
  let ls2 = select(0.0, aG / a, a > 0.0);
  var col = mix(
    pow(max(crgb, vec3f(0.0)), vec3f(g)),
    crgb * ls2,
    clamp(T.vibrancy, 0.0, 1.0)
  );

  // Highlights roll toward white instead of clipping per-channel (hue shift).
  let m = max(col.r, max(col.g, col.b));
  if (m > 1.0) {
    col = mix(col / m, vec3f(1.0), clamp((m - 1.0) * 0.6, 0.0, 1.0));
  }

  let alpha = clamp(aG, 0.0, 1.0);
  let ccol = clamp(col, vec3f(0.0), vec3f(1.0));
  if (transparent) {
    return vec4f(ccol, alpha);
  }
  return vec4f(mix(T.bg.rgb, ccol, alpha), 1.0);
}
`;
