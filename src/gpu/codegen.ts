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
//                       6 yz, 6 zx, 6 yzPost, 6 zxPost (JWildfire 3D affines),
//                       then per variation: weight + params

import type { Flame, XForm, VarInstance } from '../core/flame';
import { visibleLayers } from '../core/flame';
import { VARIATIONS } from '../core/variations';

const CDF_ROW = 16;
const HEADER = 40;    // floats per xform block header
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

function genXformFn(name: string, x: XForm, B: number, palBase: number): string {
  const A = (i: number) => `xd[${B + i}u]`;
  const [pre, main, post] = varLists(x);
  let off = B + HEADER;

  // Emits one stage. The point (t), its polar precalcs and the accumulator (v)
  // are all mutable so JWildfire-style pre-priority variations can rewrite the
  // input and post-priority ones can rewrite the output; snippets run in
  // priority order (pre → normal → post), list order within a priority.
  // `z_` is the stage's input z (mutable, JWildfire's __z), `pz_` its z output
  // accumulator (__pz). Only the main stage carries z out; the WilderFire pre/post
  // sum-stages pass z through untouched. With preserve_z, 2D variations at normal
  // priority add w·z (JWildfire's isPreserveZCoordinate behaviour).
  const genStage = (list: VarInstance[], input: string, output: string, initV: string, zIn: string, zOut: string | null): string => {
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
      for (const b of bound) {
        if (b.prio !== prio) continue;
        snips += '    ' + b.def.code(b.w, b.p, A) + '\n';
        if (prio === 0 && zOut && !(b.def.flags ?? []).includes('z')) {
          snips += `    if (P.flags & 1u) != 0u { pz_ += ${b.w} * z_; }\n`;
        }
      }
    }
    return `  {
    var t = ${input};
    var z_ = ${zIn};
    var pz_ = 0.0;
    var r2 = max(dot(t, t), 1e-12);
    var r = sqrt(r2);
    var th = atan2(t.x, t.y);
    var ph = atan2(t.y, t.x);
    var v = ${initV};
${snips}    ${output} = v;${zOut ? `\n    ${zOut} = pz_;` : ''}
  }
`;
  };

  // JWildfire TransformationAffineFullStep: xy linear part, then the yz plane,
  // then the zx plane, and all translations added at the end.
  let body = `  let ax = ${A(0)}*pin.x + ${A(1)}*pin.y;
  let ay = ${A(3)}*pin.x + ${A(4)}*pin.y;
  let ay2 = ${A(16)}*ay + ${A(17)}*pin.z;
  let az2 = ${A(19)}*ay + ${A(20)}*pin.z;
  let ax3 = ${A(22)}*ax + ${A(23)}*az2;
  let az3 = ${A(25)}*ax + ${A(26)}*az2;
  var t0 = vec2f(ax3 + ${A(2)} + ${A(24)}, ay2 + ${A(5)} + ${A(18)});
  let zin = az3 + ${A(21)} + ${A(27)};
`;
  body += '  var zout = zin;\n';
  if (pre.length) body += genStage(pre, 't0', 't0', 'vec2f(0.0, 0.0)', 'zin', null);
  body += '  var vout = vec2f(0.0, 0.0);\n';
  body += genStage(main, 't0', 'vout', 'vec2f(0.0, 0.0)', 'zin', 'zout');
  if (post.length) body += genStage(post, 'vout', 'vout', 'vec2f(0.0, 0.0)', 'zout', null);

  return `fn ${name}(pin: vec3f, cp: ptr<function, f32>, rs: ptr<function, u32>, hd: ptr<function, bool>, rgb: ptr<function, vec4f>) -> vec3f {
  let PALB_: u32 = ${palBase}u; // this layer's palette base (direct-colour variations read it)
${body}  let px1 = ${A(6)}*vout.x + ${A(7)}*vout.y;
  let py1 = ${A(9)}*vout.x + ${A(10)}*vout.y;
  let py2 = ${A(28)}*py1 + ${A(29)}*zout;
  let pz2 = ${A(31)}*py1 + ${A(32)}*zout;
  let px3 = ${A(34)}*px1 + ${A(35)}*pz2;
  let pz3 = ${A(37)}*px1 + ${A(38)}*pz2;
  return vec3f(px3 + ${A(8)} + ${A(36)}, py2 + ${A(11)} + ${A(30)}, pz3 + ${A(33)} + ${A(39)});
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
      funcs += genXformFn(`applyX${li}_${i}`, x, info.bases[i], li * 256) + '\n\n';
    });
    if (ly.final) funcs += genXformFn(`applyF${li}`, ly.final, info.finalBase, li * 256) + '\n\n';

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
        np = applyX${li}_${i}(p, &c, &rs, &hide, &rgbo);
      }`;
    }).join('\n');
    const finalBlock = ly.final ? `
      {
        let fcs = xd[${info.finalBase + 13}u];
        dc = dc * (1.0 - fcs) + xd[${info.finalBase + 12}u] * fcs;
      }
      dp = applyF${li}(p, &dc, &rs, &hide, &rgbo);` : '';

    iterFns += `
fn iterLayer${li}(idx: u32) {
  var pt = pts[idx];
  var rs = rngs[idx].x;
  var prev = min(rngs[idx].y & 255u, ${info.n - 1}u);
  var fuse = f32(rngs[idx].y >> 8u);
  var p = pt.xyz;
  var c = pt.w;
  let ca = cos(P.rotation);
  let sa = sin(P.rotation);
  let offX = P.fullW * 0.5 - P.tileX;
  let offY = P.fullH * 0.5 - P.tileY;
  let cam3d = (P.flags & 2u) != 0u;

  for (var it = 0u; it < P.iters; it = it + 1u) {
    let rw = rnd(&rs);
${sel}
    var np = p;
    var op = 1.0;
    var hide = false;
    var rgbo = vec4f(0.0); // direct RGB colour from dc_* variations (w = 1 when set)
    switch xi {
${cases}
      default: {}
    }
    prev = xi;
    p = np;

    if (p.x != p.x || p.y != p.y || p.z != p.z || abs(p.x) > 1e10 || abs(p.y) > 1e10 || abs(p.z) > 1e10) {
      p = vec3f(rnd(&rs) * 2.0 - 1.0, rnd(&rs) * 2.0 - 1.0, 0.0);
      c = rnd(&rs);
      fuse = 100.0;
      continue;
    }
    if (fuse > 0.0) {
      fuse = fuse - 1.0;
      continue;
    }

    var dp = p;
    var dc = c;${finalBlock}
    if (hide) { op = 0.0; }

    var rx: f32;
    var ry: f32;
    var visible = true;
    var dz = 1.0; // dimish-z intensity
    if (cam3d) {
      // JWildfire 3D camera: rotate by the camera matrix (yaw/pitch/roll), offset,
      // then perspective-divide; the centre offset applies after projection.
      let cx = P.m0.x * dp.x + P.m0.y * dp.y + P.m0.z * dp.z + P.camPos.x;
      let cy = P.m1.x * dp.x + P.m1.y * dp.y + P.m1.z * dp.z + P.camPos.y;
      let cz = P.m2.x * dp.x + P.m2.y * dp.y + P.m2.z * dp.z + P.camPos.z;
      let zr = 1.0 - P.camPersp * cz + P.camPos.z;
      visible = zr >= 1e-9;
      // dimish-z: fade points beyond dimZDist toward the dim colour
      if (P.dimZ.x > 1e-9) {
        let zd = P.dimZ.y - cz;
        if (zd > 0.0) { dz = exp(-zd * zd * P.dimZ.x); }
      }
      var px = cx;
      var py = cy;
      // depth of field (bubble shape): random disc offset ∝ distance from focus
      if (P.dof.x != 0.0) {
        var dist = 0.0;
        if (P.dof.w > 0.5) {
          let fx_ = cx - P.focus.x; let fy_ = cy - P.focus.y; let fz_ = cz - P.focus.z;
          let d = pow(fx_ * fx_ + fy_ * fy_ + fz_ * fz_, 1.0 / P.dof.y);
          let area = P.focus.w; let fade = area / 2.25; let amf = area - fade;
          if (d > area) { dist = d; }
          else if (d > amf) { let u = clamp((d - amf) / fade, 0.0, 1.0); dist = u * u * u * (u * (u * 6.0 - 15.0) + 10.0) * d; }
        } else {
          let zd_ = P.camZ - cz;
          if (zd_ > 0.0) { dist = zd_; }
        }
        if (dist > 0.0) {
          var ff = 1.0;
          if (P.dof.z >= 0.999999) { ff = rnd(&rs); }
          else if (P.dof.z > 1e-6) { if (rnd(&rs) <= P.dof.z) { ff = rnd(&rs); } }
          let dr = ff * P.dof.x * dist;
          let ang = 6.28318530718 * rnd(&rs);
          px = cx + dr * cos(ang);
          py = cy + dr * sin(ang);
        }
      }
      rx = px / zr - P.centerX;
      ry = py / zr - P.centerY;
    } else {
      let ox = dp.x - P.centerX;
      let oy = dp.y - P.centerY;
      rx = ox * ca - oy * sa;
      ry = ox * sa + oy * ca;
    }
    // flam3/JWildfire raster convention: +y grows downward on the image.
    var fx = rx * P.ppu + offX;
    var fy = ry * P.ppu + offY;
    // JWildfire antialiasing: some samples get a random jitter (raster pixels)
    if (P.aa.x > 0.0 && rnd(&rs) > 1.0 - P.aa.x) {
      let dr = exp(P.aa.y * sqrt(-log(max(rnd(&rs), 1e-12)))) - 1.0;
      let da = 6.28318530718 * rnd(&rs);
      fx = fx + dr * cos(da);
      fy = fy + dr * sin(da);
    }
    if (visible && fx >= 0.0 && fy >= 0.0 && fx < f32(P.width) && fy < f32(P.height)) {
      let hi = (u32(fy) * P.width + u32(fx)) * 4u;
      var col = pal[${li * 256}u + min(u32(clamp(dc, 0.0, 1.0) * 255.99), 255u)];
      if (rgbo.w > 0.5) { col = vec4f(clamp(rgbo.xyz, vec3f(0.0), vec3f(1.0)), 1.0); }
      if (dz < 1.0) { col = vec4f(mix(P.dimColor.xyz, col.xyz, dz), col.w); }
      atomicAdd(&hist[hi + 0u], u32(col.x * op * 255.0));
      atomicAdd(&hist[hi + 1u], u32(col.y * op * 255.0));
      atomicAdd(&hist[hi + 2u], u32(col.z * op * 255.0));
      atomicAdd(&hist[hi + 3u], u32(op * 256.0));
    }
  }

  pts[idx] = vec4f(p, c);
  rngs[idx] = vec2u(rs, prev | (u32(fuse) << 8u));
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
  flags: u32,   // bit0: preserve_z, bit1: 3D camera active
  centerX: f32,
  centerY: f32,
  ppu: f32,
  rotation: f32,
  tileX: f32,   // tile origin within the full image (0 for on-screen render)
  tileY: f32,
  fullW: f32,   // full image dimensions the camera maps onto
  fullH: f32,
  m0: vec4f,    // camera matrix rows (JWildfire yaw/pitch/bank/roll), w unused
  m1: vec4f,
  m2: vec4f,
  camPos: vec4f, // xyz: camera position offset, w: camPersp
  camPersp: f32,
  camZ: f32,     // legacy DOF focus plane
  _p2: f32,
  _p3: f32,
  focus: vec4f,  // xyz: DOF focus point, w: DOF area
  dof: vec4f,    // x: 0.1·camDOF·scale, y: exponent, z: fade, w: newDOF flag
  dimZ: vec4f,   // x: dimishZ, y: dimZDist
  dimColor: vec4f,
  aa: vec4f,     // x: antialias amount, y: antialias radius
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

// runtime 0u (set opaquely in main): the double-float helpers (hsin_) route intermediates through an
// integer add of it so the shader compiler's fast-math reassociation cannot fold their error terms
var<private> df_zero: u32 = 0u;

${collectFuncs(flame)}

${funcs}${iterFns}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= arrayLength(&rngs)) { return; }
  df_zero = bitcast<u32>(P.ppu) >> 31u;
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
        const I = [1, 0, 0, 0, 1, 0];
        for (let i = 0; i < 6; i++) out[B + 16 + i] = x.yz?.[i] ?? I[i];
        for (let i = 0; i < 6; i++) out[B + 22 + i] = x.zx?.[i] ?? I[i];
        for (let i = 0; i < 6; i++) out[B + 28 + i] = x.yzPost?.[i] ?? I[i];
        for (let i = 0; i < 6; i++) out[B + 34 + i] = x.zxPost?.[i] ?? I[i];
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
  filterN: u32, // colour filter size (odd; <= 1 = off), weights at sfilt[0..]
  filterNI: u32, // intensity filter size (JWildfire: gaussian 0.75 for sharpening kernels), weights at sfilt[128..]
  brightness: f32,
  gamma: f32,
  vibrancy: f32,
  spp: f32,
  de: vec4f, // x: DE estimator radius in px (0 = off), y: deCurve, z: transparent bg flag, w: oversample factor
  bg: vec4f, // rgb: background, w: gamma threshold
  jw: vec4f, // x: world area of the image (W·H/ppu²), y: contrast, z: colour scale 199.2/whiteLevel, w: low-density brightness
};

// Two passes: fsA = DE + log-scale per pixel into an rgba16float texture,
// fsB = spatial filter + gamma/vibrancy/background from that texture.
@group(0) @binding(0) var<uniform> T: TP;
@group(0) @binding(1) var<storage, read> hist: array<u32>;
@group(0) @binding(2) var<storage, read> sfilt: array<f32>;
@group(0) @binding(3) var midTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(pos[vi], 0.0, 1.0);
}

// Abramowitz–Stegun 7.1.26 erf (max err 1.5e-7)
fn erf1(x: f32) -> f32 {
  let s = sign(x);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return s * y;
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

// Density-estimation + JWildfire log scale for output pixel (x, y):
// returns (r, g, b, intensity), all zero where nothing landed.
fn logScaled(x: i32, y: i32) -> vec4f {
  let c0 = blockAt(x, y);
  var rgb = c0.rgb;
  var cnt = c0.w;

  // Density estimation (JWildfire DeCalculator): average neighbours of
  // *similar* density, weighted 1/(d²+1); a neighbour is accepted when
  // |erf((n - c)/m1)| <= deCurve^dist · m2 with m1 = sqrt(8c)+5, m2 = (c+1)^-¼.
  // Empty/sparse pixels gather from a wide area (soft glow), dense ones stay sharp.
  let R = i32(T.de.x + 0.5);
  if (R > 0) {
    let os2 = max(T.de.w, 1.0) * max(T.de.w, 1.0);
    let cw = c0.w * os2;               // raw hits (per output pixel)
    let m1 = sqrt(8.0 * cw) + 5.0;
    let m2 = pow(cw + 1.0, -0.25);
    let lnc = log(max(T.de.y, 1e-4));  // deCurve
    var sumRGB = vec3f(0.0);
    var sumA = 0.0;
    var wsum = 0.0;
    for (var dy = -R; dy <= R; dy = dy + 1) {
      for (var dx = -R; dx <= R; dx = dx + 1) {
        let d2 = f32(dx * dx + dy * dy);
        let cc = blockAt(x + dx, y + dy);
        let la = cc.w * os2;
        let dev = abs(erf1((la - cw) / m1));
        if (dev <= exp(lnc * sqrt(d2)) * m2) {
          let wgt = 1.0 / (d2 + 1.0);
          sumRGB += cc.rgb * wgt;
          sumA += cc.w * wgt;
          wsum += wgt;
        }
      }
    }
    if (wsum > 1e-9) {
      rgb = sumRGB / wsum;
      // JWildfire stores the estimated density as an int: deCount = (int)(sumA + 0.5).
      // Below half a hit the pixel is empty — that is what keeps isolated stray samples
      // from spreading into a visible speckle haze; the colour sums stay unrounded.
      cnt = floor(sumA / wsum * os2 + 0.5) / os2;
    }
  }

  if (cnt <= 0.0) {
    return vec4f(0.0);
  }
  // ---- JWildfire / flam3 log-density (LogScaleCalculator) ----
  // hits per output pixel relative to the expected count (quality), then
  // divided by the world area the image covers, so the curve is zoom-invariant:
  //   intensity = 2·contrast·brightness·log10(1 + d/(contrast·area)) + glow/(hits+1)
  let os = max(T.de.w, 1.0);
  let hits = cnt * os * os;             // blockAt averages the os×os block
  let d = hits / T.spp;
  let contrast = max(T.jw.y, 1e-3);
  let k1 = 2.0 * contrast * T.brightness;
  let glow = T.jw.w / (contrast * T.spp) / (hits + 1.0);
  let a = k1 * 0.43429448 * log(1.0 + d / (contrast * T.jw.x)) + glow;
  let avg = rgb / (255.0 * cnt);        // mean palette color, 0..1
  return vec4f(avg * a * T.jw.z, a);    // ×199.2/whiteLevel (JWildfire palette scaling)
}

@fragment
fn fsA(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let x = i32(fragPos.x);
  let y = i32(fragPos.y);
  if (x >= i32(T.width) || y >= i32(T.height) || T.spp <= 0.0) { return vec4f(0.0); }
  return logScaled(x, y);
}

fn mid(x: i32, y: i32) -> vec4f {
  let px = clamp(x, 0, i32(T.width) - 1);
  let py = clamp(y, 0, i32(T.height) - 1);
  return textureLoad(midTex, vec2i(px, py), 0);
}

@fragment
fn fsB(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let transparent = T.de.z > 0.5;
  let bgOut = select(vec4f(T.bg.rgb, 1.0), vec4f(0.0), transparent);
  let x = i32(fragPos.x);
  let y = i32(fragPos.y);
  if (x >= i32(T.width) || y >= i32(T.height) || T.spp <= 0.0) {
    return bgOut;
  }
  // Spatial filter (JWildfire LogDensityFilter) over the log-scaled image:
  // colours with the primary kernel, intensity with its own (smoothing) kernel.
  var v = vec4f(0.0);
  let NC = i32(T.filterN);
  let NI = i32(T.filterNI);
  if (NC <= 1 && NI <= 1) {
    v = mid(x, y);
  } else {
    let N = max(NC, NI);
    let h = N / 2;
    let hc = NC / 2;
    let hi = NI / 2;
    for (var j = -h; j <= h; j = j + 1) {
      for (var i = -h; i <= h; i = i + 1) {
        var wc = 0.0;
        var wi = 0.0;
        if (NC > 1 && abs(i) <= hc && abs(j) <= hc) { wc = sfilt[u32((j + hc) * NC + (i + hc))]; }
        if (NI > 1 && abs(i) <= hi && abs(j) <= hi) { wi = sfilt[128u + u32((j + hi) * NI + (i + hi))]; }
        if (NC <= 1 && i == 0 && j == 0) { wc = 1.0; }
        if (NI <= 1 && i == 0 && j == 0) { wi = 1.0; }
        if (wc != 0.0 || wi != 0.0) {
          let s = mid(x + i, y + j);
          v += vec4f(s.rgb * wc, s.w * wi);
        }
      }
    }
  }
  let a = v.w;
  if (a <= 0.0) {
    return bgOut;
  }
  let crgb = v.rgb;

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

  // col is already alpha-scaled (flam3/JWildfire: logScl = alpha / intensity),
  // so composite as col + bg*(1 - alpha), NOT mix(bg, col, alpha), which would
  // apply alpha twice. Per-channel clip like JWildfire (whiteLevel < 255 fades to white).
  let alpha = clamp(aG, 0.0, 1.0);
  let ccol = clamp(col, vec3f(0.0), vec3f(1.0));
  if (transparent) {
    // straight (un-premultiplied) colour for the PNG alpha channel
    return vec4f(select(ccol, clamp(col / max(alpha, 1e-6), vec3f(0.0), vec3f(1.0)), alpha > 1e-6), alpha);
  }
  return vec4f(clamp(ccol + T.bg.rgb * (1.0 - alpha), vec3f(0.0), vec3f(1.0)), 1.0);
}
`;
