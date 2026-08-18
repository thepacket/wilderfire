// Escape-time layer renderer: a fullscreen fragment shader iterates the layer's formula per pixel (supersampled),
// colours the escape data through the gradient and writes straight-alpha rgba16f into the layer texture. The
// shader is generated per structure (formula, colourings, trap shape, AA — `escapeSignature`); numbers are
// uniforms. Live rendering is progressive: the picture is drawn in horizontal bands over successive ticks
// (a band budget per tick keeps the UI fluid at high iteration counts), then stays static until a parameter
// changes. Export tiles render fully in one go and read back like FlameRenderer.renderRegion.

import { type EscapeLayerData, escapeSignature, escapeFormulaWgsl, escapeTier } from '../core/escape';
import { COMPLEX_WGSL, DS_WGSL } from '../core/formula';
import { bitsForZoom, centreFixed, fromNumber, referenceOrbit, fxSub, toNumber } from '../core/bigfloat';
import type { RenderStats } from './renderer';

const OUTSIDE = ['smooth', 'iterations', 'exp-smooth', 'orbit-trap', 'distance', 'angle', 'solid'];
const INSIDE = ['solid', 'orbit-trap', 'final-mag', 'final-angle', 'exp-smooth'];

function trapWgsl(shape: string): string {
  // distance from z to the trap shape (centre tx,ty; size s)
  switch (shape) {
    case 'point': return 'length(q)';
    case 'cross': return 'min(abs(q.x), abs(q.y))';
    case 'lines': return 'abs(q.x)';
    case 'circle': return 'abs(length(q) - P.trap.z)';
    case 'square': return 'max(abs(q.x), abs(q.y))';
    case 'ring': return 'abs(max(abs(q.x), abs(q.y)) - P.trap.z)';
    default: return 'length(q)';
  }
}

export function buildEscapeWgsl(e: EscapeLayerData): { code: string; error?: string; tier: 'f32' | 'ds' | 'perturb' } {
  const f = escapeFormulaWgsl(e);
  const tier = escapeTier(e);
  const mandel = e.mode === 'mandelbrot';
  const powerBased = e.formula === 'mandelbrot' || e.formula === 'custom' || e.formula === 'newton' || e.formula === 'nova';
  const needDist = e.coloring.outside === 'distance' && tier !== 'perturb';
  const needTrap = e.coloring.outside === 'orbit-trap' || e.coloring.inside === 'orbit-trap';
  const needExp = e.coloring.outside === 'exp-smooth' || e.coloring.inside === 'exp-smooth';
  const aa = Math.max(1, Math.min(3, Math.round(e.antialias)));
  const transfer = { linear: 't0', sqrt: 'sqrt(max(t0, 0.0))', cuberoot: 'pow(max(t0, 0.0), 1.0 / 3.0)', log: 'log(1.0 + max(t0, 0.0))' }[e.coloring.transfer] ?? 't0';
  const outIdx = OUTSIDE.indexOf(e.coloring.outside), inIdx = INSIDE.indexOf(e.coloring.inside);
  const outIdxEff = outIdx === 4 && tier === 'perturb' ? 0 : outIdx; // no derivative in the perturbation path
  const pw = Math.max(2, Math.min(8, Math.round(e.power)));
  const trapLine = needTrap ? `{ let q = zf - P.trap.xy; let td = ${trapWgsl(e.coloring.trap.shape)}; trapv = select(td, min(trapv, td), P.trap.w > 0.5); }` : '';
  const expLine = needExp ? 'expsum = expsum + exp(-length(zf));' : '';
  // the colouring tail: from (zf: final z as f32, n, escaped, trapv, expsum, dz)
  const tail = `
  var t = 0.0;
  if (escaped) {
    ${outIdxEff === 6 ? 'return P.solid;' : ''}
    ${outIdxEff === 0 ? `{ let m2 = max(dot(zf, zf), 1.0000001); let lp = ${powerBased ? 'log(max(abs(P.power), 1.0001))' : 'log(2.0)'}; ${f.convergent ? 't = n;' : 't = n + 1.0 - log(0.5 * log(m2)) / lp;'} }` : ''}
    ${outIdxEff === 1 ? 't = n;' : ''}
    ${outIdxEff === 2 ? 't = expsum;' : ''}
    ${outIdxEff === 3 ? 't = trapv;' : ''}
    ${outIdxEff === 4 ? '{ let m = length(zf); let dd = m * log(max(m, 1.0000001)) / max(length(dz), 1e-30); t = dd; }' : ''}
    ${outIdxEff === 5 ? 't = atan2(zf.y, zf.x) / 6.283185307 + 0.5;' : ''}
    return vec4f(gradient(t), P.outAlpha);
  }
  ${inIdx === 0 ? 'return P.inside;' : ''}
  ${inIdx === 1 ? 't = trapv;' : ''}
  ${inIdx === 2 ? 't = length(zf);' : ''}
  ${inIdx === 3 ? 't = atan2(zf.y, zf.x) / 6.283185307 + 0.5;' : ''}
  ${inIdx === 4 ? 't = expsum;' : ''}
  return vec4f(gradient(t), P.inside.w);`;

  let shade = '';
  if (tier === 'f32') {
    shade = `
fn shade(pixel: vec2f, dr: vec2f) -> vec4f {
  var z = ${mandel ? 'P.seed' : 'pixel'};
  let c = ${mandel ? 'pixel' : 'P.cc'};
  let p1 = P.p1; let p2 = P.p2; let p3 = P.p3; let p4 = P.p4;
  var zprev = z;
  var n: f32 = 0.0;
  var escaped = false;
  var trapv = 1e30;
  var expsum = 0.0;
  var dz = vec2f(1.0, 0.0);
  var zf = z;
  for (var i = 0u; i < P.maxIter; i = i + 1u) {
    ${needDist ? `dz = P.power * cmul(cpow(z, vec2f(P.power - 1.0, 0.0)), dz)${mandel ? ' + vec2f(1.0, 0.0)' : ''};` : ''}
    let zn = ${f.wgsl};
    zprev = z; z = zn; n = f32(i + 1u); zf = z;
    ${trapLine}
    ${expLine}
    if (dot(z, z) > P.bail2) { escaped = true; break; }
    ${f.convergent ? 'if (dot(z - zprev, z - zprev) < 1e-9) { escaped = true; break; }' : ''}
    if (z.x != z.x || z.y != z.y) { escaped = true; break; }
  }
  ${tail}
}`;
  } else if (tier === 'ds') {
    shade = `
fn shade(pixel0: vec2f, dr: vec2f) -> vec4f {
  // the pixel in double-single: centre (DS) + rotated pixel offset · 1/ppu (DS)
  let pixel = DC(ds_add(P.centerDS.xy, ds_mulf(P.invPpu, dr.x)), ds_add(P.centerDS.zw, ds_mulf(P.invPpu, dr.y)));
  var z = ${mandel ? 'DC(vec2f(P.seed.x, P.seedLo.x), vec2f(P.seed.y, P.seedLo.y))' : 'pixel'};
  let c = ${mandel ? 'pixel' : 'DC(vec2f(P.cc.x, P.ccLo.x), vec2f(P.cc.y, P.ccLo.y))'};
  let p1 = dc_c(P.p1); let p2 = dc_c(P.p2); let p3 = dc_c(P.p3); let p4 = dc_c(P.p4);
  var zprev = z;
  var n: f32 = 0.0;
  var escaped = false;
  var trapv = 1e30;
  var expsum = 0.0;
  var dz = vec2f(1.0, 0.0);
  var zf = dc_to(z);
  for (var i = 0u; i < P.maxIter; i = i + 1u) {
    ${needDist ? `dz = P.power * cmul(cpow(zf, vec2f(P.power - 1.0, 0.0)), dz)${mandel ? ' + vec2f(1.0, 0.0)' : ''};` : ''}
    let zn = ${f.ds};
    zprev = z; z = zn; n = f32(i + 1u); zf = dc_to(z);
    ${trapLine}
    ${expLine}
    if (dot(zf, zf) > P.bail2) { escaped = true; break; }
    ${f.convergent ? '{ let dd = zf - dc_to(zprev); if (dot(dd, dd) < 1e-9) { escaped = true; break; } }' : ''}
    if (zf.x != zf.x || zf.y != zf.y) { escaped = true; break; }
  }
  ${tail}
}`;
  } else {
    // perturbation with rebasing: z_n = Z_m + Δ_n, Δ' = Σ_k C(p,k) Z^{p−k} Δ^k (+ δc); Δ carried as mantissa·2^e so a
    // 1e-30 pixel offset never underflows; when |z| < |Δ| (or the reference ends) the pixel rebases onto Z_0
    shade = `
struct SC { m: vec2f, e: i32 }
fn sc_norm(a: SC) -> SC {
  let mx = max(abs(a.m.x), abs(a.m.y));
  if (mx == 0.0) { return SC(vec2f(0.0), 0); }
  let k = i32(floor(log2(mx)));
  return SC(ldexp(a.m, vec2i(-k)), a.e + k);
}
fn sc_add(a: SC, b: SC) -> SC {
  if (a.m.x == 0.0 && a.m.y == 0.0) { return b; }
  if (b.m.x == 0.0 && b.m.y == 0.0) { return a; }
  if (a.e >= b.e) { let d = a.e - b.e; if (d > 60) { return a; } return sc_norm(SC(a.m + ldexp(b.m, vec2i(-d)), a.e)); }
  let d = b.e - a.e; if (d > 60) { return b; } return sc_norm(SC(b.m + ldexp(a.m, vec2i(-d)), b.e));
}
fn sc_mulc(a: SC, s: vec2f) -> SC { return sc_norm(SC(cmul(a.m, s), a.e)); }
fn sc_mul(a: SC, b: SC) -> SC { return sc_norm(SC(cmul(a.m, b.m), a.e + b.e)); }
fn sc_scale(a: SC, s: f32) -> SC { return sc_norm(SC(a.m * s, a.e)); }
fn sc_to(a: SC) -> vec2f { return ldexp(a.m, vec2i(a.e)); }
fn sc_from(v: vec2f) -> SC { return sc_norm(SC(v, 0)); }
fn shade(pixel0: vec2f, dr: vec2f) -> vec4f {
  let delta0 = sc_add(sc_norm(SC(dr * P.dscale, P.dexp)), sc_norm(SC(P.refOff, P.refExp)));  // the pixel's offset from the reference point
  ${mandel ? 'var dl = SC(vec2f(0.0), 0); let dc = delta0;' : 'var dl = delta0;'}
  var m = 0u;                                             // reference index
  var n: f32 = 0.0;
  var escaped = false;
  var trapv = 1e30;
  var expsum = 0.0;
  var dz = vec2f(1.0, 0.0);
  let z0 = refo[0].xz;
  var zf = z0 + sc_to(dl);
  for (var i = 0u; i < P.maxIter; i = i + 1u) {
    let Z = refo[m].xz + refo[m].yw;
    // Δ' = Σ_{k=1..p} C(p,k) Z^{p−k} Δ^k
    var acc = SC(vec2f(0.0), 0);
    var dk = dl;                     // Δ^k
    var Zp = vec2f(1.0, 0.0);        // Z^{p-k}, built from Z^{p-1} downwards
    var Zpow = array<vec2f, ${pw + 1}>();
    Zpow[0] = vec2f(1.0, 0.0);
    for (var j = 1; j <= ${pw}; j = j + 1) { Zpow[j] = cmul(Zpow[j - 1], Z); }
    var binom = 1.0;
    for (var k = 1; k <= ${pw}; k = k + 1) {
      binom = binom * f32(${pw} - k + 1) / f32(k);
      Zp = Zpow[${pw} - k];
      acc = sc_add(acc, sc_scale(sc_mulc(dk, Zp), binom));
      dk = sc_mul(dk, dl);
    }
    ${mandel ? 'dl = sc_add(acc, dc);' : 'dl = acc;'}
    m = m + 1u;
    n = f32(i + 1u);
    let Zn = refo[m].xz;
    let dfull = sc_to(dl);
    zf = Zn + dfull;
    ${trapLine}
    ${expLine}
    if (dot(zf, zf) > P.bail2) { escaped = true; break; }
    if (zf.x != zf.x || zf.y != zf.y) { escaped = true; break; }
    // rebase: the orbit is closer to the reference start than to its current reference point, or the reference ended
    let rel = zf - z0;
    if (m >= P.refN || dot(rel, rel) < dot(dfull, dfull)) { dl = sc_from(rel); m = 0u; }
  }
  ${tail}
}`;
  }
  const code = `
struct EP {
  size: vec2f, tile: vec2f,          // tile size, tile origin in the full image
  full: vec2f, center: vec2f,        // full image size, view centre (f32)
  ppu: f32, cs: f32, sn: f32, power: f32,
  seed: vec2f, cc: vec2f,
  p1: vec2f, p2: vec2f, p3: vec2f, p4: vec2f,
  bail2: f32, maxIter: u32, density: f32, offset: f32,
  trap: vec4f,                        // x, y, size, min flag
  inside: vec4f, solid: vec4f,        // colours with alpha
  outAlpha: f32, dscale: f32, dexp: i32, refN: u32,   // perturbation: pixel offset = dr·dscale·2^dexp; reference length
  centerDS: vec4f,                    // centre as double-single: x hi, x lo, y hi, y lo
  invPpu: vec2f, seedLo: vec2f,       // 1/ppu as double-single; lo words of the seed
  ccLo: vec2f, refOff: vec2f,         // lo words of the Julia constant; view centre − reference point (mantissa)
  refExp: i32, pad1: f32, pad2: f32, pad3: f32, // … and its exponent (perturbation: the reference may sit off-centre)
}
@group(0) @binding(0) var<uniform> P: EP;
@group(0) @binding(1) var<storage, read> pal: array<vec4f>;
@group(0) @binding(2) var<storage, read> refo: array<vec4f>;
var<private> df_zero: u32 = 0u;
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
${COMPLEX_WGSL}
${tier === 'ds' ? DS_WGSL : ''}
fn gradient(t0: f32) -> vec3f {
  let t = ${transfer};
  let g = fract(t * P.density + P.offset);
  let x = g * 256.0;
  let i0 = u32(x) % 256u; let i1 = (i0 + 1u) % 256u;
  return mix(pal[i0].xyz, pal[i1].xyz, fract(x));
}
${shade}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  df_zero = bitcast<u32>(P.ppu) >> 31u; // 0 at run time, opaque to the compiler (keeps the double-single error terms alive)
  var acc = vec4f(0.0);
  for (var sy = 0u; sy < ${aa}u; sy = sy + 1u) {
    for (var sx = 0u; sx < ${aa}u; sx = sx + 1u) {
      let sub = (vec2f(f32(sx), f32(sy)) + 0.5) / ${aa}.0;
      let px = in.pos.xy - 0.5 + sub + P.tile;              // pixel in the full image
      let d = vec2f(px.x - P.full.x * 0.5, P.full.y * 0.5 - px.y);
      let dr = vec2f(P.cs * d.x - P.sn * d.y, P.sn * d.x + P.cs * d.y); // rotated pixel offset, in pixels
      let w = P.center + dr / P.ppu;
      acc = acc + shade(w, dr);
    }
  }
  return acc / ${aa * aa}.0;
}
`;
  return { code, error: f.error, tier };
}

const UNIFORM_FLOATS = 60; // 240 bytes

// While a new view renders band by band, the display shows the last complete picture warped into the new view
// (scale/rotate/shift about the centre) — a zoom or pan never flashes; sharp bands then overwrite it top-down.
const REPROJ_WGSL = `
struct RP { size: vec2f, shift: vec2f, k: f32, cs: f32, sn: f32, pad: f32 }   // k = oldPpu/newPpu; shift = old-pixel offset of the new centre; rotation delta
@group(0) @binding(0) var<uniform> P: RP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let c = P.size * 0.5;
  let d = in.pos.xy - c;                                   // new-view pixel offset (y down)
  let r = vec2f(P.cs * d.x - P.sn * d.y, P.sn * d.x + P.cs * d.y);
  let q = r * P.k + P.shift + c;                           // where it was in the old picture
  if (q.x < 0.0 || q.y < 0.0 || q.x >= P.size.x || q.y >= P.size.y) { return vec4f(0.0); }
  return textureSampleLevel(src, samp, q / P.size, 0.0);
}
`;

export class EscapeRenderer {
  private device: GPUDevice;
  private tex: GPUTexture | null = null;    // display texture (what the composer blends)
  private back: GPUTexture | null = null;   // the picture being rendered, band by band
  private front: GPUTexture | null = null;  // the last complete picture
  private frontView: { cx: number; cy: number; hi?: [string, string]; zoom: number; rot: number } | null = null;
  private view: GPUTextureView | null = null; // back's view (band target)
  private w = 2;
  private h = 2;
  private reprojPipe: GPURenderPipeline;
  private reprojBuf: GPUBuffer;
  private reprojSampler: GPUSampler;
  private needsReproject = false;
  // GPU timestamps around a band (when the device has 'timestamp-query'): the only clean measure of the band's own cost
  private tsQuery: GPUQuerySet | null = null;
  private tsResolve: GPUBuffer | null = null;
  private tsRead: GPUBuffer | null = null;
  private sig = '';
  private pipe: GPURenderPipeline | null = null;      // → rgba16float (layer texture)
  private pipeExport: GPURenderPipeline | null = null; // → rgba8unorm (tiles)
  private module: GPUShaderModule | null = null;
  private uBuf: GPUBuffer;
  private palBuf: GPUBuffer;
  private bg: GPUBindGroup | null = null;
  private data: EscapeLayerData | null = null;
  private nextRow = 0; // progressive: next band start (h = done)
  private offTex: GPUTexture | null = null;
  private offW = 0; private offH = 0;
  private uOff: GPUBuffer;
  private bgOff: GPUBindGroup | null = null;
  private refBuf: GPUBuffer;
  private refKey = '';
  private refN = 0;
  /** where the reference orbit starts (fixed point) and the zoom it was made for; reused while the view stays near */
  private refCentre: [import('../core/bigfloat').Fixed, import('../core/bigfloat').Fixed] | null = null;
  private refZoom = 0;
  /** view centre − reference point, as (mantissa, exponent) for the shader */
  private refOff: [number, number, number] = [0, 0, 0];
  /** perturbation: the reference orbit's length (0 = none) — the panel shows it */
  get referenceLength() { return this.refN; }

  // FlameRenderer-shaped surface used by the composer
  driven = true;
  transparentBg = false;
  presented = false;
  exporting = false;
  onError: ((msg: string) => void) | null = null;
  /** last compile error of a custom formula (shown by the panel) */
  formulaError = '';

  constructor(device: GPUDevice) {
    this.device = device;
    this.uBuf = device.createBuffer({ size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uOff = device.createBuffer({ size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.palBuf = device.createBuffer({ size: 256 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.refBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const rm = device.createShaderModule({ code: REPROJ_WGSL });
    this.reprojPipe = device.createRenderPipeline({ layout: 'auto', vertex: { module: rm, entryPoint: 'vs' }, fragment: { module: rm, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] }, primitive: { topology: 'triangle-list' } });
    this.reprojBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reprojSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    if (device.features.has('timestamp-query')) {
      this.tsQuery = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.tsResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.tsRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
  }

  get layerTexture(): GPUTexture | null { return this.tex; }
  get width() { return this.w; }
  get height() { return this.h; }

  resize(w: number, h: number) {
    w = Math.max(2, w); h = Math.max(2, h);
    if (this.tex && w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    const mk = () => this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
    this.tex?.destroy(); this.back?.destroy(); this.front?.destroy();
    this.tex = mk(); this.back = mk(); this.front = null; this.frontView = null;
    this.view = this.back.createView();
    this.nextRow = 0;
    this.needsReproject = true;
  }

  private compile(e: EscapeLayerData) {
    const { code, error } = buildEscapeWgsl(e);
    this.formulaError = error ?? '';
    const module = this.device.createShaderModule({ code });
    module.getCompilationInfo().then((info) => {
      for (const m of info.messages) if (m.type === 'error') { console.error('escape WGSL error:', m.message, `line ${m.lineNum}`); this.onError?.(`Escape shader error: ${m.message}`); }
    });
    this.module = module;
    const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ] })] });
    const mk = (format: GPUTextureFormat) => this.device.createRenderPipeline({ layout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    this.pipe = mk('rgba16float');
    this.pipeExport = mk('rgba8unorm');
    this.rebuildBindGroups();
  }
  private rebuildBindGroups() {
    if (!this.pipe) return;
    const bgl = this.pipe.getBindGroupLayout(0);
    const entries = (u: GPUBuffer) => [{ binding: 0, resource: { buffer: u } }, { binding: 1, resource: { buffer: this.palBuf } }, { binding: 2, resource: { buffer: this.refBuf } }];
    this.bg = this.device.createBindGroup({ layout: bgl, entries: entries(this.uBuf) });
    this.bgOff = this.device.createBindGroup({ layout: bgl, entries: entries(this.uOff) });
  }

  /** perturbation: the reference orbit (BigInt fixed point on the CPU). It is kept while the view stays within a
   *  couple of frames of it and the zoom within a wide band — a pan or wheel step then costs nothing on the CPU;
   *  the shader iterates deltas from the reference point, wherever it sits in the frame (rebasing keeps them sane). */
  private ensureReference(e: EscapeLayerData) {
    if (escapeTier(e) !== 'perturb') { this.refOff = [0, 0, 0]; return; }
    const P = bitsForZoom(e.zoom);
    const [cx, cy] = centreFixed(e.centerX, e.centerY, e.centerHi, P);
    const dyn = JSON.stringify([e.mode, Math.round(e.power), e.seed, e.c, e.maxIter, e.bailout]);
    let reuse = false;
    if (this.refCentre && this.refKey === dyn && this.refZoom > 0) {
      const dx = toNumber(fxSub(cx, this.refCentre[0])), dy = toNumber(fxSub(cy, this.refCentre[1]));
      const halfFrame = 2 / e.zoom; // the ±2 frame at zoom 1
      reuse = Math.hypot(dx, dy) < 2 * halfFrame && e.zoom < this.refZoom * 1e6 && e.zoom > this.refZoom / 1e4;
      if (reuse) {
        const m = Math.max(Math.abs(dx), Math.abs(dy));
        const ex = m > 0 ? Math.floor(Math.log2(m)) : 0;
        this.refOff = [dx / 2 ** ex, dy / 2 ** ex, ex];
      }
    }
    if (reuse) return;
    const z0: [typeof cx, typeof cy] = e.mode === 'mandelbrot' ? [fromNumber(e.seed[0], P), fromNumber(e.seed[1], P)] : [cx, cy];
    const c: [typeof cx, typeof cy] = e.mode === 'mandelbrot' ? [cx, cy] : [fromNumber(e.c[0], P), fromNumber(e.c[1], P)];
    const t0 = performance.now();
    const orb = referenceOrbit(z0, c, e.power, e.maxIter, e.bailout);
    if (orb.data.byteLength > this.refBuf.size) { this.refBuf.destroy(); this.refBuf = this.device.createBuffer({ size: Math.max(64, orb.data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); this.rebuildBindGroups(); }
    this.device.queue.writeBuffer(this.refBuf, 0, orb.data as Float32Array<ArrayBuffer>);
    this.refN = orb.n;
    this.refKey = dyn;
    this.refCentre = [cx, cy];
    this.refZoom = e.zoom;
    this.refOff = [0, 0, 0];
    if (performance.now() - t0 > 200) console.info(`reference orbit: ${orb.n} steps, ${P} bits, ${(performance.now() - t0).toFixed(0)} ms`);
  }
  private writeUniforms(buf: GPUBuffer, e: EscapeLayerData, tile: { x: number; y: number; w: number; h: number; fullW: number; fullH: number }) {
    const f = new Float32Array(UNIFORM_FLOATS);
    const u = new Uint32Array(f.buffer);
    const i32 = new Int32Array(f.buffer);
    const ppu = 0.25 * Math.min(tile.fullW, tile.fullH) * e.zoom;
    f.set([tile.w, tile.h, tile.x, tile.y, tile.fullW, tile.fullH, e.centerX, e.centerY], 0);
    f[8] = ppu; f[9] = Math.cos(e.rotation); f[10] = Math.sin(e.rotation); f[11] = e.power;
    f.set([e.seed[0], e.seed[1], e.c[0], e.c[1]], 12);
    f.set([e.params[0][0], e.params[0][1], e.params[1][0], e.params[1][1], e.params[2][0], e.params[2][1], e.params[3][0], e.params[3][1]], 16);
    f[24] = e.bailout * e.bailout; u[25] = e.maxIter; f[26] = e.coloring.density; f[27] = e.coloring.offset;
    f.set([e.coloring.trap.x, e.coloring.trap.y, e.coloring.trap.size, e.coloring.trap.min ? 1 : 0], 28);
    f.set([...e.coloring.insideColor, e.coloring.insideAlpha], 32);
    f.set([...e.coloring.solidColor, e.coloring.outsideAlpha], 36);
    f[40] = e.coloring.outsideAlpha;
    // perturbation: pixel offset (in pixels, rotated) × dscale·2^dexp = world offset (1/ppu split into mantissa/exponent)
    const inv = 1 / ppu;
    const dexp = Math.floor(Math.log2(inv));
    f[41] = inv / 2 ** dexp; i32[42] = dexp; u[43] = this.refN;
    // double-single centre / 1/ppu / seed / constant (hi = f32, lo = the f64 remainder)
    const ds = (v: number): [number, number] => { const hi = Math.fround(v); return [hi, Math.fround(v - hi)]; };
    const cxs = ds(e.centerX), cys = ds(e.centerY);
    f.set([cxs[0], cxs[1], cys[0], cys[1]], 44);
    const invs = ds(inv); const s0 = ds(e.seed[0]), s1 = ds(e.seed[1]), c0 = ds(e.c[0]), c1 = ds(e.c[1]);
    f.set([invs[0], invs[1], s0[1], s1[1], c0[1], c1[1], this.refOff[0], this.refOff[1]], 48);
    i32[56] = this.refOff[2];
    this.device.queue.writeBuffer(buf, 0, f);
  }

  /** Push the layer data (recompiles on a structural change) and start a fresh progressive render. */
  setLayer(e: EscapeLayerData) {
    this.data = e;
    const sig = escapeSignature(e);
    if (sig !== this.sig || !this.pipe) { this.sig = sig; this.compile(e); }
    this.ensureReference(e);
    const pal = new Float32Array(256 * 4);
    for (let i = 0; i < 256; i++) { const c = e.palette[i] ?? [0, 0, 0]; pal.set([c[0], c[1], c[2], 1], i * 4); }
    this.device.queue.writeBuffer(this.palBuf, 0, pal);
    this.writeUniforms(this.uBuf, e, { x: 0, y: 0, w: this.w, h: this.h, fullW: this.w, fullH: this.h });
    this.nextRow = 0;
    this.needsReproject = true;
    if (sig !== this.lastSigForRows) { this.rows = 0; this.lastSigForRows = sig; }
  }

  /** Display update after bands were drawn into `back`: on the first band of a new render the old picture is warped
   *  into the new view, then the finished rows [0, nextRow) are copied over it; a complete render becomes the new front. */
  private encodeDisplay(enc: GPUCommandEncoder) {
    if (!this.tex || !this.back || !this.data) return;
    if (this.needsReproject) {
      this.needsReproject = false;
      if (this.front && this.frontView) {
        const e = this.data, fv = this.frontView;
        const ppuNew = 0.25 * Math.min(this.w, this.h) * e.zoom, ppuOld = 0.25 * Math.min(this.w, this.h) * fv.zoom;
        // offset of the new centre from the old one, in world units (exact-centre safe: differences of decimal strings)
        let dx = e.centerX - fv.cx, dy = e.centerY - fv.cy;
        if (e.centerHi || fv.hi) {
          const P = bitsForZoom(Math.max(e.zoom, fv.zoom));
          const [nx, ny] = centreFixed(e.centerX, e.centerY, e.centerHi, P), [ox, oy] = centreFixed(fv.cx, fv.cy, fv.hi, P);
          dx = toNumber(fxSub(nx, ox)); dy = toNumber(fxSub(ny, oy));
        }
        // new-view pixel → world offset r/ppuNew (rotated by newRot) → old-view pixel: R(−oldRot)·(w + dcentre)·ppuOld
        const drot = e.rotation - fv.rot;
        const cs = Math.cos(drot), sn = Math.sin(drot);
        const oc = Math.cos(-fv.rot), os = Math.sin(-fv.rot);
        // shift (old pixels, y down): the new centre seen from the old centre
        const sx = (oc * dx - os * dy) * ppuOld, sy = -(os * dx + oc * dy) * ppuOld;
        const f = new Float32Array(8);
        f.set([this.w, this.h, sx, sy, ppuOld / ppuNew, cs, -sn, 0], 0); // y-down pixel frame: rotation sign flips
        this.device.queue.writeBuffer(this.reprojBuf, 0, f);
        const bg = this.device.createBindGroup({ layout: this.reprojPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.reprojBuf } }, { binding: 1, resource: this.front.createView() }, { binding: 2, resource: this.reprojSampler }] });
        const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.tex.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
        pass.setPipeline(this.reprojPipe); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
      }
      // (no previous picture: the display keeps whatever it shows until the bands arrive)
    }
    if (this.nextRow > 0) enc.copyTextureToTexture({ texture: this.back }, { texture: this.tex }, { width: this.w, height: this.nextRow });
    if (this.nextRow >= this.h) {
      // complete: remember it as the front picture for the next view change
      if (!this.front) this.front = this.device.createTexture({ size: [this.w, this.h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
      enc.copyTextureToTexture({ texture: this.back }, { texture: this.front }, { width: this.w, height: this.h });
      const e = this.data;
      this.frontView = { cx: e.centerX, cy: e.centerY, hi: e.centerHi ? [e.centerHi[0], e.centerHi[1]] : undefined, zoom: e.zoom, rot: e.rotation };
    }
  }
  setFlame(_f: unknown) { /* not a flame renderer */ }

  private encodeBands(enc: GPUCommandEncoder, rows: number, timed = false) {
    if (!this.pipe || !this.bg || !this.view || this.nextRow >= this.h) return false;
    const y0 = this.nextRow, y1 = Math.min(this.h, y0 + rows);
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.view, loadOp: 'load', storeOp: 'store' }],
      ...(timed && this.tsQuery ? { timestampWrites: { querySet: this.tsQuery, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {}),
    });
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.bg);
    pass.setScissorRect(0, y0, this.w, y1 - y0);
    pass.draw(3);
    pass.end();
    this.nextRow = y1;
    return true;
  }

  // progressive budget: rows per band start from an iteration-count estimate and adapt to the measured GPU time of the
  // last band (target ~12 ms), one band in flight at a time so the queue never backs up behind a heavy render
  private rows = 0;
  private rows0 = 0;
  private bandInFlight = false;
  private lastSigForRows = '';
  private rowsPerTick(): number {
    const e = this.data!;
    if (this.rows <= 0) {
      const aa = e.antialias * e.antialias;
      const cost = escapeTier(e) === 'perturb' ? 12 : escapeTier(e) === 'ds' ? 6 : 1; // relative per-iteration cost
      this.rows = Math.max(Math.ceil(this.h / 24), Math.min(this.h, Math.round(4e8 / Math.max(1, this.w * e.maxIter * aa * cost))));
      this.rows0 = this.rows;
    }
    return this.rows;
  }

  /** One live step: draw the next band while the picture is incomplete. */
  tick(_t: number): RenderStats | null {
    if (this.exporting || !this.data || !this.pipe) return null;
    if (this.nextRow < this.h && !this.bandInFlight) {
      const rows = this.rowsPerTick();
      const enc = this.device.createCommandEncoder();
      const timed = !!this.tsQuery && !!this.tsResolve && !!this.tsRead;
      this.encodeBands(enc, rows, timed);
      this.encodeDisplay(enc);
      if (timed) { enc.resolveQuerySet(this.tsQuery!, 0, 2, this.tsResolve!, 0); enc.copyBufferToBuffer(this.tsResolve!, 0, this.tsRead!, 0, 16); }
      this.device.queue.submit([enc.finish()]);
      this.presented = true;
      this.bandInFlight = true;
      const target = 10; // ms of GPU time per band
      if (timed) {
        this.tsRead!.mapAsync(GPUMapMode.READ).then(() => {
          const t = new BigUint64Array(this.tsRead!.getMappedRange());
          const ms = Number(t[1] - t[0]) / 1e6;
          this.tsRead!.unmap();
          this.bandInFlight = false;
          if (ms > 0 && isFinite(ms)) {
            // aim at the target with a damped step; the measure is this band's own GPU time, other layers excluded
            const f = Math.max(0.25, Math.min(3, target / ms));
            this.rows = Math.max(2, Math.min(this.h, Math.round(this.rows * (0.5 + 0.5 * f))));
          }
        }, () => { this.bandInFlight = false; });
      } else {
        // no timestamps: the estimate stands (never shrink on queue completion, which includes the flames' work),
        // grow only when the whole queue was clearly idle
        const t0 = performance.now();
        this.device.queue.onSubmittedWorkDone().then(() => {
          const ms = performance.now() - t0;
          this.bandInFlight = false;
          if (ms < 8) this.rows = Math.min(this.h, Math.round(this.rows * 1.6));
        }, () => { this.bandInFlight = false; });
      }
    }
    return { spp: 0, samplesPerSec: 0, paused: false, converged: this.nextRow >= this.h, budgetScale: 1, gpuMs: 0 };
  }

  /** Finish the picture now (capture paths). */
  presentNow() {
    if (!this.data || !this.pipe || (this.nextRow >= this.h && !this.needsReproject)) return;
    const enc = this.device.createCommandEncoder();
    this.encodeBands(enc, this.h);
    this.encodeDisplay(enc);
    this.device.queue.submit([enc.finish()]);
    this.presented = true;
  }

  // parity with FlameRenderer for the composer
  setPaused(_p: boolean) {}
  resetAccumulation() { this.nextRow = 0; }
  invalidate() { this.nextRow = 0; }
  async ready(): Promise<void> {}
  async stepExport(_passes: number): Promise<void> { this.presentNow(); await this.device.queue.onSubmittedWorkDone(); }
  setOversample(os: number): number { return os; }

  /** Render a tile of the layer at an arbitrary resolution and read the straight-alpha rgba8 pixels back. */
  async renderRegion(o: { fullW: number; fullH: number; tileX: number; tileY: number; tileW: number; tileH: number; spp: number; transparent?: boolean }): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const d = this.device;
    if (!this.data || !this.pipeExport || !this.bgOff) throw new Error('No escape layer.');
    if (!this.offTex || this.offW !== o.tileW || this.offH !== o.tileH) {
      this.offTex?.destroy();
      this.offTex = d.createTexture({ size: [o.tileW, o.tileH], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      this.offW = o.tileW; this.offH = o.tileH;
    }
    this.writeUniforms(this.uOff, this.data, { x: o.tileX, y: o.tileY, w: o.tileW, h: o.tileH, fullW: o.fullW, fullH: o.fullH });
    const bpr = Math.ceil((o.tileW * 4) / 256) * 256;
    const rb = d.createBuffer({ size: bpr * o.tileH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.offTex.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
    pass.setPipeline(this.pipeExport);
    pass.setBindGroup(0, this.bgOff);
    pass.draw(3);
    pass.end();
    enc.copyTextureToBuffer({ texture: this.offTex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: o.tileH }, { width: o.tileW, height: o.tileH });
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(rb.getMappedRange());
    const out = new Uint8ClampedArray(o.tileW * o.tileH * 4);
    for (let y = 0; y < o.tileH; y++) out.set(src.subarray(y * bpr, y * bpr + o.tileW * 4), y * o.tileW * 4);
    rb.unmap(); rb.destroy();
    return out;
  }

  destroy() {
    this.tex?.destroy(); this.tex = null;
    this.tsQuery?.destroy(); this.tsResolve?.destroy(); this.tsRead?.destroy();
    this.back?.destroy(); this.back = null;
    this.front?.destroy(); this.front = null;
    this.offTex?.destroy(); this.offTex = null;
    this.data = null;
  }
}
