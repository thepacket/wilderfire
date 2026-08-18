// Escape-time layer renderer: a fullscreen fragment shader iterates the layer's formula per pixel (supersampled),
// colours the escape data through the gradient and writes straight-alpha rgba16f into the layer texture. The
// shader is generated per structure (formula, colourings, trap shape, AA — `escapeSignature`); numbers are
// uniforms. Live rendering is progressive: the picture is drawn in horizontal bands over successive ticks
// (a band budget per tick keeps the UI fluid at high iteration counts), then stays static until a parameter
// changes. Export tiles render fully in one go and read back like FlameRenderer.renderRegion.

import { type EscapeLayerData, escapeSignature, escapeFormulaWgsl } from '../core/escape';
import { COMPLEX_WGSL } from '../core/formula';
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

export function buildEscapeWgsl(e: EscapeLayerData): { code: string; error?: string } {
  const f = escapeFormulaWgsl(e);
  const mandel = e.mode === 'mandelbrot';
  const powerBased = e.formula === 'mandelbrot' || e.formula === 'custom' || e.formula === 'newton' || e.formula === 'nova';
  const needDist = e.coloring.outside === 'distance';
  const needTrap = e.coloring.outside === 'orbit-trap' || e.coloring.inside === 'orbit-trap';
  const needExp = e.coloring.outside === 'exp-smooth' || e.coloring.inside === 'exp-smooth';
  const aa = Math.max(1, Math.min(3, Math.round(e.antialias)));
  const transfer = { linear: 't0', sqrt: 'sqrt(max(t0, 0.0))', cuberoot: 'pow(max(t0, 0.0), 1.0 / 3.0)', log: 'log(1.0 + max(t0, 0.0))' }[e.coloring.transfer] ?? 't0';
  const outIdx = OUTSIDE.indexOf(e.coloring.outside), inIdx = INSIDE.indexOf(e.coloring.inside);
  const code = `
struct EP {
  size: vec2f, tile: vec2f,          // tile size, tile origin in the full image
  full: vec2f, center: vec2f,        // full image size, view centre
  ppu: f32, cs: f32, sn: f32, power: f32,
  seed: vec2f, cc: vec2f,
  p1: vec2f, p2: vec2f, p3: vec2f, p4: vec2f,
  bail2: f32, maxIter: u32, density: f32, offset: f32,
  trap: vec4f,                        // x, y, size, min flag
  inside: vec4f, solid: vec4f,        // colours with alpha
  outAlpha: f32, pad0: f32, pad1: f32, pad2: f32,
}
@group(0) @binding(0) var<uniform> P: EP;
@group(0) @binding(1) var<storage, read> pal: array<vec4f>;
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
${COMPLEX_WGSL}
fn gradient(t0: f32) -> vec3f {
  let t = ${transfer};
  let g = fract(t * P.density + P.offset);
  let x = g * 256.0;
  let i0 = u32(x) % 256u; let i1 = (i0 + 1u) % 256u;
  return mix(pal[i0].xyz, pal[i1].xyz, fract(x));
}
fn shade(pixel: vec2f) -> vec4f {
  var z = ${mandel ? 'P.seed' : 'pixel'};
  let c = ${mandel ? 'pixel' : 'P.cc'};
  let p1 = P.p1; let p2 = P.p2; let p3 = P.p3; let p4 = P.p4;
  var zprev = z;
  var n: f32 = 0.0;
  var escaped = false;
  var trapv = 1e30;
  var expsum = 0.0;
  var dz = vec2f(1.0, 0.0);
  for (var i = 0u; i < P.maxIter; i = i + 1u) {
    ${needDist ? `dz = P.power * cmul(cpow(z, vec2f(P.power - 1.0, 0.0)), dz)${mandel ? ' + vec2f(1.0, 0.0)' : ''};` : ''}
    let zn = ${f.wgsl};
    zprev = z; z = zn; n = f32(i + 1u);
    ${needTrap ? `{ let q = z - P.trap.xy; let td = ${trapWgsl(e.coloring.trap.shape)}; trapv = select(td, min(trapv, td), P.trap.w > 0.5); }` : ''}
    ${needExp ? 'expsum = expsum + exp(-length(z));' : ''}
    if (dot(z, z) > P.bail2) { escaped = true; break; }
    ${f.convergent ? 'if (dot(z - zprev, z - zprev) < 1e-9) { escaped = true; break; }' : ''}
    if (z.x != z.x || z.y != z.y) { escaped = true; break; }
  }
  var t = 0.0;
  if (escaped) {
    ${outIdx === 6 ? 'return P.solid;' : ''}
    ${outIdx === 0 ? `{ let m2 = max(dot(z, z), 1.0000001); let lp = ${powerBased ? 'log(max(abs(P.power), 1.0001))' : 'log(2.0)'}; ${f.convergent ? 't = n;' : 't = n + 1.0 - log(0.5 * log(m2)) / lp;'} }` : ''}
    ${outIdx === 1 ? 't = n;' : ''}
    ${outIdx === 2 ? 't = expsum;' : ''}
    ${outIdx === 3 ? 't = trapv;' : ''}
    ${outIdx === 4 ? '{ let m = length(z); let dd = m * log(max(m, 1.0000001)) / max(length(dz), 1e-30); t = dd; }' : ''}
    ${outIdx === 5 ? 't = atan2(z.y, z.x) / 6.283185307 + 0.5;' : ''}
    return vec4f(gradient(t), P.outAlpha);
  }
  ${inIdx === 0 ? 'return P.inside;' : ''}
  ${inIdx === 1 ? 't = trapv;' : ''}
  ${inIdx === 2 ? 't = length(z);' : ''}
  ${inIdx === 3 ? 't = atan2(z.y, z.x) / 6.283185307 + 0.5;' : ''}
  ${inIdx === 4 ? 't = expsum;' : ''}
  return vec4f(gradient(t), P.inside.w);
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  var acc = vec4f(0.0);
  for (var sy = 0u; sy < ${aa}u; sy = sy + 1u) {
    for (var sx = 0u; sx < ${aa}u; sx = sx + 1u) {
      let sub = (vec2f(f32(sx), f32(sy)) + 0.5) / ${aa}.0;
      let px = in.pos.xy - 0.5 + sub + P.tile;              // pixel in the full image
      let d = vec2f(px.x - P.full.x * 0.5, P.full.y * 0.5 - px.y) / P.ppu;
      let w = P.center + vec2f(P.cs * d.x - P.sn * d.y, P.sn * d.x + P.cs * d.y);
      acc = acc + shade(w);
    }
  }
  return acc / ${aa * aa}.0;
}
`;
  return { code, error: f.error };
}

const UNIFORM_FLOATS = 44; // 176 bytes

export class EscapeRenderer {
  private device: GPUDevice;
  private tex: GPUTexture | null = null;
  private view: GPUTextureView | null = null;
  private w = 2;
  private h = 2;
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
  }

  get layerTexture(): GPUTexture | null { return this.tex; }
  get width() { return this.w; }
  get height() { return this.h; }

  resize(w: number, h: number) {
    w = Math.max(2, w); h = Math.max(2, h);
    if (this.tex && w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.tex?.destroy();
    this.tex = this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.view = this.tex.createView();
    this.nextRow = 0;
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
    ] })] });
    const mk = (format: GPUTextureFormat) => this.device.createRenderPipeline({ layout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    this.pipe = mk('rgba16float');
    this.pipeExport = mk('rgba8unorm');
    const bgl = this.pipe.getBindGroupLayout(0);
    this.bg = this.device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: this.uBuf } }, { binding: 1, resource: { buffer: this.palBuf } }] });
    this.bgOff = this.device.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: this.uOff } }, { binding: 1, resource: { buffer: this.palBuf } }] });
  }

  private writeUniforms(buf: GPUBuffer, e: EscapeLayerData, tile: { x: number; y: number; w: number; h: number; fullW: number; fullH: number }) {
    const f = new Float32Array(UNIFORM_FLOATS);
    const u = new Uint32Array(f.buffer);
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
    this.device.queue.writeBuffer(buf, 0, f);
  }

  /** Push the layer data (recompiles on a structural change) and start a fresh progressive render. */
  setLayer(e: EscapeLayerData) {
    this.data = e;
    const sig = escapeSignature(e);
    if (sig !== this.sig || !this.pipe) { this.sig = sig; this.compile(e); }
    const pal = new Float32Array(256 * 4);
    for (let i = 0; i < 256; i++) { const c = e.palette[i] ?? [0, 0, 0]; pal.set([c[0], c[1], c[2], 1], i * 4); }
    this.device.queue.writeBuffer(this.palBuf, 0, pal);
    this.writeUniforms(this.uBuf, e, { x: 0, y: 0, w: this.w, h: this.h, fullW: this.w, fullH: this.h });
    this.nextRow = 0;
  }
  setFlame(_f: unknown) { /* not a flame renderer */ }

  private encodeBands(enc: GPUCommandEncoder, rows: number) {
    if (!this.pipe || !this.bg || !this.view || this.nextRow >= this.h) return false;
    const y0 = this.nextRow, y1 = Math.min(this.h, y0 + rows);
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.view, loadOp: y0 === 0 ? 'clear' : 'load', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.bg);
    pass.setScissorRect(0, y0, this.w, y1 - y0);
    pass.draw(3);
    pass.end();
    this.nextRow = y1;
    return true;
  }

  private rowsPerTick(): number {
    const e = this.data!;
    const aa = e.antialias * e.antialias;
    return Math.max(8, Math.min(this.h, Math.round(1.5e8 / Math.max(1, this.w * e.maxIter * aa))));
  }

  /** One live step: draw the next band(s) while the picture is incomplete. */
  tick(_t: number): RenderStats | null {
    if (this.exporting || !this.data || !this.pipe) return null;
    if (this.nextRow < this.h) {
      const enc = this.device.createCommandEncoder();
      this.encodeBands(enc, this.rowsPerTick());
      this.device.queue.submit([enc.finish()]);
      this.presented = true;
    }
    return { spp: 0, samplesPerSec: 0, paused: false, converged: this.nextRow >= this.h, budgetScale: 1, gpuMs: 0 };
  }

  /** Finish the picture now (capture paths). */
  presentNow() {
    if (!this.data || !this.pipe || this.nextRow >= this.h) return;
    const enc = this.device.createCommandEncoder();
    this.encodeBands(enc, this.h);
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
    this.offTex?.destroy(); this.offTex = null;
    this.data = null;
  }
}
