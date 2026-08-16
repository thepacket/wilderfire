// WebGPU progressive fractal-flame renderer.
// One compute dispatch per frame runs the chaos game across N persistent points,
// accumulating into an atomic RGBA histogram; a fullscreen pass tonemaps it.

import type { Flame } from '../core/flame';
import { flameSignature, visibleLayers, MAX_LAYERS } from '../core/flame';
import { compileFlame, TONEMAP_WGSL, type CompiledFlame } from './codegen';

const XD_FLOATS = 8192;

export interface RenderStats {
  spp: number;           // samples per pixel accumulated
  samplesPerSec: number;
  paused: boolean;
  converged: boolean;
}

export class FlameRenderer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private canvas: HTMLCanvasElement;

  private histBuf!: GPUBuffer;
  private ptsBuf!: GPUBuffer;
  private rngBuf!: GPUBuffer;
  private xdBuf!: GPUBuffer;
  private palBuf!: GPUBuffer;
  private paramsBuf!: GPUBuffer;
  private tmBuf!: GPUBuffer;

  private computePipeline: GPUComputePipeline | null = null;
  private renderPipeline!: GPURenderPipeline;
  private computeBG: GPUBindGroup | null = null;
  private renderBG: GPUBindGroup | null = null;

  private compiled: CompiledFlame | null = null;
  private sig = '';
  private xdData = new Float32Array(XD_FLOATS);
  private palData = new Float32Array(MAX_LAYERS * 256 * 4);

  flame: Flame | null = null;

  nPoints = 1 << 16;
  itersPerPass = 64;
  passesPerFrame = 2;
  targetQuality = 4000; // spp cap
  deMaxRadius = 1;      // density-estimation filter radius (0 = off)
  deAlpha = 0.4;        // how fast the DE radius shrinks with density
  oversample = 1;       // 1 or 2 — supersampled histogram, box-downsampled in tonemap
  private maxHistBytes = 128 << 20;

  private samples = 0;
  private paused = false;
  /** While true, the rAF loop idles so an offline export can drive frames. */
  exporting = false;
  private raf = 0;
  private lastT = 0;
  private emaSps = 0;

  onError: ((msg: string) => void) | null = null;
  onFrame: ((stats: RenderStats) => void) | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  /** The GPU device (available after init). Used by dev tooling. */
  get gpuDevice(): GPUDevice { return this.device; }

  async init(): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    // Raise storage limits so 2× oversampled histograms fit on capable GPUs.
    const wantBind = Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30);
    const wantBuf = Math.min(adapter.limits.maxBufferSize, 1 << 30);
    this.maxHistBytes = Math.min(wantBind, wantBuf);
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: wantBind,
        maxBufferSize: wantBuf,
      },
    });
    this.device.addEventListener('uncapturederror', (e) => {
      console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message);
      this.onError?.((e as GPUUncapturedErrorEvent).error.message);
    });

    this.context = this.canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });

    const d = this.device;
    this.ptsBuf = d.createBuffer({ size: this.nPoints * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.rngBuf = d.createBuffer({ size: this.nPoints * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.xdBuf = d.createBuffer({ size: XD_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.palBuf = d.createBuffer({ size: MAX_LAYERS * 256 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.paramsBuf = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tmBuf = d.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const tmModule = d.createShaderModule({ code: TONEMAP_WGSL });
    this.renderPipeline = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: tmModule, entryPoint: 'vs' },
      fragment: { module: tmModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });

    this.allocHistogram();
    this.raf = requestAnimationFrame(this.frame);
  }

  get width() { return this.canvas.width; }
  get height() { return this.canvas.height; }

  /** Change the oversample factor (recreates the histogram). Returns the factor actually applied. */
  setOversample(os: number): number {
    const wanted = os === 2 ? 2 : 1;
    const bytes = this.width * wanted * this.height * wanted * 16;
    this.oversample = bytes <= this.maxHistBytes ? wanted : 1;
    this.allocHistogram();
    this.resetAccumulation();
    return this.oversample;
  }

  private allocHistogram() {
    const os = this.oversample;
    if (this.width * os * this.height * os * 16 > this.maxHistBytes) this.oversample = 1;
    const size = Math.max(this.width * this.oversample * this.height * this.oversample, 1) * 16;
    this.histBuf?.destroy();
    this.histBuf = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.renderBG = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tmBuf } },
        { binding: 1, resource: { buffer: this.histBuf } },
      ],
    });
    this.rebuildComputeBG();
  }

  private rebuildComputeBG() {
    if (!this.computePipeline) return;
    this.computeBG = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.xdBuf } },
        { binding: 2, resource: { buffer: this.ptsBuf } },
        { binding: 3, resource: { buffer: this.rngBuf } },
        { binding: 4, resource: { buffer: this.histBuf } },
        { binding: 5, resource: { buffer: this.palBuf } },
      ],
    });
  }

  resize(w: number, h: number) {
    if (w === this.width && h === this.height) return;
    this.canvas.width = Math.max(2, w);
    this.canvas.height = Math.max(2, h);
    this.allocHistogram();
    this.resetAccumulation();
  }

  /** Push the current flame to the GPU. Recompiles the kernel on structural change. */
  setFlame(flame: Flame) {
    this.flame = flame;
    const sig = flameSignature(flame);
    if (sig !== this.sig || !this.compiled) {
      this.sig = sig;
      this.compiled = compileFlame(flame, this.nPoints);
      if (this.compiled.dataSize > XD_FLOATS) {
        this.onError?.('Flame too complex for parameter buffer.');
        return;
      }
      const module = this.device.createShaderModule({ code: this.compiled.wgsl });
      module.getCompilationInfo().then((info) => {
        for (const m of info.messages) {
          if (m.type === 'error') {
            console.error('WGSL error:', m.message, `line ${m.lineNum}`);
            this.onError?.(`Shader error: ${m.message}`);
          }
        }
      });
      this.computePipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
      this.rebuildComputeBG();
    }
    this.compiled.writeData(flame, this.xdData);
    this.device.queue.writeBuffer(this.xdBuf, 0, this.xdData, 0, this.compiled.dataSize);
    visibleLayers(flame).forEach((ly, li) => {
      if (li >= MAX_LAYERS) return;
      for (let i = 0; i < 256; i++) {
        const c = ly.palette[i] ?? [0, 0, 0];
        const o = (li * 256 + i) * 4;
        this.palData[o] = c[0];
        this.palData[o + 1] = c[1];
        this.palData[o + 2] = c[2];
        this.palData[o + 3] = 1;
      }
    });
    this.device.queue.writeBuffer(this.palBuf, 0, this.palData);
    this.resetAccumulation();
  }

  /** Tone-only changes (brightness/gamma/vibrancy/background) need no reset. */
  touchTone() { /* uniforms are rewritten every frame */ }

  private reseedPoints() {
    const pts = new Float32Array(this.nPoints * 4);
    const rng = new Uint32Array(this.nPoints * 2); // [state, prev xform]
    for (let i = 0; i < this.nPoints; i++) {
      pts[i * 4] = Math.random() * 2 - 1;
      pts[i * 4 + 1] = Math.random() * 2 - 1;
      pts[i * 4 + 2] = Math.random();
      pts[i * 4 + 3] = 20; // fuse
      rng[i * 2] = (Math.random() * 0xffffffff) >>> 0 || 1;
      rng[i * 2 + 1] = 0;
    }
    this.device.queue.writeBuffer(this.ptsBuf, 0, pts);
    this.device.queue.writeBuffer(this.rngBuf, 0, rng);
  }

  resetAccumulation() {
    this.samples = 0;
    this.emaSps = 0;
    this.reseedPoints();
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.histBuf);
    this.device.queue.submit([enc.finish()]);
  }

  setPaused(p: boolean) { this.paused = p; }
  isPaused() { return this.paused; }

  private writeUniforms(
    spp: number,
    tile?: { tileX: number; tileY: number; fullW: number; fullH: number; tileW: number; tileH: number },
    transparent = false,
  ) {
    const f = this.flame!;
    const os = tile ? 1 : this.oversample;
    const w = tile ? tile.tileW : this.width * os;
    const h = tile ? tile.tileH : this.height * os;
    const fullW = tile ? tile.fullW : w;
    const fullH = tile ? tile.fullH : h;
    const ppu = 0.25 * Math.min(fullW, fullH) * f.zoom;
    const pu32 = new Uint32Array(12);
    const pf32 = new Float32Array(pu32.buffer);
    pu32[0] = w; pu32[1] = h; pu32[2] = this.itersPerPass; pu32[3] = 0;
    pf32[4] = f.centerX; pf32[5] = f.centerY; pf32[6] = ppu; pf32[7] = f.rotation;
    pf32[8] = tile ? tile.tileX : 0; pf32[9] = tile ? tile.tileY : 0;
    pf32[10] = fullW; pf32[11] = fullH;
    this.device.queue.writeBuffer(this.paramsBuf, 0, pu32);

    const tu32 = new Uint32Array(16);
    const tf32 = new Float32Array(tu32.buffer);
    tu32[0] = tile ? tile.tileW : this.width; // output pixels; hist rows are width×os
    tu32[1] = tile ? tile.tileH : this.height;
    tf32[4] = f.brightness; tf32[5] = f.gamma; tf32[6] = f.vibrancy;
    tf32[7] = Math.max(spp, 1e-6);
    tf32[8] = this.deMaxRadius; tf32[9] = this.deAlpha;
    tf32[10] = transparent ? 1 : 0;
    tf32[11] = os;
    tf32[12] = f.background[0]; tf32[13] = f.background[1]; tf32[14] = f.background[2];
    tf32[15] = f.gammaThreshold ?? 0.04; // packed into bg.w
    this.device.queue.writeBuffer(this.tmBuf, 0, tu32);
  }

  /** Offline stepping for video export: accumulate `passes` compute dispatches
   *  and resolve when the GPU is done. Pair with captureSync() to grab pixels. */
  async stepExport(passes: number): Promise<void> {
    if (!this.flame || !this.computePipeline || !this.computeBG) return;
    const CHUNK = 24; // keep single submissions short to stay watchdog-friendly
    let done = 0;
    while (done < passes) {
      const nowPasses = Math.min(CHUNK, passes - done);
      this.samples += this.nPoints * this.itersPerPass * nowPasses;
      this.writeUniforms(this.samples / Math.max(this.width * this.height, 1));
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBG);
      for (let i = 0; i < nowPasses; i++) {
        pass.dispatchWorkgroups(Math.ceil(this.nPoints / 256));
      }
      pass.end();
      done += nowPasses;
      this.device.queue.submit([enc.finish()]);
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  private exportPipeline: GPURenderPipeline | null = null;
  private offTex: GPUTexture | null = null;
  private offTexW = 0;
  private offTexH = 0;
  private offHist: GPUBuffer | null = null;
  private offHistSize = 0;

  /** Render an arbitrary tile of an arbitrary-resolution image offscreen and
   *  read the RGBA pixels back. Used by hi-res export and the mutation grid.
   *  Caller must set `exporting = true` around batches and restore the screen
   *  flame afterwards (this reseeds the shared walker state). */
  async renderRegion(o: {
    fullW: number; fullH: number;
    tileX: number; tileY: number; tileW: number; tileH: number;
    spp: number; transparent?: boolean;
  }): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const d = this.device;
    if (!this.flame || !this.computePipeline) throw new Error('No compiled flame.');

    const need = o.tileW * o.tileH * 16;
    if (!this.offHist || this.offHistSize < need) {
      this.offHist?.destroy();
      this.offHist = d.createBuffer({ size: need, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.offHistSize = need;
    }
    const computeBG = d.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.xdBuf } },
        { binding: 2, resource: { buffer: this.ptsBuf } },
        { binding: 3, resource: { buffer: this.rngBuf } },
        { binding: 4, resource: { buffer: this.offHist } },
        { binding: 5, resource: { buffer: this.palBuf } },
      ],
    });
    if (!this.exportPipeline) {
      const tmModule = d.createShaderModule({ code: TONEMAP_WGSL });
      this.exportPipeline = d.createRenderPipeline({
        layout: 'auto',
        vertex: { module: tmModule, entryPoint: 'vs' },
        fragment: { module: tmModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
      });
    }
    const renderBG = d.createBindGroup({
      layout: this.exportPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tmBuf } },
        { binding: 1, resource: { buffer: this.offHist } },
      ],
    });
    if (!this.offTex || this.offTexW !== o.tileW || this.offTexH !== o.tileH) {
      this.offTex?.destroy();
      this.offTex = d.createTexture({
        size: { width: o.tileW, height: o.tileH },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      this.offTexW = o.tileW;
      this.offTexH = o.tileH;
    }

    this.reseedPoints();
    const tile = { tileX: o.tileX, tileY: o.tileY, fullW: o.fullW, fullH: o.fullH, tileW: o.tileW, tileH: o.tileH };
    const perPass = this.nPoints * this.itersPerPass;
    let passes = Math.max(2, Math.min(Math.ceil((o.spp * o.fullW * o.fullH) / perPass), 40000));
    let done = 0;
    const CHUNK = 32;
    let first = true;
    while (passes > 0) {
      const nowP = Math.min(CHUNK, passes);
      passes -= nowP;
      done += nowP * perPass;
      this.writeUniforms(done / (o.fullW * o.fullH), tile, o.transparent);
      const enc = d.createCommandEncoder();
      if (first) { enc.clearBuffer(this.offHist); first = false; }
      const cp = enc.beginComputePass();
      cp.setPipeline(this.computePipeline);
      cp.setBindGroup(0, computeBG);
      for (let i = 0; i < nowP; i++) cp.dispatchWorkgroups(Math.ceil(this.nPoints / 256));
      cp.end();
      d.queue.submit([enc.finish()]);
      // Wait between chunks so submissions stay short and the tab responsive.
      await d.queue.onSubmittedWorkDone();
    }

    this.writeUniforms(done / (o.fullW * o.fullH), tile, o.transparent);
    const bpr = Math.ceil((o.tileW * 4) / 256) * 256;
    const rb = d.createBuffer({ size: bpr * o.tileH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: this.offTex.createView(),
        loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store',
      }],
    });
    rp.setPipeline(this.exportPipeline);
    rp.setBindGroup(0, renderBG);
    rp.draw(3);
    rp.end();
    enc.copyTextureToBuffer(
      { texture: this.offTex },
      { buffer: rb, bytesPerRow: bpr, rowsPerImage: o.tileH },
      { width: o.tileW, height: o.tileH },
    );
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(rb.getMappedRange());
    const out = new Uint8ClampedArray(o.tileW * o.tileH * 4);
    for (let y = 0; y < o.tileH; y++) {
      out.set(src.subarray(y * bpr, y * bpr + o.tileW * 4), y * o.tileW * 4);
    }
    rb.unmap();
    rb.destroy();
    return out;
  }

  /** Draw the tonemap pass and hand the canvas to `fn` synchronously, in the
   *  same task as the submit — a WebGPU canvas is cleared once the task ends,
   *  so captures (VideoFrame, toBlob, drawImage) must not happen after an
   *  await. Queue ordering guarantees `fn` sees the finished image. */
  captureSync<T>(fn: (canvas: HTMLCanvasElement) => T): T {
    this.writeUniforms(this.samples / Math.max(this.width * this.height, 1));
    const enc = this.device.createCommandEncoder();
    const view = this.context.getCurrentTexture().createView();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    });
    rp.setPipeline(this.renderPipeline);
    rp.setBindGroup(0, this.renderBG!);
    rp.draw(3);
    rp.end();
    this.device.queue.submit([enc.finish()]);
    return fn(this.canvas);
  }

  private frame = (t: number) => {
    this.raf = requestAnimationFrame(this.frame);
    if (this.exporting) return;
    const f = this.flame;
    if (!f || !this.computePipeline || !this.computeBG || !this.renderBG) return;

    const w = this.width, h = this.height;
    const os2 = this.oversample * this.oversample;
    const spp = this.samples / Math.max(w * h * os2, 1);
    const converged = spp >= this.targetQuality;
    const accumulate = !this.paused && !converged;

    const dt = this.lastT ? (t - this.lastT) / 1000 : 0;
    this.lastT = t;

    this.writeUniforms(spp);

    const enc = this.device.createCommandEncoder();
    if (accumulate) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBG);
      for (let i = 0; i < this.passesPerFrame; i++) {
        pass.dispatchWorkgroups(Math.ceil(this.nPoints / 256));
      }
      pass.end();
      this.samples += this.nPoints * this.itersPerPass * this.passesPerFrame;
    }
    const view = this.context.getCurrentTexture().createView();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    });
    rp.setPipeline(this.renderPipeline);
    rp.setBindGroup(0, this.renderBG);
    rp.draw(3);
    rp.end();
    this.device.queue.submit([enc.finish()]);

    if (accumulate && dt > 0 && dt < 0.5) {
      const sps = (this.nPoints * this.itersPerPass * this.passesPerFrame) / dt;
      this.emaSps = this.emaSps ? this.emaSps * 0.95 + sps * 0.05 : sps;
    }
    this.onFrame?.({ spp, samplesPerSec: this.emaSps, paused: this.paused, converged });
  };

  async exportPNG(): Promise<Blob | null> {
    return new Promise((resolve) => {
      this.captureSync((cv) => cv.toBlob((b) => resolve(b), 'image/png'));
    });
  }

  destroy() {
    cancelAnimationFrame(this.raf);
  }
}
