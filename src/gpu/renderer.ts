// WebGPU progressive fractal-flame renderer.
// One compute dispatch per frame runs the chaos game across N persistent points,
// accumulating into an atomic RGBA histogram; a fullscreen pass tonemaps it.

import type { Flame } from '../core/flame';
import { flameSignature, visibleLayers, MAX_LAYERS } from '../core/flame';
import { compileFlame, TONEMAP_WGSL, type CompiledFlame } from './codegen';
import { buildSpatialFilters, FILT_FLOATS } from './filters';

const XD_FLOATS = 8192;

export interface RenderStats {
  spp: number;           // samples per pixel accumulated
  samplesPerSec: number;
  paused: boolean;
  converged: boolean;
  /** adaptive preview budget: fraction of the configured iterations per pass currently used (1 = full) */
  budgetScale: number;
  /** measured GPU time per preview frame (ms, smoothed) */
  gpuMs: number;
}

export class FlameRenderer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private canvas: HTMLCanvasElement;

  private histBuf!: GPUBuffer;
  private ptsBuf!: GPUBuffer;
  private rngBuf!: GPUBuffer;
  private modsBuf!: GPUBuffer; // per-point JWildfire colour modifiers (bound only when the compiled flame uses them)
  private xdBuf!: GPUBuffer;
  private palBuf!: GPUBuffer;
  private paramsBuf!: GPUBuffer;
  private tmBuf!: GPUBuffer;
  private filtBuf!: GPUBuffer;
  private filtKey = '';

  private computePipeline: GPUComputePipeline | null = null;
  private renderPipeline!: GPURenderPipeline; // pass B → canvas
  private pipeA!: GPURenderPipeline;           // pass A → mid texture
  private tmModule!: GPUShaderModule;
  private midTex: GPUTexture | null = null;
  private midW = 0;
  private midH = 0;
  private bgB: GPUBindGroup | null = null;      // pass B for the canvas format
  private bgBExport: GPUBindGroup | null = null; // pass B for rgba8 export
  private computeBG: GPUBindGroup | null = null;
  private renderBG: GPUBindGroup | null = null;

  private compiled: CompiledFlame | null = null;
  private sig = '';
  private xdData = new Float32Array(XD_FLOATS);
  private palData = new Float32Array(MAX_LAYERS * 256 * 4);

  flame: Flame | null = null;

  nPoints = 1 << 16;
  itersPerPass = 64;
  /** Adaptive preview budget: on a heavy kernel (many layers × variations) a full pass can take longer than a
   *  frame and the UI stalls behind the GPU queue. When on, the iterations per preview pass are scaled down
   *  until the measured GPU time per frame fits `frameBudgetMs`, and scaled back up when there is headroom.
   *  Exports always use the full `itersPerPass`. */
  adaptiveBudget = true;
  frameBudgetMs = 14;
  private budgetScale = 1;
  private gpuMs = 0;
  private gpuProbeInFlight = false;
  passesPerFrame = 2;
  targetQuality = 4000; // spp cap
  /** Live-preview cap on the DE estimator radius (px); exports use the flame's full radius. */
  deLiveCap = 6;
  /** Don't present a freshly reset accumulation until it has this many samples per
   *  pixel; the previous image stays on screen meanwhile (0 = present every frame). */
  minDisplaySpp = 10;
  private hasPresented = false;
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
    this.modsBuf = d.createBuffer({ size: this.nPoints * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.xdBuf = d.createBuffer({ size: XD_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.palBuf = d.createBuffer({ size: MAX_LAYERS * 256 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.paramsBuf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tmBuf = d.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.filtBuf = d.createBuffer({ size: FILT_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const tmModule = d.createShaderModule({ code: TONEMAP_WGSL });
    this.tmModule = tmModule;
    // Two-pass tonemap: A (DE + log scale) → rgba16float, B (filter + gamma) → target
    this.pipeA = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: tmModule, entryPoint: 'vs' },
      fragment: { module: tmModule, entryPoint: 'fsA', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.renderPipeline = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: tmModule, entryPoint: 'vs' },
      fragment: { module: tmModule, entryPoint: 'fsB', targets: [{ format: this.format }] },
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
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.renderBG = this.device.createBindGroup({
      layout: this.pipeA.getBindGroupLayout(0),
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
        ...(this.compiled?.usesMods ? [{ binding: 6, resource: { buffer: this.modsBuf } }] : []),
      ],
    });
  }

  resize(w: number, h: number) {
    if (w === this.width && h === this.height) return;
    this.canvas.width = Math.max(2, w);
    this.canvas.height = Math.max(2, h);
    this.allocHistogram();
    this.resetAccumulation();
    this.hasPresented = false; // canvas was cleared by the resize: present right away
    this.needsPresent = true;
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

  /** Re-seed every walker at a random point. `fuse` = iterations to run before plotting.
   *  flam3/JWildfire use 20/42, but they run few walkers for millions of steps each, so
   *  the transient is negligible; we run 65k walkers for a few hundred steps each per
   *  export, and a slowly contracting flame (e.g. an affine shrinking y by 0.9/step)
   *  is still visibly off the attractor after 20 — the strays became a speckle haze
   *  around dark regions. 200 steps cost ~one frame of work per re-seed. */
  /** unplotted (fuse) iterations still owed after a re-seed; excluded from the sample count */
  private fuseDebt = 0;
  /** account `iters` dispatched iterations: returns how many of them plot (the rest pay the fuse debt) */
  private countIters(iters: number): number {
    const paid = Math.min(this.fuseDebt, iters);
    this.fuseDebt -= paid;
    return iters - paid;
  }
  private reseedPoints(fuse = 200) {
    this.fuseDebt = fuse * this.nPoints;
    const pts = new Float32Array(this.nPoints * 4);
    const rng = new Uint32Array(this.nPoints * 2); // [state, prev xform]
    for (let i = 0; i < this.nPoints; i++) {
      pts[i * 4] = Math.random() * 2 - 1;      // x
      pts[i * 4 + 1] = Math.random() * 2 - 1;  // y
      pts[i * 4 + 2] = 0;                      // z
      pts[i * 4 + 3] = Math.random();          // color
      rng[i * 2] = (Math.random() * 0xffffffff) >>> 0 || 1;
      rng[i * 2 + 1] = fuse << 8; // prev xform (low 8 bits) | fuse (high bits)
    }
    this.device.queue.writeBuffer(this.ptsBuf, 0, pts);
    this.device.queue.writeBuffer(this.rngBuf, 0, rng);
    if (this.compiled?.usesMods) this.device.queue.writeBuffer(this.modsBuf, 0, new Float32Array(this.nPoints * 4)); // JWildfire starts every point with zero modifiers
  }

  /** Something tone-related changed (or the view): redraw even when the
   *  accumulation is finished. Without this the renderer idles once converged. */
  invalidate() { this.needsPresent = true; }
  private needsPresent = true;

  resetAccumulation() {
    this.samples = 0;
    this.emaSps = 0;
    this.needsPresent = true;
    this.reseedPoints(100); // live: a shorter fuse keeps drags snappy (exports use the full 200)
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.histBuf);
    this.device.queue.submit([enc.finish()]);
  }

  setPaused(p: boolean) { this.paused = p; this.needsPresent = true; }
  isPaused() { return this.paused; }

  private writeUniforms(
    spp: number,
    tile?: { tileX: number; tileY: number; fullW: number; fullH: number; tileW: number; tileH: number },
    transparent = false,
    iters = this.itersPerPass,
  ) {
    const f = this.flame!;
    const os = tile ? 1 : this.oversample;
    const w = tile ? tile.tileW : this.width * os;
    const h = tile ? tile.tileH : this.height * os;
    const fullW = tile ? tile.fullW : w;
    const fullH = tile ? tile.fullH : h;
    const ppu = 0.25 * Math.min(fullW, fullH) * f.zoom;
    const pu32 = new Uint32Array(52);
    const pf32 = new Float32Array(pu32.buffer);
    const dofOn = Math.abs(f.camDOF ?? 0) > 1e-9;
    const dimOn = (f.dimishZ ?? 0) > 1e-9;
    const cam3d = Math.abs(f.camPitch) > 1e-9 || Math.abs(f.camYaw) > 1e-9 || Math.abs(f.camBank) > 1e-9 || Math.abs(f.camPersp) > 1e-9
      || Math.abs(f.camPosX) > 1e-9 || Math.abs(f.camPosY) > 1e-9 || Math.abs(f.camPosZ) > 1e-9 || dofOn || dimOn;
    pu32[0] = w; pu32[1] = h; pu32[2] = iters; pu32[3] = (f.preserveZ ? 1 : 0) | (cam3d ? 2 : 0);
    pf32[4] = f.centerX; pf32[5] = f.centerY; pf32[6] = ppu; pf32[7] = f.rotation;
    pf32[8] = tile ? tile.tileX : 0; pf32[9] = tile ? tile.tileY : 0;
    pf32[10] = fullW; pf32[11] = fullH;
    // JWildfire camera matrix (FlameRendererView.createProjectionMatrix; bank = 0,
    // roll = our rotation, yaw negated as JWildfire does). Rows m0..m2 map to
    // camPoint = (m0·p, m1·p, m2·p).
    const yaw = (-f.camYaw * Math.PI) / 180, pitch = (f.camPitch * Math.PI) / 180, roll = f.rotation, bank = (f.camBank * Math.PI) / 180;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll), cb = Math.cos(bank), sb = Math.sin(bank);
    const m00 = -cp * sr * sy - (sp * sb * sr - cb * cr) * cy;
    const m10 = -cp * cy * sr + (sp * sb * sr - cb * cr) * sy;
    const m20 = cb * sp * sr + cr * sb;
    const m01 = cp * cr * sy + (cr * sp * sb + cb * sr) * cy;
    const m11 = cp * cr * cy - (cr * sp * sb + cb * sr) * sy;
    const m21 = -cb * cr * sp + sb * sr;
    const m02 = -cp * cy * sb + sp * sy;
    const m12 = cp * sb * sy + cy * sp;
    const m22 = cp * cb;
    pf32.set([m00, m10, m20, 0], 12);
    pf32.set([m01, m11, m21, 0], 16);
    pf32.set([m02, m12, m22, 0], 20);
    pf32.set([f.camPosX, f.camPosY, f.camPosZ, f.camPersp], 24);
    pf32[28] = f.camPersp;
    pf32[29] = f.camZ ?? 0;
    pf32.set([f.focusX ?? 0, f.focusY ?? 0, f.focusZ ?? 0, f.camDOFArea ?? 0.5], 32);
    pf32.set([dofOn ? 0.1 * f.camDOF * (f.camDOFScale ?? 1) : 0, f.camDOFExponent ?? 2, f.camDOFFade ?? 1, f.newDOF ? 1 : 0], 36);
    pf32.set([f.dimishZ ?? 0, f.dimZDist ?? 0, 0, 0], 40);
    const dc = f.dimZColor ?? [0, 0, 0];
    pf32.set([dc[0], dc[1], dc[2], 0], 44);
    pf32.set([f.antialiasAmount ?? 0.25, f.antialiasRadius ?? 0.5, 0, 0], 48);
    this.device.queue.writeBuffer(this.paramsBuf, 0, pu32);

    const tu32 = new Uint32Array(24);
    const tf32 = new Float32Array(tu32.buffer);
    tu32[0] = tile ? tile.tileW : this.width; // output pixels; hist rows are width×os
    tu32[1] = tile ? tile.tileH : this.height;
    { const [nc, ni] = this.uploadFilters(f.filterRadius ?? 0, f.filterKernel ?? 'mitchell'); tu32[2] = nc; tu32[3] = ni; }
    // world area covered by the full image (zoom-invariant density normalisation)
    const ppuOut = ppu / os;
    tf32[16] = (fullW / os) * (fullH / os) / (ppuOut * ppuOut);
    tf32[17] = f.contrast ?? 1;
    // JWildfire RenderColor pre-scales palette entries by 200/256, then divides by whiteLevel
    tf32[18] = (255 * 200 / 256) / Math.max(f.whiteLevel ?? 220, 1);
    tf32[19] = f.lowDensityBrightness ?? 0.24;
    tf32[4] = f.brightness; tf32[5] = f.gamma; tf32[6] = f.vibrancy;
    tf32[7] = Math.max(spp, 1e-6);
    {
      // JWildfire: estimatorRadius = de_radius · 9 · pixelsPerUnitScale (capped 18)
      const scl = tile ? tile.fullW / this.width : 1;
      let R = Math.round((f.deRadius ?? 1) * 9 * scl);
      R = Math.min(R, 18);
      if (!tile && this.deLiveCap >= 0) R = Math.min(R, this.deLiveCap);
      tf32[8] = R; tf32[9] = f.deCurve ?? 0.8;
    }
    tf32[10] = transparent ? 1 : 0;
    tf32[11] = os;
    tf32[12] = f.background[0]; tf32[13] = f.background[1]; tf32[14] = f.background[2];
    tf32[15] = f.gammaThreshold ?? 0.04; // packed into bg.w
    this.device.queue.writeBuffer(this.tmBuf, 0, tu32);
  }

  /** Intermediate rgba16float texture for the two-pass tonemap, sized to the target. */
  private ensureMid(w: number, h: number) {
    if (this.midTex && this.midW === w && this.midH === h) return;
    this.midTex?.destroy();
    this.midTex = this.device.createTexture({
      size: { width: Math.max(w, 1), height: Math.max(h, 1) },
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.midW = w; this.midH = h;
    this.bgB = null; this.bgBExport = null;
  }

  /** Encode both tonemap passes: histogram (via bgA) → mid → `target`. */
  private encodeTonemap(enc: GPUCommandEncoder, bgA: GPUBindGroup, pipeB: GPURenderPipeline, target: GPUTextureView, w: number, h: number, exportFmt: boolean, clear: GPUColor) {
    this.ensureMid(w, h);
    const rpA = enc.beginRenderPass({
      colorAttachments: [{ view: this.midTex!.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
    });
    rpA.setPipeline(this.pipeA);
    rpA.setBindGroup(0, bgA);
    rpA.draw(3);
    rpA.end();
    let bgB = exportFmt ? this.bgBExport : this.bgB;
    if (!bgB) {
      bgB = this.device.createBindGroup({
        layout: pipeB.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.tmBuf } },
          { binding: 2, resource: { buffer: this.filtBuf } },
          { binding: 3, resource: this.midTex!.createView() },
        ],
      });
      if (exportFmt) this.bgBExport = bgB; else this.bgB = bgB;
    }
    const rpB = enc.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', clearValue: clear, storeOp: 'store' }],
    });
    rpB.setPipeline(pipeB);
    rpB.setBindGroup(0, bgB);
    rpB.draw(3);
    rpB.end();
  }

  /** Upload the JWildfire spatial-filter kernels for the flame's settings; returns [Ncolour, Nintensity]. */
  private uploadFilters(radius: number, kernel: 'mitchell' | 'gaussian'): [number, number] {
    const { weights, nc, ni, key } = buildSpatialFilters(radius, kernel);
    if (nc === 0 && ni === 0) return [0, 0];
    if (key !== this.filtKey) {
      this.filtKey = key;
      this.device.queue.writeBuffer(this.filtBuf, 0, weights);
    }
    return [nc, ni];
  }

  /** Offline stepping for video export: accumulate `passes` compute dispatches
   *  and resolve when the GPU is done. Pair with captureSync() to grab pixels. */
  async stepExport(passes: number): Promise<void> {
    if (!this.flame || !this.computePipeline || !this.computeBG) return;
    const CHUNK = 24; // keep single submissions short to stay watchdog-friendly
    let done = 0;
    while (done < passes) {
      const nowPasses = Math.min(CHUNK, passes - done);
      this.samples += this.countIters(this.nPoints * this.itersPerPass * nowPasses);
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
        ...(this.compiled?.usesMods ? [{ binding: 6, resource: { buffer: this.modsBuf } }] : []),
      ],
    });
    if (!this.exportPipeline) {
      this.exportPipeline = d.createRenderPipeline({
        layout: 'auto',
        vertex: { module: this.tmModule, entryPoint: 'vs' },
        fragment: { module: this.tmModule, entryPoint: 'fsB', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
      });
    }
    const renderBG = d.createBindGroup({
      layout: this.pipeA.getBindGroupLayout(0),
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
    // enough passes for spp plotted samples per pixel on top of the fuse iterations
    let passes = Math.max(2, Math.min(Math.ceil((o.spp * o.fullW * o.fullH + this.fuseDebt) / perPass), 40000));
    let done = 0;
    const CHUNK = 32;
    let first = true;
    while (passes > 0) {
      const nowP = Math.min(CHUNK, passes);
      passes -= nowP;
      done += this.countIters(nowP * perPass);
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
    this.encodeTonemap(enc, renderBG, this.exportPipeline, this.offTex.createView(), o.tileW, o.tileH, true, { r: 0, g: 0, b: 0, a: 0 });
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
    this.encodeTonemap(enc, this.renderBG!, this.renderPipeline, view, this.width, this.height, false, { r: 0, g: 0, b: 0, a: 1 });
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
    if (dt > 0.5) this.gpuMs = 0; // we were away (hidden tab, stall): the last probe says nothing about the GPU load

    // Idle: converged (or paused) and nothing to redraw → do no GPU work at all.
    // The canvas keeps its last presented frame. invalidate() wakes us up.
    if (!accumulate && !this.needsPresent) {
      this.onFrame?.({ spp, samplesPerSec: 0, paused: this.paused, converged, budgetScale: this.budgetScale, gpuMs: this.gpuMs });
      return;
    }

    // Adaptive budget: shrink the iterations per preview pass while the GPU time per frame overshoots the
    // budget, grow them back (up to the configured count) when there is headroom. Measured with
    // onSubmittedWorkDone on the frame's own submission (queue backlog included — exactly the stall we avoid).
    if (this.adaptiveBudget && accumulate && this.gpuMs > 0) {
      if (this.gpuMs > this.frameBudgetMs * 1.3) this.budgetScale = Math.max(1 / 16, this.budgetScale * 0.7);
      else if (this.gpuMs < this.frameBudgetMs * 0.6 && this.budgetScale < 1) this.budgetScale = Math.min(1, this.budgetScale * 1.2);
    } else if (!this.adaptiveBudget) this.budgetScale = 1;
    const iters = Math.max(4, Math.round(this.itersPerPass * this.budgetScale));

    // The tonemap must see the sample count *including* this frame's passes:
    // the compute pass runs before the tonemap in the same submission, and
    // using the pre-dispatch count (0 right after a reset, i.e. on every frame
    // while a triangle is being dragged) blows the whole image out to white.
    // Burst: right after a reset run extra passes (bounded) so the new image
    // reaches minDisplaySpp within a frame or two instead of freezing the view
    // during a fast drag.
    const perPass = this.nPoints * iters;
    let passes = this.passesPerFrame;
    if (accumulate && this.minDisplaySpp > 0) {
      const need = Math.ceil((this.minDisplaySpp * w * h - this.samples) / perPass);
      if (need > passes) passes = Math.min(need, this.passesPerFrame * 4);
    }
    if (accumulate) this.samples += this.countIters(perPass * passes);
    this.writeUniforms(this.samples / Math.max(w * h, 1), undefined, false, iters); // tonemap spp is per output pixel

    const enc = this.device.createCommandEncoder();
    if (accumulate) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBG);
      for (let i = 0; i < passes; i++) {
        pass.dispatchWorkgroups(Math.ceil(this.nPoints / 256));
      }
      pass.end();
    }
    // Present gate: after a reset (every frame while a triangle is dragged) the
    // first few frames are too sparse to look like anything. Keep accumulating
    // in the background and leave the last presented image on the canvas until
    // the new one has reached minDisplaySpp — a WebGPU canvas keeps its last
    // frame as long as we don't fetch a new current texture.
    const sppAfter = this.samples / Math.max(w * h, 1);
    const present = !accumulate || sppAfter >= this.minDisplaySpp || !this.hasPresented;
    if (present) {
      const view = this.context.getCurrentTexture().createView();
      this.encodeTonemap(enc, this.renderBG, this.renderPipeline, view, w, h, false, { r: 0, g: 0, b: 0, a: 1 });
      this.hasPresented = true;
      this.needsPresent = false;
    }
    this.device.queue.submit([enc.finish()]);
    if (accumulate && !this.gpuProbeInFlight) {
      // one probe at a time: how long until this frame's work (and whatever was queued before it) is done
      this.gpuProbeInFlight = true;
      const t0 = performance.now();
      this.device.queue.onSubmittedWorkDone().then(() => {
        const ms = Math.min(performance.now() - t0, 200); // capped: one stall must not poison the average for long
        this.gpuMs = this.gpuMs ? this.gpuMs * 0.8 + ms * 0.2 : ms;
        this.gpuProbeInFlight = false;
      }, () => { this.gpuProbeInFlight = false; });
    }

    if (accumulate && dt > 0 && dt < 0.5) {
      const sps = (perPass * passes) / dt;
      this.emaSps = this.emaSps ? this.emaSps * 0.95 + sps * 0.05 : sps;
    }
    this.onFrame?.({ spp, samplesPerSec: this.emaSps, paused: this.paused, converged, budgetScale: this.budgetScale, gpuMs: this.gpuMs });
  };

  /** Dev diagnostic: total opacity-weighted hits in the live histogram and the count at a pixel. */
  async debugHistStats(px = -1, py = -1): Promise<{ hits: number; atPixel: number; cells: number }> {
    const os = this.oversample;
    const cells = this.width * os * this.height * os;
    const staging = this.device.createBuffer({ size: cells * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.histBuf, 0, staging, 0, cells * 16);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const u = new Uint32Array(staging.getMappedRange());
    let hits = 0;
    for (let i = 3; i < u.length; i += 4) hits += u[i];
    const at = px >= 0 ? u[((py * os) * this.width * os + px * os) * 4 + 3] / 256 : -1;
    staging.unmap(); staging.destroy();
    return { hits: hits / 256, atPixel: at, cells };
  }

  async exportPNG(): Promise<Blob | null> {
    return new Promise((resolve) => {
      this.captureSync((cv) => cv.toBlob((b) => resolve(b), 'image/png'));
    });
  }

  destroy() {
    cancelAnimationFrame(this.raf);
  }
}
