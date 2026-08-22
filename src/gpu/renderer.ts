// WebGPU progressive fractal-flame renderer.
// One compute dispatch per frame runs the chaos game across N persistent points,
// accumulating into an atomic RGBA histogram; a fullscreen pass tonemaps it.

import type { Flame } from '../core/flame';
import { flameSignature, visibleLayers, MAX_LAYERS, LIGHT_DIFF_FUNCS } from '../core/flame';
import { compileFlame, TONEMAP_WGSL, type CompiledFlame } from './codegen';
import { buildSpatialFilters, solidFilterWeights, gaussianFilter1D, normFilterKernel, kernelCoeff, kernelSupport, FILT_FLOATS, type FilterKernel } from './filters';
import { SOLID_POST_WGSL, SOLID_TONEMAP_WGSL, SOLID_AO_WGSL, SOLID_SHADOW_WGSL, SOLID_PDOF_WGSL, SOLID_PDOF_BLIT_WGSL, SOLID_PAY_WORDS, SOLID_MAX_LIGHTS, SOLID_MAX_MATS, SOLID_FILT_FLOATS } from './solid.wgsl';
import { flameMeshKeys, ensureMesh, meshSampler, meshLayout } from '../core/meshes';
import { flameReflMaps, loadReflMap, reflMapKey, REFL_SIZE } from '../core/reflMaps';

const XD_FLOATS = 8192;
/** bytes per raster cell: density histogram (rgba u32) vs solid z-buffer (key + payload + normal) */
const HIST_CELL_BYTES = 16;
const SOLID_CELL_BYTES = 4 + SOLID_PAY_WORDS * 4 + 4 + 4 + 4 + 4 + 4; // key, payload, normal, depth (raster units), AO out, AO raw, AO temp
/** AO smoothing kernel: gaussian FilterHolder up to 45×45 raster cells */
const AO_FILT_MAX_N = 45;

/** Solid-rendering z-buffer set for one raster (live view or an offscreen tile). */
interface SolidBufs {
  key: GPUBuffer; pay: GPUBuffer; nrm: GPUBuffer; zr: GPUBuffer;
  ao: GPUBuffer; aoRaw: GPUBuffer | null; aoTmp: GPUBuffer | null; aoValid: boolean;
  /** shadow maps: light-space bounds (4 keys per light), maps (nCast × size²), per-cell step results + visibility (nCast × cells) */
  smaps: GPUBuffer | null; sacc: GPUBuffer | null; svis: GPUBuffer | null; shKey: string;
  bg: { post: GPUBindGroup; tm: GPUBindGroup; ao: GPUBindGroup | null; sh: GPUBindGroup } | null;
  /** the shadow visibility buffer reflects the current z-buffer (live view refreshes it every few presents) */
  shValid: boolean;
  cells: number;
}

export interface RenderStats {
  spp: number;           // samples per pixel accumulated
  samplesPerSec: number;
  paused: boolean;
  converged: boolean;
  /** converged because the wall-clock limit ran out, not because the quality cap was reached */
  timedOut: boolean;
  /** seconds the live view has spent accumulating since the last reset (paused / hidden time excluded) */
  elapsedS: number;
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

  // ---- JWildfire solid rendering (z-buffer + shading; see solid.wgsl.ts) ----
  private matsBuf!: GPUBuffer;      // per-point material index (bound when the compiled flame tracks materials)
  private meshBuf!: GPUBuffer;      // obj_mesh_primitive_wf samplers (face CDFs + triangles), packed per flame
  private meshPacked = '';          // keys packed into meshBuf
  private solidLive: SolidBufs | null = null;
  private solidOff: SolidBufs | null = null;
  private solidPostPipe!: GPUComputePipeline;
  private solidPipe!: GPURenderPipeline;        // → canvas format
  private solidExportPipe!: GPURenderPipeline; // → rgba8
  private solidTmLayout!: GPUBindGroupLayout;
  /** reflection maps: one REFL_SIZE² layer per image the solid materials use (src/core/reflMaps.ts) */
  private reflTex!: GPUTexture;
  private reflLayers = new Map<string, number>(); // image key (name@version) → layer
  private reflPacked = '';                         // keys currently uploaded, joined
  private reflLoading = '';
  // Post-process DOF for solid flames (PostDOFCalculator): scatter compute pass + present blit
  private solidMidPipe: GPURenderPipeline | null = null; // solid tonemap into midTex (rgba16float)
  private pdofPipe!: GPUComputePipeline;
  private pdofBlitLive: GPURenderPipeline | null = null;
  private pdofBlitExport: GPURenderPipeline | null = null;
  private pdofBlitModule!: GPUShaderModule;
  private solidModule!: GPUShaderModule;
  private solidLayout!: GPUPipelineLayout;
  private pdofUni!: GPUBuffer;
  private pdofLut!: GPUBuffer;
  private pdofLutKey = '';
  private pdofAcc: GPUBuffer | null = null;
  private pdofAccSize = 0;
  private sppBuf!: GPUBuffer;       // post-pass params
  private spBuf!: GPUBuffer;        // solid tonemap params
  private lightsBuf!: GPUBuffer;    // lights + materials
  private sfiltBuf!: GPUBuffer;     // JWildfire raster-cell filter kernel
  private sfiltKey = '';
  private aoRawPipe!: GPUComputePipeline;
  private aoBlurHPipe!: GPUComputePipeline;
  private aoBlurVPipe!: GPUComputePipeline;
  private aopBuf!: GPUBuffer;       // AO pass params
  private aoFiltBuf!: GPUBuffer;    // AO smoothing kernel
  private aoFiltKey = '';
  private aoBlurN = 0;
  private aoOn = false;
  private shAccPipe!: GPUComputePipeline;
  private shSmoothPipe!: GPUComputePipeline;
  private shpBuf!: GPUBuffer;       // shadow lookup params
  private shOn = false;             // shadows on for the current flame (casting lights > 0)
  private shSmoothR = 0;            // SMOOTH radius in raster cells (< 1 = FAST)
  /** shadow maps: kernel mode for the next dispatches — 1 collects the light-space bounds until enough points have
   *  plotted since the reseed (`SHADOW_BOUNDS_SAMPLES` per walker), 2 splats with the frozen bounds */
  private shadowMode = 1;
  private static readonly SHADOW_BOUNDS_ITERS = 32;

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
  private dtMs = 0; // smoothed rAF interval — the UI-stall symptom the budget reacts to
  private gpuProbeInFlight = false;
  /** Present cadence. The tonemap pass (density estimation: (2R+1)² taps per pixel) costs far more than
   *  the chaos game on a big canvas — ~35 ms at 5 Mpx against 5 ms of compute — and presenting it every
   *  frame drags the frame rate down, which the budget controller would read as compute overload and
   *  answer by throttling the chaos game to its floor. So the picture is presented no more often than
   *  every 2× its measured cost (a third of the GPU at most, whatever the display's refresh rate), while
   *  the compute keeps running every frame. */
  private presentMs = 0;
  private presentSkips = 0;
  private lastPresentT = 0;
  private lastPresented = false;
  private probeDoneT = 0;
  passesPerFrame = 2;
  targetQuality = 1000; // spp cap (the live view stops accumulating here; exports pick their own quality)
  /** Wall-clock limit on the live view, seconds of accumulation since the last reset (0 = none). Whichever
   *  of the two caps comes first stops the render — a heavy flame that would take minutes to reach the
   *  quality cap stops here instead of keeping the GPU at full load. Exports ignore it. */
  timeLimitS = 30;
  private liveMs = 0;
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
        // the solid kernel binds up to 9 storage buffers (xd, pts, rngs, pal, zkey, zpay, shadow maps + optional mods, mats)
        maxStorageBuffersPerShaderStage: Math.min(Math.max(adapter.limits.maxStorageBuffersPerShaderStage, 8), 10),
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
    this.paramsBuf = d.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tmBuf = d.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.filtBuf = d.createBuffer({ size: FILT_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.matsBuf = d.createBuffer({ size: this.nPoints * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.meshBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.sppBuf = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.spBuf = d.createBuffer({ size: 192, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lightsBuf = d.createBuffer({ size: (SOLID_MAX_LIGHTS * 3 + SOLID_MAX_MATS * 3) * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.shpBuf = d.createBuffer({ size: 32 + 12 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    {
      const shModule = d.createShaderModule({ code: SOLID_SHADOW_WGSL });
      const C = GPUShaderStage.COMPUTE;
      const shLayout = d.createPipelineLayout({ bindGroupLayouts: [d.createBindGroupLayout({ entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: C, buffer: { type: 'storage' } },
        { binding: 5, visibility: C, buffer: { type: 'storage' } },
      ] })] });
      this.shAccPipe = d.createComputePipeline({ layout: shLayout, compute: { module: shModule, entryPoint: 'accPass' } });
      this.shSmoothPipe = d.createComputePipeline({ layout: shLayout, compute: { module: shModule, entryPoint: 'smoothPass' } });
    }
    this.sfiltBuf = d.createBuffer({ size: SOLID_FILT_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.solidPostPipe = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: SOLID_POST_WGSL }), entryPoint: 'main' } });
    this.aopBuf = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.aoFiltBuf = d.createBuffer({ size: AO_FILT_MAX_N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    {
      const aoModule = d.createShaderModule({ code: SOLID_AO_WGSL });
      const C = GPUShaderStage.COMPUTE;
      const aoLayout = d.createPipelineLayout({ bindGroupLayouts: [d.createBindGroupLayout({ entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: C, buffer: { type: 'storage' } },
        { binding: 4, visibility: C, buffer: { type: 'storage' } },
        { binding: 5, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: C, buffer: { type: 'storage' } },
      ] })] });
      this.aoRawPipe = d.createComputePipeline({ layout: aoLayout, compute: { module: aoModule, entryPoint: 'aoRawPass' } });
      this.aoBlurHPipe = d.createComputePipeline({ layout: aoLayout, compute: { module: aoModule, entryPoint: 'aoBlurH' } });
      this.aoBlurVPipe = d.createComputePipeline({ layout: aoLayout, compute: { module: aoModule, entryPoint: 'aoBlurV' } });
    }
    const solidModule = d.createShaderModule({ code: SOLID_TONEMAP_WGSL });
    // explicit layout so one bind group serves both the canvas and the rgba8 export pipeline
    const F = GPUShaderStage.FRAGMENT;
    this.solidTmLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: F, buffer: { type: 'uniform' } },
        { binding: 6, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 7, visibility: F, buffer: { type: 'read-only-storage' } },
        { binding: 8, visibility: F, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      ],
    });
    this.reflTex = d.createTexture({ size: [REFL_SIZE, REFL_SIZE, SOLID_MAX_MATS], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const solidLayout = d.createPipelineLayout({ bindGroupLayouts: [this.solidTmLayout] });
    this.solidPipe = d.createRenderPipeline({
      layout: solidLayout,
      vertex: { module: solidModule, entryPoint: 'vs' },
      fragment: { module: solidModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.solidExportPipe = d.createRenderPipeline({
      layout: solidLayout,
      vertex: { module: solidModule, entryPoint: 'vs' },
      fragment: { module: solidModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });

    this.solidModule = solidModule;
    this.solidLayout = solidLayout;
    this.pdofPipe = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: SOLID_PDOF_WGSL }), entryPoint: 'scatter' } });
    this.pdofBlitModule = d.createShaderModule({ code: SOLID_PDOF_BLIT_WGSL });
    this.pdofUni = d.createBuffer({ size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.pdofLut = d.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

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
    const bytes = this.width * wanted * this.height * wanted * this.cellBytes;
    this.oversample = bytes <= this.maxHistBytes ? wanted : 1;
    this.allocHistogram();
    this.resetAccumulation();
    return this.oversample;
  }

  /** The current flame renders solid (JWildfire z-buffer shading instead of density). */
  get solid(): boolean { return !!this.compiled?.solid; }
  /** JWildfire runs PostDOFCalculator when the flame is solid and cam_dof > 0 (plot-time jitter is skipped). */
  private get pdofOn(): boolean { return this.solid && ((this.flame?.camDOF ?? 0) > 1e-9); }
  private get cellBytes(): number { return this.solid ? SOLID_CELL_BYTES : HIST_CELL_BYTES; }

  private makeSolidBufs(cells: number): SolidBufs {
    const d = this.device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC; // COPY_SRC: dev readbacks
    return {
      key: d.createBuffer({ size: cells * 4, usage }),
      pay: d.createBuffer({ size: cells * SOLID_PAY_WORDS * 4, usage }),
      nrm: d.createBuffer({ size: cells * 4, usage }),
      zr: d.createBuffer({ size: cells * 4, usage }),
      ao: d.createBuffer({ size: cells * 4, usage }),
      aoRaw: null, // allocated the first time AO runs on this set
      aoTmp: null,
      aoValid: false,
      smaps: null, sacc: null, svis: null, shKey: '',
      bg: null,
      shValid: false,
      cells,
    };
  }
  private destroySolidBufs(b: SolidBufs | null) {
    if (!b) return;
    for (const buf of [b.key, b.pay, b.nrm, b.zr, b.ao, b.aoRaw, b.aoTmp, b.smaps, b.sacc, b.svis]) buf?.destroy();
  }
  /** Shadow-map storage for `nCast` casting lights of `size²` cells (re-created when either changes). Returns true when re-created. */
  private ensureShadowBufs(b: SolidBufs, nCast: number, size: number): boolean {
    const key = `${nCast}x${size}`;
    if (b.shKey === key) return false;
    const d = this.device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    b.smaps?.destroy(); b.sacc?.destroy(); b.svis?.destroy();
    b.smaps = d.createBuffer({ size: SOLID_MAX_LIGHTS * 16 + nCast * size * size * 4, usage }); // 16 bounds words + maps
    b.sacc = d.createBuffer({ size: Math.max(16, nCast * b.cells * 4), usage });
    b.svis = d.createBuffer({ size: Math.max(16, nCast * b.cells * 4), usage });
    b.shKey = key;
    b.bg = null;
    b.shValid = false;
    return true;
  }
  /** Clear the shadow maps + light-space bounds (min keys to 0xFFFFFFFF, max to 0) of a set. */
  private clearShadowBufs(enc: GPUCommandEncoder, b: SolidBufs) {
    if (!b.smaps) return;
    // the maps are cleared by the encoder (runs at submit); the bounds words are written now — a queue write
    // executes before the later submit, so it must not overlap the clear
    enc.clearBuffer(b.smaps, SOLID_MAX_LIGHTS * 16);
    const init = new Uint32Array(SOLID_MAX_LIGHTS * 4);
    for (let i = 0; i < SOLID_MAX_LIGHTS; i++) { init[i * 4] = 0xffffffff; init[i * 4 + 2] = 0xffffffff; }
    this.device.queue.writeBuffer(b.smaps, 0, init);
  }

  private allocHistogram() {
    const os = this.oversample;
    if (this.width * os * this.height * os * this.cellBytes > this.maxHistBytes) this.oversample = 1;
    const cells = Math.max(this.width * this.oversample * this.height * this.oversample, 1);
    // solid mode: the histogram is not written (a 1-cell stub stays bound), the z-buffer set takes its place
    const size = (this.solid ? 1 : cells) * HIST_CELL_BYTES;
    this.histBuf?.destroy();
    this.histBuf = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.bgB = null; // the filter pass binds the histogram (adaptive kernel selection)
    this.destroySolidBufs(this.solidLive);
    this.solidLive = this.solid ? this.makeSolidBufs(cells) : null;
    if (this.solidLive) this.ensureShadowBufs(this.solidLive, this.shadowCasters(), this.shadowMapSize());
    this.renderBG = this.device.createBindGroup({
      layout: this.pipeA.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tmBuf } },
        { binding: 1, resource: { buffer: this.histBuf } },
      ],
    });
    this.rebuildComputeBG();
  }

  /** Shadow-casting lights of the current flame (0 when shadows are off). */
  private shadowCasters(): number {
    const s = this.flame?.solid;
    if (!s?.enabled || s.shadows.type === 'OFF') return 0;
    return Math.min(s.lights.filter((l) => l.castShadows).length, SOLID_MAX_LIGHTS);
  }
  /** JWildfire: FTOI(shadowmapSize · pixelsPerUnitScale), ≥ 64; capped at 4096² (64 MB per light) here. */
  private shadowMapSize(): number { return Math.min(4096, Math.max(64, Math.round(this.flame?.solid?.shadows.mapSize ?? 2048))); }

  private computeEntries(hist: GPUBuffer, solid: SolidBufs | null): GPUBindGroupEntry[] {
    return [
      { binding: 0, resource: { buffer: this.paramsBuf } },
      { binding: 1, resource: { buffer: this.xdBuf } },
      { binding: 2, resource: { buffer: this.ptsBuf } },
      { binding: 3, resource: { buffer: this.rngBuf } },
      ...(this.compiled?.solid ? [] : [{ binding: 4, resource: { buffer: hist } }]), // the solid kernel never touches the histogram (an 'auto' layout drops it)
      { binding: 5, resource: { buffer: this.palBuf } },
      ...(this.compiled?.usesMods ? [{ binding: 6, resource: { buffer: this.modsBuf } }] : []),
      ...(this.compiled?.solid && solid ? [
        { binding: 7, resource: { buffer: solid.key } }, { binding: 8, resource: { buffer: solid.pay } },
        { binding: 10, resource: { buffer: solid.smaps! } },
      ] : []),
      ...(this.compiled?.usesMat ? [{ binding: 9, resource: { buffer: this.matsBuf } }] : []),
      ...(this.compiled?.usesMesh ? [{ binding: 12, resource: { buffer: this.meshBuf } }] : []),
    ];
  }

  /** obj_mesh_primitive_wf: make sure every mesh the flame uses is prepared and packed into `meshBuf`
   *  (cdf + triangles per key, offsets in `meshLayout` for the data writer). Returns false when something is
   *  still loading — the flame is re-set once it is. */
  private ensureMeshes(flame: Flame): boolean {
    const keys = flameMeshKeys(flame);
    if (!keys.length) return true;
    const packKey = keys.join('|');
    if (keys.every((k) => meshSampler(k))) {
      if (packKey !== this.meshPacked) {
        let total = 0;
        for (const k of keys) { const sm = meshSampler(k)!; total += sm.cdf.length + sm.tris.length; }
        const data = new Float32Array(Math.max(total, 4));
        let o = 0;
        meshLayout.clear();
        for (const k of keys) {
          const sm = meshSampler(k)!;
          data.set(sm.cdf, o); const cdfBase = o; o += sm.cdf.length;
          data.set(sm.tris, o); const triBase = o; o += sm.tris.length;
          meshLayout.set(k, { cdfBase, triBase, faces: sm.faces });
        }
        if (this.meshBuf.size < data.byteLength) {
          this.meshBuf.destroy();
          this.meshBuf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          this.rebuildComputeBG();
        }
        this.device.queue.writeBuffer(this.meshBuf, 0, data);
        this.meshPacked = packKey;
      }
      return true;
    }
    Promise.all(keys.map(ensureMesh)).then(() => { if (this.flame && flameMeshKeys(this.flame).join('|') === packKey) this.setFlame(this.flame); }).catch((e) => this.onError?.(String(e)));
    return false;
  }

  private rebuildComputeBG() {
    if (!this.computePipeline) return;
    if (this.solid && !this.solidLive) { this.allocHistogram(); return; } // allocHistogram calls back into here
    this.computeBG = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: this.computeEntries(this.histBuf, this.solidLive),
    });
  }

  /** Bind groups of the solid passes for one z-buffer set (cached on the set; dropped when a buffer is re-created). */
  private solidBindGroups(b: SolidBufs, withAO: boolean): NonNullable<SolidBufs['bg']> {
    if (b.bg && (!withAO || b.bg.ao)) return b.bg;
    const d = this.device;
    if (withAO && !b.aoRaw) { const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC; b.aoRaw = d.createBuffer({ size: b.cells * 4, usage }); b.aoTmp = d.createBuffer({ size: b.cells * 4, usage }); }
    b.bg = {
      post: b.bg?.post ?? d.createBindGroup({
        layout: this.solidPostPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.sppBuf } },
          { binding: 1, resource: { buffer: b.key } },
          { binding: 2, resource: { buffer: b.pay } },
          { binding: 3, resource: { buffer: b.nrm } },
          { binding: 4, resource: { buffer: b.zr } },
        ],
      }),
      ao: b.aoRaw ? d.createBindGroup({
        layout: this.aoRawPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.aopBuf } },
          { binding: 1, resource: { buffer: b.zr } },
          { binding: 2, resource: { buffer: b.nrm } },
          { binding: 3, resource: { buffer: b.aoRaw } },
          { binding: 4, resource: { buffer: b.ao } },
          { binding: 5, resource: { buffer: this.aoFiltBuf } },
          { binding: 6, resource: { buffer: b.aoTmp! } },
        ],
      }) : null,
      sh: b.bg?.sh ?? d.createBindGroup({
        layout: this.shAccPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.shpBuf } },
          { binding: 1, resource: { buffer: b.key } },
          { binding: 2, resource: { buffer: b.pay } },
          { binding: 3, resource: { buffer: b.smaps! } },
          { binding: 4, resource: { buffer: b.sacc! } },
          { binding: 5, resource: { buffer: b.svis! } },
        ],
      }),
      tm: b.bg?.tm ?? d.createBindGroup({
        layout: this.solidTmLayout,
        entries: [
          { binding: 0, resource: { buffer: this.spBuf } },
          { binding: 1, resource: { buffer: b.key } },
          { binding: 2, resource: { buffer: b.pay } },
          { binding: 3, resource: { buffer: b.nrm } },
          { binding: 4, resource: { buffer: this.sfiltBuf } },
          { binding: 5, resource: { buffer: this.lightsBuf } },
          { binding: 6, resource: { buffer: b.ao } },
          { binding: 7, resource: { buffer: b.svis! } },
          { binding: 8, resource: this.reflTex.createView({ dimension: '2d-array' }) },
        ],
      }),
    };
    return b.bg;
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
      const wasSolid = this.solid;
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
      if (this.compiled.solid !== wasSolid) this.allocHistogram(); // histogram ↔ z-buffer set (rebuilds the bind group)
      else this.rebuildComputeBG();
    }
    if (this.compiled.usesMesh) this.ensureMeshes(flame); // (writeData sees 0 faces for meshes still loading)
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
    if (this.solidLive && this.ensureShadowBufs(this.solidLive, this.shadowCasters(), this.shadowMapSize())) this.rebuildComputeBG();
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
    if (this.compiled?.usesMat) this.device.queue.writeBuffer(this.matsBuf, 0, Float32Array.from({ length: this.nPoints }, () => Math.random())); // JWildfire: p.material = random()
  }

  /** Something tone-related changed (or the view): redraw even when the
   *  accumulation is finished. Without this the renderer idles once converged. */
  invalidate() { this.needsPresent = true; }
  private needsPresent = true;

  /** offscreen shadow maps hold the current flame's light-space depth (tiles of one export share them) */
  private offShadowsReady = false;

  resetAccumulation() {
    this.samples = 0;
    this.liveMs = 0;
    this.offShadowsReady = false;
    this.emaSps = 0;
    this.needsPresent = true;
    this.reseedPoints(100); // live: a shorter fuse keeps drags snappy (exports use the full 200)
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.histBuf);
    if (this.solidLive) { enc.clearBuffer(this.solidLive.key); this.solidLive.aoValid = false; this.solidLive.shValid = false; this.clearShadowBufs(enc, this.solidLive); } // key 0 = empty cell; payload/normals are ignored until a key is set
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
    const pu32 = new Uint32Array(104);
    const pf32 = new Float32Array(pu32.buffer);
    const dofOn = Math.abs(f.camDOF ?? 0) > 1e-9;
    const dimOn = (f.dimishZ ?? 0) > 1e-9;
    const cam3d = Math.abs(f.camPitch) > 1e-9 || Math.abs(f.camYaw) > 1e-9 || Math.abs(f.camBank) > 1e-9 || Math.abs(f.camPersp) > 1e-9
      || Math.abs(f.camPosX) > 1e-9 || Math.abs(f.camPosY) > 1e-9 || Math.abs(f.camPosZ) > 1e-9 || dofOn || dimOn
      || this.solid; // JWildfire: solid rendering always projects through the 3D camera (its depth is the z-buffer key)
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
    // JWildfire mixes the dimish-z colour (0..255) with palette colours held on its 200/256 scale (RenderColor),
    // so on our palette-relative scale the dim colour is ×256/200
    const dc = f.dimZColor ?? [0, 0, 0];
    pf32.set([dc[0] * 1.28, dc[1] * 1.28, dc[2] * 1.28, 0], 44);
    // JWildfire's FlameRenderer switches antialiasing off for solid flames (the z-buffer keeps every jittered hit)
    pf32.set([this.solid ? 0 : f.antialiasAmount ?? 0.25, this.solid ? 0 : f.antialiasRadius ?? 0.5, 0, 0], 48);
    this.device.queue.writeBuffer(this.paramsBuf, 0, pu32);

    const tu32 = new Uint32Array(52);
    const tf32 = new Float32Array(tu32.buffer);
    tu32[0] = tile ? tile.tileW : this.width; // output pixels; hist rows are width×os
    tu32[1] = tile ? tile.tileH : this.height;
    {
      const fk = this.uploadFilters(f.filterRadius ?? 0, normFilterKernel(f.filterKernel));
      tu32[2] = fk.nc; tu32[3] = fk.ni;
      // JWildfire runs the adaptive kernel only on density renders (never solid)
      const adaptive = fk.adaptive && !this.solid;
      tf32[48] = adaptive ? 1 : 0; tf32[49] = fk.nLow; tf32[50] = fk.nSmooth; tf32[51] = fk.nDetail;
    }
    // world area covered by the full image (zoom-invariant density normalisation)
    const ppuOut = ppu / os;
    tf32[16] = (fullW / os) * (fullH / os) / (ppuOut * ppuOut);
    tf32[17] = f.contrast ?? 1;
    // JWildfire RenderColor pre-scales palette entries by 200/256, then divides by whiteLevel
    tf32[18] = (255 * 200 / 256) / Math.max(f.whiteLevel ?? 220, 1);
    tf32[19] = f.lowDensityBrightness ?? 0.24;
    // Post symmetry plots N copies of every point; JWildfire compensates by dividing the
    // brightness constant k1 by the copy count (LogScaleCalculator's constructor: POINT → /order,
    // X/Y_AXIS → /2 — note /order even though order+1 points are plotted). The glow term (bg_glow,
    // built from k2) is deliberately NOT divided, matching JWildfire.
    const symDiv = f.postSymmetry ? (f.postSymmetry.type === 'POINT' ? Math.max(1, f.postSymmetry.order) : 2) : 1;
    tf32[4] = f.brightness / symDiv; tf32[5] = f.gamma; tf32[6] = f.vibrancy;
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
    // JWildfire background gradient (corner colours + kind in bgUL.w; the gradient spans the FULL image, tiles offset into it)
    const bgWords = this.bgGradientWords(f, tile ? tile.tileX : 0, tile ? tile.tileY : 0, fullW / os, fullH / os);
    tf32.set(bgWords, 20);
    // GammaCorrectionFilter: saturation is an HSL shift of (saturation − 1), clamped at −1;
    // foreground opacity scales alpha by 1 − atan(3·(v − 1))/1.25.
    tf32[44] = Math.max(-1, (f.saturation ?? 1) - 1);
    tf32[45] = 1 - Math.atan(3 * ((f.fgOpacity ?? 1) - 1)) / 1.25;
    tf32[46] = f.filterLowDensity ?? 0.025;
    tf32[47] = f.filterSharpness ?? 4;
    this.device.queue.writeBuffer(this.tmBuf, 0, tu32);

    if (this.solid) this.writeSolidUniforms(f, w, h, os, transparent, [m00, m10, m20, m01, m11, m21, m02, m12, m22], ppu, Math.hypot(fullW, fullH), bgWords);
  }

  /** Solid rendering: post-pass params, tonemap params, lights/materials, raster-cell filter kernel. */
  /** TP/SP background-gradient block: UL, UR, LL, LR, CC (w of UL = kind), geometry (tile origin, full size in output pixels). */
  private bgGradientWords(f: Flame, tileX: number, tileY: number, fullW: number, fullH: number): number[] {
    const g = f.bgGradient;
    const kind = g ? (g.type === 'GRADIENT_2X2_C' ? 2 : 1) : 0;
    const c = (v?: [number, number, number]) => (v ? [v[0], v[1], v[2]] : [0, 0, 0]);
    return [...c(g?.ul), kind, ...c(g?.ur), 0, ...c(g?.ll), 0, ...c(g?.lr), 0, ...c(g?.cc), 0, tileX, tileY, fullW, fullH];
  }

  private writeSolidUniforms(f: Flame, rasterW: number, rasterH: number, os: number, transparent: boolean, cam: number[], ppu: number, imgSize: number, bgWords: number[]) {
    const s = f.solid!;
    const m2 = cam.slice(6, 9);
    const q = new Uint32Array(8);
    const qf = new Float32Array(q.buffer);
    q[0] = rasterW; q[1] = rasterH; qf[2] = ppu;
    qf.set([m2[0], m2[1], m2[2], f.camPosZ], 4);
    this.device.queue.writeBuffer(this.sppBuf, 0, q);

    // Post-process DOF (PostDOFCalculator): scatter-pass uniforms + the bokeh kernel LUT
    if (this.pdofOn) {
      const pb = s.postBokeh ?? { filterKernel: 'SINEPOW15', intensity: 0.005, brightness: 1, size: 2, activation: 0.2 };
      const kern = normFilterKernel(pb.filterKernel);
      const support = kernelSupport(kern);
      if (this.pdofLutKey !== kern) {
        this.pdofLutKey = kern;
        const lut = new Float32Array(256);
        // JWildfire evaluates getFilterCoeff far beyond the support (glint discs shrink r by
        // plainRadius/radius), and skips non-positive coefficients — bake that into the LUT
        for (let i = 0; i < 256; i++) lut[i] = Math.max(0, kernelCoeff(kern, (i / 255) * support * 8));
        this.device.queue.writeBuffer(this.pdofLut, 0, lut);
      }
      const u = new Uint32Array(44);
      const uf = new Float32Array(u.buffer);
      u[0] = rasterW / os; u[1] = rasterH / os; u[2] = os; u[3] = 0x9e3779b9; // stable seed: glints must not flicker in the live preview
      uf.set([cam[0], cam[1], cam[2], f.camPosX], 4);
      uf.set([cam[3], cam[4], cam[5], f.camPosY], 8);
      uf.set([cam[6], cam[7], cam[8], f.camPosZ], 12);
      const area = f.camDOFArea ?? 0.5;
      uf.set([f.newDOF ? 1 : 0, area, f.camDOFExponent ?? 2, f.camZ ?? 0], 16);
      uf.set([f.focusX ?? 0, f.focusY ?? 0, f.focusZ ?? 0, area / 2.25], 20);
      uf.set([pb.intensity * 1000 / Math.max(imgSize, 1), pb.brightness, pb.size, pb.activation], 24);
      uf.set([support, support * 8, 256, Math.max(1, Math.floor(imgSize))], 28);
      // RasterFloatIntForSolidRendering: dofDist = |dist| · cam_dof · diagonal / 1000
      uf[32] = (f.camDOF ?? 0) * imgSize / 1000;
      this.device.queue.writeBuffer(this.pdofUni, 0, u);
    }

    // AOCalculator params (Tools.limitValue ranges); radii scale with the raster diagonal / 500
    this.aoOn = !!s.ao.enabled;
    if (this.aoOn) {
      const clampN = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
      // imgSize = the FULL raster's diagonal (a hi-res tile keeps the whole image's AO radius)
      const searchR = clampN(s.ao.searchRadius, 0.25, 120), blurR = clampN(s.ao.blurRadius, 0.25, 10) * imgSize / 500;
      const rs = clampN(Math.round(s.ao.radiusSamples), 1, 128), as = clampN(Math.round(s.ao.azimuthSamples), 1, 128);
      const sphere = searchR * imgSize / 500;
      let blurN = 0;
      if (blurR >= 0.42) {
        const { n, w } = gaussianFilter1D(blurR, AO_FILT_MAX_N);
        blurN = n;
        const key = blurR.toFixed(4);
        if (n && key !== this.aoFiltKey) { this.aoFiltKey = key; this.device.queue.writeBuffer(this.aoFiltBuf, 0, w); }
      }
      this.aoBlurN = blurN;
      const a = new Uint32Array(12);
      const af = new Float32Array(a.buffer);
      a[0] = rasterW; a[1] = rasterH; a[2] = rs; a[3] = as;
      af[4] = sphere; af[5] = sphere / rs; af[6] = (2 * Math.PI) / as; af[7] = clampN(s.ao.falloff, 0, 10);
      a[8] = blurN; a[9] = 32851137;
      this.device.queue.writeBuffer(this.aopBuf, 0, a);
    }

    const { n, w: fw } = solidFilterWeights(f.filterRadius ?? 0, normFilterKernel(f.filterKernel), os);
    const fkey = `${f.filterKernel}:${(f.filterRadius ?? 0).toFixed(4)}:${os}`;
    if (n && fkey !== this.sfiltKey) { this.sfiltKey = fkey; this.device.queue.writeBuffer(this.sfiltBuf, 0, fw); }

    const p = new Uint32Array(44);
    const pf = new Float32Array(p.buffer);
    p[0] = rasterW / os; p[1] = rasterH / os; p[2] = os; p[3] = n;
    pf[4] = f.gamma; pf[5] = transparent ? 1 : 0;
    const nL = Math.min(s.lights.length, SOLID_MAX_LIGHTS), nM = Math.min(s.materials.length, SOLID_MAX_MATS);
    p[6] = nL; p[7] = nM;
    pf.set([f.background[0], f.background[1], f.background[2], 0], 8);

    // LightViewCalculator: lightDir = aᵀ·(0,0,−1) with a = rotation from alpha = −altitude, beta = −azimuth;
    // a's rows are the light-space projection (x, y → shadow map, z → depth toward the light)
    const lm = new Float32Array((SOLID_MAX_LIGHTS * 3 + SOLID_MAX_MATS * 3) * 4);
    const rows = new Float32Array(12 * 4); // casting lights' rows for the kernel splat + lookups
    const shOn = this.shOn = s.shadows.type !== 'OFF' && s.lights.some((l) => l.castShadows);
    let nCast = 0;
    for (let i = 0; i < nL; i++) {
      const l = s.lights[i];
      const al = (-l.altitude * Math.PI) / 180, be = (-l.azimuth * Math.PI) / 180;
      const sa = Math.sin(al), ca = Math.cos(al), sb = Math.sin(be), cb = Math.cos(be);
      // aᵀ·(0,0,−1) = −(row 2 of a) = −(−ca·sb, sa, ca·cb)
      lm.set([ca * sb, -sa, -ca * cb, l.intensity], i * 12);
      lm.set([l.color[0], l.color[1], l.color[2], l.castShadows ? 1 : 0], i * 12 + 4);
      const casts = shOn && l.castShadows && nCast < SOLID_MAX_LIGHTS;
      lm.set([casts ? nCast : -1, 1 - Math.min(1, Math.max(0, l.shadowIntensity)), 0, 0], i * 12 + 8);
      if (casts) {
        rows.set([cb, 0, sb, 0], nCast * 12);
        rows.set([sa * sb, ca, -sa * cb, 0], nCast * 12 + 4);
        rows.set([-ca * sb, sa, ca * cb, 0], nCast * 12 + 8);
        nCast++;
      }
    }
    const mo = SOLID_MAX_LIGHTS * 12;
    void this.ensureReflMaps(f);
    for (let i = 0; i < nM; i++) {
      const m = s.materials[i];
      lm.set([m.diffuse, m.ambient, m.phong, m.phongSize], mo + i * 12);
      lm.set([m.phongColor[0], m.phongColor[1], m.phongColor[2], Math.max(0, LIGHT_DIFF_FUNCS.indexOf(m.diffFunc))], mo + i * 12 + 4);
      const layer = m.reflMapFilename && m.reflMapIntensity > 1e-9 ? this.reflLayers.get(reflMapKey(m.reflMapFilename)) : undefined;
      lm.set([m.reflMapIntensity, m.reflMapping === 'SPHERICAL' ? 1 : 0, layer === undefined ? 0 : layer + 1, 0], mo + i * 12 + 8);
    }
    this.device.queue.writeBuffer(this.lightsBuf, 0, lm);

    // ShadowCalculator: map size (FTOI(shadowmapSize · pixelsPerUnitScale), ≥ 64), bias, SMOOTH radius
    // FTOI(smoothRadius · 6 · imgSize / 1000) capped 128 (< 1 → FAST); shadow maps + bounds live on the z-buffer set
    const size = this.shadowMapSize();
    let smoothR = 0;
    if (s.shadows.type === 'SMOOTH') {
      const raw = s.shadows.smoothRadius < 1e-8 ? 0 : s.shadows.smoothRadius;
      smoothR = Math.min(128, Math.round(raw * 6 * imgSize / 1000));
    }
    this.shSmoothR = smoothR;
    pf.set([this.aoOn ? 1 : 0, Math.min(4, Math.max(0, s.ao.intensity)), Math.max(0, s.ao.affectDiffuse), 0], 12);
    p[16] = shOn ? 1 : 0; p[17] = rasterW * rasterH; p[18] = nCast; p[19] = 0;
    pf.set(bgWords, 20);
    this.device.queue.writeBuffer(this.spBuf, 0, p);
    const hp = new Uint32Array(8 + 12 * 4);
    const hf = new Float32Array(hp.buffer);
    hp[0] = rasterW; hp[1] = rasterH; hp[2] = size; hp[3] = nCast;
    hf[4] = s.shadows.bias; hp[5] = smoothR >>> 0; // i32 in the shader; smoothR ≥ 0
    hf.set(rows, 8);
    this.device.queue.writeBuffer(this.shpBuf, 0, hp);
    // kernel: shadow mode + rows (Params.shadow at u32 52, Params.lm at 56)
    const ku = new Uint32Array(4 + 12 * 4);
    const kf = new Float32Array(ku.buffer);
    ku[0] = shOn ? this.shadowMode : 0; ku[1] = size; ku[2] = nCast;
    kf.set(rows, 4);
    this.device.queue.writeBuffer(this.paramsBuf, 52 * 4, ku);
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
          // adaptive filtering reads raw raster cells (density + colour sums) to pick a kernel per
          // pixel, so the filter pass needs the histogram too — the export pass its own copy
          { binding: 1, resource: { buffer: exportFmt ? this.offHist! : this.histBuf } },
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

  /** Solid rendering: post pass (key repair + normals) over the raster, then the shading/filter/composite pass to `target`. */
  private encodeSolid(enc: GPUCommandEncoder, bufs: SolidBufs, rasterW: number, rasterH: number, pipe: GPURenderPipeline, target: GPUTextureView, clear: GPUColor, refreshAO = true) {
    const bg = this.solidBindGroups(bufs, this.aoOn);
    const cp = enc.beginComputePass();
    cp.setPipeline(this.solidPostPipe);
    cp.setBindGroup(0, bg.post);
    const gx = Math.ceil(rasterW / 16), gy = Math.ceil(rasterH / 16);
    cp.dispatchWorkgroups(gx, gy);
    if (this.aoOn && bg.ao && (refreshAO || !bufs.aoValid)) {
      bufs.aoValid = true;
      // AOCalculator: raw occlusion from the z-buffer, then (blur radius ≥ 0.42 cells) the gaussian smoothing × 0.1
      cp.setPipeline(this.aoRawPipe);
      cp.setBindGroup(0, bg.ao);
      cp.dispatchWorkgroups(gx, gy);
      if (this.aoBlurN > 0) {
        cp.setPipeline(this.aoBlurHPipe);
        cp.dispatchWorkgroups(gx, gy);
        cp.setPipeline(this.aoBlurVPipe);
        cp.dispatchWorkgroups(gx, gy);
      }
    }
    if (this.shOn && (refreshAO || !bufs.shValid)) {
      // ShadowCalculator: accelerateShadows per cell (+ smoothing for SMOOTH shadows); refreshed with the AO cadence live
      bufs.shValid = true;
      cp.setPipeline(this.shAccPipe);
      cp.setBindGroup(0, bg.sh);
      cp.dispatchWorkgroups(gx, gy);
      if (this.shSmoothR >= 1) {
        cp.setPipeline(this.shSmoothPipe);
        cp.dispatchWorkgroups(gx, gy);
      }
    }
    cp.end();
    if (!this.pdofOn) {
      const rp = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: 'clear', clearValue: clear, storeOp: 'store' }] });
      rp.setPipeline(pipe);
      rp.setBindGroup(0, bg.tm);
      rp.draw(3);
      rp.end();
      return;
    }
    // Post-process DOF (PostDOFCalculator): tonemap into midTex, scatter every pixel as a bokeh
    // disc into a fixed-point buffer, then blit the buffer to the target.
    const d = this.device;
    const exportFmt = pipe === this.solidExportPipe;
    const outW = exportFmt ? rasterW : this.width, outH = exportFmt ? rasterH : this.height;
    this.ensureMid(outW, outH);
    if (!this.solidMidPipe) {
      this.solidMidPipe = d.createRenderPipeline({
        layout: this.solidLayout,
        vertex: { module: this.solidModule, entryPoint: 'vs' },
        fragment: { module: this.solidModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] },
        primitive: { topology: 'triangle-list' },
      });
    }
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.midTex!.createView(), loadOp: 'clear', clearValue: clear, storeOp: 'store' }] });
    rp.setPipeline(this.solidMidPipe);
    rp.setBindGroup(0, bg.tm);
    rp.draw(3);
    rp.end();
    const need = outW * outH * 16;
    if (!this.pdofAcc || this.pdofAccSize < need) {
      this.pdofAcc?.destroy();
      this.pdofAcc = d.createBuffer({ size: need, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.pdofAccSize = need;
    }
    enc.clearBuffer(this.pdofAcc);
    const scatterBG = d.createBindGroup({
      layout: this.pdofPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.pdofUni } },
        { binding: 1, resource: { buffer: bufs.key } },
        { binding: 2, resource: { buffer: bufs.pay } },
        { binding: 3, resource: this.midTex!.createView() },
        { binding: 4, resource: { buffer: this.pdofLut } },
        { binding: 5, resource: { buffer: this.pdofAcc } },
      ],
    });
    const sp = enc.beginComputePass();
    sp.setPipeline(this.pdofPipe);
    sp.setBindGroup(0, scatterBG);
    sp.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    sp.end();
    let blit = exportFmt ? this.pdofBlitExport : this.pdofBlitLive;
    if (!blit) {
      blit = d.createRenderPipeline({
        layout: 'auto',
        vertex: { module: this.pdofBlitModule, entryPoint: 'vs' },
        fragment: { module: this.pdofBlitModule, entryPoint: 'fs', targets: [{ format: exportFmt ? 'rgba8unorm' : this.format }] },
        primitive: { topology: 'triangle-list' },
      });
      if (exportFmt) this.pdofBlitExport = blit; else this.pdofBlitLive = blit;
    }
    const blitBG = d.createBindGroup({
      layout: blit.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.pdofUni } }, // PB reads only width/height — PD's first words
        { binding: 1, resource: { buffer: this.pdofAcc } },
        { binding: 2, resource: this.midTex!.createView() },
      ],
    });
    const bp = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: 'clear', clearValue: clear, storeOp: 'store' }] });
    bp.setPipeline(blit);
    bp.setBindGroup(0, blitBG);
    bp.draw(3);
    bp.end();
  }

  /** Upload the JWildfire spatial-filter kernels for the flame's settings; returns [Ncolour, Nintensity]. */
  private uploadFilters(radius: number, kernel: FilterKernel) {
    const built = buildSpatialFilters(radius, kernel);
    const { weights, nc, ni, key } = built;
    if (nc === 0 && ni === 0) return { ...built, nc: 0, ni: 0 };
    if (key !== this.filtKey) {
      this.filtKey = key;
      this.device.queue.writeBuffer(this.filtBuf, 0, weights);
    }
    return built;
  }

  /** Resolves once everything the current flame needs is on the GPU (mesh primitives load asynchronously;
   *  the live view simply shows them when they arrive, exports wait here). */
  /** Upload the reflection maps a flame's materials name (resampled to REFL_SIZE², one texture layer each);
   *  asynchronous — the materials read layer 0 = none until the images are in, then the view is redrawn. */
  private ensureReflMaps(flame: Flame): Promise<void> {
    const names = flameReflMaps(flame);
    const keys = names.map(reflMapKey);
    const packKey = keys.join('|');
    if (packKey === this.reflPacked || packKey === this.reflLoading) return Promise.resolve();
    this.reflLoading = packKey;
    return Promise.all(names.map(loadReflMap)).then((imgs) => {
      if (this.reflLoading !== packKey) return; // superseded
      this.reflLayers.clear();
      let layer = 0;
      imgs.forEach((img, i) => {
        if (!img || layer >= SOLID_MAX_MATS) return;
        this.device.queue.writeTexture({ texture: this.reflTex, origin: [0, 0, layer] }, img.data, { bytesPerRow: REFL_SIZE * 4, rowsPerImage: REFL_SIZE }, [REFL_SIZE, REFL_SIZE, 1]);
        this.reflLayers.set(keys[i], layer++);
      });
      this.reflPacked = packKey;
      this.reflLoading = '';
      this.invalidate();
    });
  }

  async ready(): Promise<void> {
    if (!this.flame || !this.compiled?.usesMesh) return;
    const keys = flameMeshKeys(this.flame);
    await Promise.all(keys.map(ensureMesh));
    if (this.solid) await this.ensureReflMaps(this.flame);
    if (!keys.every((k) => meshLayout.has(k)) || keys.join('|') !== this.meshPacked) this.setFlame(this.flame);
  }

  /** Offline stepping for video export: accumulate `passes` compute dispatches
   *  and resolve when the GPU is done. Pair with captureSync() to grab pixels. */
  async stepExport(passes: number): Promise<void> {
    await this.ready();
    if (!this.flame || !this.computePipeline || !this.computeBG) return;
    const CHUNK = 24; // keep single submissions short to stay watchdog-friendly
    let done = 0;
    while (done < passes) {
      const nowPasses = Math.min(this.shadowCasters() > 0 && this.samples < this.nPoints * FlameRenderer.SHADOW_BOUNDS_ITERS ? 2 : CHUNK, passes - done);
      this.shadowMode = this.samples >= this.nPoints * FlameRenderer.SHADOW_BOUNDS_ITERS ? 2 : 1;
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
    await this.ready();

    const solid = this.solid;
    const cells = o.tileW * o.tileH;
    const need = (solid ? 1 : cells) * HIST_CELL_BYTES;
    if (!this.offHist || this.offHistSize < need) {
      this.offHist?.destroy();
      // COPY_SRC: dev readbacks of the raw raster (renderCheck-style numeric debugging) — without it a
      // copyBufferToBuffer readback silently returns zeros (same trap as the solid buffers, 2026-08-18)
      this.offHist = d.createBuffer({ size: need, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      this.offHistSize = need;
      this.bgBExport = null; // the filter pass binds this buffer (adaptive kernel selection)
    }
    if (solid && (!this.solidOff || this.solidOff.cells < cells)) {
      this.destroySolidBufs(this.solidOff);
      this.solidOff = this.makeSolidBufs(cells);
      this.offShadowsReady = false;
    }
    if (solid && this.solidOff && this.ensureShadowBufs(this.solidOff, this.shadowCasters(), this.shadowMapSize())) this.offShadowsReady = false;
    const computeBG = d.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: this.computeEntries(this.offHist, this.solidOff),
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
      // solid shadows: until enough points have plotted (fuse + SHADOW_BOUNDS_ITERS per walker) the kernel only
      // collects the light-space bounds (mode 1) — short chunks so little is lost to that phase
      const collecting = this.shadowCasters() > 0 && !this.offShadowsReady && done < this.nPoints * FlameRenderer.SHADOW_BOUNDS_ITERS;
      this.shadowMode = collecting ? 1 : 2;
      const nowP = Math.min(collecting ? 2 : CHUNK, passes);
      passes -= nowP;
      done += this.countIters(nowP * perPass);
      this.writeUniforms(done / (o.fullW * o.fullH), tile, o.transparent);
      const enc = d.createCommandEncoder();
      // the shadow maps are light-space (view independent): the tiles of one export share them — cleared only when
      // the flame changed (setFlame) or the buffers were re-created
      if (first) { enc.clearBuffer(this.offHist); if (solid && this.solidOff) { enc.clearBuffer(this.solidOff.key); if (!this.offShadowsReady) this.clearShadowBufs(enc, this.solidOff); } first = false; }
      const cp = enc.beginComputePass();
      cp.setPipeline(this.computePipeline);
      cp.setBindGroup(0, computeBG);
      for (let i = 0; i < nowP; i++) cp.dispatchWorkgroups(Math.ceil(this.nPoints / 256));
      cp.end();
      d.queue.submit([enc.finish()]);
      // Wait between chunks so submissions stay short and the tab responsive.
      await d.queue.onSubmittedWorkDone();
    }

    if (solid && this.shadowCasters() > 0) this.offShadowsReady = true;
    this.writeUniforms(done / (o.fullW * o.fullH), tile, o.transparent);
    const bpr = Math.ceil((o.tileW * 4) / 256) * 256;
    const rb = d.createBuffer({ size: bpr * o.tileH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    if (solid && this.solidOff) this.encodeSolid(enc, this.solidOff, o.tileW, o.tileH, this.solidExportPipe, this.offTex.createView(), { r: 0, g: 0, b: 0, a: 0 });
    else this.encodeTonemap(enc, renderBG, this.exportPipeline, this.offTex.createView(), o.tileW, o.tileH, true, { r: 0, g: 0, b: 0, a: 0 });
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

  /** Encode the live view's tonemap (density or solid) to `view`. */
  private aoTick = 0;
  private presentTo(enc: GPUCommandEncoder, view: GPUTextureView, live = false) {
    const os = this.oversample;
    // live view: the AO passes (the costliest part of a solid present) refresh every third presented frame while
    // accumulating — the z-buffer changes slowly once the surface is in; captures/exports always refresh
    const refreshAO = !live || (this.aoTick++ % 3) === 0;
    if (this.solid && this.solidLive) this.encodeSolid(enc, this.solidLive, this.width * os, this.height * os, this.solidPipe, view, { r: 0, g: 0, b: 0, a: 1 }, refreshAO);
    else this.encodeTonemap(enc, this.renderBG!, this.renderPipeline, view, this.width, this.height, false, { r: 0, g: 0, b: 0, a: 1 });
  }

  /** Draw the tonemap pass and hand the canvas to `fn` synchronously, in the
   *  same task as the submit — a WebGPU canvas is cleared once the task ends,
   *  so captures (VideoFrame, toBlob, drawImage) must not happen after an
   *  await. Queue ordering guarantees `fn` sees the finished image. */
  captureSync<T>(fn: (canvas: HTMLCanvasElement) => T): T {
    this.writeUniforms(this.samples / Math.max(this.width * this.height, 1));
    const enc = this.device.createCommandEncoder();
    const view = this.context.getCurrentTexture().createView();
    this.presentTo(enc, view);
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
    const timedOut = this.timeLimitS > 0 && this.liveMs >= this.timeLimitS * 1000;
    const converged = spp >= this.targetQuality || timedOut;
    const accumulate = !this.paused && !converged;
    const stats = (samplesPerSec: number): RenderStats => ({ spp, samplesPerSec, paused: this.paused, converged, timedOut, elapsedS: this.liveMs / 1000, budgetScale: this.budgetScale, gpuMs: this.gpuMs });

    const dt = this.lastT ? (t - this.lastT) / 1000 : 0;
    this.lastT = t;
    if (accumulate && dt > 0 && dt < 0.5) this.liveMs += dt * 1000; // rendering time only: a hidden tab or a pause does not count
    if (dt > 0.5) { this.gpuMs = 0; this.dtMs = 0; } // we were away (hidden tab, stall): the last probe says nothing about the GPU load
    else if (dt > 0) this.dtMs = this.dtMs ? this.dtMs * 0.8 + dt * 1000 * 0.2 : dt * 1000;

    // Idle: converged (or paused) and nothing to redraw → do no GPU work at all.
    // The canvas keeps its last presented frame. invalidate() wakes us up.
    if (!accumulate && !this.needsPresent) {
      this.onFrame?.(stats(0));
      return;
    }

    // Adaptive budget: shrink the iterations per preview pass while the GPU time per frame overshoots the
    // budget, grow them back (up to the configured count) when there is headroom. Measured with
    // onSubmittedWorkDone on the frame's own submission (queue backlog included — exactly the stall we avoid).
    // Throttle only when both say so — the GPU probe overshoots the budget AND the frame interval has
    // stretched past ~60 fps (a healthy vsync-paced loop never throttles, whatever the probe reads);
    // recover as soon as either shows headroom.
    if (this.adaptiveBudget && accumulate && this.gpuMs > 0) {
      if (this.gpuMs > this.frameBudgetMs * 1.3 && this.dtMs > 20) this.budgetScale = Math.max(1 / 16, this.budgetScale * 0.7);
      else if ((this.gpuMs < this.frameBudgetMs * 0.6 || this.dtMs < 17.5) && this.budgetScale < 1) this.budgetScale = Math.min(1, this.budgetScale * 1.2);
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
    let probing = false;
    if (accumulate && this.minDisplaySpp > 0) {
      const need = Math.ceil((this.minDisplaySpp * w * h - this.samples) / perPass);
      if (need > passes) passes = Math.min(need, this.passesPerFrame * 4);
    }
    this.shadowMode = this.samples >= this.nPoints * FlameRenderer.SHADOW_BOUNDS_ITERS ? 2 : 1; // bounds from the points plotted so far
    if (accumulate) this.samples += this.countIters(perPass * passes);
    this.writeUniforms(this.samples / Math.max(w * h, 1), undefined, false, iters); // tonemap spp is per output pixel

    if (accumulate) {
      // the compute passes go in their own submission so the budget probe measures them (plus any
      // backlog) without the presentation that follows
      const cenc = this.device.createCommandEncoder();
      const pass = cenc.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBG);
      for (let i = 0; i < passes; i++) {
        pass.dispatchWorkgroups(Math.ceil(this.nPoints / 256));
      }
      pass.end();
      this.device.queue.submit([cenc.finish()]);
      // Probe the compute cost on frames whose queue holds no present from the previous frame (the queue
      // is FIFO, so a present ahead of the compute would be timed with it) — unless presents are cheap,
      // in which case every frame presents and the probe includes it, as it always did.
      if (!this.gpuProbeInFlight && (this.presentMs < 4 || !this.lastPresented)) {
        this.gpuProbeInFlight = true;
        probing = true;
        const t0 = performance.now();
        this.device.queue.onSubmittedWorkDone().then(() => {
          const ms = Math.min(performance.now() - t0, 200); // capped: one stall must not poison the average for long
          this.gpuMs = this.gpuMs ? this.gpuMs * 0.8 + ms * 0.2 : ms;
          this.probeDoneT = performance.now();
          this.gpuProbeInFlight = false;
        }, () => { this.gpuProbeInFlight = false; });
      }
    }
    const enc = this.device.createCommandEncoder();
    // Present gate: after a reset (every frame while a triangle is dragged) the
    // first few frames are too sparse to look like anything. Keep accumulating
    // in the background and leave the last presented image on the canvas until
    // the new one has reached minDisplaySpp — a WebGPU canvas keeps its last
    // frame as long as we don't fetch a new current texture.
    const sppAfter = this.samples / Math.max(w * h, 1);
    const due = performance.now() - this.lastPresentT >= 2 * this.presentMs - 1;
    const present = !accumulate || !this.hasPresented || this.needsPresent || (due && sppAfter >= this.minDisplaySpp);
    if (present) {
      const view = this.context.getCurrentTexture().createView();
      this.presentTo(enc, view, accumulate);
      this.hasPresented = true;
      this.needsPresent = false;
      this.lastPresentT = performance.now();
      this.device.queue.submit([enc.finish()]);
      if (probing) {
        // this present follows the probed compute in the queue: its own cost is the gap between the two
        // completions (callbacks that arrive in the same batch say nothing — skipped, and a run of skips
        // lets the estimate decay so a canvas that got smaller presents more often again)
        this.device.queue.onSubmittedWorkDone().then(() => {
          const own = performance.now() - this.probeDoneT;
          if (own > 0.3) { this.presentMs = this.presentMs ? this.presentMs * 0.7 + own * 0.3 : own; this.presentSkips = 0; }
          else if (++this.presentSkips > 10) { this.presentMs *= 0.8; this.presentSkips = 0; }
        }, () => {});
      }
    }
    this.lastPresented = present;

    if (accumulate && dt > 0 && dt < 0.5) {
      const sps = (perPass * passes) / dt;
      this.emaSps = this.emaSps ? this.emaSps * 0.95 + sps * 0.05 : sps;
    }
    this.onFrame?.(stats(this.emaSps));
  };

  /** Dev diagnostic: total opacity-weighted hits in the live histogram and the count at a pixel. */
  async debugHistStats(px = -1, py = -1): Promise<{ hits: number; atPixel: number; cells: number }> {
    const os = this.oversample;
    if (this.solid) return { hits: 0, atPixel: -1, cells: 0 };
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
