// The composer owns the screen canvas and drives one FlameRenderer per composition layer, each rendering
// into its own offscreen rgba16float texture (straight alpha); every frame in which a layer presented, the
// stack is blended bottom-up (W3C blend modes, opacity, optional clip to what is below) into the canvas —
// one render pass per layer, ping-ponging between two backdrop textures, the last one targeting the canvas.
// It also composites export tiles (renderRegion, on the CPU with the same formulas) and offers the
// synchronous capture the video/thumbnail paths need.

import { FlameRenderer, type RenderStats } from './renderer';
import type { Composition, CompLayer, BlendMode } from '../core/composition';
import { BLEND_MODES, BLEND_WGSL, blendPixel } from '../core/composition';

const COMPOSITE_WGSL = `
struct CP { mode: u32, opacity: f32, flags: u32, pad: u32, bg: vec4f }
@group(0) @binding(0) var<uniform> P: CP;
@group(0) @binding(1) var backdrop: texture_2d<f32>;
@group(0) @binding(2) var source: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
${BLEND_WGSL}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let xy = vec2i(in.pos.xy);
  // flags: 1 = first layer (backdrop is the composition background), 2 = clip to the backdrop's alpha
  var b = P.bg;
  if ((P.flags & 1u) == 0u) { b = textureLoad(backdrop, xy, 0); }
  var s = textureLoad(source, xy, 0);
  if ((P.flags & 2u) != 0u) { s.a = s.a * b.a; }
  return blendOver(P.mode, b, s, P.opacity);
}
`;

interface Slot {
  id: string;
  renderer: FlameRenderer;
  layer: CompLayer;
  flameJson: string;
}

export interface CompRegionOpts {
  fullW: number; fullH: number; tileX: number; tileY: number; tileW: number; tileH: number;
  spp: number; transparent?: boolean;
}

export class Composer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private pipeCanvas!: GPURenderPipeline;
  private pipeTex!: GPURenderPipeline;
  private bgLayout!: GPUBindGroupLayout;
  private ping: GPUTexture[] = [];
  private pingW = 0;
  private pingH = 0;
  private cpBufs: GPUBuffer[] = [];
  private raf = 0;
  private needsComposite = true;
  private hasComposited = false;

  readonly canvas: HTMLCanvasElement;
  slots: Slot[] = [];
  comp: Composition | null = null;
  /** index of the layer whose stats are reported (the active one) */
  active = 0;
  private bg: [number, number, number] = [0, 0, 0];

  onError: ((msg: string) => void) | null = null;
  onFrame: ((stats: RenderStats) => void) | null = null;

  // ---- render settings shared by every layer renderer (the render panel edits them here) ----
  private _exporting = false;
  get exporting() { return this._exporting; }
  set exporting(v: boolean) { this._exporting = v; for (const s of this.slots) s.renderer.exporting = v; }
  private settings: { targetQuality: number; minDisplaySpp: number; deLiveCap: number; adaptiveBudget: boolean; passesPerFrame: number } = { targetQuality: 1000, minDisplaySpp: 10, deLiveCap: 6, adaptiveBudget: true, passesPerFrame: 2 };
  private os = 1;
  private applySetting<K extends keyof Composer['settings']>(key: K, value: Composer['settings'][K]) {
    this.settings[key] = value;
    for (const s of this.slots) (s.renderer as any)[key] = value;
  }
  // FlameRenderer-shaped settings, applied to every layer renderer (and to layers created later)
  get targetQuality() { return this.settings.targetQuality; } set targetQuality(v: number) { this.applySetting('targetQuality', v); }
  get minDisplaySpp() { return this.settings.minDisplaySpp; } set minDisplaySpp(v: number) { this.applySetting('minDisplaySpp', v); }
  get deLiveCap() { return this.settings.deLiveCap; } set deLiveCap(v: number) { this.applySetting('deLiveCap', v); }
  get adaptiveBudget() { return this.settings.adaptiveBudget; } set adaptiveBudget(v: boolean) { this.applySetting('adaptiveBudget', v); }
  get passesPerFrame() { return this.settings.passesPerFrame; } set passesPerFrame(v: number) { this.applySetting('passesPerFrame', v); }
  setOversample(os: number): number {
    let applied = os;
    for (const s of this.slots) applied = Math.min(applied, s.renderer.setOversample(os));
    this.os = applied;
    return applied;
  }
  get oversample() { return this.os; }
  get width() { return this.canvas.width; }
  get height() { return this.canvas.height; }
  get gpuDevice(): GPUDevice { return this.device; }
  /** the active layer's own renderer — single-flame paths (mutation grid, batch export of flames, dev tools) */
  get layerRenderer(): FlameRenderer { const r = this.slots[this.active]?.renderer ?? this.slots[0]?.renderer; if (!r) throw new Error('no layer renderer yet'); return r; }
  /** Render `f` in the active layer's renderer (exports/previews driving frames themselves — the document is untouched);
   *  every other layer re-accumulates its own flame. Restore afterwards with `restore()` or by setting the document's flame. */
  setFlame(f: import('../core/flame').Flame) {
    this.slots.forEach((s, i) => { s.renderer.setFlame(i === this.active ? f : s.layer.flame); s.flameJson = i === this.active ? '' : s.flameJson; });
    this.needsComposite = true;
  }
  /** total spp of the active layer, budget etc. come through onFrame */
  paused = false;
  setPaused(p: boolean) { this.paused = p; for (const s of this.slots) s.renderer.setPaused(p); this.needsComposite = true; }
  isPaused() { return this.paused; }
  resetAccumulation() { for (const s of this.slots) s.renderer.resetAccumulation(); }
  invalidate() { for (const s of this.slots) s.renderer.invalidate(); this.needsComposite = true; }

  constructor(canvas: HTMLCanvasElement) { this.canvas = canvas; }

  async init(): Promise<void> {
    this.device = await FlameRenderer.createDevice();
    this.device.addEventListener('uncapturederror', (e) => {
      console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message);
      this.onError?.((e as GPUUncapturedErrorEvent).error.message);
    });
    this.context = this.canvas.getContext('webgpu')!;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    const d = this.device;
    const module = d.createShaderModule({ code: COMPOSITE_WGSL });
    const F = GPUShaderStage.FRAGMENT;
    this.bgLayout = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: F, buffer: { type: 'uniform' } },
      { binding: 1, visibility: F, texture: { sampleType: 'float' } },
      { binding: 2, visibility: F, texture: { sampleType: 'float' } },
    ] });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgLayout] });
    const mk = (format: GPUTextureFormat) => d.createRenderPipeline({ layout, vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    this.pipeCanvas = mk(this.format);
    this.pipeTex = mk('rgba16float');
    this.raf = requestAnimationFrame(this.frame);
  }

  private ensurePing(w: number, h: number) {
    if (this.pingW === w && this.pingH === h && this.ping.length === 2) return;
    for (const t of this.ping) t.destroy();
    this.ping = [0, 1].map(() => this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING }));
    this.pingW = w; this.pingH = h;
  }

  resize(w: number, h: number) {
    w = Math.max(2, w); h = Math.max(2, h);
    if (w === this.canvas.width && h === this.canvas.height) return;
    this.canvas.width = w; this.canvas.height = h;
    for (const s of this.slots) s.renderer.resize(w, h);
    this.hasComposited = false;
    this.needsComposite = true;
  }

  private async makeRenderer(): Promise<FlameRenderer> {
    const r = new FlameRenderer(null);
    r.driven = true;
    r.onError = (m) => this.onError?.(m);
    r.resize(this.canvas.width, this.canvas.height); // sets the offscreen size before init creates the texture
    await r.init(this.device);
    for (const [k, v] of Object.entries(this.settings)) (r as any)[k] = v;
    r.setPaused(this.paused);
    r.exporting = this._exporting;
    if (this.os !== 1) r.setOversample(this.os);
    return r;
  }

  /** Reconcile the layer renderers with the composition (by layer id) and push every layer's flame/settings.
   *  Flames that did not change (JSON-equal) keep accumulating; `force` re-sets them all (after an export). */
  async setComposition(comp: Composition, active: number, force = false): Promise<void> {
    this.comp = comp;
    this.active = Math.max(0, Math.min(active, comp.layers.length - 1));
    this.bg = comp.background;
    const byId = new Map(this.slots.map((s) => [s.id, s]));
    const next: Slot[] = [];
    for (const layer of comp.layers) {
      let slot = byId.get(layer.id);
      if (!slot) {
        slot = { id: layer.id, renderer: await this.makeRenderer(), layer, flameJson: '' };
      }
      byId.delete(layer.id);
      const json = JSON.stringify(layer.flame);
      slot.renderer.transparentBg = !layer.ownBackground;
      if (force || json !== slot.flameJson) { slot.flameJson = json; slot.renderer.setFlame(layer.flame); }
      else if (slot.layer.ownBackground !== layer.ownBackground) slot.renderer.invalidate();
      slot.layer = layer;
      next.push(slot);
    }
    for (const gone of byId.values()) gone.renderer.destroy();
    this.slots = next;
    this.needsComposite = true;
  }

  /** Re-push every layer flame (after renderRegion/exports changed the renderers' flames). */
  restore() { if (this.comp) void this.setComposition(this.comp, this.active, true); }

  private cpBuf(i: number): GPUBuffer {
    while (this.cpBufs.length <= i) this.cpBufs.push(this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    return this.cpBufs[i];
  }

  /** Blend the visible layers into `target` (the canvas view unless given). */
  private encodeComposite(enc: GPUCommandEncoder, target: GPUTextureView) {
    const vis = this.slots.filter((s) => s.layer.visible && s.renderer.layerTexture);
    const w = this.canvas.width, h = this.canvas.height;
    if (!vis.length) {
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: target, clearValue: { r: this.bg[0], g: this.bg[1], b: this.bg[2], a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
      pass.end();
      return;
    }
    this.ensurePing(w, h);
    vis.forEach((s, i) => {
      const last = i === vis.length - 1;
      const u = new ArrayBuffer(32); const u32 = new Uint32Array(u); const f32 = new Float32Array(u);
      u32[0] = Math.max(0, BLEND_MODES.indexOf(s.layer.blend)); f32[1] = s.layer.opacity; u32[2] = (i === 0 ? 1 : 0) | (s.layer.clip ? 2 : 0);
      f32.set([this.bg[0], this.bg[1], this.bg[2], 1], 4);
      this.device.queue.writeBuffer(this.cpBuf(i), 0, u);
      const backdrop = this.ping[(i + 1) & 1].createView();
      const bg = this.device.createBindGroup({ layout: this.bgLayout, entries: [
        { binding: 0, resource: { buffer: this.cpBuf(i) } },
        { binding: 1, resource: backdrop },
        { binding: 2, resource: s.renderer.layerTexture!.createView() },
      ] });
      const view = last ? target : this.ping[i & 1].createView();
      const pass = enc.beginRenderPass({ colorAttachments: [{ view, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(last ? this.pipeCanvas : this.pipeTex);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    });
  }

  private frame = (t: number) => {
    this.raf = requestAnimationFrame(this.frame);
    if (this._exporting) return;
    let activeStats: RenderStats | null = null;
    let anyPresented = false;
    this.slots.forEach((s, i) => {
      if (!s.layer.visible) return;
      s.renderer.presented = false;
      const st = s.renderer.tick(t);
      if (s.renderer.presented) anyPresented = true;
      if (i === this.active) activeStats = st;
    });
    if (anyPresented || this.needsComposite || !this.hasComposited) {
      const enc = this.device.createCommandEncoder();
      this.encodeComposite(enc, this.context.getCurrentTexture().createView());
      this.device.queue.submit([enc.finish()]);
      this.needsComposite = false;
      this.hasComposited = true;
    }
    if (activeStats) this.onFrame?.(activeStats);
    else if (this.slots.length) this.onFrame?.({ spp: 0, samplesPerSec: 0, paused: this.paused, converged: true, budgetScale: 1, gpuMs: 0 });
  };

  /** Present every layer and composite synchronously, then hand the canvas to `fn` in the same task (see FlameRenderer.captureSync). */
  captureSync<T>(fn: (canvas: HTMLCanvasElement) => T): T {
    for (const s of this.slots) if (s.layer.visible) s.renderer.presentNow();
    const enc = this.device.createCommandEncoder();
    this.encodeComposite(enc, this.context.getCurrentTexture().createView());
    this.device.queue.submit([enc.finish()]);
    return fn(this.canvas);
  }

  /** Offline stepping (video export): every visible layer accumulates `passes` dispatches. */
  async stepExport(passes: number): Promise<void> {
    for (const s of this.slots) if (s.layer.visible) await s.renderer.stepExport(passes);
  }
  async ready(): Promise<void> { for (const s of this.slots) await s.renderer.ready(); }
  /** the visible layers' walkers all reseed (video export frame changes) */
  get nPoints() { return this.slots[0]?.renderer.nPoints ?? 1 << 16; }
  get itersPerPass() { return this.slots[0]?.renderer.itersPerPass ?? 64; }

  /** Render a tile of every visible layer offscreen and composite them on the CPU (hi-res export, thumbnails, dev tools).
   *  Straight-alpha rgba8 like FlameRenderer.renderRegion; `transparent` renders every layer without a background. */
  async renderRegion(o: CompRegionOpts): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const vis = this.slots.filter((s) => s.layer.visible);
    const n = o.tileW * o.tileH;
    const out = new Uint8ClampedArray(n * 4);
    const bg = this.bg;
    if (!o.transparent) for (let i = 0; i < n; i++) { out[i * 4] = Math.round(bg[0] * 255); out[i * 4 + 1] = Math.round(bg[1] * 255); out[i * 4 + 2] = Math.round(bg[2] * 255); out[i * 4 + 3] = 255; }
    if (!vis.length) return out;
    // one layer with its own opaque background over an opaque bg = the plain single-flame render (bit-exact with the old path)
    if (vis.length === 1 && vis[0].layer.blend === 'normal' && vis[0].layer.opacity >= 1 && !vis[0].layer.clip && (vis[0].layer.ownBackground || o.transparent)) {
      return vis[0].renderer.renderRegion({ ...o, transparent: !!o.transparent || !vis[0].layer.ownBackground });
    }
    const acc = new Float32Array(n * 4);
    for (let i = 0; i < n * 4; i++) acc[i] = out[i] / 255;
    for (let li = 0; li < vis.length; li++) {
      const s = vis[li];
      const px = await s.renderer.renderRegion({ ...o, transparent: !!o.transparent || !s.layer.ownBackground });
      const mode: BlendMode = s.layer.blend;
      for (let i = 0; i < n; i++) {
        const b: [number, number, number, number] = [acc[i * 4], acc[i * 4 + 1], acc[i * 4 + 2], acc[i * 4 + 3]];
        const src: [number, number, number, number] = [px[i * 4] / 255, px[i * 4 + 1] / 255, px[i * 4 + 2] / 255, px[i * 4 + 3] / 255 * (s.layer.clip ? b[3] : 1)];
        const r = blendPixel(mode, b, src, s.layer.opacity);
        acc[i * 4] = r[0]; acc[i * 4 + 1] = r[1]; acc[i * 4 + 2] = r[2]; acc[i * 4 + 3] = r[3];
      }
    }
    for (let i = 0; i < n * 4; i++) out[i] = Math.round(Math.min(1, Math.max(0, acc[i])) * 255);
    return out;
  }

  async exportPNG(): Promise<Blob | null> {
    return new Promise((resolve) => { this.captureSync((cv) => cv.toBlob((b) => resolve(b), 'image/png')); });
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    for (const s of this.slots) s.renderer.destroy();
    this.slots = [];
  }
}
