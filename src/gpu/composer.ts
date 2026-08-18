// The composer owns the screen canvas and drives one FlameRenderer per composition layer, each rendering
// into its own offscreen rgba16float texture (straight alpha); every frame in which a layer presented, the
// stack is blended bottom-up (W3C blend modes, opacity, optional clip to what is below) into the canvas —
// one render pass per layer, ping-ponging between two backdrop textures, the last one targeting the canvas.
// It also composites export tiles (renderRegion, on the CPU with the same formulas) and offers the
// synchronous capture the video/thumbnail paths need.

import { FlameRenderer, type RenderStats } from './renderer';
import { EscapeRenderer } from './escapeRenderer';
import { ImageRenderer } from './imageRenderer';
import type { Composition, CompLayer, BlendMode, LayerEffects } from '../core/composition';
import { BLEND_MODES, BLEND_WGSL, blendPixel, effectsActive, adjustPixel } from '../core/composition';

/** what every layer kind's renderer offers the composer */
export type LayerRenderer = FlameRenderer | EscapeRenderer | ImageRenderer;

const COMPOSITE_WGSL = `
struct CP { mode: u32, opacity: f32, flags: u32, maskMode: u32, bg: vec4f }
@group(0) @binding(0) var<uniform> P: CP;
@group(0) @binding(1) var backdrop: texture_2d<f32>;
@group(0) @binding(2) var source: texture_2d<f32>;
@group(0) @binding(3) var maskTex: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
${BLEND_WGSL}
fn luma(c: vec4f) -> f32 { return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b; }
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let xy = vec2i(in.pos.xy);
  // flags: 1 = first layer (backdrop is the composition background), 2 = clip to the backdrop's alpha
  // maskMode: 0 none, 1 backdrop alpha, 2 backdrop luma, 3 mask texture alpha, 4 mask texture luma; +8 = invert
  var b = P.bg;
  if ((P.flags & 1u) == 0u) { b = textureLoad(backdrop, xy, 0); }
  var s = textureLoad(source, xy, 0);
  if ((P.flags & 2u) != 0u) { s.a = s.a * b.a; }
  let mm = P.maskMode & 7u;
  if (mm != 0u) {
    var m = 1.0;
    if (mm == 1u) { m = b.a; } else if (mm == 2u) { m = luma(b) * b.a; }
    else { let t = textureLoad(maskTex, xy, 0); if (mm == 3u) { m = t.a; } else { m = luma(t) * t.a; } }
    if ((P.maskMode & 8u) != 0u) { m = 1.0 - m; }
    s.a = s.a * clamp(m, 0.0, 1.0);
  }
  return blendOver(P.mode, b, s, P.opacity);
}
`;

// per-layer effects: separable gaussian blur (premultiplied, so edges do not darken) + colour adjustments
const FX_WGSL = `
struct FP { dir: vec2f, radius: f32, sigma: f32, brightness: f32, contrast: f32, saturation: f32, hue: f32, gamma: f32, invert: u32, adjust: u32, pad: u32 }
@group(0) @binding(0) var<uniform> P: FP;
@group(0) @binding(1) var src: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
fn adjustRgb(c0: vec3f) -> vec3f {
  var c = c0;
  if (P.invert != 0u) { c = vec3f(1.0) - c; }
  c = c + vec3f(P.brightness);
  c = (c - vec3f(0.5)) * (1.0 + P.contrast) + vec3f(0.5);
  let l = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  c = vec3f(l) + (c - vec3f(l)) * (1.0 + P.saturation);
  if (P.hue != 0.0) {
    let a = P.hue * 3.14159265 / 180.0; let ca = cos(a); let sa = sin(a);
    let y = 0.299 * c.r + 0.587 * c.g + 0.114 * c.b; let i = 0.596 * c.r - 0.274 * c.g - 0.322 * c.b; let q = 0.211 * c.r - 0.523 * c.g + 0.312 * c.b;
    let i2 = i * ca - q * sa; let q2 = i * sa + q * ca;
    c = vec3f(y + 0.956 * i2 + 0.621 * q2, y - 0.272 * i2 - 0.647 * q2, y - 1.106 * i2 + 1.703 * q2);
  }
  if (P.gamma != 1.0) { c = pow(max(c, vec3f(0.0)), vec3f(1.0 / P.gamma)); }
  return clamp(c, vec3f(0.0), vec3f(1.0));
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let xy = vec2i(in.pos.xy);
  let dims = vec2i(textureDimensions(src));
  var acc = vec4f(0.0);
  if (P.radius > 0.0) {
    // premultiplied gaussian along P.dir
    let r = i32(ceil(P.radius));
    var wsum = 0.0;
    for (var k = -r; k <= r; k = k + 1) {
      let w = exp(-f32(k * k) / (2.0 * P.sigma * P.sigma));
      let q = clamp(xy + vec2i(P.dir) * k, vec2i(0), dims - vec2i(1));
      let t = textureLoad(src, q, 0);
      acc = acc + vec4f(t.rgb * t.a, t.a) * w;
      wsum = wsum + w;
    }
    acc = acc / wsum;
    if (acc.a > 1e-6) { acc = vec4f(acc.rgb / acc.a, acc.a); } else { acc = vec4f(0.0); }
  } else {
    acc = textureLoad(src, xy, 0);
  }
  if (P.adjust != 0u) { acc = vec4f(adjustRgb(acc.rgb), acc.a); }
  return acc;
}
`;

interface Slot {
  id: string;
  renderer: LayerRenderer;
  layer: CompLayer;
  /** JSON of the layer's content last pushed (flame or escape data) — unchanged content keeps accumulating */
  json: string;
  /** effects output (and the blur's intermediate), created when the layer has effects */
  fx?: GPUTexture;
  fxTmp?: GPUTexture;
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
  private fxPipe!: GPURenderPipeline;
  private fxBufs: GPUBuffer[] = [];
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
  /** index of the flame layer being edited (the panels' flame) — its renderer is `layerRenderer` */
  flameActive = 0;
  /** the edited flame layer's own renderer — single-flame paths (mutation grid, batch export of flames, dev tools) */
  get layerRenderer(): FlameRenderer {
    const s = this.slots[this.flameActive]?.layer.kind === 'flame' ? this.slots[this.flameActive] : this.slots.find((x) => x.layer.kind === 'flame');
    if (!s) throw new Error('no flame layer');
    return s.renderer as FlameRenderer;
  }
  /** Render `f` in the edited flame layer's renderer (exports/previews driving frames themselves — the document is untouched);
   *  every other layer re-renders its own content. Restore afterwards with `restore()` or by pushing the document again. */
  setFlame(f: import('../core/flame').Flame) {
    const target = this.slots[this.flameActive]?.layer.kind === 'flame' ? this.slots[this.flameActive] : this.slots.find((x) => x.layer.kind === 'flame');
    for (const s of this.slots) {
      if (s === target) { (s.renderer as FlameRenderer).setFlame(f); s.json = ''; }
      else this.pushContent(s, true);
    }
    this.needsComposite = true;
  }
  /** push a layer's content to its renderer (when changed, or forced) */
  private pushContent(s: Slot, force: boolean) {
    const json = JSON.stringify(s.layer.kind === 'flame' ? s.layer.flame : s.layer.kind === 'escape' ? s.layer.escape : s.layer.image);
    if (!force && json === s.json) return false;
    s.json = json;
    if (s.layer.kind === 'flame') (s.renderer as FlameRenderer).setFlame(s.layer.flame);
    else if (s.layer.kind === 'escape') (s.renderer as EscapeRenderer).setLayer(s.layer.escape);
    else (s.renderer as ImageRenderer).setLayer(s.layer.image);
    return true;
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
      { binding: 3, visibility: F, texture: { sampleType: 'float' } },
    ] });
    const fxModule = d.createShaderModule({ code: FX_WGSL });
    this.fxPipe = d.createRenderPipeline({ layout: 'auto', vertex: { module: fxModule, entryPoint: 'vs' }, fragment: { module: fxModule, entryPoint: 'fs', targets: [{ format: 'rgba16float' }] }, primitive: { topology: 'triangle-list' } });
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

  private async makeRenderer(kind: CompLayer['kind']): Promise<LayerRenderer> {
    if (kind === 'image') {
      const r = new ImageRenderer(this.device);
      r.onError = (m) => this.onError?.(m);
      r.onLoaded = () => { this.needsComposite = true; };
      r.resize(this.canvas.width, this.canvas.height);
      r.exporting = this._exporting;
      return r;
    }
    if (kind === 'escape') {
      const r = new EscapeRenderer(this.device);
      r.onError = (m) => this.onError?.(m);
      r.resize(this.canvas.width, this.canvas.height);
      r.exporting = this._exporting;
      return r;
    }
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
      if (slot && slot.layer.kind !== layer.kind) { slot.renderer.destroy(); slot = undefined; byId.delete(layer.id); }
      if (!slot) {
        slot = { id: layer.id, renderer: await this.makeRenderer(layer.kind), layer, json: '' };
      }
      byId.delete(layer.id);
      slot.renderer.transparentBg = !layer.ownBackground;
      const bgChanged = slot.layer.ownBackground !== layer.ownBackground;
      slot.layer = layer;
      if (!this.pushContent(slot, force) && bgChanged) slot.renderer.invalidate();
      next.push(slot);
    }
    for (const gone of byId.values()) { gone.renderer.destroy(); gone.fx?.destroy(); gone.fxTmp?.destroy(); }
    this.slots = next;
    this.needsComposite = true;
  }

  /** Re-push every layer flame (after renderRegion/exports changed the renderers' flames). */
  restore() { if (this.comp) void this.setComposition(this.comp, this.active, true); }

  private cpBuf(i: number): GPUBuffer {
    while (this.cpBufs.length <= i) this.cpBufs.push(this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    return this.cpBufs[i];
  }

  private fxBuf(i: number): GPUBuffer {
    while (this.fxBufs.length <= i) this.fxBufs.push(this.device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    return this.fxBufs[i];
  }
  /** Apply a slot's effects to its layer texture (blur H → tmp, blur V + adjust → fx); returns the texture to composite. */
  private encodeEffects(enc: GPUCommandEncoder, s: Slot, i: number): GPUTexture {
    const src = s.renderer.layerTexture!;
    const fx = s.layer.effects;
    if (!effectsActive(fx)) { s.fx?.destroy(); s.fx = undefined; s.fxTmp?.destroy(); s.fxTmp = undefined; return src; }
    const w = src.width, h = src.height;
    const mk = () => this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    if (!s.fx || s.fx.width !== w || s.fx.height !== h) { s.fx?.destroy(); s.fx = mk(); }
    const blur = Math.min(fx!.blur, 100);
    if (blur > 0 && (!s.fxTmp || s.fxTmp.width !== w || s.fxTmp.height !== h)) { s.fxTmp?.destroy(); s.fxTmp = mk(); }
    const write = (buf: GPUBuffer, dir: [number, number], radius: number, adjust: boolean) => {
      const f = new Float32Array(12); const u = new Uint32Array(f.buffer);
      f.set([dir[0], dir[1], radius, Math.max(0.5, radius / 2), fx!.brightness, fx!.contrast, fx!.saturation, fx!.hue, fx!.gamma], 0);
      u[9] = fx!.invert ? 1 : 0; u[10] = adjust ? 1 : 0;
      this.device.queue.writeBuffer(buf, 0, f);
    };
    const pass = (buf: GPUBuffer, from: GPUTexture, to: GPUTexture) => {
      const bg = this.device.createBindGroup({ layout: this.fxPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }, { binding: 1, resource: from.createView() }] });
      const p = enc.beginRenderPass({ colorAttachments: [{ view: to.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
      p.setPipeline(this.fxPipe); p.setBindGroup(0, bg); p.draw(3); p.end();
    };
    if (blur > 0) {
      write(this.fxBuf(i * 2), [1, 0], blur, false); pass(this.fxBuf(i * 2), src, s.fxTmp!);
      write(this.fxBuf(i * 2 + 1), [0, 1], blur, true); pass(this.fxBuf(i * 2 + 1), s.fxTmp!, s.fx);
    } else {
      write(this.fxBuf(i * 2), [0, 0], 0, true); pass(this.fxBuf(i * 2), src, s.fx);
    }
    return s.fx;
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
    // effects first (a layer used as a mask must be post-effects too)
    const texOf = new Map<Slot, GPUTexture>();
    this.slots.forEach((s, i) => { if (s.layer.visible && s.renderer.layerTexture) texOf.set(s, this.encodeEffects(enc, s, i)); });
    vis.forEach((s, i) => {
      const last = i === vis.length - 1;
      const u = new ArrayBuffer(32); const u32 = new Uint32Array(u); const f32 = new Float32Array(u);
      const mk = s.layer.mask;
      let maskMode = 0;
      let maskTex: GPUTexture | undefined;
      if (mk) {
        if (mk.source === 'below') maskMode = mk.channel === 'luma' ? 2 : 1;
        else { const ms = this.slots.find((x) => x.id === mk.layerId); maskTex = ms ? texOf.get(ms) ?? (ms.renderer.layerTexture ?? undefined) : undefined; if (maskTex) maskMode = mk.channel === 'luma' ? 4 : 3; }
        if (maskMode && mk.invert) maskMode |= 8;
      }
      u32[0] = Math.max(0, BLEND_MODES.indexOf(s.layer.blend)); f32[1] = s.layer.opacity; u32[2] = (i === 0 ? 1 : 0) | (s.layer.clip ? 2 : 0); u32[3] = maskMode;
      f32.set([this.bg[0], this.bg[1], this.bg[2], 1], 4);
      this.device.queue.writeBuffer(this.cpBuf(i), 0, u);
      const backdrop = this.ping[(i + 1) & 1].createView();
      const srcTex = texOf.get(s)!;
      const bg = this.device.createBindGroup({ layout: this.bgLayout, entries: [
        { binding: 0, resource: { buffer: this.cpBuf(i) } },
        { binding: 1, resource: backdrop },
        { binding: 2, resource: srcTex.createView() },
        { binding: 3, resource: (maskTex ?? srcTex).createView() },
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
  private get anyFlame(): FlameRenderer | undefined { return this.slots.find((s) => s.layer.kind === 'flame')?.renderer as FlameRenderer | undefined; }
  get nPoints() { return this.anyFlame?.nPoints ?? 1 << 16; }
  get itersPerPass() { return this.anyFlame?.itersPerPass ?? 64; }

  /** Render a tile of every visible layer offscreen and composite them on the CPU (hi-res export, thumbnails, dev tools)
   *  with the same blend/mask/effects maths as the live compositor. Straight-alpha rgba8 like FlameRenderer.renderRegion;
   *  `transparent` renders every layer without a background. */
  async renderRegion(o: CompRegionOpts): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const vis = this.slots.filter((s) => s.layer.visible);
    const n = o.tileW * o.tileH;
    const out = new Uint8ClampedArray(n * 4);
    const bg = this.bg;
    if (!o.transparent) for (let i = 0; i < n; i++) { out[i * 4] = Math.round(bg[0] * 255); out[i * 4 + 1] = Math.round(bg[1] * 255); out[i * 4 + 2] = Math.round(bg[2] * 255); out[i * 4 + 3] = 255; }
    if (!vis.length) return out;
    // one layer with its own opaque background over an opaque bg = the plain single-flame render (bit-exact with the old path)
    if (vis.length === 1 && vis[0].layer.blend === 'normal' && vis[0].layer.opacity >= 1 && !vis[0].layer.clip && !vis[0].layer.mask && !effectsActive(vis[0].layer.effects) && (vis[0].layer.ownBackground || o.transparent)) {
      return vis[0].renderer.renderRegion({ ...o, transparent: !!o.transparent || !vis[0].layer.ownBackground });
    }
    // blur needs pixels beyond the tile: render every layer with a margin, crop at the end
    const scale = o.fullW / Math.max(1, this.canvas.width);
    const maxBlur = Math.max(0, ...vis.map((s) => (effectsActive(s.layer.effects) ? Math.min(s.layer.effects!.blur, 100) * scale : 0)));
    const pad = Math.ceil(maxBlur * 2);
    const W = o.tileW + 2 * pad, H = o.tileH + 2 * pad, N = W * H;
    const region = { ...o, tileX: o.tileX - pad, tileY: o.tileY - pad, tileW: W, tileH: H };
    const acc = new Float32Array(N * 4);
    if (!o.transparent) for (let i = 0; i < N; i++) { acc[i * 4] = bg[0]; acc[i * 4 + 1] = bg[1]; acc[i * 4 + 2] = bg[2]; acc[i * 4 + 3] = 1; }
    // every visible layer's tile (post-effects) — masks may refer to any of them
    const tiles = new Map<string, Float32Array>();
    for (const s of vis) {
      const px = await s.renderer.renderRegion({ ...region, transparent: !!o.transparent || !s.layer.ownBackground });
      const t = new Float32Array(N * 4);
      for (let i = 0; i < N * 4; i++) t[i] = px[i] / 255;
      tiles.set(s.id, effectsActive(s.layer.effects) ? applyEffectsCPU(t, W, H, s.layer.effects!, scale) : t);
    }
    for (const s of vis) {
      const t = tiles.get(s.id)!;
      const mode: BlendMode = s.layer.blend;
      const mk = s.layer.mask;
      const maskTile = mk?.source === 'layer' && mk.layerId ? tiles.get(mk.layerId) : undefined;
      for (let i = 0; i < N; i++) {
        const b: [number, number, number, number] = [acc[i * 4], acc[i * 4 + 1], acc[i * 4 + 2], acc[i * 4 + 3]];
        let a = t[i * 4 + 3] * (s.layer.clip ? b[3] : 1);
        if (mk) {
          let m = 1;
          if (mk.source === 'below') m = mk.channel === 'luma' ? (0.2126 * b[0] + 0.7152 * b[1] + 0.0722 * b[2]) * b[3] : b[3];
          else if (maskTile) m = mk.channel === 'luma' ? (0.2126 * maskTile[i * 4] + 0.7152 * maskTile[i * 4 + 1] + 0.0722 * maskTile[i * 4 + 2]) * maskTile[i * 4 + 3] : maskTile[i * 4 + 3];
          if (mk.invert) m = 1 - m;
          a *= Math.min(1, Math.max(0, m));
        }
        const r = blendPixel(mode, b, [t[i * 4], t[i * 4 + 1], t[i * 4 + 2], a], s.layer.opacity);
        acc[i * 4] = r[0]; acc[i * 4 + 1] = r[1]; acc[i * 4 + 2] = r[2]; acc[i * 4 + 3] = r[3];
      }
    }
    for (let y = 0; y < o.tileH; y++) for (let x = 0; x < o.tileW; x++) {
      const si = ((y + pad) * W + (x + pad)) * 4, di = (y * o.tileW + x) * 4;
      for (let k = 0; k < 4; k++) out[di + k] = Math.round(Math.min(1, Math.max(0, acc[si + k])) * 255);
    }
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

/** CPU twin of the effects pass: premultiplied separable gaussian blur (radius in output pixels) + colour adjustments. */
export function applyEffectsCPU(t: Float32Array, W: number, H: number, fx: LayerEffects, scale: number): Float32Array {
  let cur = t;
  const radius = Math.min(fx.blur, 100) * scale;
  if (radius > 0) {
    const r = Math.ceil(radius), sigma = Math.max(0.5, radius / 2);
    const wts = new Float32Array(2 * r + 1);
    let wsum = 0;
    for (let k = -r; k <= r; k++) { wts[k + r] = Math.exp(-(k * k) / (2 * sigma * sigma)); wsum += wts[k + r]; }
    for (let k = 0; k < wts.length; k++) wts[k] /= wsum;
    // premultiply
    const pm = new Float32Array(cur.length);
    for (let i = 0; i < W * H; i++) { const a = cur[i * 4 + 3]; pm[i * 4] = cur[i * 4] * a; pm[i * 4 + 1] = cur[i * 4 + 1] * a; pm[i * 4 + 2] = cur[i * 4 + 2] * a; pm[i * 4 + 3] = a; }
    const tmp = new Float32Array(cur.length);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      for (let k = -r; k <= r; k++) { const xx = Math.min(W - 1, Math.max(0, x + k)); const w = wts[k + r]; const j = (y * W + xx) * 4; a0 += pm[j] * w; a1 += pm[j + 1] * w; a2 += pm[j + 2] * w; a3 += pm[j + 3] * w; }
      const j = (y * W + x) * 4; tmp[j] = a0; tmp[j + 1] = a1; tmp[j + 2] = a2; tmp[j + 3] = a3;
    }
    const outp = new Float32Array(cur.length);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      for (let k = -r; k <= r; k++) { const yy = Math.min(H - 1, Math.max(0, y + k)); const w = wts[k + r]; const j = (yy * W + x) * 4; a0 += tmp[j] * w; a1 += tmp[j + 1] * w; a2 += tmp[j + 2] * w; a3 += tmp[j + 3] * w; }
      const j = (y * W + x) * 4;
      if (a3 > 1e-6) { outp[j] = a0 / a3; outp[j + 1] = a1 / a3; outp[j + 2] = a2 / a3; outp[j + 3] = a3; }
    }
    cur = outp;
  }
  if (fx.brightness || fx.contrast || fx.saturation || fx.hue || fx.gamma !== 1 || fx.invert) {
    const o = cur === t ? new Float32Array(cur) : cur;
    for (let i = 0; i < W * H; i++) { const c = adjustPixel(fx, o[i * 4], o[i * 4 + 1], o[i * 4 + 2]); o[i * 4] = c[0]; o[i * 4 + 1] = c[1]; o[i * 4 + 2] = c[2]; }
    cur = o;
  }
  return cur;
}
