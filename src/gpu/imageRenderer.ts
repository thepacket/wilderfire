// Image layer renderer: draws a picture from the image store into the layer texture (fit/scale/offset/rotation,
// optional tiling), and renders export tiles at any resolution the same way. Straight alpha, bilinear sampling.

import type { ImageLayerData } from '../core/composition';
import { imageGet } from '../core/libraryStore';
import type { RenderStats } from './renderer';

const IMG_WGSL = `
struct IP { size: vec2f, tile: vec2f, full: vec2f, img: vec2f, scale: vec2f, offset: vec2f, cs: f32, sn: f32, tiled: u32, pad: u32 }
@group(0) @binding(0) var<uniform> P: IP;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
struct VOut { @builtin(position) pos: vec4f }
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VOut; o.pos = vec4f(p[vi], 0.0, 1.0); return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4f {
  // pixel in the full image, relative to its centre, un-rotated and un-offset → image space
  let px = in.pos.xy - 0.5 + P.tile;
  let d = px - P.full * 0.5 - P.offset * P.full;
  let r = vec2f(P.cs * d.x + P.sn * d.y, -P.sn * d.x + P.cs * d.y);   // inverse rotation
  let uv = r / (P.scale * P.img) + 0.5;                                // scale = displayed pixels per image pixel (per axis)
  if (P.tiled == 0u && (uv.x < 0.0 || uv.y < 0.0 || uv.x >= 1.0 || uv.y >= 1.0)) { return vec4f(0.0); }
  let c = textureSampleLevel(tex, samp, select(uv, fract(uv), P.tiled != 0u), 0.0);
  // straight alpha out (the sampled texture is straight too)
  return c;
}
`;

const cache = new Map<string, Promise<ImageBitmap | null>>();
async function loadBitmap(key: string): Promise<ImageBitmap | null> {
  let p = cache.get(key);
  if (!p) {
    p = imageGet(key).then((blob) => (blob ? createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }) : null)).catch((e) => { console.warn('image layer:', e); return null; });
    cache.set(key, p);
  }
  return p;
}
/** forget a cached bitmap (after re-storing under the same key) */
export function forgetImage(key: string) { cache.delete(key); }

export class ImageRenderer {
  private device: GPUDevice;
  private tex: GPUTexture | null = null;   // layer texture
  private view: GPUTextureView | null = null;
  private w = 2; private h = 2;
  private img: GPUTexture | null = null;   // the picture
  private imgKey = '';
  private imgW = 1; private imgH = 1;
  private pipe: GPURenderPipeline;
  private pipeExport: GPURenderPipeline;
  private uBuf: GPUBuffer;
  private uOff: GPUBuffer;
  private sampler: GPUSampler;
  private data: ImageLayerData | null = null;
  private dirty = true;
  private loading: Promise<void> | null = null;
  private offTex: GPUTexture | null = null;
  private offW = 0; private offH = 0;

  driven = true;
  transparentBg = false;
  presented = false;
  exporting = false;
  onError: ((msg: string) => void) | null = null;
  /** called when the picture finished loading (the composer re-blends) */
  onLoaded: (() => void) | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ code: IMG_WGSL });
    const mk = (format: GPUTextureFormat) => device.createRenderPipeline({ layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    this.pipe = mk('rgba16float');
    this.pipeExport = mk('rgba8unorm');
    this.uBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uOff = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });
  }

  get layerTexture(): GPUTexture | null { return this.tex; }
  get width() { return this.w; }
  get height() { return this.h; }
  get loaded() { return !!this.img; }

  resize(w: number, h: number) {
    w = Math.max(2, w); h = Math.max(2, h);
    if (this.tex && w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.tex?.destroy();
    this.tex = this.device.createTexture({ size: [w, h], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.view = this.tex.createView();
    this.dirty = true;
  }

  setLayer(d: ImageLayerData) {
    this.data = d;
    this.dirty = true;
    if (d.key !== this.imgKey) {
      this.imgKey = d.key;
      this.img?.destroy(); this.img = null;
      const key = d.key;
      this.loading = loadBitmap(key).then((bmp) => {
        if (!bmp || this.imgKey !== key) return;
        this.imgW = bmp.width; this.imgH = bmp.height;
        this.img = this.device.createTexture({ size: [bmp.width, bmp.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
        this.device.queue.copyExternalImageToTexture({ source: bmp }, { texture: this.img }, [bmp.width, bmp.height]);
        this.dirty = true;
        this.onLoaded?.();
      });
    }
  }
  setFlame(_f: unknown) {}

  private writeUniforms(buf: GPUBuffer, tile: { x: number; y: number; w: number; h: number; fullW: number; fullH: number }) {
    const d = this.data!;
    const f = new Float32Array(16); const u = new Uint32Array(f.buffer);
    // displayed pixels per image pixel, per axis
    const iw = this.imgW || d.w, ih = this.imgH || d.h;
    let sx = 1, sy = 1;
    if (d.fit === 'contain') { const s = Math.min(tile.fullW / iw, tile.fullH / ih); sx = sy = s; }
    else if (d.fit === 'cover') { const s = Math.max(tile.fullW / iw, tile.fullH / ih); sx = sy = s; }
    else if (d.fit === 'stretch') { sx = tile.fullW / iw; sy = tile.fullH / ih; }
    else { const s = tile.fullW / this.w; sx = sy = s; } // 'none': image pixels = screen pixels (scaled with the export resolution)
    sx *= d.scale; sy *= d.scale;
    f.set([tile.w, tile.h, tile.x, tile.y, tile.fullW, tile.fullH, iw, ih, sx, sy, d.offsetX, -d.offsetY], 0);
    f[12] = Math.cos(d.rotation); f[13] = Math.sin(d.rotation); u[14] = d.tile ? 1 : 0;
    this.device.queue.writeBuffer(buf, 0, f);
  }

  private bindGroup(pipe: GPURenderPipeline, u: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: u } }, { binding: 1, resource: this.img!.createView() }, { binding: 2, resource: this.sampler }] });
  }

  private draw(enc: GPUCommandEncoder) {
    if (!this.view || !this.data) return;
    if (!this.img) { // nothing loaded yet: transparent layer
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
      pass.end();
      return;
    }
    this.writeUniforms(this.uBuf, { x: 0, y: 0, w: this.w, h: this.h, fullW: this.w, fullH: this.h });
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.bindGroup(this.pipe, this.uBuf));
    pass.draw(3);
    pass.end();
  }

  tick(_t: number): RenderStats | null {
    if (this.exporting || !this.data) return null;
    if (this.dirty) {
      const enc = this.device.createCommandEncoder();
      this.draw(enc);
      this.device.queue.submit([enc.finish()]);
      this.dirty = false;
      this.presented = true;
    }
    return { spp: 0, samplesPerSec: 0, paused: false, converged: !!this.img, budgetScale: 1, gpuMs: 0 };
  }
  presentNow() { if (this.dirty && this.data) { const enc = this.device.createCommandEncoder(); this.draw(enc); this.device.queue.submit([enc.finish()]); this.dirty = false; this.presented = true; } }
  setPaused(_p: boolean) {}
  resetAccumulation() { this.dirty = true; }
  invalidate() { this.dirty = true; }
  async ready(): Promise<void> { await this.loading; }
  async stepExport(_passes: number): Promise<void> { await this.loading; this.presentNow(); }
  setOversample(os: number): number { return os; }

  async renderRegion(o: { fullW: number; fullH: number; tileX: number; tileY: number; tileW: number; tileH: number; spp: number; transparent?: boolean }): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const d = this.device;
    if (!this.data) throw new Error('No image layer.');
    await this.loading;
    const out = new Uint8ClampedArray(o.tileW * o.tileH * 4);
    if (!this.img) return out;
    if (!this.offTex || this.offW !== o.tileW || this.offH !== o.tileH) {
      this.offTex?.destroy();
      this.offTex = d.createTexture({ size: [o.tileW, o.tileH], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      this.offW = o.tileW; this.offH = o.tileH;
    }
    this.writeUniforms(this.uOff, { x: o.tileX, y: o.tileY, w: o.tileW, h: o.tileH, fullW: o.fullW, fullH: o.fullH });
    const bpr = Math.ceil((o.tileW * 4) / 256) * 256;
    const rb = d.createBuffer({ size: bpr * o.tileH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.offTex.createView(), loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }] });
    pass.setPipeline(this.pipeExport);
    pass.setBindGroup(0, this.bindGroup(this.pipeExport, this.uOff));
    pass.draw(3);
    pass.end();
    enc.copyTextureToBuffer({ texture: this.offTex }, { buffer: rb, bytesPerRow: bpr, rowsPerImage: o.tileH }, { width: o.tileW, height: o.tileH });
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(rb.getMappedRange());
    for (let y = 0; y < o.tileH; y++) out.set(src.subarray(y * bpr, y * bpr + o.tileW * 4), y * o.tileW * 4);
    rb.unmap(); rb.destroy();
    return out;
  }

  destroy() { this.tex?.destroy(); this.tex = null; this.img?.destroy(); this.img = null; this.offTex?.destroy(); this.offTex = null; this.data = null; }
}
