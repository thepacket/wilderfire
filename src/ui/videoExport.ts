// Video export: renders an animation timeline frame by frame and encodes it with WebCodecs
// (VideoEncoder) into WebM (VP9/VP8) or MP4 (H.264) via webm-muxer / mp4-muxer — fully in
// the browser. Frames come either from the live canvas (its size, cheapest) or from an
// offscreen tiled render at a fixed size (720p … 4K). Shared by the Anim tab's Export video
// button and the batch export queue.
import { Muxer as WebMMuxer, ArrayBufferTarget as WebMTarget } from 'webm-muxer';
import { Muxer as Mp4Muxer, ArrayBufferTarget as Mp4Target } from 'mp4-muxer';
import type { Flame } from '../core/flame';
import { renderMotionBlurred } from './motionBlur';
import type { FlameRenderer } from '../gpu/renderer';

/** What the Anim panel exposes for rendering: the timeline's span and the flame at a time. */
export interface Timeline {
  /** timeline start (s) */
  t0: number;
  /** timeline length (s) */
  total: number;
  evalAt(t: number): Flame;
}

export type VideoFormat = 'webm' | 'mp4';

/** Fixed video sizes (16:9); '' = the live canvas size. */
export const VIDEO_SIZE_OPTIONS: { label: string; value: string }[] = [
  { label: 'Canvas', value: '' }, { label: '720p', value: '1280x720' }, { label: '1080p', value: '1920x1080' },
  { label: '1440p', value: '2560x1440' }, { label: '4K', value: '3840x2160' },
];
export const VIDEO_QUALITY_OPTIONS: { label: string; value: string }[] = [
  { label: 'Draft', value: '24' }, { label: 'Good', value: '72' }, { label: 'High', value: '160' },
];

export interface VideoOpts {
  fps: number;
  /** accumulation passes per frame at canvas size (the fixed-size path renders the equivalent samples per pixel) */
  passes: number;
  format: VideoFormat;
  /** fixed output size; omitted = the live canvas */
  size?: { w: number; h: number };
  onFrame?: (i: number, n: number) => void;
  onStatus?: (s: string) => void;
  /** stop after the current frame (throws an AbortError) */
  signal?: AbortSignal;
}

export const videoFileExt = (f: VideoFormat) => (f === 'mp4' ? '.mp4' : '.webm');
export const videoMime = (f: VideoFormat) => (f === 'mp4' ? 'video/mp4' : 'video/webm');

/**
 * Render + encode. The caller owns the renderer state: set `renderer.exporting = true`
 * before and restore the flame (`renderer.setFlame(app.flame)`) after.
 */
export async function renderVideo(renderer: FlameRenderer, tl: Timeline, o: VideoOpts): Promise<Blob> {
  if (!('VideoEncoder' in window)) throw new Error('WebCodecs (VideoEncoder) is not available in this browser.');
  const { fps, passes, format } = o;
  const nFrames = Math.max(2, Math.round(Math.max(tl.total, 0.01) * fps) + 1);
  const canvasW = renderer.width & ~1, canvasH = renderer.height & ~1;
  const width = o.size ? o.size.w & ~1 : canvasW;
  const height = o.size ? o.size.h & ~1 : canvasH;

  let addChunk: (chunk: EncodedVideoChunk, meta?: EncodedVideoChunkMetadata) => void;
  let finalize: () => ArrayBuffer;
  const encCfg: VideoEncoderConfig = { codec: '', width, height, bitrate: Math.round(12_000_000 * Math.max(1, (width * height) / (1920 * 1080))), framerate: fps };
  if (format === 'mp4') {
    let codec = '';
    for (const c of ['avc1.640028', 'avc1.4D0028', 'avc1.420028', 'avc1.640033']) {
      const s = await VideoEncoder.isConfigSupported({ ...encCfg, codec: c });
      if (s.supported) { codec = c; break; }
    }
    if (!codec) throw new Error(`H.264 encoding at ${width}×${height} not supported here — use WebM or a smaller size.`);
    encCfg.codec = codec;
    (encCfg as VideoEncoderConfig & { avc?: { format: string } }).avc = { format: 'avc' };
    const muxer = new Mp4Muxer({ target: new Mp4Target(), video: { codec: 'avc', width, height, frameRate: fps }, fastStart: 'in-memory' });
    addChunk = (c, m) => muxer.addVideoChunk(c, m);
    finalize = () => { muxer.finalize(); return muxer.target.buffer; };
  } else {
    let codec = 'vp09.00.10.08', muxCodec = 'V_VP9';
    const support = await VideoEncoder.isConfigSupported({ ...encCfg, codec });
    if (!support.supported) { codec = 'vp8'; muxCodec = 'V_VP8'; }
    encCfg.codec = codec;
    const muxer = new WebMMuxer({ target: new WebMTarget(), video: { codec: muxCodec, width, height, frameRate: fps } });
    addChunk = (c, m) => muxer.addVideoChunk(c, m!);
    finalize = () => { muxer.finalize(); return muxer.target.buffer; };
  }

  const encoder = new VideoEncoder({ output: (chunk, meta) => addChunk(chunk, meta), error: (e) => console.error('VideoEncoder:', e) });
  encoder.configure(encCfg);

  // fixed size: same samples per pixel as `passes` would give the canvas
  const spp = Math.max(4, Math.round((passes * renderer.nPoints * renderer.itersPerPass) / Math.max(canvasW * canvasH, 1)));
  const off = o.size || tl.evalAt(tl.t0).motionBlur ? document.createElement('canvas') : null; // motion blur renders offscreen (sub-frames averaged)
  if (off) { off.width = width; off.height = height; }
  const offCtx = off?.getContext('2d') ?? null;
  const TILE = 1024, PAD = 8;

  try {
    for (let i = 0; i < nFrames; i++) {
      if (o.signal?.aborted) throw new DOMException('Export cancelled', 'AbortError');
      const t = tl.t0 + Math.min(i / fps, tl.total);
      const flameAt = tl.evalAt(t);
      renderer.setFlame(flameAt);
      const stamp = { timestamp: Math.round((i * 1e6) / fps), duration: Math.round(1e6 / fps) };
      let frame: VideoFrame;
      if (flameAt.motionBlur && off && offCtx && width <= TILE && height <= TILE) {
        // motion blur: the frame is the weighted average of its sub-frames (offscreen, one tile)
        const px = await renderMotionBlurred(renderer, flameAt, { evalAt: tl.evalAt, t, fps }, width, height, spp);
        offCtx.putImageData(new ImageData(px, width, height), 0, 0);
        frame = new VideoFrame(off, stamp);
      } else if (!off || !offCtx) {
        await renderer.stepExport(passes);
        // Capture must stay in the same task as the tonemap submit.
        frame = renderer.captureSync((cv) => new VideoFrame(cv, stamp));
      } else {
        for (let ty = 0; ty < Math.ceil(height / TILE); ty++) {
          for (let tx = 0; tx < Math.ceil(width / TILE); tx++) {
            const x0 = tx * TILE, y0 = ty * TILE;
            const tw = Math.min(TILE, width - x0), th = Math.min(TILE, height - y0);
            const pw = tw + 2 * PAD, ph = th + 2 * PAD;
            const px = await renderer.renderRegion({ fullW: width, fullH: height, tileX: x0 - PAD, tileY: y0 - PAD, tileW: pw, tileH: ph, spp });
            const img = new ImageData(tw, th);
            for (let y = 0; y < th; y++) { const so = ((y + PAD) * pw + PAD) * 4; img.data.set(px.subarray(so, so + tw * 4), y * tw * 4); }
            offCtx.putImageData(img, x0, y0);
          }
        }
        frame = new VideoFrame(off, stamp);
      }
      encoder.encode(frame, { keyFrame: i % (fps * 2) === 0 });
      frame.close();
      while (encoder.encodeQueueSize > 4) await new Promise((r) => setTimeout(r, 4));
      o.onFrame?.(i + 1, nFrames);
    }
    o.onStatus?.('Encoding…');
    await encoder.flush();
    return new Blob([finalize()], { type: videoMime(format) });
  } finally {
    try { encoder.close(); } catch { /* already closed */ }
  }
}
