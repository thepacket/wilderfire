// Dev harness: renders a set of flames offscreen at a fixed size/quality and saves
// them (PNG + the exact .flame XML) into compare-out/ through the dev-server sink,
// so scripts/jwf-port/Compare.java can render the same XML with headless JWildfire
// and compare the images numerically (see the header of Compare.java).
//
//   await window.wilderfire.flameCompare()                       // fixtures + samples + presets
//   await window.wilderfire.flameCompare({ only: ['Gnarl_0'] })
//   await window.wilderfire.flameCompare({ width: 512, quality: 100 })

import { flameToXML, importFlameText } from '../core/flameXML';
import { PRESETS } from '../core/presets';
import { JWF_SAMPLES } from '../core/samples';
import { FIXTURES } from './flameTest';
import type { App } from '../ui/common';

export interface CompareOpts { only?: string[]; width?: number; quality?: number; sets?: ('fixtures' | 'samples' | 'presets')[]; files?: string[]; prefix?: string }
export interface CompareItem { id: string; set: string; xml: string }

async function collect(app: App, sets: string[], files: string[] = []): Promise<CompareItem[]> {
  const items: CompareItem[] = [];
  if (sets.includes('fixtures') || files.length) {
    for (const f of sets.includes('fixtures') ? [...FIXTURES, ...files] : files) {
      const r = await fetch(`/scripts/jwf-port/testflames/${f}.flame`);
      if (r.ok) items.push({ id: f, set: 'fixtures', xml: await r.text() });
    }
  }
  if (sets.includes('samples')) {
    for (const s of JWF_SAMPLES) {
      const r = await fetch(`/flames/${s.file}`);
      if (r.ok) items.push({ id: s.file.replace(/\.flame$/, ''), set: 'samples', xml: await r.text() });
    }
  }
  if (sets.includes('presets')) {
    for (const p of PRESETS) {
      // presets are built in-app; export them the same way "Save .flame" does so JWildfire sees the same XML
      const xml = flameToXML(p.make());
      items.push({ id: 'preset_' + p.name.replace(/\W+/g, '_'), set: 'presets', xml });
    }
  }
  return items;
}

async function save(name: string, body: Blob | string): Promise<void> {
  const r = await fetch(`/__jwf/save?name=${encodeURIComponent(name)}`, { method: 'POST', body });
  if (!r.ok) throw new Error(`save ${name}: ${r.status} ${await r.text()}`);
}

export async function runFlameCompare(app: App, opts: CompareOpts = {}): Promise<{ id: string; ok: boolean; msg?: string; ms: number }[]> {
  const W = opts.width ?? 512, quality = opts.quality ?? 100;
  const items = (await collect(app, opts.sets ?? ['fixtures', 'samples', 'presets'], opts.files)).filter((it) => !opts.only || opts.only.includes(it.id));
  const out: { id: string; ok: boolean; msg?: string; ms: number }[] = [];
  const saved = app.flame;
  const manifest: { id: string; set: string; w: number; h: number; quality: number }[] = [];
  for (const it of items) {
    const t0 = performance.now();
    try {
      // the flame's own aspect ratio, so pixels-per-unit maps identically in both engines
      const sm = /size="(\d+)\s+(\d+)"/.exec(it.xml);
      const fw = sm ? Number(sm[1]) : 800, fh = sm ? Number(sm[2]) : 600;
      const H = Math.max(16, Math.round(W * fh / fw));
      const { flame } = importFlameText(it.xml, app.activeLayer.palette);
      app.setFlame(flame);
      app.renderer.setFlame(app.flame);
      const px = await app.renderer.renderRegion({ fullW: W, fullH: H, tileX: 0, tileY: 0, tileW: W, tileH: H, spp: quality });
      const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
      cv.getContext('2d')!.putImageData(new ImageData(px, W, H), 0, 0);
      const blob = await new Promise<Blob | null>((res) => cv.toBlob(res, 'image/png'));
      if (!blob) throw new Error('toBlob failed');
      const id = (opts.prefix ?? '') + it.id; // e.g. prefix 'full_' keeps a full-size run apart from the 512 px one
      await save(`${id}.wf.png`, blob);
      await save(`${id}.flame`, it.xml);
      manifest.push({ id, set: it.set, w: W, h: H, quality });
      out.push({ id: it.id, ok: true, ms: performance.now() - t0 });
    } catch (err) {
      out.push({ id: it.id, ok: false, msg: String((err as Error).message ?? err), ms: performance.now() - t0 });
    }
  }
  // partial runs (only/files) keep the full manifest intact
  await save(opts.prefix ? `manifest.${opts.prefix}json` : opts.only || opts.files ? 'manifest.part.json' : 'manifest.json', JSON.stringify(manifest, null, 1));
  app.setFlame(saved);
  app.renderer.setFlame(app.flame);
  console.log('flameCompare:', out.filter((o) => o.ok).length, 'ok,', out.filter((o) => !o.ok).map((o) => `${o.id}: ${o.msg}`).join('; '));
  return out;
}
