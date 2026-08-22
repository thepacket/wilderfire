// Bundle the gradient packs JWildfire ships (resources/org/jwildfire/create/tina/io/scripts/*.ugr — the classic
// Apophysis / UltraFractal packs) into one compact binary for the Gradient tab's library:
//   u32 LE header length, JSON header { packs: [...], names: [...], pack: [packIndex per gradient] }, then 768 bytes
//   (256 × RGB) per gradient in the same order. Identical packs (carr = full2) and identical gradients are dropped.
//   node scripts/jwf-port/ugr2bin.mjs <jwildfire>/resources/org/jwildfire/create/tina/io/scripts public/gradients/jwildfire.bin
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, basename } from 'node:path';

const [dir, out] = process.argv.slice(2);
const ORDER = ['carr', 'floral', 'universe', 'skygradients', 'stargradients'];
const files = readdirSync(dir).filter((f) => f.endsWith('.ugr') && f !== 'full2.ugr' /* byte-identical copy of carr.ugr */).sort((a, b) => ORDER.indexOf(basename(a, '.ugr')) - ORDER.indexOf(basename(b, '.ugr')));
const seenFile = new Set();
const packs = [], names = [], packIdx = [], blocks = [];
const seenGrad = new Map();
for (const f of files) {
  const text = readFileSync(join(dir, f), 'latin1');
  const fileKey = text.length + ':' + text.slice(0, 4000);
  if (seenFile.has(fileKey)) { console.log(`${f}: identical to an earlier pack, skipped`); continue; }
  seenFile.add(fileKey);
  const pack = basename(f, '.ugr').replace('gradients', '');
  packs.push(pack);
  const re = /([^\n{]*?)\s*\{\s*gradient:([\s\S]*?)\}/gi;
  let g, n = 0, dup = 0;
  while ((g = re.exec(text))) {
    const seg = g[2];
    const title = (/title="([^"]*)"/.exec(seg)?.[1] ?? g[1]).trim();
    const stops = [];
    const sr = /index=(\d+)\s+color=(\d+)/g;
    let m;
    while ((m = sr.exec(seg))) { const c = parseInt(m[2]); stops.push([Math.min(parseInt(m[1]) / 399, 1), c & 255, (c >> 8) & 255, (c >> 16) & 255]); }
    if (stops.length < 2) continue;
    stops.sort((a, b) => a[0] - b[0]);
    // expand to 256 entries (linear between stops, like the app's expandStops)
    const rgb = new Uint8Array(768);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let k = 0; while (k < stops.length - 1 && stops[k + 1][0] < t) k++;
      const a = stops[k], b = stops[Math.min(k + 1, stops.length - 1)];
      const span = b[0] - a[0]; const u = span > 1e-9 ? Math.min(1, Math.max(0, (t - a[0]) / span)) : 0;
      for (let c = 0; c < 3; c++) rgb[i * 3 + c] = Math.round(a[1 + c] + (b[1 + c] - a[1 + c]) * u);
    }
    const key = Buffer.from(rgb).toString('base64');
    if (seenGrad.has(key)) { dup++; continue; }
    seenGrad.set(key, names.length);
    names.push(!title || /^\d+$/.test(title) ? `${pack} ${title || n + 1}` : title); packIdx.push(packs.length - 1); blocks.push(rgb); n++; // bare numbers get their pack's name
  }
  console.log(`${f}: ${n} gradients${dup ? ` (+${dup} duplicates dropped)` : ''}`);
}
const header = Buffer.from(JSON.stringify({ packs, names, pack: packIdx }), 'utf8');
const len = Buffer.alloc(4); len.writeUInt32LE(header.length);
writeFileSync(out, Buffer.concat([len, header, ...blocks.map((b) => Buffer.from(b))]));
console.log(`${names.length} gradients, ${packs.length} packs → ${out} (${(4 + header.length + blocks.length * 768) / 1e6} MB)`);
