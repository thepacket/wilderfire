#!/usr/bin/env node
// Survey .flame files for JWildfire features WilderFire's importer never reads.
//
//   node scripts/jwf-port/attr-survey.mjs <dir|file> [more dirs…]
//
// The importer reports unknown *variations* but silently ignores unknown attributes,
// so a flame can import "cleanly" and still render differently. Part 1 counts flames
// whose settings actually switch an unported feature ON (the number that matters when
// deciding what to port next); part 2 dumps every unread attribute for the record.
import { readFileSync, readdirSync, statSync } from 'fs';
import { join } from 'path';

// Flame-level attributes flameXML.ts reads (sld_render_*, *Curve* and the xform level are handled elsewhere).
const KNOWN_FLAME = new Set(`name size center scale cam_zoom rotate cam_pitch cam_yaw cam_roll cam_persp cam_perspective
cam_pos_x cam_pos_y cam_pos_z preserve_z cam_dof cam_dof_area cam_dof_exponent cam_dof_scale cam_dof_fade new_dof
cam_xfocus cam_yfocus cam_zfocus cam_zpos cam_zdimish cam_zdimdist cam_zdimcolor brightness gamma gamma_threshold
vibrancy contrast white_level low_density_brightness filter filter_kernel antialias_amount antialias_radius de_radius
de_curve background background_type background_ul background_ur background_ll background_lr background_cc
fps frame frame_count version`.split(/\s+/));

// Editor state and render-job settings: in the file, but not part of the picture.
const NOISE = /^(grad_edit_|mixer_[a-z]{2}_curve_|meta_info_|resolution_profile|quality_profile|quality$|temporal_samples|swarm_size|orientation|zbuffer_|de_comparison_line|filter_comparison_line|filter_indicator|smooth_gradient)/;

// Unported features, and what in the file switches each one on. `note` records what is
// still unverified against the JWildfire source (rebuild scratchpad/jwf to settle those).
const num = (a, k, d) => { const v = parseFloat(a[k]); return Number.isFinite(v) ? v : d; };
const FEATURES = [
  { id: 'post_symmetry', on: (a) => a.post_symmetry_type && a.post_symmetry_type !== 'NONE', what: 'mirrored/rotated copies — changes the geometry' },
  { id: 'filter_type', on: (a) => !!a.filter_type, what: 'JWildfire 7 sharpening/smoothing/adaptive filter model', group: (a) => a.filter_type },
  { id: 'ai_denoiser', on: (a) => a.ai_post_denoiser && a.ai_post_denoiser !== 'NONE', what: 'OptiX/OIDN denoise of the final image', group: (a) => a.ai_post_denoiser },
  { id: 'oversample', on: (a) => num(a, 'oversample', 1) > 1, what: 'supersampled histogram (edge sharpness)' },
  { id: 'color_oversample', on: (a) => num(a, 'color_oversample', 1) > 1, what: 'colour oversampling' },
  { id: 'post_bokeh', on: (a) => num(a, 'post_bokeh_intensity', 0) > 0, what: 'highlight bloom', note: 'default 0.005 — gate unverified' },
  { id: 'post_noise_filter', on: (a) => num(a, 'post_noise_filter', 0) > 0, what: 'post denoise' },
  { id: 'post_blur', on: (a) => num(a, 'post_blur_radius', 0) > 0, what: 'post blur' },
  { id: 'saturation', on: (a) => num(a, 'saturation', 1) !== 1, what: 'flame-level saturation' },
  { id: 'hue/lightness', on: (a) => num(a, 'hue', 1) !== 1 || num(a, 'lightness', 1) !== 1, what: 'flame-level hue / lightness' },
  { id: 'balancing_rgb', on: (a) => ['red', 'green', 'blue'].some((c) => num(a, `balancing_${c}`, 1) !== 1), what: 'per-channel colour balance' },
  { id: 'mixer', on: (a) => a.mixer_mode && a.mixer_mode !== 'OFF', what: 'RGB curve mixer', group: (a) => a.mixer_mode },
  { id: 'fg_opacity', on: (a) => num(a, 'fg_opacity', 1) !== 1, what: 'foreground opacity' },
  { id: 'bg_transparency', on: (a) => num(a, 'bg_transparency', 0) !== 0, what: 'transparent background' },
  { id: 'dof_shape', on: (a) => a.cam_dof_shape && a.cam_dof_shape !== 'BUBBLE', what: 'non-bubble DOF bokeh shape', group: (a) => a.cam_dof_shape },
  { id: 'dof_rotate', on: (a) => num(a, 'cam_dof_rotate', 0) !== 0, what: 'DOF shape rotation' },
  { id: 'background_image', on: (a) => !!a.background_image, what: 'image background (needs the file — usually unreachable)' },
];

const files = [];
const walk = (p) => {
  const st = statSync(p);
  if (st.isDirectory()) for (const e of readdirSync(p)) walk(join(p, e));
  else if (/\.flames?$/i.test(p)) files.push(p);
};
const roots = process.argv.slice(2);
if (!roots.length) { console.error('usage: attr-survey.mjs <dir|file> [more…]'); process.exit(1); }
for (const r of roots) walk(r);

const hits = new Map(FEATURES.map((f) => [f.id, { n: 0, groups: new Map() }]));
const attrs = new Map(); // attr → Map(value → count)
let flames = 0, unreadable = 0;
for (const f of files) {
  let text;
  try { text = readFileSync(f, 'utf8'); } catch { unreadable++; continue; }
  for (const m of text.matchAll(/<(?:jwf-)?flame\s([^>]*)>/g)) {
    flames++;
    const a = Object.fromEntries([...m[1].matchAll(/([a-zA-Z_0-9]+)="([^"]*)"/g)].map(([, k, v]) => [k, v]));
    for (const feat of FEATURES) {
      if (!feat.on(a)) continue;
      const h = hits.get(feat.id);
      h.n++;
      const g = feat.group?.(a);
      if (g) h.groups.set(g, (h.groups.get(g) ?? 0) + 1);
    }
    for (const [k, v] of Object.entries(a)) {
      if (KNOWN_FLAME.has(k) || k.startsWith('sld_render_') || k.includes('Curve') || NOISE.test(k)) continue;
      if (!attrs.has(k)) attrs.set(k, new Map());
      attrs.get(k).set(v, (attrs.get(k).get(v) ?? 0) + 1);
    }
  }
}

const pct = (n) => `${((100 * n) / Math.max(flames, 1)).toFixed(1)}%`;
console.log(`${files.length} files · ${flames} flames${unreadable ? ` · ${unreadable} unreadable` : ''}\n`);
console.log('Unported features actually switched on:');
for (const f of FEATURES.slice().sort((x, y) => hits.get(y.id).n - hits.get(x.id).n)) {
  const h = hits.get(f.id);
  if (!h.n) continue;
  const groups = h.groups.size ? '  [' + [...h.groups].sort((a, b) => b[1] - a[1]).map(([g, c]) => `${g} ${c}`).join(', ') + ']' : '';
  console.log(`  ${String(h.n).padStart(6)}  ${pct(h.n).padStart(6)}  ${f.id.padEnd(18)} ${f.what}${groups}${f.note ? `  (${f.note})` : ''}`);
}

console.log('\nEvery unread attribute (editor state and render-job settings filtered out):');
const rows = [...attrs].map(([k, vals]) => ({
  k,
  total: [...vals.values()].reduce((a, b) => a + b, 0),
  top: [...vals].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([v, c]) => `${v || '""'}×${c}`).join(' | '),
})).sort((a, b) => b.total - a.total);
for (const r of rows) console.log(String(r.total).padStart(7), ' ', r.k.padEnd(32), r.top.slice(0, 80));
