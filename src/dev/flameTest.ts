// Dev harness: imports each JWildfire fixture flame (scripts/jwf-port/testflames),
// reports unresolved variations, compiles the flame's WGSL kernel and checks
// it for shader errors.
//
//   await window.wilderfire.flameTest()                 // all fixtures
//   await window.wilderfire.flameTest({ show: 'Gnarl_0' })  // load one into the editor

import { compileFlame } from '../gpu/codegen';
import { importFlameText } from '../core/flameXML';
import type { RGB } from '../core/flame';

export interface FlameTestResult { file: string; unknown: string[]; wgslError?: string; variations: string[] }

export const FIXTURES = [
  'Bokeh_0', 'Bokeh_1', 'Brokat_0', 'Brokat_1', 'Bubbles_0', 'Bubbles_1', 'Cross_0', 'Cross_1', 'Duality_0', 'Duality_1',
  'Duckies_0', 'Duckies_1', 'EDisc_0', 'EDisc_1', 'Galaxies_0', 'Galaxies_1', 'Ghosts_0', 'Ghosts_1', 'Gnarl_0', 'Gnarl_1',
  'Julians_0', 'Julians_1', 'Layers_0', 'Layers_1', 'Machine_0', 'Machine_1', 'Orchids_0', 'Orchids_1', 'Outlines_0', 'Outlines_1',
  'Painterly_0', 'Painterly_1', 'Phoenix_0', 'Phoenix_1', 'Rays_0', 'Rays_1', 'Sierpinsky_0', 'Sierpinsky_1',
  'Spherical_0', 'Spherical_1', 'Spirals_0', 'Spirals_1', 'Splits_0', 'Splits_1',
  'yflip', // hand-made Sierpinski with the wide base at +y: must render base-down like flam3/JWildfire
  // Sierpinsky_0 with a JWildfire weighting field on one xform (amount / jitter / int+amount param modulation / cellular colour / value-fractal jitter)
  'wfield_amount', 'wfield_jitter', 'wfield_ppow', 'wfield_cell', 'wfield_value',
  // JWildfire solid rendering (z-buffer + lights/materials): two materials, a per-xform material index, coloured lights, filter on/off
  'Solid_0', 'Solid_1', 'Solid_2', 'Solid_3', // Solid_2 = Solid_0 with ambient occlusion, Solid_3 = + smooth shadow maps from both lights
  'Solid_4', // obj_mesh_primitive_wf: torus + subdivided ball + flat box, two materials, AO, fast shadows
  'Sub_0', // subflame_wf (JWildfire's default sub-flame, colour mode 0, scale/angle/offset/colorscale_z) + a forced-post julian (enforced-priority preserve-z) + post_crop final (pre/post preserve-z)
];

export async function runFlameTest(device: GPUDevice, palette: RGB[], opts: { files?: string[]; verbose?: boolean } = {}): Promise<FlameTestResult[]> {
  const results: FlameTestResult[] = [];
  for (const f of opts.files ?? FIXTURES) {
    const res: FlameTestResult = { file: f, unknown: [], variations: [] };
    results.push(res);
    let text: string;
    try {
      const r = await fetch(`/scripts/jwf-port/testflames/${f}.flame`);
      if (!r.ok) { res.wgslError = `fetch ${r.status}`; continue; }
      text = await r.text();
    } catch (err) { res.wgslError = 'fetch: ' + String(err); continue; }
    try {
      const { flame, unknown } = importFlameText(text, palette);
      res.unknown = unknown;
      const names = new Set<string>();
      for (const ly of flame.layers) for (const x of [...ly.xforms, ...(ly.final ? [ly.final] : [])]) {
        for (const v of [...(x.preVariations ?? []), ...x.variations, ...(x.postVariations ?? [])]) names.add(v.name);
      }
      res.variations = [...names].sort();
      const compiled = compileFlame(flame, 1024);
      device.pushErrorScope('validation');
      const module = device.createShaderModule({ code: compiled.wgsl });
      const info = await module.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === 'error');
      await device.popErrorScope();
      if (errs.length) {
        const ls = compiled.wgsl.split('\n');
        res.wgslError = errs.slice(0, 2).map((m) => `${m.lineNum}: ${m.message} | ${ls[m.lineNum - 1]?.trim()}`).join('\n');
      }
    } catch (err) {
      res.wgslError = 'import/compile: ' + String((err as Error).message ?? err);
    }
    if (opts.verbose) console.log(f, res.unknown.length ? 'unknown: ' + res.unknown.join(',') : '', res.wgslError ?? 'ok');
  }
  const bad = results.filter((r) => r.wgslError);
  const unk = results.filter((r) => r.unknown.length);
  console.log(`flameTest: ${results.length} fixtures, ${bad.length} with shader errors, ${unk.length} with unsupported variations`);
  if (unk.length) console.log('unsupported:', [...new Set(unk.flatMap((r) => r.unknown))].sort().join(', '));
  if (bad.length) console.log('errors:', bad.map((r) => `${r.file}: ${r.wgslError}`).join('\n'));
  (window as any).wilderfire.flameTestReport = results;
  return results;
}
