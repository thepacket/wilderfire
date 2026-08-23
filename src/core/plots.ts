// JWildfire's formula plot family — yplot2d_wf, yplot3d_wf, parplot2d_wf, polarplot2d_wf, polarplot3d_wf,
// isosfplot3d_wf (Andreas Maschke; LGPL 2.1+, see NOTICE.md). Each samples a random point of its parameter range,
// evaluates a user formula there (JWildfire compiles the text with Janino at run time) and plots the result; the
// formula is a "ressource" of the instance, or the preset's when `preset_id` ≥ 0 (JWildfire reads the ressources
// before the params, so a preset id ≥ 0 overrides the formula text — and the editor sets it to −1 whenever the
// text or the preset's ranges are edited). WilderFire compiles the formula into the kernel (src/core/formula.ts
// → a WGSL expression inlined in the variation snippet; `sigKey` makes the kernel recompile when it changes).
// The colormap / displacement-map ressources (images) are kept for round-tripping but not rendered: their colour
// modes leave the colour as JWildfire does when no map is loaded.
import { formulaToWgsl, FORMULA_WGSL_FUNCS, FORMULA_WGSL_FUNC_NAMES } from './formula';
import { YPLOT2D_PRESETS, YPLOT3D_PRESETS, PARPLOT2D_PRESETS, POLARPLOT2D_PRESETS, POLARPLOT3D_PRESETS, ISOSFPLOT3D_PRESETS, KNOTS3D_PRESETS, type PlotPreset } from './plotPresets';

export interface PlotFamily {
  presets: PlotPreset[];
  /** the formula ressource names (`formula`, or `xformula`/`yformula`/`zformula`) */
  formulas: string[];
  /** the variables a formula may use besides param_a…param_f */
  vars: string[];
  /** JWildfire's createDefaultPreset — what a preset id outside the table resolves to */
  dflt: PlotPreset;
  /** params `refreshFormulaFromPreset` copies from the preset (besides the formulas and param_a…) */
  refresh: string[];
  /** the preset-id param (`preset_id`; knots3D: `presetId`) */
  idParam: string;
  /** the param_<letter> set (a…f; knots3D: a…h) */
  letters: string;
}
const P6 = { param_a: 0, param_b: 0, param_c: 0, param_d: 0, param_e: 0, param_f: 0 };
const P8 = { ...P6, param_g: 0, param_h: 0 };
const AF = { idParam: 'preset_id', letters: 'abcdef' };
/** The six plot variations, plus knots3D (src/core/knots.ts, a CPU mesh) which shares the preset/ressource mechanism. */
export const PLOT_FAMILIES: Record<string, PlotFamily> = {
  yplot2d_wf: { ...AF, presets: YPLOT2D_PRESETS, formulas: ['formula'], vars: ['x'], dflt: { id: -1, f: { formula: '0.0' }, p: { xmin: -1, xmax: 1, ...P6 } }, refresh: ['xmin', 'xmax'] },
  yplot3d_wf: { ...AF, presets: YPLOT3D_PRESETS, formulas: ['formula'], vars: ['x', 'z'], dflt: { id: -1, f: { formula: '0.0' }, p: { xmin: -1, xmax: 1, zmin: -1, zmax: 1, ...P6 } }, refresh: ['xmin', 'xmax', 'zmin', 'zmax'] },
  parplot2d_wf: { ...AF, presets: PARPLOT2D_PRESETS, formulas: ['xformula', 'yformula', 'zformula'], vars: ['u', 'v'], dflt: { id: -1, f: { xformula: 'u', yformula: '0.0', zformula: 'v' }, p: { umin: -1, umax: 1, vmin: -1, vmax: 1, ...P6 } }, refresh: ['umin', 'umax', 'vmin', 'vmax'] },
  polarplot2d_wf: { ...AF, presets: POLARPLOT2D_PRESETS, formulas: ['formula'], vars: ['t'], dflt: { id: -1, f: { formula: '1.0' }, p: { tmin: -Math.PI, tmax: Math.PI, ...P6 } }, refresh: ['tmin', 'tmax'] },
  polarplot3d_wf: { ...AF, presets: POLARPLOT3D_PRESETS, formulas: ['formula'], vars: ['t', 'u'], dflt: { id: -1, f: { formula: '1.0' }, p: { tmin: -3.14159, tmax: 3.14159, umin: 0, umax: 3.14159, cylindrical: 0, ...P6 } }, refresh: ['tmin', 'tmax', 'umin', 'umax', 'cylindrical'] },
  isosfplot3d_wf: { ...AF, presets: ISOSFPLOT3D_PRESETS, formulas: ['formula'], vars: ['x', 'y', 'z'], dflt: { id: -1, f: { formula: 'x' }, p: { xmin: -1, xmax: 1, ymin: -1, ymax: 1, zmin: -1, zmax: 1, ...P6 } }, refresh: ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'] },
  // Knots3DWFFuncPresets.createDefaultPreset; refreshFormulaFromPreset copies steps/radius/facets too (an empty formula is 0.0)
  knots3D: { idParam: 'presetId', letters: 'abcdefgh', presets: KNOTS3D_PRESETS, formulas: ['xformula', 'yformula', 'zformula'], vars: ['t'], dflt: { id: -1, f: { xformula: '100 * cos(t)', yformula: '100 * sin(t)', zformula: 'cos(t)' }, p: { steps: 1000, radius: 1, facets: 4, ...P8, param_a: 1, param_b: 1, param_c: 1, param_d: 1, param_e: 1, param_f: 1, param_g: 1, param_h: 1 } }, refresh: ['steps', 'radius', 'facets'] },
};
/** the kernel-compiled plot variations (the registry entries below) */
export const PLOT_NAMES = Object.keys(PLOT_FAMILIES).filter((n) => n !== 'knots3D');

/** The preset an id resolves to (JWildfire's getPreset: the default preset when the id is not in the table). */
export function plotPreset(name: string, presetId: number): PlotPreset {
  const fam = PLOT_FAMILIES[name];
  return fam.presets.find((pr) => pr.id === Math.round(presetId)) ?? fam.dflt;
}

/** The formulas an instance evaluates: the preset's when preset_id ≥ 0 (JWildfire's reading order), else its own
 *  ressources (an absent one is the default preset's). */
export function plotFormulas(name: string, presetId: number, res?: Record<string, string>): Record<string, string> {
  const fam = PLOT_FAMILIES[name];
  const out: Record<string, string> = {};
  const pr = Math.round(presetId) >= 0 ? plotPreset(name, presetId) : null;
  for (const k of fam.formulas) out[k] = (pr ? pr.f[k] : res?.[k]?.trim()) || (name === 'knots3D' ? '0.0' : fam.dflt.f[k]);
  return out;
}

type Inst = { params: Record<string, number>; res?: Record<string, string> };
/** param_a…f with preset 0's values as defaults (JWildfire's constructor loads a preset, params included) */
const paramsAF = (name: string) => ['param_a', 'param_b', 'param_c', 'param_d', 'param_e', 'param_f'].map((k) => ({ name: k, def: PLOT_FAMILIES[name].presets[0]?.p[k] ?? 0 }));
const MAP_PARAMS = [{ name: 'blend_colormap', def: 1, int: true }, { name: 'displ_amount', def: 0.1 }, { name: 'blend_displ_map', def: 1, int: true }];
const RES = ['formula', 'colormap_filename', 'displ_map_filename', 'preset_id_reference'];
const XYZ_RES = ['xformula', 'yformula', 'zformula', 'colormap_filename', 'displ_map_filename', 'preset_id_reference'];

/** A formula as WGSL, or a formula that evaluates to 0 with a console warning when the text is outside the subset
 *  (JWildfire renders nothing from an instance whose formula does not compile; we plot the zero curve). */
function wgsl(name: string, key: string, text: string, vars: Record<string, string>): string {
  try { return formulaToWgsl(text, vars); }
  catch (e) { console.warn(`${name}: ${key} "${text}" not understood (${(e as Error).message}); using 0`); return '0.0'; }
}
/** the param_a…f slots of a snippet (p[] indices `base`..`base+5`) as formula variables */
const paramVars = (p: string[], base: number): Record<string, string> => ({ param_a: p[base], param_b: p[base + 1], param_c: p[base + 2], param_d: p[base + 3], param_e: p[base + 4], param_f: p[base + 5] });
/** `var lo_/hi_/d_` of a range with JWildfire's init swap (min > max → exchanged) */
const range = (tag: string, lo: string, hi: string) => `var ${tag}0 = ${lo}; var ${tag}1 = ${hi}; if (${tag}0 > ${tag}1) { let tt_ = ${tag}0; ${tag}0 = ${tag}1; ${tag}1 = tt_; } let ${tag}d = ${tag}1 - ${tag}0;`;
const clampColor = (expr: string) => `(*cp) = clamp(${expr}, 0.0, 1.0);`;
/** the part of the registry entry every plot variation shares */
const common = (name: string) => ({
  flags: ['3d', 'z', 'dc', 'formula'],
  types: ['3D', 'DC', 'BASE_SHAPE'],
  funcNames: FORMULA_WGSL_FUNC_NAMES,
  funcs: FORMULA_WGSL_FUNCS,
  /** the effective formulas are compiled into the kernel */
  sigKey: (inst: Inst) => Object.values(plotFormulas(name, inst.params.preset_id ?? -1, inst.res)).join(''),
});
const preset0 = (name: string, key: string) => PLOT_FAMILIES[name].presets[0]?.p[key] ?? PLOT_FAMILIES[name].dflt.p[key] ?? 0;

export const PLOT_VARIATIONS = {
  // y = f(x) over x ∈ [xmin, xmax], extruded along z ∈ [zmin, zmax]; colour by x (1), y (2) or the colormap (0)
  yplot2d_wf: {
    ...common('yplot2d_wf'),
    params: [
      { name: 'preset_id', def: 0, int: true }, { name: 'xmin', def: preset0('yplot2d_wf', 'xmin') }, { name: 'xmax', def: preset0('yplot2d_wf', 'xmax') },
      { name: 'ymin', def: -4 }, { name: 'ymax', def: 4 }, { name: 'zmin', def: -2 }, { name: 'zmax', def: 2 },
      { name: 'direct_color', def: 1, int: true }, { name: 'color_mode', def: 1, int: true }, ...MAP_PARAMS, ...paramsAF('yplot2d_wf'),
    ],
    res: RES,
    code: (w: string, p: string[], _A: unknown, inst?: Inst) => {
      const f = plotFormulas('yplot2d_wf', inst?.params.preset_id ?? 0, inst?.res).formula;
      return `{ ${range('plx', p[1], p[2])} ${range('ply', p[3], p[4])} ${range('plz', p[5], p[6])}
    let plu = rnd(rs); let plv = rnd(rs);
    let pl_x = plx0 + plu * plxd; let pl_z = plz0 + plv * plzd;
    let pl_y = ${wgsl('yplot2d_wf', 'formula', f, { x: 'pl_x', ...paramVars(p, 12) })};
    if (i32(${p[7]}) > 0) { let cm_ = i32(${p[8]}); if (cm_ == 2) { ${clampColor('(pl_y - ply0) / plyd')} } else if (cm_ != 0) { ${clampColor('(pl_x - plx0) / plxd')} } }
    v += ${w} * vec2f(pl_x, pl_y); pz_ += ${w} * pl_z; }`;
    },
  },
  // y = f(x, z) over the x and z ranges; colour by x (1), y (2), z (3, default), x·z (4) or the colormap (0)
  yplot3d_wf: {
    ...common('yplot3d_wf'),
    params: [
      { name: 'preset_id', def: 0, int: true }, { name: 'xmin', def: preset0('yplot3d_wf', 'xmin') }, { name: 'xmax', def: preset0('yplot3d_wf', 'xmax') },
      { name: 'ymin', def: -4 }, { name: 'ymax', def: 4 }, { name: 'zmin', def: preset0('yplot3d_wf', 'zmin') }, { name: 'zmax', def: preset0('yplot3d_wf', 'zmax') },
      { name: 'direct_color', def: 1, int: true }, { name: 'color_mode', def: 3, int: true }, ...MAP_PARAMS, ...paramsAF('yplot3d_wf'),
    ],
    res: RES,
    code: (w: string, p: string[], _A: unknown, inst?: Inst) => {
      const f = plotFormulas('yplot3d_wf', inst?.params.preset_id ?? 0, inst?.res).formula;
      return `{ ${range('plx', p[1], p[2])} ${range('ply', p[3], p[4])} ${range('plz', p[5], p[6])}
    let plu = rnd(rs); let plv = rnd(rs);
    let pl_x = plx0 + plu * plxd; let pl_z = plz0 + plv * plzd;
    let pl_y = ${wgsl('yplot3d_wf', 'formula', f, { x: 'pl_x', z: 'pl_z', ...paramVars(p, 12) })};
    if (i32(${p[7]}) > 0) { let cm_ = i32(${p[8]});
      if (cm_ == 1) { ${clampColor('(pl_x - plx0) / plxd')} } else if (cm_ == 2) { ${clampColor('(pl_y - ply0) / plyd')} }
      else if (cm_ == 4) { ${clampColor('(pl_x - plx0) / plxd * (pl_z - plz0) / plzd')} } else if (cm_ != 0) { ${clampColor('(pl_z - plz0) / plzd')} } }
    v += ${w} * vec2f(pl_x, pl_y); pz_ += ${w} * pl_z; }`;
    },
  },
  // a parametric surface (x, y, z) = f(u, v); with solid = 0 the affine point's x/y stand in for the random u/v;
  // colour by u (1), v (2), u·v (3, default) or the colormap (0)
  parplot2d_wf: {
    ...common('parplot2d_wf'),
    params: [
      { name: 'preset_id', def: 0, int: true }, { name: 'umin', def: preset0('parplot2d_wf', 'umin') }, { name: 'umax', def: preset0('parplot2d_wf', 'umax') },
      { name: 'vmin', def: preset0('parplot2d_wf', 'vmin') }, { name: 'vmax', def: preset0('parplot2d_wf', 'vmax') },
      { name: 'direct_color', def: 1, int: true }, { name: 'color_mode', def: 3, int: true }, ...MAP_PARAMS, { name: 'solid', def: 1, int: true }, ...paramsAF('parplot2d_wf'),
    ],
    res: XYZ_RES,
    code: (w: string, p: string[], _A: unknown, inst?: Inst) => {
      const f = plotFormulas('parplot2d_wf', inst?.params.preset_id ?? 0, inst?.res);
      const vars = { u: 'pl_u', v: 'pl_v', ...paramVars(p, 11) };
      return `{ ${range('plu', p[1], p[2])} ${range('plv', p[3], p[4])}
    var plru = t.x; var plrv = t.y; if (i32(${p[10]}) != 0) { plru = rnd(rs); plrv = rnd(rs); }
    let pl_u = plu0 + plru * plud; let pl_v = plv0 + plrv * plvd;
    let pl_x = ${wgsl('parplot2d_wf', 'xformula', f.xformula, vars)};
    let pl_y = ${wgsl('parplot2d_wf', 'yformula', f.yformula, vars)};
    let pl_z = ${wgsl('parplot2d_wf', 'zformula', f.zformula, vars)};
    if (i32(${p[5]}) > 0) { let cm_ = i32(${p[6]});
      if (cm_ == 2) { ${clampColor('(pl_v - plv0) / plvd')} } else if (cm_ == 1) { ${clampColor('(pl_u - plu0) / plud')} } else if (cm_ != 0) { ${clampColor('(pl_v - plv0) / plvd * (pl_u - plu0) / plud')} } }
    v += ${w} * vec2f(pl_x, pl_y); pz_ += ${w} * pl_z; }`;
    },
  },
  // r = f(t) over t ∈ [tmin, tmax], plotted at (r·cos t, r·sin t) and extruded along z; colour by t (1), r (2) or the colormap (0)
  polarplot2d_wf: {
    ...common('polarplot2d_wf'),
    params: [
      { name: 'preset_id', def: 0, int: true }, { name: 'tmin', def: preset0('polarplot2d_wf', 'tmin') }, { name: 'tmax', def: preset0('polarplot2d_wf', 'tmax') },
      { name: 'rmin', def: 0 }, { name: 'rmax', def: 2 }, { name: 'zmin', def: -2 }, { name: 'zmax', def: 2 },
      { name: 'direct_color', def: 1, int: true }, { name: 'color_mode', def: 1, int: true }, ...MAP_PARAMS, ...paramsAF('polarplot2d_wf'),
    ],
    res: RES,
    code: (w: string, p: string[], _A: unknown, inst?: Inst) => {
      const f = plotFormulas('polarplot2d_wf', inst?.params.preset_id ?? 0, inst?.res).formula;
      return `{ ${range('plt', p[1], p[2])} ${range('plr', p[3], p[4])} ${range('plz', p[5], p[6])}
    let plu = rnd(rs); let plv = rnd(rs);
    let pl_t = plt0 + plu * pltd; let pl_z = plz0 + plv * plzd;
    let pl_r = ${wgsl('polarplot2d_wf', 'formula', f, { t: 'pl_t', ...paramVars(p, 12) })};
    let pl_x = pl_r * cos(pl_t); let pl_y = pl_r * sin(pl_t);
    if (i32(${p[7]}) > 0) { let cm_ = i32(${p[8]}); if (cm_ == 2) { ${clampColor('(pl_r - plr0) / plrd')} } else if (cm_ != 0) { ${clampColor('(pl_t - plt0) / pltd')} } }
    v += ${w} * vec2f(pl_x, pl_y); pz_ += ${w} * pl_z; }`;
    },
  },
  // r = f(t, u): spherical (t = θ, u = φ) or, with cylindrical = 1, (r·cos t, r·sin t, u); colour by t (1), u (2,
  // default), r (3), t·u (4) or the colormap (0)
  polarplot3d_wf: {
    ...common('polarplot3d_wf'),
    params: [
      { name: 'preset_id', def: 0, int: true }, { name: 'tmin', def: preset0('polarplot3d_wf', 'tmin') }, { name: 'tmax', def: preset0('polarplot3d_wf', 'tmax') },
      { name: 'umin', def: preset0('polarplot3d_wf', 'umin') }, { name: 'umax', def: preset0('polarplot3d_wf', 'umax') }, { name: 'rmin', def: -2 }, { name: 'rmax', def: 2 },
      { name: 'cylindrical', def: preset0('polarplot3d_wf', 'cylindrical'), int: true },
      { name: 'direct_color', def: 1, int: true }, { name: 'color_mode', def: 2, int: true }, ...MAP_PARAMS, ...paramsAF('polarplot3d_wf'),
    ],
    res: ['formula', 'colormap_filename', 'displ_map_filename'],
    code: (w: string, p: string[], _A: unknown, inst?: Inst) => {
      const f = plotFormulas('polarplot3d_wf', inst?.params.preset_id ?? 0, inst?.res).formula;
      return `{ ${range('plt', p[1], p[2])} ${range('plu', p[3], p[4])} ${range('plr', p[5], p[6])}
    let plrt = rnd(rs); let plru = rnd(rs);
    let pl_t = plt0 + plrt * pltd; let pl_u = plu0 + plru * plud;
    let pl_r = ${wgsl('polarplot3d_wf', 'formula', f, { t: 'pl_t', u: 'pl_u', ...paramVars(p, 13) })};
    var pl_x: f32; var pl_y: f32; var pl_z: f32;
    if (i32(${p[7]}) == 0) { pl_x = pl_r * sin(pl_u) * cos(pl_t); pl_y = pl_r * sin(pl_u) * sin(pl_t); pl_z = pl_r * cos(pl_u); }
    else { pl_x = pl_r * cos(pl_t); pl_y = pl_r * sin(pl_t); pl_z = pl_u; }
    if (i32(${p[8]}) > 0) { let cm_ = i32(${p[9]});
      if (cm_ == 1) { ${clampColor('(pl_t - plt0) / pltd')} } else if (cm_ == 2) { ${clampColor('(pl_u - plu0) / plud')} } else if (cm_ == 3) { ${clampColor('(pl_r - plr0) / plrd')} }
      else if (cm_ == 4) { ${clampColor('(pl_t - plt0) / pltd * (pl_u - plu0) / plud')} } }
    v += ${w} * vec2f(pl_x, pl_y); pz_ += ${w} * pl_z; }`;
    },
  },
  // an implicit surface f(x, y, z) = 0: up to max_iter random points of the box, the first with |f| ≤ thickness is
  // plotted, none hides the point; colour by x (3), y (4), z (5), xy (6), yz (7), zx (8), xyz (9, default) or the colormap (0..2)
  isosfplot3d_wf: {
    ...common('isosfplot3d_wf'),
    flags: ['3d', 'z', 'dc', 'hide', 'formula'],
    params: [
      { name: 'preset_id', def: 0, int: true }, { name: 'xmin', def: preset0('isosfplot3d_wf', 'xmin') }, { name: 'xmax', def: preset0('isosfplot3d_wf', 'xmax') },
      { name: 'ymin', def: preset0('isosfplot3d_wf', 'ymin') }, { name: 'ymax', def: preset0('isosfplot3d_wf', 'ymax') }, { name: 'zmin', def: preset0('isosfplot3d_wf', 'zmin') }, { name: 'zmax', def: preset0('isosfplot3d_wf', 'zmax') },
      { name: 'thickness', def: 0.05 }, { name: 'max_iter', def: 160, int: true },
      { name: 'direct_color', def: 1, int: true }, { name: 'color_mode', def: 9, int: true }, { name: 'blend_colormap', def: 1, int: true }, ...paramsAF('isosfplot3d_wf'),
    ],
    res: ['formula', 'colormap_filename', 'preset_id_reference'],
    code: (w: string, p: string[], _A: unknown, inst?: Inst) => {
      const f = plotFormulas('isosfplot3d_wf', inst?.params.preset_id ?? 0, inst?.res).formula;
      return `{ ${range('plx', p[1], p[2])} ${range('ply', p[3], p[4])} ${range('plz', p[5], p[6])}
    *hd = true;
    var pl_x = 0.0; var pl_y = 0.0; var pl_z = 0.0;
    let pl_n = clamp(i32(${p[8]}), 0, 4096); let pl_th = ${p[7]};
    for (var pl_i = 0; pl_i < pl_n; pl_i++) {
      pl_x = plx0 + rnd(rs) * plxd; pl_y = ply0 + rnd(rs) * plyd; pl_z = plz0 + rnd(rs) * plzd;
      let pl_e = ${wgsl('isosfplot3d_wf', 'formula', f, { x: 'pl_x', y: 'pl_y', z: 'pl_z', ...paramVars(p, 12) })};
      if (abs(pl_e) <= pl_th) { v += ${w} * vec2f(pl_x, pl_y); pz_ += ${w} * pl_z; *hd = false; break; }
    }
    if (!(*hd) && i32(${p[9]}) > 0) { let cm_ = i32(${p[10]});
      let cx_ = (pl_x - plx0) / plxd; let cy_ = (pl_y - ply0) / plyd; let cz_ = (pl_z - plz0) / plzd;
      if (cm_ == 3) { ${clampColor('cx_')} } else if (cm_ == 4) { ${clampColor('cy_')} } else if (cm_ == 5) { ${clampColor('cz_')} }
      else if (cm_ == 6) { ${clampColor('cx_ * cy_')} } else if (cm_ == 7) { ${clampColor('cy_ * cz_')} } else if (cm_ == 8) { ${clampColor('cz_ * cx_')} }
      else if (cm_ < 0 || cm_ > 2) { ${clampColor('cx_ * cy_ * cz_')} } } }`;
    },
  },
};
