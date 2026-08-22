// AI context shaping: what we send about the flame (full JSON / compact summary,
// palette as stops or full), which variations we describe, and a small "edits"
// language the model can use instead of re-emitting the whole flame — orders of
// magnitude fewer output tokens, hence faster and cheaper.

import type { Flame, RGB, XForm } from '../core/flame';
import { cloneFlame, defaultXForm, expandStops } from '../core/flame';
import { VARIATIONS } from '../core/variations';
import { getParam, setParam } from '../core/motion';

export type FlameMode = 'json' | 'summary' | 'none';
export type PaletteMode = 'none' | 'stops' | 'full';
export type VarsMode = 'none' | 'used' | 'all';
export type ReplyMode = 'edits' | 'json' | 'text';

export interface ContextOpts {
  flame: FlameMode;
  palette: PaletteMode;
  vars: VarsMode;
  screenshot: boolean;
  memory: boolean;
  reply: ReplyMode;
  /** let the model act through function tools (edit, camera, library, screenshot…) in a loop */
  tools: boolean;
}

export const DEFAULT_CONTEXT: ContextOpts = {
  flame: 'summary', palette: 'stops', vars: 'used', screenshot: true, memory: true, reply: 'edits', tools: true,
};

const r3 = (v: number) => Math.round(v * 1000) / 1000;
const fmtN = (v: number) => String(r3(v));

/** Down-sample a 256-entry palette to `n` evenly spaced stops [t, r, g, b]. */
export function paletteStops(pal: RGB[], n = 8): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const c = pal[Math.min(255, Math.round(t * 255))] ?? [0, 0, 0];
    out.push([r3(t), r3(c[0]), r3(c[1]), r3(c[2])]);
  }
  return out;
}

/** Flame JSON with the palette replaced by stops (or dropped) — same shape normalizeFlame accepts back. */
export function flameJSONFor(f: Flame, palette: PaletteMode): string {
  const c: any = cloneFlame(f);
  for (const ly of c.layers) {
    if (palette === 'full') { ly.palette = ly.palette.map((v: RGB) => v.map(r3)); }
    else {
      delete ly.palette;
      if (palette === 'stops') ly.paletteStops = paletteStops(f.layers[c.layers.indexOf(ly)].palette, 8);
    }
    for (const x of [...ly.xforms, ...(ly.final ? [ly.final] : [])]) {
      x.affine = x.affine.map(r3); x.post = x.post.map(r3);
      for (const k of ['weight', 'color', 'colorSpeed', 'opacity']) x[k] = r3(x[k]);
      for (const list of [x.variations, x.preVariations, x.postVariations]) {
        for (const v of list ?? []) { v.weight = r3(v.weight); for (const p in v.params) v.params[p] = r3(v.params[p]); }
      }
    }
  }
  return JSON.stringify(c);
}

function xformSummary(x: XForm, label: string, path: string): string {
  const vars = (list: XForm['variations'] | undefined, tag: string) =>
    (list ?? []).map((v) => {
      const ps = Object.entries(v.params ?? {}).map(([k, val]) => `${k}=${fmtN(val)}`).join(' ');
      return `${tag}${v.name} ${fmtN(v.weight)}${ps ? ' (' + ps + ')' : ''}`;
    });
  const all = [...vars(x.preVariations, 'pre:'), ...vars(x.variations, ''), ...vars(x.postVariations, 'post:')];
  const post = x.post.every((v, i) => Math.abs(v - [1, 0, 0, 0, 1, 0][i]) < 1e-9) ? '' : ` post [${x.post.map(fmtN).join(',')}]`;
  const xaos = x.xaos && x.xaos.some((v) => v !== 1) ? ` xaos [${x.xaos.map(fmtN).join(',')}]` : '';
  return `${label} (${path}): weight ${fmtN(x.weight)}, color ${fmtN(x.color)}, colorSpeed ${fmtN(x.colorSpeed)}, opacity ${fmtN(x.opacity)}, affine [${x.affine.map(fmtN).join(',')}]${post}${xaos}; variations: ${all.join(', ') || 'none'}`;
}

/** Compact human/LLM-readable description of the flame with the paths edits use. */
export function flameSummary(f: Flame, palette: PaletteMode): string {
  const lines: string[] = [];
  lines.push(`name "${f.name}"; zoom ${fmtN(f.zoom)}, center (${fmtN(f.centerX)}, ${fmtN(f.centerY)}), rotation ${fmtN(f.rotation)} rad; brightness ${fmtN(f.brightness)}, gamma ${fmtN(f.gamma)}, gammaThreshold ${fmtN(f.gammaThreshold)}, vibrancy ${fmtN(f.vibrancy)}, background [${f.background.map(fmtN).join(',')}]` +
    (f.camPitch || f.camYaw || f.camPersp ? `; 3D camera pitch ${fmtN(f.camPitch)}°, yaw ${fmtN(f.camYaw)}°, perspective ${fmtN(f.camPersp)}` : '') +
    (f.camDOF ? `; DOF ${fmtN(f.camDOF)} (focusZ ${fmtN(f.focusZ)}, area ${fmtN(f.camDOFArea)})` : ''));
  f.layers.forEach((ly, li) => {
    const L = f.layers.length > 1 ? `Layer ${li + 1} (layers.${li}, weight ${fmtN(ly.weight)}${ly.visible ? '' : ', hidden'}): ` : '';
    if (L) lines.push(L.trim());
    ly.xforms.forEach((x, xi) => lines.push('  ' + xformSummary(x, `T${xi + 1}`, `layers.${li}.xforms.${xi}`)));
    if (ly.final) lines.push('  ' + xformSummary(ly.final, 'Final', `layers.${li}.final`));
    if (palette !== 'none') {
      const stops = paletteStops(ly.palette, palette === 'full' ? 16 : 8);
      lines.push(`  palette stops (t,r,g,b): ${JSON.stringify(stops)}`);
    }
  });
  return lines.join('\n');
}

/** Variation catalogue text: all, or only those used in the flame. */
export function variationCatalogue(f: Flame, mode: VarsMode): string {
  if (mode === 'none') return '';
  let names: string[];
  if (mode === 'all') names = Object.keys(VARIATIONS);
  else {
    const s = new Set<string>();
    for (const ly of f.layers) for (const x of [...ly.xforms, ...(ly.final ? [ly.final] : [])])
      for (const list of [x.variations, x.preVariations, x.postVariations]) for (const v of list ?? []) s.add(v.name);
    for (const n of ['linear', 'spherical', 'swirl', 'julian', 'curl', 'bubble', 'blur', 'gaussian_blur', 'sinusoidal', 'polar', 'disc', 'spiral', 'ngon', 'pdj']) s.add(n);
    names = [...s].filter((n) => VARIATIONS[n]);
  }
  const items = names.map((name) => {
    const def = VARIATIONS[name];
    const ps = (def.params ?? []).map((p) => `${p.name}=${p.def}`).join(', ');
    return ps ? `${name}(${ps})` : name;
  });
  return (mode === 'all' ? 'AVAILABLE VARIATIONS (params with defaults): ' : `VARIATIONS IN USE + CLASSICS (params with defaults; ${Object.keys(VARIATIONS).length} exist in total — the user can name others): `) + items.join(', ');
}

/** Rough token estimate (chars/4 + a flat cost per image). */
export const estimateTokens = (chars: number, images = 0) => Math.round(chars / 4) + images * 800;

// ---------------- edits language ----------------

export const EDITS_SPEC = `TO APPLY CHANGES reply with ONE fenced block tagged \`edits\`, one command per line (numbers only, no prose inside):
  set <path> <number>            e.g. set brightness 4 / set T2.weight 0.8 / set T1.variations.0.params.power 3 / set F.affine.2 0.1 / set layers.0.xforms.1.color 0.4
  set <T#|F>.affine [a,b,c,d,e,f]  (or .post) the whole matrix at once
  addvar <T#|F> <name> [weight] [param=value ...]   e.g. addvar T2 julian 0.7 power=3 dist=1
  delvar <T#|F> <name>
  addxform [weight] [color=0.5 colorSpeed=0.3 opacity=1 affine=[a,b,c,d,e,f] variations=name:weight,name:weight]   (new transform, becomes the next T#)
  delxform <T#>
  palette [[t,r,g,b], ...]       3-8 stops, t and rgb in 0..1 (whole gradient of the active layer)
  name <text>
Paths: T1..Tn = transforms of the active layer, F = its final transform; fields weight, color, colorSpeed, opacity, affine.0-5 (a,b,c,d,e,f: x'=a·x+b·y+c, y'=d·x+e·y+f), post.0-5, variations.<i>.weight, variations.<i>.params.<name>; flame fields zoom, centerX, centerY, rotation, brightness, gamma, gammaThreshold, vibrancy, camPitch, camYaw, camPersp, camDOF, focusZ. Only list what changes. If the user only asks a question, answer without a block.`;

export interface EditsResult { applied: number; errors: string[]; flame: Flame }

/** Apply an ```edits block to a copy of the flame. */
export function applyEdits(base: Flame, block: string, activeLayer: number): EditsResult {
  const f = cloneFlame(base);
  const li = Math.min(Math.max(activeLayer, 0), f.layers.length - 1);
  const ly = f.layers[li];
  const errors: string[] = [];
  let applied = 0;
  const xformRef = (tok: string): { x: XForm; base: string } | null => {
    if (/^F(inal)?$/i.test(tok)) return ly.final ? { x: ly.final, base: `layers.${li}.final` } : null;
    const m = /^T(\d+)$/i.exec(tok);
    if (m) { const i = parseInt(m[1]) - 1; return ly.xforms[i] ? { x: ly.xforms[i], base: `layers.${li}.xforms.${i}` } : null; }
    return null;
  };
  const normPath = (p: string): string => {
    const m = /^(T\d+|F(?:inal)?)\.(.+)$/i.exec(p);
    if (m) { const r = xformRef(m[1]); if (!r) return p; return `${r.base}.${m[2]}`; }
    return p;
  };
  for (const raw of block.split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || line.startsWith('//')) continue;
    const [cmd, ...rest] = line.split(/\s+/);
    try {
      // the models' spelling variants: add_xform, del-var, SET …
      switch (cmd.toLowerCase().replace(/[-_]/g, '')) {
        case 'set': {
          const path = normPath(rest[0] ?? '');
          const valueText = line.slice(line.indexOf(rest[0] ?? '') + (rest[0] ?? '').length).trim();
          if (valueText.startsWith('[')) {
            // a whole affine / post matrix: set T4.affine [a,b,c,d,e,f]
            const arr = JSON.parse(valueText.replace(/,\s*]/, ']'));
            if (!Array.isArray(arr) || !arr.every((v) => typeof v === 'number' && isFinite(v))) throw new Error('the array needs numbers');
            let n = 0;
            arr.forEach((v: number, i: number) => { if (setParam(f, `${path}.${i}`, v)) n++; });
            if (!n) throw new Error(`unknown array path ${path}`);
            applied++;
            break;
          }
          const val = parseFloat(rest[1] ?? '');
          if (!isFinite(val)) throw new Error('needs a number');
          if (!setParam(f, path, val)) throw new Error(`unknown path ${path}`);
          applied++;
          break;
        }
        case 'addvar': {
          const r = xformRef(rest[0] ?? '');
          if (!r) throw new Error(`no transform ${rest[0]}`);
          const name = rest[1];
          const def = VARIATIONS[name];
          if (!def) throw new Error(`unknown variation ${name}`);
          let w = 1;
          const params: Record<string, number> = {};
          for (const pd of def.params ?? []) params[pd.name] = pd.def;
          for (const t of rest.slice(2)) {
            const kv = /^([A-Za-z_]\w*)=(-?[\d.eE+-]+)$/.exec(t);
            if (kv) { if (kv[1] in params) params[kv[1]] = parseFloat(kv[2]); }
            else if (isFinite(parseFloat(t))) w = parseFloat(t);
          }
          const existing = r.x.variations.find((v) => v.name === name);
          if (existing) { existing.weight = w; Object.assign(existing.params, params); }
          else r.x.variations.push({ name, weight: w, params });
          applied++;
          break;
        }
        case 'delvar': {
          const r = xformRef(rest[0] ?? '');
          if (!r) throw new Error(`no transform ${rest[0]}`);
          const n = r.x.variations.length;
          r.x.variations = r.x.variations.filter((v) => v.name !== rest[1]);
          if (r.x.variations.length === n) throw new Error(`${rest[1]} not on ${rest[0]}`);
          if (!r.x.variations.length) r.x.variations.push({ name: 'linear', weight: 1, params: {} });
          applied++;
          break;
        }
        case 'addxform': {
          const x = defaultXForm();
          // `addxform 0.5` or key=value pairs (weight, color, colorSpeed, opacity, affine=[…], post=[…], variations=name:w,name:w);
          // a leading layer token such as layers.0 is accepted and ignored (edits always target the active layer)
          const argText = rest.filter((t) => !/^layers\.\d+$/.test(t)).join(' ');
          for (const m of argText.matchAll(/([A-Za-z_]\w*)=(\[[^\]]*\]|[^\s]+)|(-?\d*\.?\d+(?:e[-+]?\d+)?)(?=\s|$)/gi)) {
            if (m[3] !== undefined) { x.weight = parseFloat(m[3]); continue; }
            const k = m[1], v = m[2];
            if (k === 'affine' || k === 'post') {
              const arr = JSON.parse(v);
              if (!Array.isArray(arr) || arr.length !== 6 || !arr.every((n) => typeof n === 'number')) throw new Error(`${k} needs 6 numbers`);
              x[k] = arr as typeof x.affine;
            } else if (k === 'variations') {
              x.variations = [];
              for (const spec of v.split(',')) {
                const vm = /^([A-Za-z_]\w*)(?::(-?[\d.eE+-]+))?(?:\{[^}]*\})?$/.exec(spec.trim());
                if (!vm || !VARIATIONS[vm[1]]) throw new Error(`unknown variation ${spec}`);
                const params: Record<string, number> = {};
                for (const pd of VARIATIONS[vm[1]].params ?? []) params[pd.name] = pd.def;
                x.variations.push({ name: vm[1], weight: vm[2] !== undefined ? parseFloat(vm[2]) : 1, params });
              }
              if (!x.variations.length) x.variations.push({ name: 'linear', weight: 1, params: {} });
            } else if (k === 'weight' || k === 'color' || k === 'colorSpeed' || k === 'opacity') {
              const n = parseFloat(v);
              if (!isFinite(n)) throw new Error(`${k} needs a number`);
              x[k] = n;
            } else throw new Error(`unknown field ${k}`);
          }
          ly.xforms.push(x);
          applied++;
          break;
        }
        case 'delxform': {
          const m = /^T(\d+)$/i.exec(rest[0] ?? '');
          if (!m || !ly.xforms[parseInt(m[1]) - 1]) throw new Error(`no transform ${rest[0]}`);
          if (ly.xforms.length <= 1) throw new Error('cannot delete the last transform');
          ly.xforms.splice(parseInt(m[1]) - 1, 1);
          applied++;
          break;
        }
        case 'palette': {
          const stops = JSON.parse(line.slice(cmd.length).trim());
          if (!Array.isArray(stops) || stops.length < 2) throw new Error('needs ≥ 2 stops');
          ly.palette = expandStops(stops.map((s: number[]) => [s[0], s[1], s[2], s[3]] as [number, number, number, number]));
          applied++;
          break;
        }
        case 'name': {
          f.name = rest.join(' ');
          applied++;
          break;
        }
        default:
          throw new Error('unknown command');
      }
    } catch (e) {
      errors.push(`${line} — ${(e as Error).message}`);
    }
  }
  return { applied, errors, flame: f };
}

/** True if a numeric path exists on the flame (used to validate before applying). */
export const hasParam = (f: Flame, path: string) => getParam(f, path) !== undefined;

/** Does a user message consist of edit commands (a fenced `edits` block, or only command lines and # comments)?
 *  Returns the command text to apply, or null. Lets a block copied from an earlier answer be pasted and applied
 *  directly, without a round trip to the model. */
export function editsFromUserText(text: string): string | null {
  const fenced = /```edits?\s*\n([\s\S]*?)```/i.exec(text);
  if (fenced) return fenced[1];
  // a bare ``` fence (or any other tag) around command lines counts too; fence lines are ignored
  const lines = text.split('\n').map((l) => l.trim()).filter((l) => l && !/^```/.test(l));
  if (!lines.length) return null;
  const cmd = /^(set|add_?var|del_?var|add_?xform|del_?xform|palette|name)\s/i;
  const commands = lines.filter((l) => cmd.test(l));
  if (!commands.length) return null;
  if (lines.every((l) => cmd.test(l) || l.startsWith('#') || l.startsWith('//'))) return lines.join('\n');
  return null;
}
