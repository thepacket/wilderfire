// Function tools the assistant can call (OpenAI-compatible tool calling via OpenRouter or a local
// server). Each tool acts on the app and returns text for the model — and, when screenshots are
// enabled, a fresh render so the model can judge its own change and iterate (the look → act → look
// loop that a single edits block cannot do). Everything runs in the browser.
import type { App } from '../ui/common';
import type { ToolDef } from './openrouter';
import { applyEdits, flameJSONFor, flameSummary, type ContextOpts } from './context';
import { normalizeFlame } from '../core/flame';
import { VARIATIONS } from '../core/variations';

export interface ToolResult { text: string; image?: string; changed?: boolean }
export interface ToolEnv {
  app: App;
  ctx: ContextOpts;
  /** small JPEG of the current render as a data URL (null when it cannot be captured) */
  screenshot: () => string | null;
  confirm: (message: string) => boolean;
}

/** How many tool rounds one user request may take before the model must answer. */
export const MAX_TOOL_ROUNDS = 8;

const num = { type: 'number' } as const;
const str = { type: 'string' } as const;

export const TOOL_DEFS: ToolDef[] = [
  { type: 'function', function: { name: 'get_flame', description: 'The current flame as a compact summary (transform paths T1…, F, variations, palette stops, camera, tone) — enough for every edit. The complete JSON is only available when the user has set Send → Flame to "full JSON"; otherwise the summary is returned.',
    parameters: { type: 'object', properties: { detail: { type: 'string', enum: ['summary', 'json'], description: 'summary (default). json is honoured only when the user enabled full JSON.' } } } } },
  { type: 'function', function: { name: 'apply_edits', description: 'Change the flame with edit commands, one per line (set <path> <number> · addvar <T#|F> <name> [weight] [param=value…] · delvar · addxform [weight] · delxform <T#> · palette [[t,r,g,b],…] · name <text>). Returns how many applied, any errors, and the new render when screenshots are on.',
    parameters: { type: 'object', properties: { edits: { type: 'string', description: 'the edit commands' } }, required: ['edits'] } } },
  { type: 'function', function: { name: 'set_flame_json', description: 'Replace the whole flame with complete flame JSON (same shape get_flame returns with detail=json). Use apply_edits for small changes — this costs far more tokens.',
    parameters: { type: 'object', properties: { flame: { type: 'object', description: 'the complete flame' } }, required: ['flame'] } } },
  { type: 'function', function: { name: 'get_engine', description: 'The live-render engine settings: mode (draft/final/custom), quality cap (spp), stop-after seconds, speed, oversample, DE preview, preview hold, adaptive budget, paused.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'set_engine', description: 'Change live-render engine settings (not the flame): any subset of the fields; the others stay. mode=final gives full quality on screen (2× oversampling, full DE radius, cap 4000), draft is responsive editing. rerender restarts the accumulation, reset puts every setting back to its default.',
    parameters: { type: 'object', properties: {
      mode: { type: 'string', enum: ['draft', 'final'] },
      qualityCap: { type: 'number', description: 'samples per pixel at which the live view stops (50…10000; snapped to the menu)' },
      stopAfterS: { type: 'number', description: 'wall-clock limit in seconds, 0 = none' },
      speed: { type: 'string', enum: ['eco', 'balanced', 'fast', 'furnace'] },
      oversample: { type: 'integer', enum: [1, 2] },
      dePreview: { type: 'string', enum: ['fast', 'balanced', 'full'], description: 'density-estimation radius cap for the live view' },
      previewHold: { type: 'number', description: 'spp before a fresh render replaces the previous image (0…60)' },
      adaptiveBudget: { type: 'boolean' },
      paused: { type: 'boolean' },
      rerender: { type: 'boolean' },
      reset: { type: 'boolean' },
    } } } },
  { type: 'function', function: { name: 'set_camera', description: 'Move the camera. Any subset of the fields; the others stay.',
    parameters: { type: 'object', properties: { zoom: { ...num, description: 'multiplier, ~0.3–4 (1 = default framing)' }, centerX: num, centerY: num, rotationDeg: { ...num, description: 'roll in degrees' }, pitch: { ...num, description: '3D pitch in degrees' }, yaw: { ...num, description: '3D yaw in degrees' }, perspective: { ...num, description: '0 = none, ~0.2 mild' } } } } },
  { type: 'function', function: { name: 'screenshot', description: 'A fresh small render of the current flame (vision models only). Use it to check a result before deciding what to change next.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'variation_lookup', description: 'Search the variation catalogue (938 JWildfire/flam3 variations) by name substring; returns names and their parameter names.',
    parameters: { type: 'object', properties: { query: { ...str, description: 'substring of the name, e.g. "julia", "blur", "wave"' }, limit: { ...num, description: 'max results (default 15)' } }, required: ['query'] } } },
  { type: 'function', function: { name: 'library_search', description: "Search the user's flame library by a substring of the name, author, source pack or tag (query \"★\" lists favourites). Returns id, ★, name, author, source, tags for each match.",
    parameters: { type: 'object', properties: { query: { ...str, description: 'substring to look for ("" lists the newest)' }, limit: { ...num, description: 'max results (default 20)' } }, required: ['query'] } } },
  { type: 'function', function: { name: 'library_similar', description: "More like this: the library flames most similar to the current flame (or to a library entry by id) by variations, palette and structure. Returns id, name, similarity for each.",
    parameters: { type: 'object', properties: { id: { ...str, description: 'library entry id to compare against (default: the flame in the editor)' }, limit: { ...num, description: 'max results (default 10)' } } } } },
  { type: 'function', function: { name: 'library_load', description: 'Load a flame from the library into the editor by its id (from library_search).',
    parameters: { type: 'object', properties: { id: str }, required: ['id'] } } },
  { type: 'function', function: { name: 'library_save', description: 'Save the current flame into the library (with a thumbnail), optionally renaming it first.',
    parameters: { type: 'object', properties: { name: { ...str, description: 'new name (optional)' } } } } },
  { type: 'function', function: { name: 'randomize', description: 'Replace the flame with a fresh random flame (a starting point; the old one is in undo), sampled like JWildfire\'s random batch (up to 16 candidates, the first that covers enough of the picture). Styles: any (default, one of JWildfire\'s 48 generators), wilderfire (the built-in contractive randomizer), or a style id — bubbles, julians, splits, spherical, ghosts, tentacle, linear, sierpinsky, galaxies, machine, brokat, spirals, phoenix, juliandisc, julianrings, xenomorph, outlines, duality, simple, simple_experimental, affine3d, edisc, mandelbrot, orchids, simpletiling, synth, tileball, underwater, gnarl, gnarl_experimental, bubbles3d, bubbles3d_experimental, flowers3d, flowers3d_filled, spherical3d, cross, brokat3d, gnarl3d, spirals3d, subflame, blackandwhite, bokeh, solid_experimental, solid_julia3d, solid_labyrinth, solid_stunning, solid_recursive, solid_shadows.',
    parameters: { type: 'object', properties: { style: { ...str, description: 'style id (default any)' }, symmetry: { type: 'string', enum: ['none', 'all', 'sparse', 'xaxis', 'yaxis', 'point'], description: 'post symmetry generator (default sparse: a third of the flames get one)' }, wfield: { type: 'string', enum: ['none', 'all', 'sparse', 'basic', 'cellular', 'fractal'], description: 'weighting-field generator (default sparse)' } } } } },
  { type: 'function', function: { name: 'mutate', description: 'Apply a MutaGen mutation to the current flame (a variation on the theme; the old one is in undo). Types: all (default, one at random), add_transform, add_variation, change_weight (xaos), gradient_position, local_gamma, affine, affine_3d, bokeh, random_bg_color, random_flame, random_ztransform, random_gradient, random_parameter, similar_gradient, weighting_field, color_type.',
    parameters: { type: 'object', properties: { type: { ...str, description: 'mutation type (default all)' }, strength: { ...num, description: '0.1…3, default 1' }, count: { ...num, description: 'how many mutations to apply in a row (1…5, default 1)' } } } } },
  { type: 'function', function: { name: 'undo', description: 'Undo the last change to the flame.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'redo', description: 'Redo the change undone last.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'share_link', description: 'A link that opens the current flame in WilderFire (the flame is encoded in the URL itself; nothing is uploaded). Show it to the user.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'export_png', description: 'Save a PNG of the current flame — the user is asked to confirm. Without a size it saves the live render as shown; with width/height it renders offscreen at that size (hi-res, seconds to minutes) at `spp` samples per pixel (default 2000).',
    parameters: { type: 'object', properties: { width: { ...num, description: 'pixels (optional, ≤ 16384)' }, height: { ...num, description: 'pixels (optional)' }, spp: { ...num, description: 'samples per pixel for a hi-res render (100…10000, default 2000)' }, transparent: { type: 'boolean' } } } } },
  { type: 'function', function: { name: 'import_flame', description: 'Load a flame from text: JWildfire/Apophysis/flam3 .flame XML (one flame or a pack) or WilderFire JSON. The first flame goes into the editor; with toLibrary=true every flame of a pack is added to the library instead (thumbnails are rendered, a few seconds per dozen).',
    parameters: { type: 'object', properties: { text: { ...str, description: 'the .flame XML or JSON' }, toLibrary: { type: 'boolean' } }, required: ['text'] } } },
  { type: 'function', function: { name: 'library_update', description: 'Change a library entry: rename, favourite on/off, add or remove tags. Use ids from library_search.',
    parameters: { type: 'object', properties: { id: str, name: str, fav: { type: 'boolean' }, addTags: { type: 'array', items: str }, removeTags: { type: 'array', items: str } }, required: ['id'] } } },
  { type: 'function', function: { name: 'library_delete', description: 'Remove library entries (the user is asked to confirm; cannot be undone).',
    parameters: { type: 'object', properties: { ids: { type: 'array', items: str } }, required: ['ids'] } } },
  { type: 'function', function: { name: 'get_animation', description: 'The animation state: keyframes and the per-parameter motion curves (path, interpolation, points as [seconds, value]).', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'animate', description: 'Set a motion curve on one flame parameter (the same dotted paths apply_edits uses, e.g. camPitch, zoom, rotation, layers.0.xforms.1.weight, layers.0.xforms.0.variations.0.params.power, layers.0.final.affine.2): points [[seconds, value], …] replace the curve\'s points; remove=true deletes the curve. The timeline spans the curves\' times (and the keyframes, if any).',
    parameters: { type: 'object', properties: { path: str, points: { type: 'array', items: { type: 'array', items: num } }, interp: { type: 'string', enum: ['spline', 'linear', 'smooth', 'step'] }, remove: { type: 'boolean' } }, required: ['path'] } } },
  { type: 'function', function: { name: 'animation_control', description: 'Drive the animation: keyframe (store the current flame as a keyframe at the end of the timeline), play, stop.',
    parameters: { type: 'object', properties: { action: { type: 'string', enum: ['keyframe', 'play', 'stop'] } }, required: ['action'] } } },
  { type: 'function', function: { name: 'set_theme', description: 'Switch the interface theme.', parameters: { type: 'object', properties: { theme: { type: 'string', enum: ['dark', 'light'] } }, required: ['theme'] } } },
];

function parseArgs(json: string): Record<string, unknown> {
  if (!json || !json.trim()) return {};
  try { const v = JSON.parse(json); return v && typeof v === 'object' ? v as Record<string, unknown> : {}; }
  catch { throw new Error('arguments are not valid JSON'); }
}

const settle = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** After a change: let the render accumulate a little, then capture (when screenshots are on). */
async function afterChange(env: ToolEnv, text: string): Promise<ToolResult> {
  if (!env.ctx.screenshot) return { text, changed: true };
  await settle(1800);
  const image = env.screenshot() ?? undefined;
  return { text: text + (image ? ' The new render is attached.' : ''), image, changed: true };
}

export async function runTool(name: string, argsJson: string, env: ToolEnv): Promise<ToolResult> {
  const { app } = env;
  let a: Record<string, unknown>;
  try { a = parseArgs(argsJson); } catch (e) { return { text: `Error: ${(e as Error).message}` }; }
  const n = (k: string) => (typeof a[k] === 'number' && Number.isFinite(a[k]) ? a[k] as number : undefined);
  const s = (k: string) => (typeof a[k] === 'string' ? a[k] as string : undefined);
  try {
    switch (name) {
      case 'get_flame':
        // the full JSON costs thousands of tokens per call: only when the user asked for it in Send → Flame
        if (s('detail') === 'json' && env.ctx.flame !== 'json') return { text: flameSummary(app.flame, env.ctx.palette) + '\n(full JSON is off — the user\'s Send → Flame setting is "summary"; work from the summary and the edit paths.)' };
        return { text: s('detail') === 'json' ? flameJSONFor(app.flame, env.ctx.palette) : flameSummary(app.flame, env.ctx.palette) };
      case 'apply_edits': {
        const edits = s('edits') ?? '';
        const r = applyEdits(app.flame, edits, app.layerIdx);
        if (r.applied > 0) app.setFlame(r.flame, 'ai');
        const errs = r.errors.length ? ` Not understood: ${r.errors.slice(0, 5).join(' · ')}${r.errors.length > 5 ? ` (+${r.errors.length - 5} more)` : ''}.` : '';
        if (r.applied === 0) return { text: `No edits applied.${errs}` };
        return afterChange(env, `Applied ${r.applied} edit${r.applied === 1 ? '' : 's'}.${errs}`);
      }
      case 'set_flame_json': {
        const obj = typeof a.flame === 'string' ? JSON.parse(a.flame as string) : a.flame;
        if (!obj || typeof obj !== 'object') return { text: 'Error: flame must be an object.' };
        app.setFlame(normalizeFlame(obj, app.activeLayer.palette), 'ai');
        return afterChange(env, 'Flame replaced.');
      }
      case 'set_camera': {
        const f = app.flame;
        const changed: string[] = [];
        const zoom = n('zoom'); if (zoom !== undefined && zoom > 0) { f.zoom = zoom; changed.push('zoom'); }
        const cx = n('centerX'); if (cx !== undefined) { f.centerX = cx; changed.push('centerX'); }
        const cy = n('centerY'); if (cy !== undefined) { f.centerY = cy; changed.push('centerY'); }
        const rot = n('rotationDeg'); if (rot !== undefined) { f.rotation = (rot * Math.PI) / 180; changed.push('rotation'); }
        const pitch = n('pitch'); if (pitch !== undefined) { f.camPitch = pitch; changed.push('pitch'); }
        const yaw = n('yaw'); if (yaw !== undefined) { f.camYaw = yaw; changed.push('yaw'); }
        const persp = n('perspective'); if (persp !== undefined) { f.camPersp = persp; changed.push('perspective'); }
        if (!changed.length) return { text: 'Nothing to change — pass zoom, centerX, centerY, rotationDeg, pitch, yaw or perspective.' };
        app.commit('ai');
        return afterChange(env, `Camera: ${changed.join(', ')} set.`);
      }
      case 'screenshot': {
        const image = env.screenshot();
        return image ? { text: 'Screenshot attached.', image } : { text: 'Could not capture the render.' };
      }
      case 'variation_lookup': {
        const q = (s('query') ?? '').toLowerCase();
        const limit = Math.max(1, Math.min(50, n('limit') ?? 15));
        const names = Object.keys(VARIATIONS).filter((k) => k.toLowerCase().includes(q)).sort();
        if (!names.length) return { text: `No variation name contains "${q}".` };
        const lines = names.slice(0, limit).map((k) => {
          const def = VARIATIONS[k] as { params?: { name: string; def: number }[]; flags?: string[] };
          const ps = (def.params ?? []).map((p) => `${p.name}=${p.def}`);
          return `${k}${ps.length ? ` — params (defaults): ${ps.join(', ')}` : ''}${def.flags?.length ? ` [${def.flags.join(', ')}]` : ''}`;
        });
        return { text: `${names.length} match${names.length === 1 ? '' : 'es'}${names.length > limit ? ` (showing ${limit})` : ''}:\n${lines.join('\n')}` };
      }
      case 'library_search': {
        const { libAll } = await import('../core/libraryStore');
        const q = (s('query') ?? '').toLowerCase();
        const limit = Math.max(1, Math.min(100, n('limit') ?? 20));
        const all = await libAll();
        const hits = q ? all.filter((e) => e.name.toLowerCase().includes(q) || (e.author ?? '').toLowerCase().includes(q) || (e.source ?? '').toLowerCase().includes(q) || (e.tags ?? []).some((t) => t.toLowerCase().includes(q)) || (q === '★' && e.fav)) : all;
        if (!hits.length) return { text: q ? `No library flame matches "${q}" (${all.length} in the library).` : 'The library is empty.' };
        const lines = hits.slice(0, limit).map((e) => `${e.id} | ${e.fav ? '★ ' : ''}${e.name}${e.author ? ` | by ${e.author}` : ''}${e.source ? ` | ${e.source}` : ''}${e.tags?.length ? ` | tags: ${e.tags.join(', ')}` : ''}`);
        return { text: `${hits.length} of ${all.length} flames match${hits.length > limit ? ` (showing ${limit})` : ''}:\n${lines.join('\n')}` };
      }
      case 'library_similar': {
        const { libAll } = await import('../core/libraryStore');
        const { flameSignature, rankSimilar } = await import('../core/similarity');
        const { sigOf } = await import('../ui/library');
        const all = await libAll();
        const id = s('id');
        const ref = id ? all.find((e) => e.id === id) : null;
        if (id && !ref) return { text: `No library entry with id "${id}".` };
        const target = ref ? sigOf(ref) : flameSignature(app.flame);
        const limit = Math.max(1, Math.min(50, n('limit') ?? 10));
        const ranked = rankSimilar(target, all.filter((e) => e.id !== ref?.id).map((e) => ({ item: e, sig: sigOf(e) })), limit);
        if (!ranked.length) return { text: 'The library has nothing to compare against.' };
        return { text: `Most similar to ${ref ? `"${ref.name}"` : 'the current flame'}:\n` + ranked.map((r) => `${r.item.id} | ${r.item.name} | ${Math.round(r.score * 100)} %`).join('\n') };
      }
      case 'library_load': {
        const { libAll } = await import('../core/libraryStore');
        const id = s('id') ?? '';
        const e = (await libAll()).find((x) => x.id === id);
        if (!e) return { text: `No library entry with id "${id}" — use library_search first.` };
        app.flameSource = e.source ?? `library: ${e.name}`;
        app.setFlame(normalizeFlame(e.flame, app.activeLayer.palette), 'ai');
        return afterChange(env, `Loaded "${e.name}".`);
      }
      case 'library_save': {
        const { libPut } = await import('../core/libraryStore');
        const name = s('name')?.trim();
        if (name) { app.flame.name = name; app.commitTone('ai'); }
        const { jpegBlob } = await import('../ui/library');
        const thumb = await app.renderer.captureSync((cv) => {
          const size = 144;
          const c = document.createElement('canvas');
          c.width = size; c.height = size;
          const g = c.getContext('2d')!;
          const sq = Math.min(cv.width, cv.height);
          g.drawImage(cv, (cv.width - sq) / 2, (cv.height - sq) / 2, sq, sq, 0, 0, size, size);
          return jpegBlob(c);
        });
        await libPut({
          id: Math.random().toString(36).slice(2), name: app.flame.name || 'untitled', date: Date.now(),
          flame: JSON.parse(JSON.stringify(app.flame)), thumb,
          ...(app.flameSource ? { source: app.flameSource } : {}), ...(app.flame.author ? { author: app.flame.author } : {}),
        });
        return { text: `Saved "${app.flame.name || 'untitled'}" to the library.` };
      }
      case 'randomize': {
        const { sampleRandomFlame } = await import('../ui/randomSampler');
        const sym = s('symmetry'), wf = s('wfield');
        const f = await sampleRandomFlame(app, { style: s('style') ?? 'any', symmetry: (['none', 'all', 'sparse', 'xaxis', 'yaxis', 'point'].includes(sym ?? '') ? sym : 'sparse') as never, wfield: (['none', 'all', 'sparse', 'basic', 'cellular', 'fractal'].includes(wf ?? '') ? wf : 'sparse') as never });
        app.flameSource = undefined;
        app.setFlame(f, 'ai');
        return afterChange(env, `Random flame "${app.flame.name}" loaded${f.postSymmetry ? ` (post symmetry ${f.postSymmetry.type})` : ''}.`);
      }
      case 'mutate': {
        const { mutateFlameWith } = await import('../ui/mutate');
        const { MUTATION_TYPES } = await import('../core/mutations');
        const type = s('type') ?? 'all';
        if (!MUTATION_TYPES.some((t) => t.id === type)) return { text: `Unknown mutation type "${type}" — one of ${MUTATION_TYPES.map((t) => t.id).join(', ')}.` };
        const { flame, applied } = mutateFlameWith(app.flame, type as never, Math.max(0.1, Math.min(3, n('strength') ?? 1)), Math.max(1, Math.min(5, Math.round(n('count') ?? 1))));
        app.setFlame(flame, 'ai');
        return afterChange(env, `Mutation applied: ${applied.join(' + ')}.`);
      }
      case 'get_engine': {
        if (!app.engine) return { text: 'Engine settings are not available in this view.' };
        return { text: JSON.stringify(app.engine.get()) };
      }
      case 'set_engine': {
        if (!app.engine) return { text: 'Engine settings are not available in this view.' };
        const p: Record<string, unknown> = {};
        for (const k of ['mode', 'speed', 'dePreview']) if (typeof a[k] === 'string') p[k] = (a[k] as string).toLowerCase();
        for (const k of ['qualityCap', 'stopAfterS', 'previewHold', 'oversample']) if (typeof a[k] === 'number') p[k] = a[k];
        for (const k of ['adaptiveBudget', 'paused', 'rerender', 'reset']) if (typeof a[k] === 'boolean') p[k] = a[k];
        const r = app.engine.set(p);
        return { text: (r.changed.length ? `Changed ${r.changed.join(', ')}. ` : 'Nothing to change. ') + 'Engine now: ' + JSON.stringify(r.state) };
      }
      case 'undo': app.undo(); return afterChange(env, 'Undone.');
      case 'redo': app.redo(); return afterChange(env, 'Redone.');
      case 'share_link': {
        const { encodeFlameLink } = await import('../core/shareLink');
        const url = await encodeFlameLink(app.flame, app.getCurves());
        return { text: `Share link (${(url.length / 1024).toFixed(1)} KB): ${url}` };
      }
      case 'export_png': {
        const w = n('width'), h = n('height');
        const hi = w !== undefined && h !== undefined && w >= 16 && h >= 16;
        const spp = Math.max(100, Math.min(10000, n('spp') ?? 2000));
        if (!env.confirm(hi ? `Render and save a ${Math.round(w)}×${Math.round(h)} PNG of "${app.flame.name || 'untitled'}" (${spp} spp)?` : `Save a PNG of "${app.flame.name || 'untitled'}"?`)) return { text: 'The user declined the export.' };
        const { saveBlob } = await import('../ui/saveFile');
        const base = (app.flame.name || 'wilderfire').replace(/[\\/:*?"<>|]+/g, '_');
        if (hi) {
          if (w > 16384 || h > 16384) return { text: 'Error: at most 16384 pixels a side.' };
          const { renderHiRes } = await import('../ui/hiresExport');
          const r = app.renderer;
          r.exporting = true;
          let blob: Blob;
          try { blob = await renderHiRes(r, app.flame, { w: Math.round(w), h: Math.round(h), spp, transparent: a.transparent === true, curves: app.getCurves() }); }
          finally { r.exporting = false; r.setFlame(app.flame); }
          const name = `${base}-${Math.round(w)}x${Math.round(h)}.png`;
          const ok = await saveBlob(blob, { suggestedName: name, description: 'PNG image', mime: 'image/png', ext: '.png' });
          return { text: ok ? `Saved ${name} (${(blob.size / 1e6).toFixed(1)} MB).` : 'The save dialog was cancelled.' };
        }
        const blob = await app.renderer.exportPNG();
        if (!blob) return { text: 'Nothing to export yet.' };
        const { pngWithFlame } = await import('../core/pngMeta');
        const ok = await saveBlob(await pngWithFlame(blob, app.flame, app.getCurves()), { suggestedName: `${base}.png`, description: 'PNG image', mime: 'image/png', ext: '.png' });
        return { text: ok ? `Saved ${base}.png.` : 'The save dialog was cancelled.' };
      }
      case 'import_flame': {
        const text = s('text') ?? '';
        if (!text.trim()) return { text: 'Error: text is empty.' };
        const { importFlameText } = await import('../core/flameXML');
        const r = importFlameText(text, app.activeLayer.palette);
        const unknown = r.unknown.length ? ` Unsupported variations skipped: ${r.unknown.slice(0, 8).join(', ')}${r.unknown.length > 8 ? '…' : ''}.` : '';
        if (a.toLibrary === true) {
          const { addFlamesToLibrary } = await import('../ui/library');
          const added = await addFlamesToLibrary(app, r.flames, undefined, r.flames.map(() => 'assistant import'));
          return { text: `Added ${added} flame${added === 1 ? '' : 's'} to the library.${unknown}` };
        }
        app.flameSource = 'assistant import';
        app.setFlame(r.flame, 'ai');
        if (r.curves.length) app.setCurves(r.curves);
        return afterChange(env, `Loaded "${r.flame.name}"${r.count > 1 ? ` (the first of ${r.count} in the text; toLibrary=true adds them all)` : ''}.${unknown}`);
      }
      case 'library_update': {
        const { libAll, libPut } = await import('../core/libraryStore');
        const id = s('id') ?? '';
        const e = (await libAll()).find((x) => x.id === id);
        if (!e) return { text: `No library entry with id "${id}" — use library_search first.` };
        const changed: string[] = [];
        const name = s('name')?.trim(); if (name && name !== e.name) { e.name = name; changed.push('name'); }
        if (typeof a.fav === 'boolean' && !!e.fav !== a.fav) { e.fav = a.fav; changed.push(a.fav ? 'favourite' : 'unfavourite'); }
        const strs = (k: string) => (Array.isArray(a[k]) ? (a[k] as unknown[]).filter((t): t is string => typeof t === 'string').map((t) => t.trim()).filter(Boolean) : []);
        const add = strs('addTags'), rm = strs('removeTags');
        if (add.length || rm.length) { const before = (e.tags ?? []).join('\0'); e.tags = [...new Set([...(e.tags ?? []).filter((t) => !rm.includes(t)), ...add])]; if (e.tags.join('\0') !== before) changed.push('tags'); }
        if (!changed.length) return { text: 'Nothing to change.' };
        await libPut(e);
        return { text: `Updated "${e.name}": ${changed.join(', ')}.${e.tags?.length ? ` Tags now: ${e.tags.join(', ')}.` : ''}` };
      }
      case 'library_delete': {
        const { libAll, libDeleteMany } = await import('../core/libraryStore');
        const ids = Array.isArray(a.ids) ? (a.ids as unknown[]).filter((t): t is string => typeof t === 'string') : [];
        const all = await libAll();
        const hits = all.filter((e) => ids.includes(e.id));
        if (!hits.length) return { text: 'No matching library entries.' };
        if (!env.confirm(`Remove ${hits.length} flame${hits.length === 1 ? '' : 's'} from the library?\n\n${hits.slice(0, 8).map((e) => '• ' + e.name).join('\n')}${hits.length > 8 ? '\n…' : ''}`)) return { text: 'The user declined the deletion.' };
        await libDeleteMany(hits.map((e) => e.id));
        return { text: `Removed ${hits.length} flame${hits.length === 1 ? '' : 's'}: ${hits.map((e) => e.name).join(', ')}.` };
      }
      case 'get_animation': {
        const curves = app.getCurves();
        const keys = app.anim?.keyCount() ?? 0;
        if (!curves.length && !keys) return { text: 'No animation: no keyframes and no motion curves.' };
        const lines = curves.map((c) => `${c.path} [${c.interp}${c.enabled === false ? ', off' : ''}]: ${c.points.map((p) => `[${+p.t.toFixed(3)}, ${+p.v.toFixed(4)}]`).join(' ')}`);
        return { text: `${keys} keyframe${keys === 1 ? '' : 's'}; ${curves.length} motion curve${curves.length === 1 ? '' : 's'}${lines.length ? ':\n' + lines.join('\n') : '.'}` };
      }
      case 'animate': {
        const path = s('path')?.trim();
        if (!path) return { text: 'Error: path is required.' };
        const { getParam } = await import('../core/motion');
        if (getParam(app.flame, path) === undefined) return { text: `Error: "${path}" is not a numeric parameter of this flame (see get_flame for the paths).` };
        const curves = app.getCurves().map((c) => ({ ...c, points: c.points.map((p) => ({ ...p })) }));
        const i = curves.findIndex((c) => c.path === path);
        if (a.remove === true) {
          if (i < 0) return { text: `No curve on ${path}.` };
          curves.splice(i, 1); app.setCurves(curves);
          return { text: `Curve on ${path} removed.` };
        }
        const pts = Array.isArray(a.points) ? (a.points as unknown[]).filter((p): p is [number, number] => Array.isArray(p) && p.length === 2 && p.every((v) => typeof v === 'number' && Number.isFinite(v))).map(([t, v]) => ({ t, v })).sort((p, q) => p.t - q.t) : null;
        const interp = s('interp');
        const c = i >= 0 ? curves[i] : { path, points: [], interp: 'spline' as const };
        if (pts) c.points = pts;
        if (interp === 'spline' || interp === 'linear' || interp === 'smooth' || interp === 'step') c.interp = interp;
        if (!c.points.length) return { text: 'Error: give points as [[seconds, value], …].' };
        if (i < 0) curves.push(c);
        app.setCurves(curves);
        return { text: `${path}: ${c.points.length} point${c.points.length === 1 ? '' : 's'} over ${+c.points[0].t.toFixed(2)}–${+c.points[c.points.length - 1].t.toFixed(2)} s (${c.interp}). ${curves.length} curve${curves.length === 1 ? '' : 's'} in total.` };
      }
      case 'animation_control': {
        if (!app.anim) return { text: 'The animation panel is not available in this view.' };
        const act = s('action');
        if (act === 'keyframe') { app.anim.addKey(); return { text: `Keyframe added (${app.anim.keyCount()} now).` }; }
        if (act === 'play') { app.anim.play(); return { text: 'Playing.' }; }
        if (act === 'stop') { app.anim.stop(); return { text: 'Stopped.' }; }
        return { text: 'Error: action must be keyframe, play or stop.' };
      }
      case 'set_theme': {
        const t = s('theme');
        if (!app.theme || (t !== 'dark' && t !== 'light')) return { text: 'Error: theme must be dark or light.' };
        app.theme.set(t);
        return { text: `Theme: ${t}.` };
      }
      default:
        return { text: `Unknown tool "${name}".` };
    }
  } catch (e) {
    return { text: `Error in ${name}: ${(e as Error).message}` };
  }
}
