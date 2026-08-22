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
  { type: 'function', function: { name: 'get_flame', description: 'The current flame: a compact summary (transform paths T1…, F, variations, palette stops, camera, tone) or the complete JSON.',
    parameters: { type: 'object', properties: { detail: { type: 'string', enum: ['summary', 'json'], description: 'summary (default) or json' } } } } },
  { type: 'function', function: { name: 'apply_edits', description: 'Change the flame with edit commands, one per line (set <path> <number> · addvar <T#|F> <name> [weight] [param=value…] · delvar · addxform [weight] · delxform <T#> · palette [[t,r,g,b],…] · name <text>). Returns how many applied, any errors, and the new render when screenshots are on.',
    parameters: { type: 'object', properties: { edits: { type: 'string', description: 'the edit commands' } }, required: ['edits'] } } },
  { type: 'function', function: { name: 'set_flame_json', description: 'Replace the whole flame with complete flame JSON (same shape get_flame returns with detail=json). Use apply_edits for small changes — this costs far more tokens.',
    parameters: { type: 'object', properties: { flame: { type: 'object', description: 'the complete flame' } }, required: ['flame'] } } },
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
  { type: 'function', function: { name: 'randomize', description: 'Replace the flame with a fresh random flame (a starting point; the old one is in undo).', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'mutate', description: 'Apply one random mutation to the current flame (a variation on the theme; the old one is in undo).', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'undo', description: 'Undo the last change to the flame.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'redo', description: 'Redo the change undone last.', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'export_png', description: 'Save the current render as a PNG file — the user is asked to confirm.', parameters: { type: 'object', properties: {} } } },
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
        const thumb = app.renderer.captureSync((cv) => {
          const size = 144;
          const c = document.createElement('canvas');
          c.width = size; c.height = size;
          const g = c.getContext('2d')!;
          const sq = Math.min(cv.width, cv.height);
          g.drawImage(cv, (cv.width - sq) / 2, (cv.height - sq) / 2, sq, sq, 0, 0, size, size);
          return c.toDataURL('image/jpeg', 0.72);
        });
        await libPut({
          id: Math.random().toString(36).slice(2), name: app.flame.name || 'untitled', date: Date.now(),
          flame: JSON.parse(JSON.stringify(app.flame)), thumb,
          ...(app.flameSource ? { source: app.flameSource } : {}), ...(app.flame.author ? { author: app.flame.author } : {}),
        });
        return { text: `Saved "${app.flame.name || 'untitled'}" to the library.` };
      }
      case 'randomize': {
        const { randomFlame } = await import('../core/random');
        app.flameSource = undefined;
        app.setFlame(randomFlame(), 'ai');
        return afterChange(env, `Random flame "${app.flame.name}" loaded.`);
      }
      case 'mutate': {
        const { mutateFlame } = await import('../ui/mutate');
        app.setFlame(mutateFlame(app.flame), 'ai');
        return afterChange(env, 'Mutation applied.');
      }
      case 'undo': app.undo(); return afterChange(env, 'Undone.');
      case 'redo': app.redo(); return afterChange(env, 'Redone.');
      case 'export_png': {
        if (!env.confirm(`Save a PNG of "${app.flame.name || 'untitled'}"?`)) return { text: 'The user declined the export.' };
        const blob = await app.renderer.exportPNG();
        if (!blob) return { text: 'Nothing to export yet.' };
        const { saveBlob } = await import('../ui/saveFile');
        const base = (app.flame.name || 'wilderfire').replace(/[\\/:*?"<>|]+/g, '_');
        const ok = await saveBlob(blob, { suggestedName: `${base}.png`, description: 'PNG image', mime: 'image/png', ext: '.png' });
        return { text: ok ? `Saved ${base}.png.` : 'The save dialog was cancelled.' };
      }
      default:
        return { text: `Unknown tool "${name}".` };
    }
  } catch (e) {
    return { text: `Error in ${name}: ${(e as Error).message}` };
  }
}
