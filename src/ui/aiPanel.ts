// Right panel — AI assistant tab, backed by OpenRouter.ai (user-supplied key).
// The model edits the flame by emitting a ```flame fenced JSON block, which we
// parse, normalize, and apply live.
import { App, el } from './common';
import { normalizeFlame, type Flame } from '../core/flame';
import { streamChat, fetchLocalModels, SUGGESTED_MODELS, type ChatMessage, type ChatPart } from '../ai/openrouter';
import { TOOL_DEFS, MAX_TOOL_ROUNDS, runTool, type ToolEnv } from '../ai/tools';
import { createModelPicker } from './modelPicker';
import {
  DEFAULT_CONTEXT, EDITS_SPEC, applyEdits, estimateTokens, flameJSONFor, flameSummary, variationCatalogue,
  type ContextOpts, type FlameMode, type PaletteMode, type VarsMode, type ReplyMode,
} from '../ai/context';

const LS_KEY = 'wilderfire.openrouter.key';
const LS_MODEL = 'wilderfire.openrouter.model';
const LS_CTX = 'wilderfire.ai.context';
const LS_ENDPOINT = 'wilderfire.ai.endpoint';      // 'openrouter' | 'local'
const LS_LOCAL_URL = 'wilderfire.ai.localUrl';
const LS_LOCAL_MODEL = 'wilderfire.ai.localModel';

/** System prompt assembled from the user's context choices (see ../ai/context.ts). */
function systemPrompt(flame: Flame, o: ContextOpts): string {
  const intro = `You are the WilderFire assistant inside a browser-based fractal flame editor (WebGPU, .flame-compatible with flam3 / Apophysis). ` +
    (o.flame !== 'none' ? `You see the user's current flame${o.screenshot ? ' and a screenshot of the current render' : ''} and you can change it. ` : `You may get a screenshot of the current render. `) +
    (o.screenshot ? 'When a screenshot is attached, look at it and let what you actually see guide your edits. ' : '');
  const useTools = o.tools && o.reply !== 'text';
  const reply = o.reply === 'text'
    ? 'Answer questions and give advice in prose only; do NOT output flame JSON or edit blocks (the user has disabled edits).'
    : useTools
      ? `You ACT through function tools: get_flame, apply_edits, set_camera, screenshot, variation_lookup, library_search, library_load, library_save, randomize, mutate, undo, redo, export_png. ` +
        `Make changes with apply_edits; its "edits" argument is the command language below. After a change you get the result${o.screenshot ? ' and a screenshot of the new render — look at it, judge it against the request, and refine' : ''}; ` +
        `use at most ${MAX_TOOL_ROUNDS} tool rounds, then finish with a short summary of what you did. Never paste JSON or edit commands into your prose — only into tool arguments.\n` +
        EDITS_SPEC.split('\n').slice(1).join('\n')
      : o.reply === 'edits'
      ? EDITS_SPEC
      : `TO APPLY CHANGES: reply with exactly one fenced code block tagged \`flame\` containing the COMPLETE updated flame JSON (not a diff). The app parses and renders it instantly. Keep prose outside the block brief and friendly. If the user only asks a question, answer without a block.`;
  const shape = o.reply === 'json' || o.flame === 'json' ? `

FLAME JSON SHAPE:
{
  "name": string,
  "layers": [ { "xforms": [...], "final": null | xform, "paletteStops": [...], "weight": 0-2, "visible": true } ],  // 1-8 layers; each layer has its own transforms + gradient; weight = colour intensity multiplier (JWildfire semantics)
  "centerX": 0, "centerY": 0, "zoom": ~0.6-1.5, "rotation": radians,
  "brightness": ~3.5-5 (4 typical), "gamma": ~3.5-4.5 (4 typical), "gammaThreshold": ~0.01, "vibrancy": 0..1, "background": [r,g,b]  // JWildfire-style tone mapping
}
XFORM SHAPE: { "affine": [a,b,c,d,e,f], "post": [1,0,0,0,1,0], "weight": 0.5-1.5, "color": 0..1, "colorSpeed": 0..1, "opacity": 0..1, "xaos": [per-target multipliers, optional], "variations": [ { "name": "...", "weight": number, "params": { ... } } ], "preVariations": optional list evaluated on the affine result BEFORE the main sum (include linear 1 for a pass-through), "postVariations": optional list evaluated on the main output }
PALETTE: "paletteStops": [[pos0..1, r, g, b], ...] with 3-8 stops, rgb 0..1 — preferred way to set colors (per layer).
For simple single-layer flames you may instead put "xforms" / "final" / "paletteStops" at the top level (the app wraps them into one layer).

Affine convention: x' = a·x + b·y + c, y' = d·x + e·y + f.` : '';
  const vars = variationCatalogue(flame, o.vars);
  return `${intro}

${reply}${shape}

Keep linear parts of affines contractive (|scale| mostly < 1) or the attractor escapes. 2-5 xforms per layer is typical. Spread "color" values across xforms for rich gradients. Layers combine a structural base with a soft glow layer (low weight, blur variations) in different colors.
${vars ? '\n' + vars + '\n' : ''}
Design tips: pair a structural variation (linear/spherical/julian/curl) with a flavor one at lower weight; julian power should be a small integer; blur/gaussian_blur make soft glows on low-weight xforms; a "final" xform with spherical or julia wraps everything in a lens.`;
}

function loadCtx(): ContextOpts {
  try { return { ...DEFAULT_CONTEXT, ...JSON.parse(localStorage.getItem(LS_CTX) ?? '{}') }; } catch { return { ...DEFAULT_CONTEXT }; }
}

interface Turn { role: 'user' | 'assistant'; content: string; }

export function buildAIPanel(app: App, root: HTMLElement) {
  root.classList.add('ai-panel');

  const cfg = el('div', 'section');
  // ---- Endpoint: OpenRouter (default) or a local OpenAI-compatible server ----
  const epRow = el('div', 'row');
  epRow.append(el('label', '', 'Endpoint'));
  const epSel = el('select') as HTMLSelectElement;
  for (const [v, txt] of [['openrouter', 'OpenRouter.ai'], ['local', 'Local server']] as const) { const o = el('option', '', txt) as HTMLOptionElement; o.value = v; epSel.append(o); }
  epSel.value = localStorage.getItem(LS_ENDPOINT) === 'local' ? 'local' : 'openrouter';
  epSel.title = 'Local server: any OpenAI-compatible endpoint on your machine — Ollama (http://localhost:11434/v1, start it with OLLAMA_ORIGINS=* for browser access), LM Studio (http://localhost:1234/v1, enable CORS), llama.cpp, vLLM. No key needed unless the server wants one.';
  epRow.append(epSel);
  const urlRow = el('div', 'row');
  urlRow.append(el('label', '', 'Base URL'));
  const urlInp = el('input') as HTMLInputElement;
  urlInp.placeholder = 'http://localhost:11434/v1';
  urlInp.style.flex = '1';
  urlInp.value = localStorage.getItem(LS_LOCAL_URL) ?? 'http://localhost:11434/v1';
  urlInp.addEventListener('change', () => { localStorage.setItem(LS_LOCAL_URL, urlInp.value.trim()); refreshLocalModels(); });
  urlRow.append(urlInp);
  const localModelRow = el('div', 'row');
  localModelRow.append(el('label', '', 'Model'));
  const localModelInp = el('input') as HTMLInputElement;
  localModelInp.placeholder = 'model id (list loads from …/models)';
  localModelInp.style.flex = '1';
  localModelInp.setAttribute('list', 'wf-local-models');
  const localList = el('datalist') as HTMLDataListElement;
  localList.id = 'wf-local-models';
  localModelInp.value = localStorage.getItem(LS_LOCAL_MODEL) ?? '';
  localModelInp.addEventListener('change', () => localStorage.setItem(LS_LOCAL_MODEL, localModelInp.value.trim()));
  localModelRow.append(localModelInp, localList);
  const localHint = el('div', 'hint', '');
  const isLocal = () => epSel.value === 'local';
  const refreshLocalModels = async () => {
    if (!isLocal()) return;
    localList.textContent = '';
    localHint.textContent = 'Fetching model list…';
    try {
      const ids = await fetchLocalModels(urlInp.value.trim());
      for (const id of ids) { const o = el('option') as HTMLOptionElement; o.value = id; localList.append(o); }
      if (!localModelInp.value && ids[0]) { localModelInp.value = ids[0]; localStorage.setItem(LS_LOCAL_MODEL, ids[0]); }
      localHint.textContent = ids.length ? `${ids.length} model${ids.length > 1 ? 's' : ''} on the server.` : 'The server lists no models.';
    } catch (e) {
      localHint.textContent = `⚠ Could not reach ${urlInp.value.trim()}/models (${(e as Error).message}). Is the server running and allowing browser (CORS) requests? Ollama: OLLAMA_ORIGINS=* ollama serve.`;
    }
  };
  const keyRow = el('div', 'row');
  keyRow.append(el('label', '', 'API key'));
  const keyInp = el('input') as HTMLInputElement;
  keyInp.type = 'password';
  keyInp.placeholder = 'sk-or-v1-…';
  keyInp.style.flex = '1';
  keyInp.value = localStorage.getItem(LS_KEY) ?? '';
  keyInp.addEventListener('change', () => localStorage.setItem(LS_KEY, keyInp.value.trim()));
  keyRow.append(keyInp);

  const modelRow = el('div', 'row');
  modelRow.append(el('label', '', 'Model'));
  // Searchable picker fed by OpenRouter's live catalogue; any custom ID is accepted too.
  const modelPicker = createModelPicker(
    localStorage.getItem(LS_MODEL) || SUGGESTED_MODELS[0],
    (id) => localStorage.setItem(LS_MODEL, id),
  );
  modelPicker.root.style.flex = '1';
  modelPicker.root.style.maxWidth = 'none';
  modelRow.append(modelPicker.root);

  // ---- Context: what goes in, what comes back (each choice costs tokens) ----
  const ctx = loadCtx();
  const saveCtx = () => { localStorage.setItem(LS_CTX, JSON.stringify(ctx)); updateEstimate(); };
  const ctxBox = el('div', 'ai-ctx');
  const mkSel = <T extends string>(label: string, opts: [T, string][], cur: T, on: (v: T) => void, title: string) => {
    const row = el('div', 'row');
    const lab = el('label', '', label);
    lab.title = title;
    const sel = el('select') as HTMLSelectElement;
    for (const [v, txt] of opts) { const o = el('option', '', txt) as HTMLOptionElement; o.value = v; sel.append(o); }
    sel.value = cur;
    sel.onchange = () => { on(sel.value as T); saveCtx(); };
    sel.title = title;
    row.append(lab, sel);
    ctxBox.append(row);
    return sel;
  };
  const mkChk = (label: string, cur: boolean, on: (v: boolean) => void, title: string) => {
    const row = el('div', 'row');
    const chk = el('input') as HTMLInputElement;
    chk.type = 'checkbox'; chk.checked = cur;
    const lab = el('label', 'check', ' ' + label);
    lab.prepend(chk); lab.title = title;
    chk.onchange = () => { on(chk.checked); saveCtx(); };
    row.append(lab);
    ctxBox.append(row);
    return chk;
  };
  ctxBox.append(el('div', 'ai-ctx-head', 'Send'));
  mkSel<FlameMode>('Flame', [['summary', 'summary (compact)'], ['json', 'full JSON'], ['none', 'nothing']], ctx.flame, (v) => { ctx.flame = v; },
    'How the current flame is described to the model. Summary is a few hundred tokens with the paths edits use; full JSON is thousands.');
  mkSel<PaletteMode>('Palette', [['stops', '8 stops'], ['full', 'full'], ['none', 'nothing']], ctx.palette, (v) => { ctx.palette = v; },
    'Gradient detail sent with the flame. 8 stops is enough for colour work; full = 256 colours.');
  mkSel<VarsMode>('Variations', [['used', 'in use + classics'], ['all', `all (${'≈10k tokens'})`], ['none', 'nothing']], ctx.vars, (v) => { ctx.vars = v; },
    'Which variation names/params the model is told about. "All" lets it pick from the whole catalogue but costs ~10k tokens per turn.');
  const visChk = mkChk('Screenshot of the render (vision models)', ctx.screenshot, (v) => { ctx.screenshot = v; },
    'Attach a small JPEG of the current render so vision models can see what they are editing (~800 tokens).');
  mkChk('Conversation memory (earlier turns)', ctx.memory, (v) => { ctx.memory = v; },
    'Send the previous messages of this conversation. Off = every request stands alone (cheapest); use Clear to reset when on.');
  mkChk('Tools — the assistant can act (edit, camera, library, screenshot…)', ctx.tools, (v) => { ctx.tools = v; },
    `Function calling: the model edits the flame, moves the camera, searches and loads your library, randomizes, mutates, saves and exports through tools, sees each result (a screenshot when enabled) and iterates — up to ${MAX_TOOL_ROUNDS} rounds per request. Needs a model that supports tools (most do; some local servers do not). Adds ~1.5k tokens of tool descriptions per request.`);
  ctxBox.append(el('div', 'ai-ctx-head', 'Reply'));
  mkSel<ReplyMode>('Edits as', [['edits', 'edit commands (fast, cheap)'], ['json', 'complete flame JSON'], ['text', 'text only, no edits']], ctx.reply, (v) => { ctx.reply = v; },
    'Edit commands: the model only lists what changes (a few lines). Complete JSON: it re-emits the whole flame (slow, thousands of tokens). Text only: questions and advice, nothing is applied.');
  const autoRow = el('div', 'row');
  autoRow.append(el('label', '', 'Auto-refine'));
  const autoSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['off', '0'], ['×1', '1'], ['×2', '2'], ['×3', '3']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    autoSel.append(o);
  }
  autoSel.title = 'After applying an edit, show the model its own render and ask it to improve — this many extra rounds (each is a full extra request)';
  autoRow.append(autoSel);
  ctxBox.append(autoRow);
  const estimate = el('div', 'hint ai-estimate', '');
  ctxBox.append(estimate);
  const updateEstimate = () => {
    const f = app.flame;
    const sys = systemPrompt(f, ctx).length;
    const body = ctx.flame === 'json' ? flameJSONFor(f, ctx.palette).length : ctx.flame === 'summary' ? flameSummary(f, ctx.palette).length : 0;
    const hist = ctx.memory ? history.reduce((a, t) => a + t.content.length, 0) : 0;
    const tok = estimateTokens(sys + body + hist, ctx.screenshot ? 1 : 0) + (ctx.tools && ctx.reply !== 'text' ? 1500 : 0);
    const out = ctx.reply === 'json' ? '2–8k' : ctx.reply === 'edits' ? '~50–300' : '~100–500';
    estimate.textContent = `≈ ${tok >= 1000 ? (tok / 1000).toFixed(1) + 'k' : tok} tokens sent per turn · reply ${out} tokens`;
  };

  const orHint = el('div', 'hint', 'Runs fully in your browser via OpenRouter.ai — pick a model or type any model ID. Get a key at openrouter.ai/keys.');
  cfg.append(epRow, keyRow, modelRow, urlRow, localModelRow, localHint, ctxBox, orHint);
  const applyEndpoint = () => {
    const local = isLocal();
    localStorage.setItem(LS_ENDPOINT, local ? 'local' : 'openrouter');
    for (const r of [urlRow, localModelRow, localHint]) r.style.display = local ? '' : 'none';
    modelRow.style.display = local ? 'none' : '';
    orHint.style.display = local ? 'none' : '';
    keyInp.placeholder = local ? 'optional' : 'sk-or-v1-…';
    if (local) refreshLocalModels();
  };
  epSel.onchange = applyEndpoint;
  applyEndpoint();
  /** the model id for the active endpoint */
  const currentModel = () => isLocal() ? localModelInp.value.trim() : (modelPicker.value.trim() || SUGGESTED_MODELS[0]);
  /** true when a request can be sent (key present, or local endpoint with a model) — else a hint is shown */
  const readyToSend = (): boolean => {
    if (isLocal()) { if (!currentModel()) { addMsg('system', 'Enter (or pick) the local model id first.'); return false; } return true; }
    if (!keyInp.value.trim()) { addMsg('system', 'Enter your OpenRouter API key first.'); return false; }
    return true;
  };

  const msgs = el('div', 'ai-msgs');
  const inputRow = el('div', 'ai-input-row');
  const ta = el('textarea') as HTMLTextAreaElement;
  ta.placeholder = 'e.g. "what does the julian variation do, and what do power and dist control?"';
  const sendBtn = el('button', 'primary', 'Send');
  const clearBtn = el('button', '', 'Clear');
  clearBtn.title = 'Forget the conversation so far — the next message starts a fresh context (the flame is untouched)';
  const stopBtn = el('button', 'danger', 'Stop');
  stopBtn.title = 'Abort the request in progress (changes already applied stay; use undo to revert)';
  stopBtn.style.display = 'none';
  let abortCtl: AbortController | null = null;
  stopBtn.onclick = () => abortCtl?.abort();
  const explainBtn = el('button', '', 'Explain');
  explainBtn.title = 'Ask the assistant to describe the current flame — what each transform and variation contributes, how the layers, final transform, palette and camera shape the look — in prose, without changing anything';
  const btnCol = el('div', 'ai-btn-col');
  btnCol.append(explainBtn, clearBtn, stopBtn, sendBtn);
  inputRow.append(ta, btnCol);

  root.append(cfg, msgs, inputRow);

  const history: Turn[] = [];
  let busy = false;

  const addMsg = (cls: string, text: string) => {
    const m = el('div', `ai-msg ${cls}`, text);
    msgs.append(m);
    msgs.scrollTop = msgs.scrollHeight;
    return m;
  };

  addMsg('system', 'Ask about a transform or variation, or ask for a change — the assistant can edit the flame live.');

  function tryApplyFlameBlocks(text: string, c: ContextOpts = ctx): boolean {
    if (c.reply === 'text') return false;
    // edits block(s) — apply all, in order
    const er = /```edits?\s*\n([\s\S]*?)```/g;
    let em: RegExpExecArray | null;
    let editsApplied = false;
    let f = app.flame;
    const errs: string[] = [];
    let n = 0;
    while ((em = er.exec(text))) {
      const r = applyEdits(f, em[1], app.layerIdx);
      f = r.flame; n += r.applied; errs.push(...r.errors);
    }
    if (n > 0) { app.setFlame(f, 'ai'); editsApplied = true; }
    if (errs.length) addMsg('system', 'Some edits were not understood: ' + errs.slice(0, 4).join(' · ') + (errs.length > 4 ? ` (+${errs.length - 4} more)` : ''));
    if (editsApplied) return true;
    const re = /```(?:flame|json)\s*\n([\s\S]*?)```/g;
    let applied = false;
    let match: RegExpExecArray | null;
    let last: string | null = null;
    while ((match = re.exec(text))) last = match[1];
    if (last) {
      try {
        const obj = JSON.parse(last);
        if (obj && (Array.isArray(obj.layers) || Array.isArray(obj.xforms) || obj.paletteStops || obj.palette)) {
          const f = normalizeFlame(obj, app.activeLayer.palette);
          app.setFlame(f, 'ai');
          applied = true;
        }
      } catch (e) {
        console.warn('Flame block parse failed:', e);
      }
    }
    return applied;
  }

  /** Hide big JSON blocks in the visible transcript. */
  const displayText = (text: string) =>
    text.replace(/```(?:flame|json)\s*\n[\s\S]*?```/g, '⟨flame JSON⟩').replace(/```edits?\s*\n([\s\S]*?)```/g, (_m, b: string) => '⟨edits⟩\n' + b.trim()).trim();

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  /** A small JPEG of the current render (≤ 448 px) for vision models; null when it cannot be captured. */
  const captureJpeg = (): string | null => {
    try {
      return app.renderer.captureSync((cv) => {
        const scale = Math.min(1, 448 / Math.max(cv.width, cv.height));
        const c = document.createElement('canvas');
        c.width = Math.max(2, Math.round(cv.width * scale));
        c.height = Math.max(2, Math.round(cv.height * scale));
        c.getContext('2d')!.drawImage(cv, 0, 0, c.width, c.height);
        return c.toDataURL('image/jpeg', 0.8);
      });
    } catch { return null; }
  };
  const shortArgs = (json: string) => { const t = json.replace(/\s+/g, ' ').trim(); return t.length > 140 ? t.slice(0, 137) + '…' : t; };

  /** One user request: a request/response round, or — with tools on — a loop of tool calls until the
   *  model answers in prose (or the round budget runs out). Returns whether the flame was changed.
   *  `over` adjusts the context options for this turn only (Explain: prose reply, flame always described). */
  async function runTurn(q: string, shownAs: string, over: Partial<ContextOpts> = {}): Promise<boolean> {
    const key = keyInp.value.trim();
    const c: ContextOpts = { ...ctx, ...over };
    const useTools = c.tools && c.reply !== 'text';
    addMsg('user', shownAs);
    history.push({ role: 'user', content: q });

    let bubble = addMsg('assistant', '…');
    let acc = '';
    let applied = false;
    abortCtl = new AbortController();
    const env: ToolEnv = { app, ctx: c, screenshot: captureJpeg, confirm: (m) => window.confirm(m) };
    try {
      const desc = c.flame === 'json'
        ? `Current flame JSON:\n\`\`\`json\n${flameJSONFor(app.flame, c.palette)}\n\`\`\`\n\n`
        : c.flame === 'summary' ? `Current flame:\n${flameSummary(app.flame, c.palette)}\n\n` : '';
      const finalText = `${desc}Request: ${q}`;
      let finalContent: string | ChatPart[] = finalText;
      if (visChk.checked) {
        const dataUrl = captureJpeg();
        if (dataUrl) {
          finalContent = [
            { type: 'text', text: finalText + '\n\n(Attached: a screenshot of the current render.)' },
            { type: 'image_url', image_url: { url: dataUrl } },
          ];
        }
      }
      const messages: ChatMessage[] = [
        { role: 'system', content: systemPrompt(app.flame, c) },
        ...(ctx.memory ? history.slice(0, -1) : []),
        { role: 'user', content: finalContent },
      ];
      for (let round = 0; ; round++) {
        acc = '';
        const r = await streamChat({
          apiKey: key,
          baseUrl: isLocal() ? urlInp.value.trim() : undefined,
          model: currentModel(),
          messages,
          tools: useTools && round < MAX_TOOL_ROUNDS ? TOOL_DEFS : undefined,
          signal: abortCtl.signal,
          onDelta: (d) => {
            acc += d;
            bubble.textContent = displayText(acc) || '…';
            msgs.scrollTop = msgs.scrollHeight;
          },
        });
        if (!r.toolCalls.length) { acc = r.text; break; }
        // the model asked for tools: run them, feed the results (and the render) back, go again
        if (r.text.trim()) bubble.textContent = displayText(r.text); else bubble.remove();
        messages.push({ role: 'assistant', content: r.text || null, tool_calls: r.toolCalls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.arguments } })) });
        let image: string | undefined;
        for (const tc of r.toolCalls) {
          const line = addMsg('tool', `⚙ ${tc.name} ${shortArgs(tc.arguments)}`);
          const res = await runTool(tc.name, tc.arguments, env);
          line.textContent += `\n   → ${res.text.split('\n')[0].slice(0, 200)}`;
          messages.push({ role: 'tool', tool_call_id: tc.id, content: res.text });
          if (res.image) image = res.image;
          if (res.changed) applied = true;
          msgs.scrollTop = msgs.scrollHeight;
        }
        if (image) messages.push({ role: 'user', content: [{ type: 'text', text: '(The render after your tool calls is attached.)' }, { type: 'image_url', image_url: { url: image } }] });
        bubble = addMsg('assistant', '…');
      }
      history.push({ role: 'assistant', content: acc });
      bubble.textContent = displayText(acc) || (applied ? 'Done.' : '(empty reply)');
      if (tryApplyFlameBlocks(acc, c)) applied = true; // models that answer with a block instead of a tool
      if (applied) bubble.append(el('span', 'applied', '✦ flame changed'));
    } catch (e) {
      const aborted = (e as Error).name === 'AbortError';
      bubble.textContent = aborted ? '■ stopped' : '⚠ ' + (e as Error).message;
      if (!aborted) bubble.style.color = 'var(--danger)';
      history.pop();
      if (!aborted) throw e;
    } finally {
      abortCtl = null;
      msgs.scrollTop = msgs.scrollHeight;
    }
    return applied;
  }

  async function send() {
    const q = ta.value.trim();
    if (!q || busy) return;
    if (!readyToSend()) return;
    busy = true;
    sendBtn.disabled = true;
    stopBtn.style.display = '';
    ta.value = '';
    try {
      let applied = await runTurn(q, q);
      const rounds = parseInt(autoSel.value);
      for (let r = 1; r <= rounds && applied; r++) {
        if (!visChk.checked) break; // refining blind is pointless
        addMsg('system', `auto-refine ${r}/${rounds} — letting the render settle…`);
        await sleep(2600);
        applied = await runTurn(
          `This is auto-refine round ${r} of ${rounds}. The attached screenshot shows the render after your last edit. ` +
          `Critique it against the original request and output an improved COMPLETE flame JSON in a \`\`\`flame block. ` +
          `If it already looks excellent, reply with a short confirmation and no block.\n\nOriginal request: ${q}`,
          `⟲ auto-refine ${r}/${rounds}`,
        );
      }
    } catch { /* already surfaced in the bubble */ } finally {
      busy = false;
      sendBtn.disabled = false;
      stopBtn.style.display = 'none';
    }
  }

  sendBtn.onclick = send;

  const EXPLAIN_PROMPT = 'Explain this flame to me as a fractal-flame artist would. Walk through each transform: what its variations do (name the variation and what its weight and parameters contribute), how its affine (scale, rotation, offset) places it, and what its weight, colour and opacity mean for the picture. ' +
    'Then explain how the transforms interact (xaos if any), what the final transform(s) do, how the layers combine, what mood the palette gives, and how the camera and tone mapping (zoom, rotation, brightness, gamma, vibrancy) shape the look. ' +
    'Finish with 2–3 concrete tweaks worth exploring (which transform, which knob, which direction) and what to expect from each. Prose only — no JSON, no edit blocks; do not change the flame.';
  explainBtn.onclick = async () => {
    if (busy) return;
    if (!readyToSend()) return;
    busy = true;
    sendBtn.disabled = true;
    explainBtn.disabled = true;
    try {
      // the flame is always described (a summary at least) and the reply is prose, whatever the context settings say
      await runTurn(EXPLAIN_PROMPT, `✎ Explain "${app.flame.name || 'this flame'}"`, { reply: 'text', flame: ctx.flame === 'none' ? 'summary' : ctx.flame });
    } catch { /* already surfaced in the bubble */ } finally {
      busy = false;
      sendBtn.disabled = false;
      explainBtn.disabled = false;
    }
  };
  clearBtn.onclick = () => {
    if (busy) return;
    history.length = 0;
    msgs.textContent = '';
    addMsg('system', 'Context cleared — the assistant no longer remembers earlier messages.');
    updateEstimate();
  };
  updateEstimate();
  app.on('flame', () => updateEstimate());
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
}
