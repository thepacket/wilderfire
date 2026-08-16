// Right panel — AI assistant tab, backed by OpenRouter.ai (user-supplied key).
// The model edits the flame by emitting a ```flame fenced JSON block, which we
// parse, normalize, and apply live.
import { App, el } from './common';
import { normalizeFlame, flameToJSON } from '../core/flame';
import { VARIATIONS } from '../core/variations';
import { streamChat, SUGGESTED_MODELS, type ChatMessage, type ChatPart } from '../ai/openrouter';
import { createModelPicker } from './modelPicker';

const LS_KEY = 'wilderfire.openrouter.key';
const LS_MODEL = 'wilderfire.openrouter.model';

function systemPrompt(): string {
  const varList = Object.entries(VARIATIONS)
    .map(([name, def]) => {
      const ps = (def.params ?? []).map((p) => `${p.name}=${p.def}`).join(', ');
      return ps ? `${name}(${ps})` : name;
    })
    .join(', ');
  return `You are the WilderFire assistant inside a browser-based fractal flame editor (a WebGPU port in the spirit of JWildfire / Apophysis / flam3). You see the user's current flame as JSON (and usually a screenshot of the current render) and you can change it. When a screenshot is attached, look at it and let what you actually see guide your edits.

TO APPLY CHANGES: reply with exactly one fenced code block tagged \`flame\` containing the COMPLETE updated flame JSON (not a diff). The app parses and renders it instantly. Keep prose outside the block brief and friendly. If the user only asks a question, answer without a block.

FLAME JSON SHAPE:
{
  "name": string,
  "layers": [ { "xforms": [...], "final": null | xform, "paletteStops": [...], "weight": 0-2, "visible": true } ],  // 1-8 layers; each layer has its own transforms + gradient; weight = density share
  "centerX": 0, "centerY": 0, "zoom": ~0.6-1.5, "rotation": radians,
  "brightness": ~3-4.5, "gamma": ~3-4, "gammaThreshold": ~0.04, "vibrancy": 0..1, "background": [r,g,b]  // flam3-style tone mapping
}
XFORM SHAPE: { "affine": [a,b,c,d,e,f], "post": [1,0,0,0,1,0], "weight": 0.5-1.5, "color": 0..1, "colorSpeed": 0..1, "opacity": 0..1, "xaos": [per-target multipliers, optional], "variations": [ { "name": "...", "weight": number, "params": { ... } } ], "preVariations": optional list evaluated on the affine result BEFORE the main sum (include linear 1 for a pass-through), "postVariations": optional list evaluated on the main output }
PALETTE: "paletteStops": [[pos0..1, r, g, b], ...] with 3-8 stops, rgb 0..1 — preferred way to set colors (per layer).
For simple single-layer flames you may instead put "xforms" / "final" / "paletteStops" at the top level (the app wraps them into one layer).

Affine convention: x' = a·x + b·y + c, y' = d·x + e·y + f. Keep linear parts contractive (|scale| mostly < 1) or the attractor escapes. 2-5 xforms per layer is typical. Spread "color" values across xforms for rich gradients. Layers are great for combining a structural base with a soft glow layer (low weight, blur variations) in different colors.

AVAILABLE VARIATIONS (params with defaults): ${varList}.

Design tips: pair a structural variation (linear/spherical/julian/curl) with a flavor one at lower weight; julian power should be a small integer; blur/gaussian_blur make soft glows on low-weight xforms; a "final" xform with spherical or julia wraps everything in a lens.`;
}

interface Turn { role: 'user' | 'assistant'; content: string; }

export function buildAIPanel(app: App, root: HTMLElement) {
  root.classList.add('ai-panel');

  const cfg = el('div', 'section');
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

  const visRow = el('div', 'row');
  const visChk = el('input') as HTMLInputElement;
  visChk.type = 'checkbox';
  visChk.checked = true;
  const visLab = el('label', '', ' attach render (vision models see the result)');
  visLab.prepend(visChk);
  visLab.style.color = 'var(--fg-dim)';
  const autoSel = el('select') as HTMLSelectElement;
  for (const [label, v] of [['no auto-refine', '0'], ['auto-refine ×1', '1'], ['auto-refine ×2', '2'], ['auto-refine ×3', '3']] as const) {
    const o = el('option', '', label) as HTMLOptionElement;
    o.value = v;
    autoSel.append(o);
  }
  autoSel.title = 'After applying an edit, show the model its own render and ask it to improve — this many extra rounds';
  visRow.append(visLab, autoSel);

  cfg.append(keyRow, modelRow, visRow);
  cfg.append(el('div', 'hint', 'Runs fully in your browser via OpenRouter.ai — pick a model or type any model ID. Get a key at openrouter.ai/keys.'));

  const msgs = el('div', 'ai-msgs');
  const inputRow = el('div', 'ai-input-row');
  const ta = el('textarea') as HTMLTextAreaElement;
  ta.placeholder = 'e.g. "make it look like a frozen galaxy with icy blues"';
  const sendBtn = el('button', 'primary', 'Send');
  inputRow.append(ta, sendBtn);

  root.append(cfg, msgs, inputRow);

  const history: Turn[] = [];
  let busy = false;

  const addMsg = (cls: string, text: string) => {
    const m = el('div', `ai-msg ${cls}`, text);
    msgs.append(m);
    msgs.scrollTop = msgs.scrollHeight;
    return m;
  };

  addMsg('system', 'Describe the fractal you want — the assistant edits the flame live.');

  function tryApplyFlameBlocks(text: string): boolean {
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
    text.replace(/```(?:flame|json)\s*\n[\s\S]*?```/g, '⟨flame updated⟩').trim();

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  /** One request/response round; returns whether a flame block was applied. */
  async function runTurn(q: string, shownAs: string): Promise<boolean> {
    const key = keyInp.value.trim();
    addMsg('user', shownAs);
    history.push({ role: 'user', content: q });

    const bubble = addMsg('assistant', '…');
    let acc = '';
    let applied = false;
    try {
      const finalText = `Current flame JSON:\n\`\`\`json\n${flameToJSON(app.flame)}\n\`\`\`\n\nRequest: ${q}`;
      let finalContent: string | ChatPart[] = finalText;
      if (visChk.checked) {
        try {
          const dataUrl = app.renderer.captureSync((cv) => {
            const scale = Math.min(1, 448 / Math.max(cv.width, cv.height));
            const c = document.createElement('canvas');
            c.width = Math.max(2, Math.round(cv.width * scale));
            c.height = Math.max(2, Math.round(cv.height * scale));
            c.getContext('2d')!.drawImage(cv, 0, 0, c.width, c.height);
            return c.toDataURL('image/jpeg', 0.8);
          });
          finalContent = [
            { type: 'text', text: finalText + '\n\n(Attached: a screenshot of the current render.)' },
            { type: 'image_url', image_url: { url: dataUrl } },
          ];
        } catch { /* capture failed — send text only */ }
      }
      const messages: ChatMessage[] = [
        { role: 'system', content: systemPrompt() },
        ...history.slice(0, -1),
        { role: 'user', content: finalContent },
      ];
      await streamChat({
        apiKey: key,
        model: modelPicker.value.trim() || SUGGESTED_MODELS[0],
        messages,
        onDelta: (d) => {
          acc += d;
          bubble.textContent = displayText(acc) || '…';
          msgs.scrollTop = msgs.scrollHeight;
        },
      });
      history.push({ role: 'assistant', content: acc });
      bubble.textContent = displayText(acc) || '(empty reply)';
      if (tryApplyFlameBlocks(acc)) {
        applied = true;
        const tag = el('span', 'applied', '✦ flame applied');
        bubble.append(tag);
      }
    } catch (e) {
      bubble.textContent = '⚠ ' + (e as Error).message;
      bubble.style.color = 'var(--danger)';
      history.pop();
      throw e;
    } finally {
      msgs.scrollTop = msgs.scrollHeight;
    }
    return applied;
  }

  async function send() {
    const q = ta.value.trim();
    if (!q || busy) return;
    if (!keyInp.value.trim()) {
      addMsg('system', 'Enter your OpenRouter API key first.');
      return;
    }
    busy = true;
    sendBtn.disabled = true;
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
    }
  }

  sendBtn.onclick = send;
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
}
