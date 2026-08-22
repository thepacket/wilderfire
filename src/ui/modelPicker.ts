// OpenRouter model picker: a button that opens a searchable popover fed by the
// live model catalogue (fetchModels), with provider chips, per-model context /
// price / vision info, and a "use as custom ID" escape hatch for anything not
// in the list. Reuses the variation picker's popover styling (.vpick-*).

import { el } from './common';
import { fetchModels, SUGGESTED_MODELS, type ORModel } from '../ai/openrouter';

export interface ModelPicker { root: HTMLElement; readonly value: string; set(id: string): void }

const PINNED_PROVIDERS = ['anthropic', 'openai', 'google', 'x-ai', 'meta-llama', 'deepseek', 'mistralai', 'qwen'];
const PROVIDER_LABEL: Record<string, string> = {
  anthropic: 'Anthropic', openai: 'OpenAI', google: 'Google', 'x-ai': 'xAI', 'meta-llama': 'Meta',
  deepseek: 'DeepSeek', mistralai: 'Mistral', qwen: 'Qwen',
};
let lastChip = 'All';

const fmtCtx = (n: number) => n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1).replace(/\.0$/, '')}M` : n >= 1000 ? `${Math.round(n / 1000)}k` : n ? String(n) : '?';
const fmtPrice = (m: ORModel) => (m.promptPerM === 0 && m.completionPerM === 0) ? 'free' : `$${trim(m.promptPerM)}/$${trim(m.completionPerM)}`;
const trim = (v: number) => v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');

export function createModelPicker(initial: string, onPick?: (id: string) => void): ModelPicker {
  let value = initial;
  let models: ORModel[] | null = null;
  let loadError: string | null = null;

  const root = el('button', 'vpick-btn mpick-btn');
  root.type = 'button';
  root.title = 'Choose an OpenRouter model (type to search, or enter any model ID)';
  const label = el('span', 'vpick-label', value);
  root.append(label, el('span', 'vpick-caret', '▾'));

  const load = async (force = false) => {
    try { models = await fetchModels({ force }); loadError = null; }
    catch (err) { loadError = String((err as Error).message ?? err); models = null; }
  };
  const loading = load();

  let pop: HTMLElement | null = null;
  const close = () => {
    if (!pop) return;
    pop.remove(); pop = null;
    document.removeEventListener('pointerdown', onDocDown, true);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('resize', close);
  };
  const onDocDown = (e: PointerEvent) => {
    if (pop && !pop.contains(e.target as Node) && e.target !== root && !root.contains(e.target as Node)) close();
  };
  const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };
  const pick = (id: string) => { value = id; label.textContent = id; close(); onPick?.(id); };

  const open = async () => {
    if (pop) { close(); return; }
    pop = el('div', 'vpick-pop mpick-pop');
    const search = el('input') as HTMLInputElement;
    search.type = 'text';
    search.placeholder = 'Search models, or type any model ID…';
    search.spellcheck = false;
    const chips = el('div', 'vpick-chips');
    const list = el('div', 'vpick-list');
    const foot = el('div', 'mpick-foot');
    pop.append(search, chips, list, foot);
    document.body.append(pop);
    const r = root.getBoundingClientRect();
    const W = Math.min(380, window.innerWidth - 16), H = Math.min(440, window.innerHeight - 20);
    let top = r.bottom + 4;
    if (top + H > window.innerHeight - 8) top = Math.max(8, r.top - H - 4);
    pop.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - W - 8))}px`;
    pop.style.top = `${top}px`;
    pop.style.width = `${W}px`;
    pop.style.maxHeight = `${H}px`;
    setTimeout(() => {
      document.addEventListener('pointerdown', onDocDown, true);
      document.addEventListener('keydown', onKey, true);
      window.addEventListener('resize', close);
      search.focus();
    }, 0);

    list.append(el('div', 'vpick-empty', 'Loading models…'));
    await loading;
    if (!pop) return;

    // chips: All + pinned providers that exist + Other
    const providers = new Set((models ?? []).map((m) => m.provider));
    const chipNames = ['All', ...PINNED_PROVIDERS.filter((p) => providers.has(p)), 'Other'];
    if (!chipNames.includes(lastChip)) lastChip = 'All';
    const chipEls = new Map<string, HTMLElement>();
    for (const name of chipNames) {
      const c = el('button', 'vpick-chip' + (name === lastChip ? ' on' : ''), PROVIDER_LABEL[name] ?? name);
      c.type = 'button';
      c.onclick = () => { lastChip = name; for (const [n, e2] of chipEls) e2.classList.toggle('on', n === name); render(); search.focus(); };
      chipEls.set(name, c);
      chips.append(c);
    }

    let firstPick: string | null = null;
    const row = (m: ORModel): HTMLElement => {
      const it = el('div', 'vpick-item mpick-item' + (m.id === value ? ' cur' : ''));
      const main = el('div', 'mpick-main');
      main.append(el('span', 'vpick-name', m.name), el('span', 'mpick-id', m.id));
      const meta = el('span', 'vpick-tags', `${fmtCtx(m.context)} ctx · ${fmtPrice(m)}${m.vision ? ' · 👁' : ''}${m.tools ? ' · 🛠' : ''}`);
      meta.title = `context ${m.context.toLocaleString()} tokens · $${m.promptPerM.toFixed(2)} in / $${m.completionPerM.toFixed(2)} out per 1M tokens${m.vision ? ' · accepts images' : ' · text only'}`;
      it.append(main, meta);
      it.onclick = () => pick(m.id);
      return it;
    };
    const customRow = (id: string): HTMLElement => {
      const it = el('div', 'vpick-item mpick-item');
      const main = el('div', 'mpick-main');
      main.append(el('span', 'vpick-name', `Use “${id}”`), el('span', 'mpick-id', 'custom model ID — sent to OpenRouter as typed'));
      it.append(main);
      it.onclick = () => pick(id);
      return it;
    };
    const render = () => {
      list.textContent = '';
      firstPick = null;
      const q = search.value.trim().toLowerCase();
      const all = models ?? SUGGESTED_MODELS.map((id) => ({ id, name: id, provider: id.split('/')[0], context: 0, promptPerM: 0, completionPerM: 0, vision: true, tools: true, created: 0 } as ORModel));
      const inChip = (m: ORModel) => lastChip === 'All' ? true : lastChip === 'Other' ? !PINNED_PROVIDERS.includes(m.provider) : m.provider === lastChip;
      // every whitespace-separated token must appear in the id or name ("claude 5" → claude-opus-5)
      const toks = q.split(/\s+/).filter(Boolean);
      const hay = (m: ORModel) => (m.id + ' ' + m.name).toLowerCase();
      const hits = all.filter((m) => inChip(m) && (!toks.length || toks.every((t) => hay(m).includes(t))));
      // rank: exact id, id prefix, name prefix, then newest first
      const score = (m: ORModel) => !q ? 0 : m.id.toLowerCase() === q ? 0 : m.id.toLowerCase().startsWith(q) ? 1 : m.name.toLowerCase().startsWith(q) ? 2 : 3;
      hits.sort((a, b) => score(a) - score(b) || b.created - a.created || a.id.localeCompare(b.id));
      const exact = !!q && all.some((m) => m.id.toLowerCase() === q);
      if (q && !exact && q.includes('/')) { list.append(customRow(search.value.trim())); firstPick = search.value.trim(); }
      if (models === null) list.append(el('div', 'vpick-sec', loadError ? `Live list unavailable (${loadError}) — showing fallbacks` : 'Fallback list'));
      else list.append(el('div', 'vpick-sec', `${lastChip === 'All' ? 'Models' : (PROVIDER_LABEL[lastChip] ?? lastChip)} · ${hits.length}`));
      for (const m of hits) { list.append(row(m)); firstPick ??= m.id; }
      if (!hits.length && !(q && !exact)) list.append(el('div', 'vpick-empty', 'No matches'));
      else if (!hits.length && q && !q.includes('/')) list.append(el('div', 'vpick-empty', 'No matches — a custom ID needs a provider prefix, e.g. vendor/model'));
    };
    search.oninput = render;
    search.onkeydown = (e) => { if (e.key === 'Enter' && firstPick) { e.preventDefault(); pick(firstPick); } };
    render();
    pop.querySelector('.vpick-item.cur')?.scrollIntoView({ block: 'center' });

    // footer: cache age + refresh
    const refresh = el('button', 'icon', '↻ refresh list');
    refresh.type = 'button';
    refresh.onclick = async () => { refresh.disabled = true; refresh.textContent = 'refreshing…'; await load(true); refresh.disabled = false; refresh.textContent = '↻ refresh list'; render(); };
    foot.append(el('span', 'hint', models ? `${models.length} models from openrouter.ai` : 'offline fallback'), refresh);
  };
  root.onclick = () => { void open(); };

  return {
    root,
    get value() { return value; },
    set(id: string) { value = id; label.textContent = id; },
  };
}
