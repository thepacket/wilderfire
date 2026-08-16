// Variation picker: a button that opens a searchable popover with type-filter
// chips, the flam3 classics pinned on top, and the full A–Z catalogue with
// JWildfire's type tags. Replaces the old 700-entry <select>.

import { el } from './common';
import { VARIATION_NAMES, CLASSIC_VARIATIONS, variationTypes } from '../core/variations';

/** chip label → JWildfire type tags it matches */
const CHIPS: [string, string[] | null][] = [
  ['All', null],
  ['Classics', ['__classic']],
  ['2D', ['2D']],
  ['3D', ['3D', 'ZTRANSFORM']],
  ['Blur', ['BLUR']],
  ['Color', ['DC']],
  ['Pre', ['PRE']],
  ['Post', ['POST']],
  ['Crop', ['CROP']],
  ['Shape', ['BASE_SHAPE']],
  ['Sim', ['SIMULATION', 'ESCAPE_TIME_FRACTAL']],
];
const TAG_LABEL: Record<string, string> = {
  '2D': '2D', '3D': '3D', BLUR: 'blur', DC: 'color', PRE: 'pre', POST: 'post', CROP: 'crop',
  BASE_SHAPE: 'shape', SIMULATION: 'sim', ESCAPE_TIME_FRACTAL: 'fractal', ZTRANSFORM: 'z',
};

const CLASSIC_SET = new Set(CLASSIC_VARIATIONS);
// picker state survives editor rebuilds
let lastChip = 'All';
let lastQuery = '';

export interface VariationPicker { root: HTMLElement; readonly value: string; set(name: string): void }

export function createVariationPicker(initial: string, onPick?: (name: string) => void): VariationPicker {
  let value = initial in VARIATION_NAMES ? initial : (VARIATION_NAMES.includes(initial) ? initial : 'linear');
  const root = el('button', 'vpick-btn');
  root.type = 'button';
  root.title = 'Choose a variation (type to search)';
  const label = el('span', 'vpick-label', value);
  root.append(label, el('span', 'vpick-caret', '▾'));

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

  const pick = (name: string) => {
    value = name;
    label.textContent = name;
    close();
    onPick?.(name);
  };

  const open = () => {
    if (pop) { close(); return; }
    pop = el('div', 'vpick-pop');
    const search = el('input') as HTMLInputElement;
    search.type = 'text';
    search.placeholder = 'Search variations…';
    search.value = lastQuery;
    search.spellcheck = false;
    const chips = el('div', 'vpick-chips');
    const list = el('div', 'vpick-list');
    pop.append(search, chips, list);

    const chipEls = new Map<string, HTMLElement>();
    for (const [name] of CHIPS) {
      const c = el('button', 'vpick-chip' + (name === lastChip ? ' on' : ''), name);
      c.type = 'button';
      c.onclick = () => { lastChip = name; for (const [n, e2] of chipEls) e2.classList.toggle('on', n === name); render(); search.focus(); };
      chipEls.set(name, c);
      chips.append(c);
    }

    const matches = (name: string, tags: string[] | null): boolean => {
      if (!tags) return true;
      if (tags[0] === '__classic') return CLASSIC_SET.has(name);
      const t = variationTypes(name);
      return tags.some((x) => t.includes(x));
    };
    const item = (name: string): HTMLElement => {
      const it = el('div', 'vpick-item' + (name === value ? ' cur' : ''));
      it.append(el('span', 'vpick-name', name));
      const tags = variationTypes(name).map((t) => TAG_LABEL[t]).filter(Boolean);
      if (tags.length) it.append(el('span', 'vpick-tags', tags.join(' · ')));
      it.onclick = () => pick(name);
      return it;
    };
    let firstMatch: string | null = null;
    const render = () => {
      list.textContent = '';
      firstMatch = null;
      const q = search.value.trim().toLowerCase();
      lastQuery = search.value;
      const tags = CHIPS.find(([n]) => n === lastChip)?.[1] ?? null;
      const all = VARIATION_NAMES.filter((n) => matches(n, tags) && (!q || n.toLowerCase().includes(q)));
      // rank: exact/prefix hits first when searching, classics before the rest
      const score = (n: string) => (q ? (n.toLowerCase() === q ? 0 : n.toLowerCase().startsWith(q) ? 1 : 2) : 0) * 2 + (CLASSIC_SET.has(n) ? 0 : 1);
      all.sort((a, b) => score(a) - score(b) || a.localeCompare(b));
      const classics = all.filter((n) => CLASSIC_SET.has(n));
      const rest = all.filter((n) => !CLASSIC_SET.has(n));
      const section = (title: string, names: string[]) => {
        if (!names.length) return;
        list.append(el('div', 'vpick-sec', `${title} · ${names.length}`));
        for (const n of names) { list.append(item(n)); firstMatch ??= n; }
      };
      if (lastChip === 'Classics' || !classics.length || !rest.length) section(lastChip === 'Classics' ? 'Classics' : 'Variations', all);
      else { section('Classics', classics); section('All', rest); }
      if (!all.length) list.append(el('div', 'vpick-empty', 'No matches'));
    };
    search.oninput = render;
    search.onkeydown = (e) => {
      if (e.key === 'Enter' && firstMatch) { e.preventDefault(); pick(firstMatch); }
      if (e.key === 'ArrowDown') { (list.querySelector('.vpick-item') as HTMLElement | null)?.focus(); }
    };
    render();

    document.body.append(pop);
    // position under the button, clamped to the viewport
    const r = root.getBoundingClientRect();
    const W = 300, H = Math.min(420, window.innerHeight - 20);
    let left = Math.min(r.left, window.innerWidth - W - 8);
    let top = r.bottom + 4;
    if (top + H > window.innerHeight - 8) top = Math.max(8, r.top - H - 4);
    pop.style.left = `${Math.max(8, left)}px`;
    pop.style.top = `${top}px`;
    pop.style.width = `${W}px`;
    pop.style.maxHeight = `${H}px`;
    setTimeout(() => {
      document.addEventListener('pointerdown', onDocDown, true);
      document.addEventListener('keydown', onKey, true);
      window.addEventListener('resize', close);
      search.focus(); search.select();
    }, 0);
    // scroll the current value into view
    pop.querySelector('.vpick-item.cur')?.scrollIntoView({ block: 'center' });
  };
  root.onclick = open;

  return {
    root,
    get value() { return value; },
    set(name: string) { if (VARIATION_NAMES.includes(name)) { value = name; label.textContent = name; } },
  };
}
