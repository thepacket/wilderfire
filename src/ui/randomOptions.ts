// The three settings that shape the next random flame — JWildfire's generator style, its random post
// symmetry and its random weighting field. They are set once and remembered, so instead of standing
// permanently in the header (where they crowded out the flame name) they hang off Randomize's caret in a
// popover, built on the variation picker's .vpick-pop chrome.
import { el } from './common';

const LS_STYLE = 'wilderfire.randomStyle';
const LS_SYMM = 'wilderfire.randomSymmetry';
const LS_WFIELD = 'wilderfire.randomWField';

export interface RandomOptions {
  /** the caret button; sits right of Randomize inside .rand-split */
  root: HTMLButtonElement;
  readonly style: string;
  readonly symmetry: string;
  readonly wfield: string;
}

export function createRandomOptions(): RandomOptions {
  const styleSel = el('select') as HTMLSelectElement;
  styleSel.title = 'Style of the next random flame — "Any" picks one of JWildfire\'s generators at random; "WilderFire" is the built-in contractive randomizer';
  const symmSel = el('select') as HTMLSelectElement;
  symmSel.title = 'Symmetry of the next random flame (JWildfire\'s random symmetry generators): None, one of the three at random, sparse (a third of the flames get one), or X axis / Y axis / Point';
  const wfieldSel = el('select') as HTMLSelectElement;
  wfieldSel.title = 'Weighting field of the next random flame (JWildfire\'s random weighting-field generators): None, any noise at random, sparse (a third of the flames get one), or basic / cellular / fractal noise';

  const root = el('button', 'rand-caret', '▾') as HTMLButtonElement;
  root.type = 'button';

  const labelOf = (s: HTMLSelectElement) => s.selectedOptions[0]?.textContent ?? s.value;
  const syncTitle = () => {
    root.title = `Settings for the next random flame\nStyle: ${labelOf(styleSel)}\nSymmetry: ${labelOf(symmSel)}\nField: ${labelOf(wfieldSel)}`;
  };
  syncTitle();

  (async () => {
    const { RANDOM_STYLES } = await import('../core/randomStyles');
    const byName = [...RANDOM_STYLES].sort((a, b) => a.name.localeCompare(b.name)); // JWildfire lists its generators alphabetically
    for (const [value, label] of [['any', 'Any style'], ['wilderfire', 'WilderFire'], ...byName.map((s) => [s.id, s.name] as [string, string])]) {
      const o = el('option', '', label) as HTMLOptionElement; o.value = value; styleSel.append(o);
    }
    styleSel.value = localStorage.getItem(LS_STYLE) ?? 'any';
    if (!styleSel.value) styleSel.value = 'any';
    const { SYMMETRY_KINDS, WFIELD_KINDS } = await import('../core/mutations');
    // the popover labels the rows, so the option text is just the kind
    for (const k of SYMMETRY_KINDS) { const o = el('option', '', k.name) as HTMLOptionElement; o.value = k.id; symmSel.append(o); }
    for (const k of WFIELD_KINDS) { const o = el('option', '', k.name) as HTMLOptionElement; o.value = k.id; wfieldSel.append(o); }
    symmSel.value = localStorage.getItem(LS_SYMM) ?? 'sparse'; if (!symmSel.value) symmSel.value = 'sparse'; // JWildfire's defaults: "(All, sparse)" for both
    wfieldSel.value = localStorage.getItem(LS_WFIELD) ?? 'sparse'; if (!wfieldSel.value) wfieldSel.value = 'sparse';
    syncTitle();
  })();

  styleSel.onchange = () => { localStorage.setItem(LS_STYLE, styleSel.value); syncTitle(); };
  symmSel.onchange = () => { localStorage.setItem(LS_SYMM, symmSel.value); syncTitle(); };
  wfieldSel.onchange = () => { localStorage.setItem(LS_WFIELD, wfieldSel.value); syncTitle(); };

  let pop: HTMLElement | null = null;
  const close = () => {
    if (!pop) return;
    pop.remove(); pop = null;
    root.classList.remove('on');
    document.removeEventListener('pointerdown', onDocDown, true);
    document.removeEventListener('keydown', onKey, true);
    window.removeEventListener('resize', close);
  };
  const onDocDown = (e: PointerEvent) => {
    if (pop && !pop.contains(e.target as Node) && !root.contains(e.target as Node)) close();
  };
  const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };

  const open = () => {
    if (pop) { close(); return; }
    pop = el('div', 'vpick-pop rand-pop');
    const row = (label: string, sel: HTMLSelectElement) => {
      const r = el('label', 'rand-row');
      r.append(el('span', '', label), sel);
      return r;
    };
    pop.append(
      row('Style', styleSel),
      row('Symmetry', symmSel),
      row('Field', wfieldSel),
      el('div', 'hint', 'These shape the next flame Randomize makes; they are remembered between sessions.'),
    );
    document.body.append(pop);
    // under the caret, right-aligned with it, clamped to the viewport
    const r = root.getBoundingClientRect();
    const W = 260;
    pop.style.width = `${W}px`;
    pop.style.left = `${Math.max(8, Math.min(r.right - W, window.innerWidth - W - 8))}px`;
    pop.style.top = `${r.bottom + 4}px`;
    root.classList.add('on');
    setTimeout(() => {
      document.addEventListener('pointerdown', onDocDown, true);
      document.addEventListener('keydown', onKey, true);
      window.addEventListener('resize', close);
    }, 0);
  };
  root.onclick = open;

  return {
    root,
    get style() { return styleSel.value || 'any'; },
    get symmetry() { return symmSel.value || 'sparse'; },
    get wfield() { return wfieldSel.value || 'sparse'; },
  };
}
