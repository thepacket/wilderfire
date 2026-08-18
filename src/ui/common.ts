// Tiny DOM helpers + the shared App context that panels talk to.
import type { Flame, Layer, XForm } from '../core/flame';
import type { Composer } from '../gpu/composer';
import type { MotionCurve } from '../core/motion';
import type { Composition, CompLayer, FlameCompLayer, EscapeCompLayer } from '../core/composition';
import { wrapFlame, flameLayer, escapeLayer, MAX_COMP_LAYERS } from '../core/composition';
import type { EscapeLayerData } from '../core/escape';
import type { RGB } from '../core/flame';

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K, cls?: string, text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export interface SliderOpts {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  fmt?: (v: number) => string;
  onInput: (v: number) => void;
}

export function slider(o: SliderOpts): { root: HTMLElement; set: (v: number) => void } {
  const root = el('div', 'row');
  const lab = el('label', '', o.label);
  const inp = el('input') as HTMLInputElement;
  inp.type = 'range';
  inp.min = String(o.min);
  inp.max = String(o.max);
  inp.step = String(o.step);
  inp.value = String(o.value);
  const val = el('span', 'val');
  const fmt = o.fmt ?? ((v: number) => v.toFixed(2));
  val.textContent = fmt(o.value);
  inp.addEventListener('input', () => {
    const v = parseFloat(inp.value);
    val.textContent = fmt(v);
    o.onInput(v);
  });
  root.append(lab, inp, val);
  return {
    root,
    set: (v: number) => { inp.value = String(v); val.textContent = fmt(v); },
  };
}

export function numberInput(value: number, step: number, onChange: (v: number) => void): HTMLInputElement {
  const inp = el('input') as HTMLInputElement;
  inp.type = 'number';
  inp.step = String(step);
  inp.value = formatNum(value);
  inp.addEventListener('change', () => {
    const v = parseFloat(inp.value);
    if (isFinite(v)) onChange(v);
  });
  return inp;
}

export function formatNum(v: number): string {
  return String(Math.round(v * 10000) / 10000);
}

export type AppEvent = 'flame' | 'select' | 'tone' | 'history' | 'preview' | 'solo' | 'comp';

const HISTORY_MAX = 100;
const COALESCE_MS = 700;

/** Central app state; panels mutate app.flame (the active composition layer's flame) then call commit().
 *  The document is a Composition (an image stack); a plain flame is a one-layer composition. */
export class App {
  comp!: Composition;
  /** active composition layer (the one the panels edit and the overlay shows) */
  compIdx = 0;
  renderer!: Composer;
  layerIdx = 0; // active flame layer (inside the active composition layer's flame)
  selected = 0; // xform index within the active layer; -1 => final xform
  xformClipboard: XForm | null = null; // survives flame switches
  /** Motion-curve bridge, registered by the Anim panel (used by .flame export/import). */
  getCurves: () => MotionCurve[] = () => [];
  setCurves: (curves: MotionCurve[]) => void = () => {};
  /** The animation timeline (registered by the Anim panel) for offscreen video renders — null when nothing is animated. */
  timeline: () => { t0: number; total: number; evalAt: (t: number) => Flame } | null = () => null;
  private listeners: { ev: AppEvent; fn: (source: string) => void }[] = [];

  // Undo/redo: JSON snapshots (of the whole composition) with slider-gesture coalescing.
  private hist: string[] = [];
  private hp = -1;
  private lastSnapSrc = '';
  private lastSnapTime = 0;

  /** the selected composition layer (stack operations; an escape layer edited by the Escape tab) */
  get compLayer(): CompLayer { return this.comp.layers[this.compIdx] ?? this.comp.layers[0]; }
  /** the flame layer being edited by the flame panels: the selected layer when it is a flame, else the last flame
   *  layer that was selected (there is always at least one flame layer in a document) */
  flameIdx = 0;
  get flameLayer(): FlameCompLayer {
    const l = this.comp.layers[this.flameIdx];
    if (l?.kind === 'flame') return l;
    const i = this.comp.layers.findIndex((x) => x.kind === 'flame');
    this.flameIdx = Math.max(0, i);
    return this.comp.layers[this.flameIdx] as FlameCompLayer;
  }
  /** the flame being edited */
  get flame(): Flame { return this.flameLayer.flame; }
  set flame(f: Flame) { this.flameLayer.flame = f; }
  /** the selected escape layer, if the selection is one */
  get escapeLayer(): EscapeCompLayer | null { const l = this.compLayer; return l.kind === 'escape' ? l : null; }
  /** the palette the Gradient tab edits: the selected escape layer's, else the active flame layer's */
  get editPalette(): RGB[] { return this.escapeLayer?.escape.palette ?? this.activeLayer.palette; }
  setEditPalette(p: RGB[]) { const e = this.escapeLayer; if (e) e.escape.palette = p; else this.activeLayer.palette = p; }
  /** replace the document by a single flame (what setFlame does) */
  private wrap(f: Flame) {
    // keep the edited flame layer's id: the composer then reuses that layer's renderer instead of building a new one
    const keepId = this.comp?.layers[this.flameIdx]?.kind === 'flame' ? this.comp.layers[this.flameIdx].id : undefined;
    this.comp = wrapFlame(f);
    if (keepId) this.comp.layers[0].id = keepId;
    this.compIdx = 0;
    this.flameIdx = 0;
  }

  on(ev: AppEvent, fn: (source: string) => void) {
    this.listeners.push({ ev, fn });
  }

  emit(ev: AppEvent, source = '') {
    for (const l of this.listeners) if (l.ev === ev) l.fn(source);
  }

  private snapshot(source: string) {
    const json = JSON.stringify({ comp: this.comp, idx: this.compIdx });
    if (json === this.hist[this.hp]) return;
    const now = performance.now();
    const coalesce =
      source !== '' &&
      source === this.lastSnapSrc &&
      now - this.lastSnapTime < COALESCE_MS &&
      this.hp >= 1; // keep the gesture's starting state one step back
    if (coalesce) {
      this.hist[this.hp] = json;
    } else {
      this.hist.splice(this.hp + 1);
      this.hist.push(json);
      if (this.hist.length > HISTORY_MAX) this.hist.shift();
      this.hp = this.hist.length - 1;
    }
    this.lastSnapSrc = source;
    this.lastSnapTime = now;
    this.emit('history');
  }

  canUndo() { return this.hp > 0; }
  canRedo() { return this.hp < this.hist.length - 1; }

  undo() {
    if (!this.canUndo()) return;
    this.hp--;
    this.restore();
  }

  redo() {
    if (!this.canRedo()) return;
    this.hp++;
    this.restore();
  }

  get activeLayer(): Layer {
    return this.flame.layers[this.layerIdx] ?? this.flame.layers[0];
  }

  private clampSelection() {
    if (this.compIdx >= this.comp.layers.length) this.compIdx = 0;
    if (this.comp.layers[this.compIdx].kind === 'flame') this.flameIdx = this.compIdx;
    void this.flameLayer;
    if (this.layerIdx >= this.flame.layers.length) this.layerIdx = 0;
    const ly = this.activeLayer;
    if (this.selected >= ly.xforms.length) this.selected = 0;
    if (this.selected === -1 && !ly.final) this.selected = 0;
  }

  selectLayer(i: number, source = '') {
    this.layerIdx = Math.max(0, Math.min(i, this.flame.layers.length - 1));
    this.selected = 0;
    this.emit('select', source);
  }

  /** Solo preview: only the selected transform's points are plotted (its dynamics still run through all
   *  transforms — every other transform gets opacity 0 in the render copy). Not part of the flame/history. */
  solo = false;
  private renderFlame(): Flame {
    if (!this.solo || this.selected < 0) return this.flame;
    const f: Flame = { ...this.flame, layers: this.flame.layers.map((ly, li) => li !== this.layerIdx ? ly : { ...ly, xforms: ly.xforms.map((x, i) => (i === this.selected ? x : { ...x, opacity: 0 })) }) };
    return f;
  }
  /** the composition as rendered (solo applied to the edited flame) */
  private renderComp(): Composition {
    const rf = this.renderFlame();
    if (rf === this.flame) return this.comp;
    return { ...this.comp, layers: this.comp.layers.map((l, i) => (i === this.flameIdx && l.kind === 'flame' ? { ...l, flame: rf } : l)) };
  }
  /** push the document to the renderers (also what exports call to put the document back afterwards) */
  pushRender() { this.renderer.flameActive = this.flameIdx; void this.renderer.setComposition(this.renderComp(), this.compIdx); }
  private push() { this.pushRender(); }
  setSolo(on: boolean) { this.solo = on; this.push(); this.emit('solo'); }

  private restore() {
    const st = JSON.parse(this.hist[this.hp]) as { comp: Composition; idx: number };
    this.comp = st.comp;
    this.compIdx = st.idx ?? 0;
    this.clampSelection();
    this.lastSnapSrc = '';
    this.push();
    this.emit('comp');
    this.emit('flame');
    this.emit('select');
    this.emit('history');
  }

  /** Structural or numeric flame change — restarts accumulation (of the layers whose flame changed). */
  commit(source = '') {
    this.push();
    this.emit('flame', source);
    this.snapshot(source);
  }

  /** Tone-only change — no accumulation reset. */
  commitTone(source = '') {
    this.renderer.invalidate();
    this.emit('tone', source);
    this.snapshot(source);
  }

  /** Composition-level change (layer added/removed/reordered, blend, opacity, visibility, background). */
  commitComp(source = '') {
    this.clampSelection();
    this.push();
    this.emit('comp', source);
    this.emit('flame', source);
    this.snapshot(source);
  }

  /** Replace the flame being edited (AI / load / randomize / preset): the whole document when it is a single
   *  flame, only the edited flame layer in a layer stack (the stack is kept). */
  setFlame(f: Flame, source = '') {
    if (this.comp && this.comp.layers.length > 1) { const l = this.flameLayer; l.flame = f; l.name = f.name || l.name; }
    else this.wrap(f);
    this.layerIdx = 0;
    this.selected = 0;
    this.push();
    this.emit('comp', source);
    this.emit('flame', source);
    this.emit('select', source);
    this.snapshot(''); // never coalesce whole-flame replacements
  }

  /** Replace the whole document by a composition (load / library). */
  setComposition(c: Composition, source = '') {
    this.comp = c;
    this.compIdx = 0;
    this.flameIdx = 0;
    this.clampSelection();
    this.layerIdx = 0;
    this.selected = 0;
    this.push();
    this.emit('comp', source);
    this.emit('flame', source);
    this.emit('select', source);
    this.snapshot('');
  }

  /** Make composition layer `i` the active one (panels/overlay follow). */
  selectCompLayer(i: number, source = '') {
    this.compIdx = Math.max(0, Math.min(i, this.comp.layers.length - 1));
    if (this.comp.layers[this.compIdx].kind === 'flame') { this.flameIdx = this.compIdx; this.layerIdx = 0; this.selected = 0; }
    this.solo = false;
    this.push();
    this.emit('comp', source);
    this.emit('flame', source);
    this.emit('select', source);
  }

  /** Add a flame (or an escape-time fractal) as a new composition layer above the selected one and select it. */
  addCompLayer(content: Flame | { escape: EscapeLayerData }, source = ''): boolean {
    if (this.comp.layers.length >= MAX_COMP_LAYERS) return false;
    const layer: CompLayer = 'escape' in content
      ? escapeLayer(content.escape, { ownBackground: false, name: `Escape ${this.comp.layers.length + 1}` })
      : flameLayer(content, { ownBackground: false, name: content.name || `Layer ${this.comp.layers.length + 1}` });
    this.comp.layers.splice(this.compIdx + 1, 0, layer);
    this.compIdx++;
    if (layer.kind === 'flame') { this.flameIdx = this.compIdx; this.layerIdx = 0; this.selected = 0; }
    this.solo = false;
    this.push();
    this.emit('comp', source);
    this.emit('flame', source);
    this.emit('select', source);
    this.snapshot('');
    return true;
  }

  select(i: number, source = '') {
    this.selected = i;
    if (this.solo) this.push(); // solo follows the selection
    this.emit('select', source);
  }

  /** Transient flame (animation playback/scrub): renders + overlay only —
   *  no history entry, no panel rebuilds. */
  applyPreview(f: Flame) {
    this.flame = f;
    this.clampSelection();
    this.push();
    this.emit('preview');
  }
}

/** Simple modal dialog; returns the body container and a close function. */
export function openModal(title: string): { body: HTMLElement; close: () => void } {
  const backdrop = el('div', 'modal-backdrop');
  const box = el('div', 'modal-box');
  const head = el('div', 'modal-head');
  head.append(el('span', 'modal-title', title));
  const x = el('button', 'icon', '✕');
  head.append(x);
  const body = el('div', 'modal-body');
  box.append(head, body);
  backdrop.append(box);
  document.body.append(backdrop);
  const close = () => backdrop.remove();
  x.onclick = close;
  backdrop.addEventListener('pointerdown', (e) => { if (e.target === backdrop) close(); });
  return { body, close };
}

export const XFORM_COLORS = [
  '#ff5470', '#3cd070', '#4ba3ff', '#ffb454', '#c792ea',
  '#2ee6d6', '#ff8f40', '#7dd35f', '#ff6ac1', '#8892ff',
  '#e6d735', '#54d7ff', '#ff7070', '#70ffb0', '#d0a0ff', '#ffd070',
];
