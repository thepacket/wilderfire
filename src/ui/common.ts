// Tiny DOM helpers + the shared App context that panels talk to.
import type { Flame, Layer, XForm } from '../core/flame';
import type { FlameRenderer } from '../gpu/renderer';
import type { MotionCurve } from '../core/motion';

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

export type AppEvent = 'flame' | 'select' | 'tone' | 'history' | 'preview' | 'solo';

const HISTORY_MAX = 100;
const COALESCE_MS = 700;

/** Central app state; panels mutate app.flame then call commit(). */
export class App {
  flame!: Flame;
  renderer!: FlameRenderer;
  layerIdx = 0; // active layer
  selected = 0; // xform index within the active layer; -1 => final xform
  xformClipboard: XForm | null = null; // survives flame switches
  /** Provenance of the loaded flame (dropped file / zip entry / library entry's source); 💾 Save records it. */
  flameSource: string | undefined;
  /** Motion-curve bridge, registered by the Anim panel (used by .flame export/import). */
  getCurves: () => MotionCurve[] = () => [];
  setCurves: (curves: MotionCurve[]) => void = () => {};
  /** The animation timeline (registered by the Anim panel) for offscreen video renders — null when nothing is animated. */
  timeline: () => { t0: number; total: number; evalAt: (t: number) => Flame } | null = () => null;
  private listeners: { ev: AppEvent; fn: (source: string) => void }[] = [];

  // Undo/redo: JSON snapshots with slider-gesture coalescing.
  private hist: string[] = [];
  private hp = -1;
  private lastSnapSrc = '';
  private lastSnapTime = 0;

  on(ev: AppEvent, fn: (source: string) => void) {
    this.listeners.push({ ev, fn });
  }

  emit(ev: AppEvent, source = '') {
    for (const l of this.listeners) if (l.ev === ev) l.fn(source);
  }

  private snapshot(source: string) {
    const json = JSON.stringify(this.flame);
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
  setSolo(on: boolean) { this.solo = on; this.renderer.setFlame(this.renderFlame()); this.emit('solo'); }

  /** Re-arm the renderer with the current flame after an offscreen render borrowed it (pack thumbnails). */
  resumeRender() { this.renderer.setFlame(this.renderFlame()); }

  private restore() {
    this.flame = JSON.parse(this.hist[this.hp]);
    this.clampSelection();
    this.lastSnapSrc = '';
    this.renderer.setFlame(this.renderFlame());
    this.emit('flame');
    this.emit('select');
    this.emit('history');
  }

  /** Structural or numeric flame change — restarts accumulation. */
  commit(source = '') {
    this.renderer.setFlame(this.renderFlame());
    this.emit('flame', source);
    this.snapshot(source);
  }

  /** Tonemap-only change — no accumulation reset. */
  commitTone(source = '') {
    this.renderer.invalidate();
    this.emit('tone', source);
    this.snapshot(source);
  }

  /** Replace the whole flame (AI / load / randomize / preset). */
  setFlame(f: Flame, source = '') {
    this.flame = f;
    this.layerIdx = 0;
    this.selected = 0;
    this.renderer.setFlame(this.renderFlame());
    this.emit('flame', source);
    this.emit('select', source);
    this.snapshot(''); // never coalesce whole-flame replacements
  }

  select(i: number, source = '') {
    this.selected = i;
    if (this.solo) this.renderer.setFlame(this.renderFlame()); // solo follows the selection
    this.emit('select', source);
  }

  /** Transient flame (animation playback/scrub): renders + overlay only —
   *  no history entry, no panel rebuilds. */
  applyPreview(f: Flame) {
    this.flame = f;
    this.clampSelection();
    this.renderer.setFlame(f);
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
