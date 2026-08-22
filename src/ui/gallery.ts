// Gallery mode: a fullscreen slideshow through library flames, rendered live by the real renderer
// at a low quality cap (so each one resolves in a second or two). Keyboard: ← → (Page Up/Down ±10,
// Home/End), Space = play/pause, F = browser fullscreen, Esc = leave (the flame shown stays loaded).
import { App, el } from './common';
import { normalizeFlame } from '../core/flame';
import type { LibEntry } from '../core/libraryStore';

const LS_GALLERY = 'wilderfire.gallery';
interface GalleryPrefs { cap: number; intervalS: number }
const loadPrefs = (): GalleryPrefs => {
  try { return { cap: 250, intervalS: 6, ...JSON.parse(localStorage.getItem(LS_GALLERY) ?? '{}') }; } catch { return { cap: 250, intervalS: 6 }; }
};

export function openGallery(app: App, entries: LibEntry[], start = 0) {
  if (!entries.length) return;
  const root = document.getElementById('app')!;
  if (root.classList.contains('gallery')) return;
  const prefs = loadPrefs();
  const savePrefs = () => localStorage.setItem(LS_GALLERY, JSON.stringify(prefs));

  // what we put back on exit
  const prevCap = app.renderer.targetQuality;
  const prevFullscreen = !!document.fullscreenElement;
  root.classList.add('gallery');
  app.renderer.targetQuality = prefs.cap;

  // ---- HUD ----
  const hud = el('div', 'gallery-hud');
  const top = el('div', 'gallery-top');
  const counter = el('span', 'gallery-counter');
  const prevBtn = el('button', 'icon', '◀'); prevBtn.title = 'Previous (←)';
  const playBtn = el('button', 'icon', '▶'); playBtn.title = 'Auto-advance (Space)';
  const nextBtn = el('button', 'icon', '▶'); nextBtn.title = 'Next (→)';
  nextBtn.textContent = '▶'; prevBtn.textContent = '◀';
  const capSel = el('select') as HTMLSelectElement;
  for (const q of [50, 100, 250, 500, 1000]) { const o = el('option', '', `${q} spp`) as HTMLOptionElement; o.value = String(q); capSel.append(o); }
  capSel.value = String(prefs.cap);
  capSel.title = 'Quality cap while in the gallery — lower is faster';
  capSel.onchange = () => { prefs.cap = parseInt(capSel.value); app.renderer.targetQuality = prefs.cap; app.renderer.invalidate(); savePrefs(); };
  const intSel = el('select') as HTMLSelectElement;
  for (const s of [3, 6, 10, 20, 40]) { const o = el('option', '', `${s} s`) as HTMLOptionElement; o.value = String(s); intSel.append(o); }
  intSel.value = String(prefs.intervalS);
  intSel.title = 'Seconds per flame when auto-advancing';
  intSel.onchange = () => { prefs.intervalS = parseInt(intSel.value); savePrefs(); if (playing) restartTimer(); };
  const fsBtn = el('button', 'icon', '⛶'); fsBtn.title = 'Browser fullscreen (F)';
  const exitBtn = el('button', 'icon', '✕'); exitBtn.title = 'Leave the gallery (Esc) — the flame shown stays loaded';
  top.append(counter, prevBtn, playBtn, nextBtn, capSel, intSel, fsBtn, exitBtn);
  const caption = el('div', 'gallery-caption');
  const capName = el('div', 'gallery-name');
  const capProv = el('div', 'gallery-prov');
  caption.append(capName, capProv);
  const help = el('div', 'gallery-help', '← → browse · Page Up/Down ±10 · Home/End · Space play/pause · F fullscreen · Esc leave (keeps this flame)');
  hud.append(top, caption, help);
  root.append(hud);

  // ---- state ----
  let i = Math.max(0, Math.min(entries.length - 1, start));
  let playing = false;
  let timer = 0;
  let hideTimer = 0;
  const show = (idx: number) => {
    i = ((idx % entries.length) + entries.length) % entries.length;
    const e = entries[i];
    app.flameSource = e.source ?? `library: ${e.name}`;
    app.setFlame(normalizeFlame(e.flame, app.activeLayer.palette), 'gallery');
    counter.textContent = `${i + 1} / ${entries.length}`;
    capName.textContent = e.name || 'untitled';
    const prov = [e.author ? 'by ' + e.author : '', e.source ?? ''].filter(Boolean).join(' · ');
    capProv.textContent = prov;
    capProv.style.display = prov ? '' : 'none';
  };
  const restartTimer = () => {
    clearInterval(timer);
    if (playing) timer = window.setInterval(() => show(i + 1), prefs.intervalS * 1000);
  };
  const setPlaying = (on: boolean) => {
    playing = on;
    playBtn.textContent = on ? '⏸' : '▶';
    playBtn.classList.toggle('active', on);
    restartTimer();
  };
  const wake = () => {
    hud.classList.remove('hidden');
    root.classList.remove('gallery-idle');
    clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => { hud.classList.add('hidden'); root.classList.add('gallery-idle'); }, 2800);
  };
  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void document.documentElement.requestFullscreen?.();
  };
  const exit = () => {
    clearInterval(timer); clearTimeout(hideTimer);
    window.removeEventListener('keydown', onKey, true);
    window.removeEventListener('pointermove', wake);
    hud.remove();
    root.classList.remove('gallery', 'gallery-idle');
    app.renderer.targetQuality = prevCap;
    app.renderer.invalidate();
    if (document.fullscreenElement && !prevFullscreen) void document.exitFullscreen();
  };
  const onKey = (ev: KeyboardEvent) => {
    if ((ev.target as HTMLElement)?.tagName === 'SELECT') return;
    switch (ev.key) {
      case 'ArrowRight': case 'ArrowDown': show(i + 1); if (playing) restartTimer(); break;
      case 'ArrowLeft': case 'ArrowUp': show(i - 1); if (playing) restartTimer(); break;
      case 'PageDown': show(i + 10); break;
      case 'PageUp': show(i - 10); break;
      case 'Home': show(0); break;
      case 'End': show(entries.length - 1); break;
      case ' ': setPlaying(!playing); break;
      case 'f': case 'F': toggleFullscreen(); break;
      case 'Escape': exit(); break;
      default: return;
    }
    ev.preventDefault();
    ev.stopPropagation();
    wake();
  };
  prevBtn.onclick = () => { show(i - 1); if (playing) restartTimer(); };
  nextBtn.onclick = () => { show(i + 1); if (playing) restartTimer(); };
  playBtn.onclick = () => setPlaying(!playing);
  fsBtn.onclick = toggleFullscreen;
  exitBtn.onclick = exit;
  window.addEventListener('keydown', onKey, true);
  window.addEventListener('pointermove', wake);
  show(i);
  wake();
  return { exit, show };
}
