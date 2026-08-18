// Render tab — "Solid" section: z-buffer surface shading (lights + materials) instead of density accumulation.
import { App, el, slider } from './common';
import { defaultSolidRender, defaultSolidLight, defaultSolidMaterial, LIGHT_DIFF_FUNCS, type RGB, type SolidRender } from '../core/flame';

const SRC = 'solid';
const MAX_LIGHTS = 4;
const MAX_MATS = 8;

const toHex = (c: number[]) => '#' + c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0')).join('');
const fromHex = (h: string): RGB => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];

export function buildSolidSection(app: App): HTMLElement {
  const sec = el('div', 'section');
  sec.append(el('h3', '', 'Solid'));

  const solid = (): SolidRender => (app.flame.solid ??= defaultSolidRender(false));

  const enRow = el('div', 'row');
  const enChk = el('input') as HTMLInputElement;
  enChk.type = 'checkbox';
  const enLab = el('label', 'check', ' Solid rendering');
  enLab.prepend(enChk);
  enLab.title = 'Shade the nearest surface of the flame with lights and materials (z-buffer) instead of accumulating density. Needs depth: 3D variations, pitch/yaw, or 3D affines.';
  enChk.onchange = () => { solid().enabled = enChk.checked; body.style.display = enChk.checked ? '' : 'none'; app.commit(SRC); };
  enRow.append(enLab);
  sec.append(enRow);

  const body = el('div');
  sec.append(body);

  // ---- lights ----
  const lHead = el('div', 'row');
  lHead.append(el('label', '', 'Light'));
  const lSel = el('select') as HTMLSelectElement;
  const lAdd = el('button', '', '+');
  lAdd.title = 'Add a light';
  const lDel = el('button', '', '−');
  lDel.title = 'Remove this light';
  lHead.append(lSel, lAdd, lDel);
  let li = 0;
  const light = () => solid().lights[li];
  const altS = slider({ label: 'Altitude', min: -180, max: 180, step: 1, value: 0, fmt: (v) => v.toFixed(0) + '°', onInput: (v) => { if (light()) { light().altitude = v; app.commitTone(SRC); } } });
  const aziS = slider({ label: 'Azimuth', min: -180, max: 180, step: 1, value: 0, fmt: (v) => v.toFixed(0) + '°', onInput: (v) => { if (light()) { light().azimuth = v; app.commitTone(SRC); } } });
  const lIntS = slider({ label: 'Intensity', min: 0, max: 2, step: 0.01, value: 0, onInput: (v) => { if (light()) { light().intensity = v; app.commitTone(SRC); } } });
  const lColRow = el('div', 'row');
  lColRow.append(el('label', '', 'Colour'));
  const lCol = el('input') as HTMLInputElement;
  lCol.type = 'color';
  lCol.addEventListener('input', () => { if (light()) { light().color = fromHex(lCol.value); app.commitTone(SRC); } });
  lColRow.append(lCol);
  const lShRow = el('div', 'row');
  const lShChk = el('input') as HTMLInputElement;
  lShChk.type = 'checkbox';
  const lShLab = el('label', 'check', ' Casts shadows');
  lShLab.prepend(lShChk);
  lShChk.onchange = () => { if (light()) { light().castShadows = lShChk.checked; app.commit(SRC); } }; // shadow maps are re-allocated: restart
  lShRow.append(lShLab);
  const lShIntS = slider({ label: 'Shadow', min: 0, max: 1, step: 0.01, value: 0.8, onInput: (v) => { if (light()) { light().shadowIntensity = v; app.commitTone(SRC); } } });
  lShIntS.root.title = 'How dark this light\'s shadow is (1 = black)';
  const lightBox = el('div');
  lightBox.append(altS.root, aziS.root, lIntS.root, lColRow, lShRow, lShIntS.root);

  const refreshLight = () => {
    const s = solid();
    lSel.replaceChildren(...s.lights.map((_, i) => { const o = el('option', '', `Light ${i + 1}`) as HTMLOptionElement; o.value = String(i); return o; }));
    li = Math.min(li, Math.max(0, s.lights.length - 1));
    lSel.value = String(li);
    const l = s.lights[li];
    lightBox.style.display = l ? '' : 'none';
    lDel.disabled = !l;
    lAdd.disabled = s.lights.length >= MAX_LIGHTS;
    if (l) { altS.set(l.altitude); aziS.set(l.azimuth); lIntS.set(l.intensity); lCol.value = toHex(l.color); lShChk.checked = l.castShadows; lShIntS.set(l.shadowIntensity); }
  };
  lSel.onchange = () => { li = parseInt(lSel.value); refreshLight(); };
  lAdd.onclick = () => { const s = solid(); if (s.lights.length >= MAX_LIGHTS) return; s.lights.push(defaultSolidLight(s.lights.length)); li = s.lights.length - 1; refreshLight(); app.commitTone(SRC); };
  lDel.onclick = () => { const s = solid(); if (!s.lights.length) return; s.lights.splice(li, 1); refreshLight(); app.commitTone(SRC); };

  // ---- materials ----
  const mHead = el('div', 'row');
  mHead.append(el('label', '', 'Material'));
  const mSel = el('select') as HTMLSelectElement;
  const mAdd = el('button', '', '+');
  mAdd.title = 'Add a material (transforms pick materials by index; index 0 is the default)';
  const mDel = el('button', '', '−');
  mDel.title = 'Remove this material';
  mHead.append(mSel, mAdd, mDel);
  let mi = 0;
  const mat = () => solid().materials[mi];
  const ambS = slider({ label: 'Ambient', min: 0, max: 2, step: 0.01, value: 0, onInput: (v) => { if (mat()) { mat().ambient = v; app.commitTone(SRC); } } });
  const difS = slider({ label: 'Diffuse', min: 0, max: 2, step: 0.01, value: 0, onInput: (v) => { if (mat()) { mat().diffuse = v; app.commitTone(SRC); } } });
  const spcS = slider({ label: 'Specular', min: 0, max: 2, step: 0.01, value: 0, onInput: (v) => { if (mat()) { mat().phong = v; app.commitTone(SRC); } } });
  const shnS = slider({ label: 'Shininess', min: 1, max: 100, step: 1, value: 24, fmt: (v) => v.toFixed(0), onInput: (v) => { if (mat()) { mat().phongSize = v; app.commitTone(SRC); } } });
  const mColRow = el('div', 'row');
  mColRow.append(el('label', '', 'Spec. colour'));
  const mCol = el('input') as HTMLInputElement;
  mCol.type = 'color';
  mCol.addEventListener('input', () => { if (mat()) { mat().phongColor = fromHex(mCol.value); app.commitTone(SRC); } });
  mColRow.append(mCol);
  const dfRow = el('div', 'row');
  dfRow.append(el('label', '', 'Falloff'));
  const dfSel = el('select') as HTMLSelectElement;
  dfSel.title = 'Diffuse response to the light angle: cos, cos², (cos+1)/2 or its square';
  for (const [k, label] of [['COSA', 'cos'], ['COSA_SQUARE', 'cos²'], ['COSA_HALVE', '(cos+1)/2'], ['COSA_HALVE_SQUARE', '((cos+1)/2)²']] as const) {
    const o = el('option', '', label) as HTMLOptionElement; o.value = k; dfSel.append(o);
  }
  dfSel.onchange = () => { if (mat() && LIGHT_DIFF_FUNCS.includes(dfSel.value as never)) { mat().diffFunc = dfSel.value as typeof LIGHT_DIFF_FUNCS[number]; app.commitTone(SRC); } };
  dfRow.append(dfSel);
  const matBox = el('div');
  matBox.append(ambS.root, difS.root, spcS.root, shnS.root, mColRow, dfRow);

  const refreshMat = () => {
    const s = solid();
    mSel.replaceChildren(...s.materials.map((_, i) => { const o = el('option', '', `Material ${i + 1}`) as HTMLOptionElement; o.value = String(i); return o; }));
    mi = Math.min(mi, Math.max(0, s.materials.length - 1));
    mSel.value = String(mi);
    const m = s.materials[mi];
    matBox.style.display = m ? '' : 'none';
    mDel.disabled = !m;
    mAdd.disabled = s.materials.length >= MAX_MATS;
    if (m) { ambS.set(m.ambient); difS.set(m.diffuse); spcS.set(m.phong); shnS.set(m.phongSize); mCol.value = toHex(m.phongColor); dfSel.value = m.diffFunc; }
  };
  mSel.onchange = () => { mi = parseInt(mSel.value); refreshMat(); };
  mAdd.onclick = () => { const s = solid(); if (s.materials.length >= MAX_MATS) return; s.materials.push(defaultSolidMaterial()); mi = s.materials.length - 1; refreshMat(); app.commitTone(SRC); };
  mDel.onclick = () => { const s = solid(); if (!s.materials.length) return; s.materials.splice(mi, 1); refreshMat(); app.commitTone(SRC); };

  // ---- ambient occlusion ----
  const aoRow = el('div', 'row');
  const aoChk = el('input') as HTMLInputElement;
  aoChk.type = 'checkbox';
  const aoLab = el('label', 'check', ' Ambient occlusion');
  aoLab.prepend(aoChk);
  aoLab.title = 'Darken creases and cavities from the depth buffer (screen-space AO)';
  aoChk.onchange = () => { solid().ao.enabled = aoChk.checked; aoBox.style.display = aoChk.checked ? '' : 'none'; app.commitTone(SRC); };
  aoRow.append(aoLab);
  const ao = () => solid().ao;
  const aoIntS = slider({ label: 'Intensity', min: 0, max: 4, step: 0.01, value: 0.6, onInput: (v) => { ao().intensity = v; app.commitTone(SRC); } });
  const aoRadS = slider({ label: 'Radius', min: 0.25, max: 40, step: 0.25, value: 4, onInput: (v) => { ao().searchRadius = v; app.commitTone(SRC); } });
  const aoBlurS = slider({ label: 'Blur', min: 0.25, max: 10, step: 0.05, value: 1.5, onInput: (v) => { ao().blurRadius = v; app.commitTone(SRC); } });
  const aoRsS = slider({ label: 'Radius steps', min: 1, max: 32, step: 1, value: 6, fmt: (v) => v.toFixed(0), onInput: (v) => { ao().radiusSamples = v; app.commitTone(SRC); } });
  const aoAsS = slider({ label: 'Directions', min: 1, max: 32, step: 1, value: 7, fmt: (v) => v.toFixed(0), onInput: (v) => { ao().azimuthSamples = v; app.commitTone(SRC); } });
  const aoFallS = slider({ label: 'Falloff', min: 0, max: 10, step: 0.05, value: 0.5, onInput: (v) => { ao().falloff = v; app.commitTone(SRC); } });
  const aoDifS = slider({ label: 'On diffuse', min: 0, max: 1, step: 0.01, value: 0.1, onInput: (v) => { ao().affectDiffuse = v; app.commitTone(SRC); } });
  const aoBox = el('div');
  aoBox.append(aoIntS.root, aoRadS.root, aoBlurS.root, aoRsS.root, aoAsS.root, aoFallS.root, aoDifS.root);
  const refreshAO = () => {
    const a = solid().ao;
    aoChk.checked = a.enabled;
    aoBox.style.display = a.enabled ? '' : 'none';
    aoIntS.set(a.intensity); aoRadS.set(a.searchRadius); aoBlurS.set(a.blurRadius); aoRsS.set(a.radiusSamples); aoAsS.set(a.azimuthSamples); aoFallS.set(a.falloff); aoDifS.set(a.affectDiffuse);
  };

  // ---- shadows ----
  const shRow = el('div', 'row');
  shRow.append(el('label', '', 'Shadows'));
  const shSel = el('select') as HTMLSelectElement;
  for (const [k, label] of [['OFF', 'Off'], ['FAST', 'Fast'], ['SMOOTH', 'Smooth']] as const) { const o = el('option', '', label) as HTMLOptionElement; o.value = k; shSel.append(o); }
  shSel.title = 'Shadow maps from the lights that cast shadows: Fast = hard, Smooth = softened over the smooth radius';
  shSel.onchange = () => { solid().shadows.type = shSel.value as 'OFF' | 'FAST' | 'SMOOTH'; shBox.style.display = shSel.value === 'OFF' ? 'none' : ''; app.commit(SRC); }; // (re)allocates the shadow maps: restart
  shRow.append(shSel);
  const sh = () => solid().shadows;
  const shSmS = slider({ label: 'Smooth radius', min: 0, max: 5, step: 0.05, value: 1, onInput: (v) => { sh().smoothRadius = v; app.commitTone(SRC); } });
  const shBiasS = slider({ label: 'Bias', min: 0, max: 0.2, step: 0.001, value: 0.01, fmt: (v) => v.toFixed(3), onInput: (v) => { sh().bias = v; app.commitTone(SRC); } });
  const shSizeRow = el('div', 'row');
  shSizeRow.append(el('label', '', 'Map size'));
  const shSizeSel = el('select') as HTMLSelectElement;
  for (const n of [512, 1024, 2048, 4096]) { const o = el('option', '', String(n)) as HTMLOptionElement; o.value = String(n); shSizeSel.append(o); }
  shSizeSel.onchange = () => { sh().mapSize = parseInt(shSizeSel.value); app.commit(SRC); };
  shSizeRow.append(shSizeSel);
  const shBox = el('div');
  shBox.append(shSmS.root, shBiasS.root, shSizeRow);
  const refreshSh = () => {
    const t = solid().shadows;
    shSel.value = t.type;
    shBox.style.display = t.type === 'OFF' ? 'none' : '';
    shSmS.set(t.smoothRadius); shBiasS.set(t.bias);
    shSizeSel.value = String([512, 1024, 2048, 4096].reduce((a, b) => (Math.abs(b - t.mapSize) < Math.abs(a - t.mapSize) ? b : a)));
  };

  body.append(lHead, lightBox, mHead, matBox, aoRow, aoBox, shRow, shBox);
  body.append(el('div', 'hint', 'Surfaces are lit by ambient + per-light diffuse and specular terms; each transform can carry a material index (files keep it). Ambient occlusion darkens the ambient term (and, by "On diffuse", the diffuse term) where the depth buffer shows nearby occluders. Shadows come from a depth map per casting light (map size × map size); Smooth averages the shadow test over the smooth radius.'));

  const refresh = () => {
    const on = !!app.flame.solid?.enabled;
    enChk.checked = on;
    body.style.display = on ? '' : 'none';
    if (on) { refreshLight(); refreshMat(); refreshAO(); refreshSh(); }
  };
  refresh();
  app.on('flame', (src) => { if (src !== SRC) refresh(); });
  return sec;
}
