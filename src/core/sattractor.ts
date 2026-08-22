// sattractor3D — JWildfire's Strange3DFunc (Jesus Sosa; LGPL 2.1+, see NOTICE.md): a strange attractor integrated
// from its x/y/z formulas into a curve, swept into a `facets`-sided tube of `radius`, rendered as a mesh like
// obj_mesh_wf. This is the CPU side (AttractorCurve): the mesh it returns goes through src/core/meshes.ts.
import type { Mesh } from './meshes';
import { compileFormula, FV, type Formula } from './formula';
import { SATTRACTOR_PRESETS } from './sattractorPresets';

export interface SAttractorSpec {
  x: string; y: string; z: string;
  steps: number; radius: number; stepTime: number; facets: number;
  start: [number, number, number]; warmup: number;
  params: number[]; // param_a … param_h
}

/** The formulas an instance runs: its own resources when set, else the preset's (JWildfire refreshes the
 *  formulas from `presetId` and lets explicit resources override them). */
export function sattractorFormulas(presetId: number, res?: Record<string, string>): { x: string; y: string; z: string } {
  const own = { x: res?.xformula?.trim() ?? '', y: res?.yformula?.trim() ?? '', z: res?.zformula?.trim() ?? '' };
  if (own.x || own.y || own.z) return { x: own.x || '0.0', y: own.y || '0.0', z: own.z || '0.0' };
  const pr = SATTRACTOR_PRESETS[Math.round(presetId)];
  return pr ? { x: pr.x, y: pr.y, z: pr.z } : { x: '0.0', y: '0.0', z: '0.0' };
}

const MAX_POSITIONS = 200_000; // steps × 1000 curve samples; JWildfire has no cap, the browser needs one

/** AttractorCurve.build + getGeometry. */
export function buildSAttractorMesh(s: SAttractorSpec): Mesh {
  let fx: Formula, fy: Formula, fz: Formula;
  try { fx = compileFormula(s.x); fy = compileFormula(s.y); fz = compileFormula(s.z); }
  catch (e) { console.warn(`sattractor3D: formula not understood (${(e as Error).message}); rendering nothing`); return { pos: new Float32Array(0), idx: new Uint32Array(0) }; }
  const v = new Float64Array(12);
  for (let i = 0; i < 8; i++) v[3 + i] = s.params[i] ?? 0;
  // evaluateX/Y/Z(x, y, z, delta_t) = x + (formula)·delta_t — called in sequence, each seeing the previous update
  const evX = (x: number, y: number, z: number, dt: number) => { v[0] = x; v[1] = y; v[2] = z; v[11] = dt; return x + fx(v) * dt; };
  const evY = (x: number, y: number, z: number, dt: number) => { v[0] = x; v[1] = y; v[2] = z; v[11] = dt; return y + fy(v) * dt; };
  const evZ = (x: number, y: number, z: number, dt: number) => { v[0] = x; v[1] = y; v[2] = z; v[11] = dt; return z + fz(v) * dt; };

  const count = Math.max(1, Math.min(MAX_POSITIONS, Math.round(s.steps) * 1000));
  const facets = Math.max(3, Math.round(s.facets));
  const sub = 2; // subAdvance
  const dt = s.stepTime;
  let px = s.start[0], py = s.start[1], pz = s.start[2];
  for (let i = 0; i < s.warmup; i++) {
    px = evX(px, py, pz, dt / sub); py = evY(px, py, pz, dt / sub); pz = evZ(px, py, pz, dt / sub);
  }
  const pos = new Float64Array(count * 3), tan = new Float64Array(count * 3), nrm = new Float64Array(count * 3);
  const norm3 = (a: Float64Array, o: number) => { const l = Math.sqrt(a[o] * a[o] + a[o + 1] * a[o + 1] + a[o + 2] * a[o + 2]); a[o] /= l; a[o + 1] /= l; a[o + 2] /= l; };
  // v − (v·n / n·n) n, written into a[o..]
  const projectOnPlane = (a: Float64Array, o: number, n: Float64Array, no: number) => {
    const nn = n[no] * n[no] + n[no + 1] * n[no + 1] + n[no + 2] * n[no + 2];
    const k = (a[o] * n[no] + a[o + 1] * n[no + 1] + a[o + 2] * n[no + 2]) / nn;
    a[o] -= k * n[no]; a[o + 1] -= k * n[no + 1]; a[o + 2] -= k * n[no + 2];
  };
  for (let i = 0; i < count; i++) {
    const ox = px, oy = py, oz = pz;
    for (let j = 0; j < sub; j++) {
      px = evX(px, py, pz, dt / sub); py = evY(px, py, pz, dt / sub); pz = evZ(px, py, pz, dt / sub);
      tan[i * 3] = px - ox; tan[i * 3 + 1] = py - oy; tan[i * 3 + 2] = pz - oz;
      norm3(tan, i * 3);
    }
    pos[i * 3] = px; pos[i * 3 + 1] = py; pos[i * 3 + 2] = pz;
    if (i === 0) { nrm[0] = 1; nrm[1] = 0; nrm[2] = 0; }
    else {
      // the flow at p over a full step, made orthogonal to the tangent (JWildfire: normal = evaluate(p, t) − p)
      nrm[i * 3] = evX(px, py, pz, dt) - px; nrm[i * 3 + 1] = evY(px, py, pz, dt) - py; nrm[i * 3 + 2] = evZ(px, py, pz, dt) - pz;
    }
    projectOnPlane(nrm, i * 3, tan, i * 3);
    norm3(nrm, i * 3);
  }
  // 2D shape: `facets` points on a circle of `radius` (angle i / (facets/2) · π)
  const shape: [number, number][] = [];
  for (let i = 0; i < facets; i++) { const a = (i / (facets / 2)) * Math.PI; shape.push([Math.cos(a) * s.radius, Math.sin(a) * s.radius]); }
  const sc = facets;
  const nv = count * sc + 2;
  const out = new Float32Array(nv * 3);
  let bad = 0;
  const put = (k: number, x: number, y: number, z: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) { bad++; x = y = z = 0; }
    out[k * 3] = x; out[k * 3 + 1] = y; out[k * 3 + 2] = z;
  };
  for (let i = 0; i < count; i++) {
    // binormal = normalize(tangent × normal)
    const tx = tan[i * 3], ty = tan[i * 3 + 1], tz = tan[i * 3 + 2], nx = nrm[i * 3], ny = nrm[i * 3 + 1], nz = nrm[i * 3 + 2];
    let bx = ty * nz - tz * ny, by = tz * nx - tx * nz, bz = tx * ny - ty * nx;
    const bl = Math.sqrt(bx * bx + by * by + bz * bz); bx /= bl; by /= bl; bz /= bl;
    const x0 = pos[i * 3], y0 = pos[i * 3 + 1], z0 = pos[i * 3 + 2];
    for (let j = 0; j < sc; j++) {
      const [u, w] = shape[j];
      put(i * sc + j, x0 + u * nx + w * bx, y0 + u * ny + w * by, z0 + u * nz + w * bz);
    }
  }
  put(count * sc, pos[0], pos[1], pos[2]);
  put(count * sc + 1, pos[(count - 1) * 3], pos[(count - 1) * 3 + 1], pos[(count - 1) * 3 + 2]);
  const idx = new Uint32Array(((count - 1) * sc * 2 + sc * 2) * 3);
  let o = 0;
  const face = (a: number, b: number, c: number) => { idx[o++] = a; idx[o++] = b; idx[o++] = c; };
  for (let i = 0; i < count - 1; i++) {
    for (let j = 0; j < sc; j++) {
      const p1 = i * sc + j, p2 = j === sc - 1 ? p1 - (sc - 1) : p1 + 1;
      const p1n = p1 + sc, p2n = j === sc - 1 ? p1n - (sc - 1) : p1n + 1;
      face(p1, p2, p1n); face(p1n, p2, p2n);
    }
  }
  // end caps
  const first = count * sc, last = count * sc + 1;
  for (let j = 0; j < sc; j++) {
    face(j === sc - 1 ? 0 : j + 1, j, first);
    const p1 = (count - 1) * sc + j, p2 = j === sc - 1 ? (count - 1) * sc : p1 + 1;
    face(p1, p2, last);
  }
  if (bad) console.warn(`sattractor3D: ${bad} vertices were not finite (the attractor diverged); they were put at the origin`);
  return { pos: out, idx };
}
