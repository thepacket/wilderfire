// knots3D — JWildfire's Knots3DFunc (Jesus Sosa, after Jürgen Meier's knot catalogue; the cable builder is Leendert
// Ammeraal's from "Interactive 3D Computer Graphics"; LGPL 2.1+, see NOTICE.md): a closed curve (x(t), y(t), z(t)),
// t = 2π·step/steps, swept into a `facets`-sided tube of `radius` by rotating the first cross-section circle along the
// curve, rendered as a mesh like obj_mesh_wf. This is the CPU side (Knot.buildMesh, line for line); the mesh goes
// through src/core/meshes.ts. The formulas are the preset's when presetId ≥ 0 (JWildfire reads the ressources before
// the params, so the preset id overrides the text), else the instance's xformula/yformula/zformula ressources.
import type { Mesh } from './meshes';
import { compileFormula, FV, type Formula } from './formula';
import { plotFormulas } from './plots';

export interface KnotSpec {
  x: string; y: string; z: string;
  steps: number; radius: number; facets: number;
  params: number[]; // param_a … param_h
}

/** The formulas an instance runs (the preset's while presetId ≥ 0, else the ressources; an empty one is 0.0 as in Knots3DFunc). */
export function knotsFormulas(presetId: number, res?: Record<string, string>): { x: string; y: string; z: string } {
  const f = plotFormulas('knots3D', presetId, res);
  return { x: f.xformula, y: f.yformula, z: f.zformula };
}

const MAX_STEPS = 20_000; // JWildfire has no cap; presets use 500..5000

/** Knot.buildMesh: the first circle in the plane through the second curve point perpendicular to the chord of its
 *  neighbours, every next circle the previous one rotated about the axis through the intersection point of the
 *  chord planes (a translation when the direction does not change); faces between consecutive circles. */
export function buildKnotMesh(s: KnotSpec): Mesh {
  let fx: Formula, fy: Formula, fz: Formula;
  try { fx = compileFormula(s.x); fy = compileFormula(s.y); fz = compileFormula(s.z); }
  catch (e) { console.warn(`knots3D: formula not understood (${(e as Error).message}); rendering nothing`); return { pos: new Float32Array(0), idx: new Uint32Array(0) }; }
  const count = Math.max(3, Math.min(MAX_STEPS, Math.round(s.steps)));
  const n = Math.max(3, Math.round(s.facets));
  const v = new Float64Array(16);
  for (let i = 0; i < 8; i++) v[3 + i] = s.params[i] ?? 0;
  const ev = (step: number): [number, number, number] => { v[FV.t] = 2.0 * Math.PI * step / count; return [fx(v), fy(v), fz(v)]; };

  const px: number[] = [], py: number[] = [], pz: number[] = [];
  const addVertex = (x: number, y: number, z: number) => { px.push(x); py.push(y); pz.push(z); };
  // initrotate / rotate: rotation by `angle` about the axis through `start` with direction `dir` (Ammeraal)
  let r11 = 0, r12 = 0, r13 = 0, r21 = 0, r22 = 0, r23 = 0, r31 = 0, r32 = 0, r33 = 0, r41 = 0, r42 = 0, r43 = 0;
  const initrotate = (a1: number, a2: number, a3: number, v1: number, v2: number, v3: number, alpha: number) => {
    const cal = Math.cos(alpha), sal = Math.sin(alpha), cal1 = 1.0 - cal;
    const rho = Math.sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    const pi = 4.0 * Math.atan(1.0);
    let theta: number, cph: number, sph: number;
    if (rho === 0.0) { theta = 0.0; cph = 1.0; sph = 0.0; }
    else {
      if (v1 === 0.0) theta = v2 >= 0.0 ? 0.5 * pi : 1.5 * pi;
      else { theta = Math.atan(v2 / v1); if (v1 < 0) theta += pi; }
      cph = v3 / rho;
      sph = Math.sqrt(1.0 - cph * cph);
    }
    const cth = Math.cos(theta), sth = Math.sin(theta);
    const cph2 = cph * cph, sph2 = 1.0 - cph2, cth2 = cth * cth, sth2 = 1.0 - cth2;
    r11 = (cal * cph2 + sph2) * cth2 + cal * sth2;
    r12 = sal * cph + cal1 * sph2 * cth * sth;
    r13 = sph * (cph * cth * cal1 - sal * sth);
    r21 = sph2 * cth * sth * cal1 - sal * cph;
    r22 = sth2 * (cal * cph2 + sph2) + cal * cth2;
    r23 = sph * (cph * sth * cal1 + sal * cth);
    r31 = sph * (cph * cth * cal1 + sal * sth);
    r32 = sph * (cph * sth * cal1 - sal * cth);
    r33 = cal * sph2 + cph2;
    r41 = a1 - a1 * r11 - a2 * r21 - a3 * r31;
    r42 = a2 - a1 * r12 - a2 * r22 - a3 * r32;
    r43 = a3 - a1 * r13 - a2 * r23 - a3 * r33;
  };
  const rotate = (x: number, y: number, z: number): [number, number, number] => [
    x * r11 + y * r21 + z * r31 + r41,
    x * r12 + y * r22 + z * r32 + r42,
    x * r13 + y * r23 + z * r33 + r43,
  ];
  const zero = (x: number) => Math.abs(x) < 1e-5;

  const [xC0, yC0, zC0] = ev(0);
  const [xC1, yC1, zC1] = ev(1);
  const [xC2, yC2, zC2] = ev(2);
  const R = Math.abs(s.radius) < 0.01 ? 0.01 : s.radius;
  let a = xC2 - xC0, b = yC2 - yC0, c = zC2 - zC0;
  let d = a * xC1 + b * yC1 + c * zC1;
  // (rx, ry, rz): a unit vector perpendicular to (a, b, c)
  let rx: number, ry: number, rz: number;
  if (zero(a) && zero(b)) { rx = 0; ry = c; rz = -b; } else { rx = b; ry = -a; rz = 0; }
  const len = Math.sqrt(rx * rx + ry * ry + rz * rz);
  rx /= len; ry /= len; rz /= len;
  addVertex(xC1 + rx * R, yC1 + ry * R, zC1 + rz * R);
  const pi = 4.0 * Math.atan(1.0);
  const theta = 2 * pi / n;
  // the n points of the first circle
  initrotate(xC1, yC1, zC1, a, b, c, theta);
  for (let i = 1; i < n; i++) { const [x, y, z] = rotate(px[i - 1], py[i - 1], pz[i - 1]); addVertex(x, y, z); }
  const m = count;
  for (let j = 1; j < m; j++) {
    const jn0 = j * n - n;
    const [xA, yA, zA] = ev(j - 1);
    const [xB, yB, zB] = ev(j);
    const dx = xB - xA, dy = yB - yA, dz = zB - zA;
    const c1 = a * a + b * b + c * c;
    const c2 = a * dx + b * dy + c * dz;
    const c0 = d - a * xA - b * yA - c * zA;
    const xM = 0.5 * (xA + xB), yM = 0.5 * (yA + yB), zM = 0.5 * (zA + zB);
    const d0 = dx * xM + dy * yM + dz * zM;
    const e1 = dx * a + dy * b + dz * c;
    const e2 = dx * dx + dy * dy + dz * dz;
    const e0 = d0 - dx * xA - dy * yA - dz * zA;
    const denom = c1 * e2 - c2 * e1;
    if (Math.abs(denom) < 1e-12) {
      // the direction does not change: a plain copy (JWildfire translates by nothing)
      for (let i = 0; i < n; i++) addVertex(px[jn0 + i], py[jn0 + i], pz[jn0 + i]);
    } else {
      const lambda = (c0 * e2 - c2 * e0) / denom;
      const mu = (c1 * e0 - c0 * e1) / denom;
      const xP = xA + lambda * a + mu * dx, yP = yA + lambda * b + mu * dy, zP = zA + lambda * c + mu * dz;
      const xAP = xA - xP, yAP = yA - yP, zAP = zA - zP;
      const xBP = xB - xP, yBP = yB - yP, zBP = zB - zP;
      const v1 = yAP * zBP - yBP * zAP, v2 = xBP * zAP - xAP * zBP, v3 = xAP * yBP - xBP * yAP;
      const cosphi = (xAP * xBP + yAP * yBP + zAP * zBP) / Math.sqrt((xAP * xAP + yAP * yAP + zAP * zAP) * (xBP * xBP + yBP * yBP + zBP * zBP));
      const phi = cosphi === 0 ? 0.5 * pi : Math.atan(Math.sqrt(1.0 - cosphi * cosphi) / cosphi);
      initrotate(xP, yP, zP, v1, v2, v3, phi);
      for (let i = 0; i < n; i++) { const [x, y, z] = rotate(px[jn0 + i], py[jn0 + i], pz[jn0 + i]); addVertex(x, y, z); }
      initrotate(0.0, 0.0, 0.0, v1, v2, v3, phi);
      const [na, nb, nc] = rotate(a, b, c);
      a = na; b = nb; c = nc;
    }
    const [xc, yc, zc] = ev(j);
    d = a * xc + b * yc + c * zc;
  }
  const pos = new Float32Array(px.length * 3);
  let bad = 0;
  for (let i = 0; i < px.length; i++) {
    let x = px[i], y = py[i], z = pz[i];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) { bad++; x = y = z = 0; }
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
  }
  // faces between circle j−1 and circle j (JWildfire's 1-based addFace indices, minus one)
  const idx = new Uint32Array((m - 1) * n * 2 * 3);
  let o = 0;
  const face = (p: number, q: number, r: number) => { idx[o++] = p; idx[o++] = q; idx[o++] = r; };
  for (let j = 1; j < m; j++) {
    const jn = j * n + 1, jn0 = jn - n;
    for (let i = 0; i < n - 1; i++) {
      face(jn + i, jn0 + i - 1, jn0 + i);
      face(jn0 + i - 1, jn + i, jn + i - 1);
    }
    face(jn - 1, jn0 + n - 2, jn0 - 1);
    face(jn0 + n - 2, jn - 1, jn + n - 2);
  }
  if (bad) console.warn(`knots3D: ${bad} vertices were not finite; they were put at the origin`);
  return { pos, idx };
}
