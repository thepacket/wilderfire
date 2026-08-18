// Meshes for obj_mesh_primitive_wf and obj_mesh_wf (JWildfire OBJMeshPrimitiveWFFunc / OBJMeshWFFunc /
// AbstractOBJMeshWFFunc / SimpleMesh / OBJMeshUtil, ported): the 26 built-in primitives ship as compact
// binaries (public/mesh/*.bin, from JWildfire's bundled .obj files, see scripts/jwf-port/mesh2bin.ts);
// user OBJ files (obj_mesh_wf) are parsed in the browser (`parseObj`, the same reader) and kept in the
// IndexedDB mesh store under their file name — a flame's instance names one by its `obj_filename` resource
// and gets JWildfire's default ±1 cube while the name is empty or no such file was loaded (JWildfire does the
// same when its file is missing). An instance asks for (mesh, subdiv_level, smooth passes/lambda/mu); the
// mesh is loaded, subdivided + Taubin-smoothed (subdiv_level > 0) or area-distributed (subdiv_level = 0)
// exactly like JWildfire, and turned into a GPU sampler: a face CDF + flat triangle list the kernel samples
// uniformly per triangle.

import { meshGet, meshPut, meshNames, meshDelete } from './libraryStore.ts';

export const MESH_PRIMITIVES = ['ball', 'capsule', 'cone', 'diamond', 'torus', 'box', 'gear15', 'icosahedron', 'tetrahedron', 'octahedron', 'dodecahedron', 'wedge',
  'icosidodecahedron', 'cubeoctahedron', 'gears6a', 'gears6s', 'gears8a', 'gears8s', 'gears12a', 'gears12s', 'gears16a', 'gears16s', 'gears24a', 'gears24s', 'mandelbulb', 'drop'];

/** hard cap on triangles after subdivision (JWildfire allows level ≤ 6; drop at level 5 would be 12 M) */
const MAX_FACES = 600_000;

export interface Mesh { pos: Float32Array; idx: Uint32Array }
/** GPU-ready sampler: cdf[i] = cumulative face weight (0..1], tris = 9 floats per face */
export interface MeshSampler { cdf: Float32Array; tris: Float32Array; faces: number }

const ftoi = (v: number) => (v > 0 ? Math.trunc(v + 0.5) : v < 0 ? Math.trunc(v - 0.5) : 0);

/** SimpleMesh's default cube (used by JWildfire when a mesh fails to load) */
export function defaultMesh(): Mesh {
  const pos = new Float32Array([-1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1]);
  const q = [[0, 1, 2, 3], [1, 5, 6, 2], [5, 6, 7, 4], [4, 7, 3, 0], [3, 2, 6, 7], [0, 1, 5, 4]];
  const idx: number[] = [];
  for (const [a, b, c, d] of q) idx.push(a, b, c, a, c, d);
  return { pos, idx: Uint32Array.from(idx) };
}

/** OBJMeshUtil.loadMeshFromFile / SimpleMesh: `v` lines (float positions), `f` triangles + quads (fan of two),
 *  negative indices relative to the end, vertices de-duplicated at 1e-4 like SimpleMesh.addVertex. */
export function parseObj(text: string): Mesh {
  const objV: number[][] = [];
  const pos: number[] = [];
  const idx: number[] = [];
  const map = new Map<string, number>();
  const addVertex = (x: number, y: number, z: number): number => {
    const key = `${ftoi(x * 1e4)}#${ftoi(y * 1e4)}#${ftoi(z * 1e4)}`;
    const e = map.get(key);
    if (e !== undefined) return e;
    const i = pos.length / 3;
    pos.push(Math.fround(x), Math.fround(y), Math.fround(z));
    map.set(key, i);
    return i;
  };
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'v') objV.push([+t[1], +t[2], +t[3]]);
    else if (t[0] === 'f') {
      const vs = t.slice(1).map((s) => { const i = parseInt(s.split('/')[0]); return objV[i > 0 ? i - 1 : objV.length + i]; });
      if ((vs.length === 3 || vs.length === 4) && vs.every((v) => v && v.every(Number.isFinite))) {
        const ids = vs.map((v) => addVertex(v[0], v[1], v[2]));
        idx.push(ids[0], ids[1], ids[2]);
        if (vs.length === 4) idx.push(ids[0], ids[2], ids[3]);
      }
    }
  }
  return { pos: Float32Array.from(pos), idx: Uint32Array.from(idx) };
}

/** Compact binary of a mesh (the public/mesh/*.bin format; also what the user mesh store keeps). */
export function meshToBin(m: Mesh): ArrayBuffer {
  const nV = m.pos.length / 3, nF = m.idx.length / 3;
  const buf = new ArrayBuffer(12 + nV * 12 + nF * 12);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x4d455348, true); dv.setUint32(4, nV, true); dv.setUint32(8, nF, true);
  new Float32Array(buf, 12, nV * 3).set(m.pos);
  new Uint32Array(buf, 12 + nV * 12, nF * 3).set(m.idx);
  return buf;
}

export function parseMeshBin(buf: ArrayBuffer): Mesh {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x4d455348) throw new Error('not a mesh binary');
  const nV = dv.getUint32(4, true), nF = dv.getUint32(8, true);
  return { pos: new Float32Array(buf.slice(12, 12 + nV * 12)), idx: new Uint32Array(buf.slice(12 + nV * 12, 12 + nV * 12 + nF * 12)) };
}

/** SimpleMesh.interpolate: every triangle → 4 (edge midpoints), vertices de-duplicated at 1e-4 (float positions). */
export function subdivide(m: Mesh): Mesh {
  const pos: number[] = [];
  const map = new Map<string, number>();
  const add = (x: number, y: number, z: number): number => {
    const key = `${ftoi(x * 1e4)}#${ftoi(y * 1e4)}#${ftoi(z * 1e4)}`;
    const e = map.get(key);
    if (e !== undefined) return e;
    const i = pos.length / 3;
    pos.push(Math.fround(x), Math.fround(y), Math.fround(z));
    map.set(key, i);
    return i;
  };
  const idx: number[] = [];
  const P = m.pos;
  const mid = (a: number, b: number) => [Math.fround(P[a * 3] + (P[b * 3] - P[a * 3]) * 0.5), Math.fround(P[a * 3 + 1] + (P[b * 3 + 1] - P[a * 3 + 1]) * 0.5), Math.fround(P[a * 3 + 2] + (P[b * 3 + 2] - P[a * 3 + 2]) * 0.5)];
  for (let f = 0; f < m.idx.length; f += 3) {
    const a = m.idx[f], b = m.idx[f + 1], c = m.idx[f + 2];
    const v4 = mid(a, b), v5 = mid(b, c), v6 = mid(c, a);
    const n1 = add(P[a * 3], P[a * 3 + 1], P[a * 3 + 2]), n2 = add(P[b * 3], P[b * 3 + 1], P[b * 3 + 2]), n3 = add(P[c * 3], P[c * 3 + 1], P[c * 3 + 2]);
    const n4 = add(v4[0], v4[1], v4[2]), n5 = add(v5[0], v5[1], v5[2]), n6 = add(v6[0], v6[1], v6[2]);
    idx.push(n1, n4, n6, n4, n2, n5, n5, n3, n6, n4, n5, n6);
  }
  return { pos: Float32Array.from(pos), idx: Uint32Array.from(idx) };
}

/** SimpleMesh.taubinSmooth: `passes` × (laplace(lambda), laplace(mu)) over the face-adjacency neighbours. */
export function taubinSmooth(m: Mesh, passes: number, lambda: number, mu: number): void {
  const nV = m.pos.length / 3;
  // NeightboursList: unique neighbours per vertex, in first-seen order
  const nb: number[][] = Array.from({ length: nV }, () => []);
  const addN = (a: number, b: number) => { const l = nb[a]; if (!l.includes(b)) l.push(b); };
  for (let f = 0; f < m.idx.length; f += 3) {
    const a = m.idx[f], b = m.idx[f + 1], c = m.idx[f + 2];
    addN(a, b); addN(a, c); addN(b, a); addN(b, c); addN(c, a); addN(c, b);
  }
  const P = m.pos;
  const d = new Float32Array(nV * 3);
  const laplace = (strength: number) => {
    d.fill(0);
    for (let i = 0; i < nV; i++) {
      const l = nb[i];
      if (l.length > 1) {
        const w = 1 / l.length;
        let dx = 0, dy = 0, dz = 0;
        for (const n of l) { dx += (P[n * 3] - P[i * 3]) * w; dy += (P[n * 3 + 1] - P[i * 3 + 1]) * w; dz += (P[n * 3 + 2] - P[i * 3 + 2]) * w; }
        d[i * 3] = dx; d[i * 3 + 1] = dy; d[i * 3 + 2] = dz;
      }
    }
    for (let i = 0; i < nV * 3; i++) P[i] = Math.fround(P[i] + d[i] * strength); // JWildfire: float vertices
  };
  for (let p = 0; p < passes; p++) { laplace(lambda); laplace(mu); }
}

/** SimpleMesh.distributeFaces as per-face integer weights (JWildfire replicates faces min(FTOI(area/areaMin), maxFaces) times). */
export function faceWeights(m: Mesh): Float64Array {
  const nF = m.idx.length / 3;
  const w = new Float64Array(nF).fill(1);
  const areas = new Float64Array(nF);
  let amin = Infinity, amax = 0;
  const P = m.pos;
  for (let f = 0; f < nF; f++) {
    const a = m.idx[f * 3], b = m.idx[f * 3 + 1], c = m.idx[f * 3 + 2];
    const ax = P[b * 3] - P[a * 3], ay = P[b * 3 + 1] - P[a * 3 + 1], az = P[b * 3 + 2] - P[a * 3 + 2];
    const bx = P[c * 3] - P[a * 3], by = P[c * 3 + 1] - P[a * 3 + 1], bz = P[c * 3 + 2] - P[a * 3 + 2];
    const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
    const area = Math.sqrt(cx * cx + cy * cy + cz * cz);
    areas[f] = area;
    if (area < amin) amin = area;
    if (area > amax) amax = area;
  }
  if (Math.abs(amin - amax) > 1e-8) {
    const maxFaces = nF < 5000 ? 5000 : 500;
    for (let f = 0; f < nF; f++) {
      const r = areas[f] / amin;
      w[f] = Math.min(isFinite(r) ? ftoi(r) : maxFaces, maxFaces);
    }
  }
  return w;
}

/** Build the GPU sampler: cumulative normalised weights + flat triangles. */
export function buildSampler(m: Mesh, weights?: Float64Array): MeshSampler {
  const nF = m.idx.length / 3;
  const cdf = new Float32Array(nF);
  const tris = new Float32Array(nF * 9);
  let acc = 0;
  let total = 0;
  for (let f = 0; f < nF; f++) total += weights ? weights[f] : 1;
  const P = m.pos;
  for (let f = 0; f < nF; f++) {
    acc += weights ? weights[f] : 1;
    cdf[f] = acc / total;
    for (let k = 0; k < 3; k++) { const v = m.idx[f * 3 + k]; tris[f * 9 + k * 3] = P[v * 3]; tris[f * 9 + k * 3 + 1] = P[v * 3 + 1]; tris[f * 9 + k * 3 + 2] = P[v * 3 + 2]; }
  }
  if (nF) cdf[nF - 1] = 1.0001; // the search always lands
  return { cdf, tris, faces: nF };
}

/** Prepare a primitive like OBJMeshPrimitiveWFFunc.init + OBJMeshUtil.loadAndSmoothMeshFromFile. */
export function prepareMesh(base: Mesh, subdivLevel: number, smoothPasses: number, lambda: number, mu: number): MeshSampler {
  let m: Mesh = { pos: Float32Array.from(base.pos), idx: base.idx };
  const level = Math.max(0, Math.min(6, Math.round(subdivLevel)));
  if (level > 0) {
    for (let i = 0; i < level; i++) {
      if (m.idx.length / 3 * 4 > MAX_FACES) { console.warn(`obj_mesh_primitive_wf: subdivision stopped at level ${i} (${m.idx.length / 3} triangles)`); break; }
      m = subdivide(m);
      taubinSmooth(m, Math.max(0, Math.min(24, Math.round(smoothPasses))), lambda, mu);
    }
    return buildSampler(m);
  }
  return buildSampler(m, faceWeights(m));
}

/** Instance key → sampler cache + async loading. Keys: `<source>#<level>[#passes#lambda#mu]` where source is a
 *  primitive name, `default` (the cube) or `obj:<file name>@<version>` (user mesh; the version bumps on re-load
 *  so the renderer re-packs). */
export type MeshKey = string;
export function meshKey(primitive: number, subdivLevel: number, smoothPasses: number, lambda: number, mu: number): MeshKey {
  const p = Math.round(primitive);
  const name = p >= 0 && p < MESH_PRIMITIVES.length ? MESH_PRIMITIVES[p] : 'default';
  return meshKeyOf(name, subdivLevel, smoothPasses, lambda, mu);
}
function meshKeyOf(source: string, subdivLevel: number, smoothPasses: number, lambda: number, mu: number): MeshKey {
  const level = Math.max(0, Math.min(6, Math.round(subdivLevel)));
  return level > 0 ? `${source}#${level}#${Math.round(smoothPasses)}#${lambda}#${mu}` : `${source}#0`;
}
/** The key of an obj_mesh_primitive_wf / obj_mesh_wf instance (undefined for anything else). */
export function meshKeyFor(vi: { name: string; params: Record<string, number>; res?: Record<string, string> }): MeshKey | undefined {
  const P = vi.params;
  if (vi.name === 'obj_mesh_primitive_wf') return meshKey(P.primitive ?? 0, P.subdiv_level ?? 0, P.subdiv_smooth_passes ?? 12, P.subdiv_smooth_lambda ?? 0.42, P.subdiv_smooth_mu ?? -0.45);
  if (vi.name === 'obj_mesh_wf') {
    const file = vi.res?.obj_filename ?? '';
    return meshKeyOf(file ? `obj:${file}@${userMeshVersion.get(file) ?? 0}` : 'default', P.subdiv_level ?? 0, P.subdiv_smooth_passes ?? 12, P.subdiv_smooth_lambda ?? 0.42, P.subdiv_smooth_mu ?? -0.45);
  }
  return undefined;
}

const rawCache = new Map<string, Promise<Mesh>>();
const samplers = new Map<MeshKey, MeshSampler>();
const pending = new Map<MeshKey, Promise<MeshSampler>>();
const userMeshVersion = new Map<string, number>();

async function loadRaw(source: string): Promise<Mesh> {
  if (source === 'default') return defaultMesh();
  let p = rawCache.get(source);
  if (!p) {
    if (source.startsWith('obj:')) {
      const file = source.slice(4, source.lastIndexOf('@'));
      p = meshGet(file).then((bin) => {
        if (!bin) { console.warn(`obj_mesh_wf: no mesh "${file}" in the mesh store (load it in the transform editor); using the default cube`); return defaultMesh(); }
        return parseMeshBin(bin);
      }).catch((e) => { console.warn(`obj_mesh_wf: ${e}; using the default cube`); return defaultMesh(); });
    } else {
      p = fetch(`/mesh/${source}.bin`).then(async (r) => {
        if (!r.ok) throw new Error(`mesh ${source}: ${r.status}`);
        return parseMeshBin(await r.arrayBuffer());
      }).catch((e) => { console.warn(`obj_mesh_primitive_wf: ${e}; using the default cube`); return defaultMesh(); });
    }
    rawCache.set(source, p);
  }
  return p;
}

/** Parse an OBJ file and keep it in the mesh store under `name` (its file name); flames referring to that name
 *  pick it up on their next set (the key version bumps so cached samplers/packing are dropped). */
export async function storeUserMesh(name: string, objText: string): Promise<{ vertices: number; faces: number }> {
  const m = parseObj(objText);
  if (!m.idx.length) throw new Error(`"${name}": no triangles found (only v/f lines are read)`);
  await meshPut(name, meshToBin(m));
  userMeshVersion.set(name, (userMeshVersion.get(name) ?? 0) + 1);
  return { vertices: m.pos.length / 3, faces: m.idx.length / 3 };
}
export async function removeUserMesh(name: string): Promise<void> {
  await meshDelete(name);
  userMeshVersion.set(name, (userMeshVersion.get(name) ?? 0) + 1);
}
/** Names in the mesh store. */
export const userMeshNames = meshNames;

/** The sampler for a key when it is ready (synchronous lookup for the data writer). */
export function meshSampler(key: MeshKey): MeshSampler | undefined { return samplers.get(key); }

/** Start loading/preparing a key; resolves with the sampler (cached). */
export function ensureMesh(key: MeshKey): Promise<MeshSampler> {
  const have = samplers.get(key);
  if (have) return Promise.resolve(have);
  let p = pending.get(key);
  if (!p) {
    const [source, level, passes, lambda, mu] = key.split('#');
    p = loadRaw(source).then((raw) => {
      const s = prepareMesh(raw, +level, +(passes ?? 12), +(lambda ?? 0.42), +(mu ?? -0.45));
      samplers.set(key, s);
      pending.delete(key);
      return s;
    });
    pending.set(key, p);
  }
  return p;
}

/** Where each prepared sampler sits in the renderer's shared mesh buffer (set by the renderer when it packs it). */
export const meshLayout = new Map<MeshKey, { cdfBase: number; triBase: number; faces: number }>();

/** Every mesh key a flame needs (obj_mesh_primitive_wf / obj_mesh_wf instances). */
export function flameMeshKeys(flame: { layers: { xforms: XFormLike[]; final: XFormLike | null; moreFinals: XFormLike[] }[] }): MeshKey[] {
  const keys = new Set<MeshKey>();
  for (const ly of flame.layers) {
    for (const x of [...ly.xforms, ...(ly.final ? [ly.final] : []), ...ly.moreFinals]) {
      for (const list of [x.preVariations ?? [], x.variations, x.postVariations ?? []]) {
        for (const vi of list) { const k = meshKeyFor(vi); if (k) keys.add(k); }
      }
    }
  }
  return [...keys];
}
type VarLike = { name: string; params: Record<string, number>; res?: Record<string, string> };
interface XFormLike { variations: VarLike[]; preVariations?: VarLike[]; postVariations?: VarLike[] }
