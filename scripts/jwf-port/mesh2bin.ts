// Converts JWildfire's bundled mesh primitives (src/org/jwildfire/create/tina/variation/mesh/*.obj, LGPL 2.1+)
// into compact binaries for obj_mesh_primitive_wf: public/mesh/<name>.bin =
//   u32 magic 0x4d455348 ('MESH'), u32 nV, u32 nF, f32 pos[nV*3], u32 idx[nF*3]
// Vertices are de-duplicated at 1e-4 like SimpleMesh.addVertex (the neighbour lists of the Taubin smoothing
// depend on it); quads become two triangles (v0,v1,v2) + (v0,v2,v3) like OBJMeshUtil.loadAndSmoothMeshFromFile.
//   usage: node scripts/jwf-port/mesh2bin.ts <jwildfire>/src/org/jwildfire/create/tina/variation/mesh
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const NAMES = ['ball', 'capsule', 'cone', 'diamond', 'torus', 'box', 'gear15', 'icosahedron', 'tetrahedron', 'octahedron', 'dodecahedron', 'wedge',
  'icosidodecahedron', 'cubeoctahedron', 'gears6a', 'gears6s', 'gears8a', 'gears8s', 'gears12a', 'gears12s', 'gears16a', 'gears16s', 'gears24a', 'gears24s', 'mandelbulb', 'drop'];
const src = process.argv[2];
if (!src) { console.error('usage: node mesh2bin.ts <jwf mesh dir>'); process.exit(1); }
mkdirSync('public/mesh', { recursive: true });
const ftoi = (v: number) => (v > 0 ? Math.trunc(v + 0.5) : v < 0 ? Math.trunc(v - 0.5) : 0);
for (const name of NAMES) {
  const text = readFileSync(join(src, name + '.obj'), 'latin1');
  const objV: number[][] = [];
  const pos: number[] = [];
  const idx: number[] = [];
  const map = new Map<string, number>();
  const addVertex = (x: number, y: number, z: number): number => {
    const key = `${ftoi(x * 1e4)}#${ftoi(y * 1e4)}#${ftoi(z * 1e4)}`;
    const e = map.get(key);
    if (e !== undefined) return e;
    const i = pos.length / 3;
    pos.push(Math.fround(x), Math.fround(y), Math.fround(z)); // JWildfire stores float vertices
    map.set(key, i);
    return i;
  };
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'v') objV.push([+t[1], +t[2], +t[3]]);
    else if (t[0] === 'f') {
      const vs = t.slice(1).map((s) => { const i = parseInt(s.split('/')[0]); return objV[i > 0 ? i - 1 : objV.length + i]; });
      if (vs.length === 3 || vs.length === 4) {
        const ids = vs.map((v) => addVertex(v[0], v[1], v[2]));
        idx.push(ids[0], ids[1], ids[2]);
        if (vs.length === 4) idx.push(ids[0], ids[2], ids[3]);
      }
    }
  }
  const nV = pos.length / 3, nF = idx.length / 3;
  const buf = new ArrayBuffer(12 + nV * 12 + nF * 12);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x4d455348, true); dv.setUint32(4, nV, true); dv.setUint32(8, nF, true);
  new Float32Array(buf, 12, nV * 3).set(pos);
  new Uint32Array(buf, 12 + nV * 12, nF * 3).set(idx);
  writeFileSync(`public/mesh/${name}.bin`, Buffer.from(buf));
  console.log(`${name}: ${nV} vertices, ${nF} triangles, ${(buf.byteLength / 1024).toFixed(0)} KB`);
}
