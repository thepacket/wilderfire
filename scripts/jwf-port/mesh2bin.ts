// Converts JWildfire's bundled mesh primitives (src/org/jwildfire/create/tina/variation/mesh/*.obj, LGPL 2.1+)
// into compact binaries for obj_mesh_primitive_wf: public/mesh/<name>.bin =
//   u32 magic 0x4d455348 ('MESH'), u32 nV, u32 nF, f32 pos[nV*3], u32 idx[nF*3]
// Vertices are de-duplicated at 1e-4 like SimpleMesh.addVertex (the neighbour lists of the Taubin smoothing
// depend on it); quads become two triangles (v0,v1,v2) + (v0,v2,v3) like OBJMeshUtil.loadAndSmoothMeshFromFile
// (the reader is src/core/meshes.ts `parseObj`, shared with the in-browser obj_mesh_wf loader).
//   usage: node scripts/jwf-port/mesh2bin.ts <jwildfire>/src/org/jwildfire/create/tina/variation/mesh
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { parseObj, meshToBin } from '../../src/core/meshes.ts';

const NAMES = ['ball', 'capsule', 'cone', 'diamond', 'torus', 'box', 'gear15', 'icosahedron', 'tetrahedron', 'octahedron', 'dodecahedron', 'wedge',
  'icosidodecahedron', 'cubeoctahedron', 'gears6a', 'gears6s', 'gears8a', 'gears8s', 'gears12a', 'gears12s', 'gears16a', 'gears16s', 'gears24a', 'gears24s', 'mandelbulb', 'drop'];
const src = process.argv[2];
if (!src) { console.error('usage: node mesh2bin.ts <jwf mesh dir>'); process.exit(1); }
mkdirSync('public/mesh', { recursive: true });
for (const name of NAMES) {
  const m = parseObj(readFileSync(join(src, name + '.obj'), 'latin1'));
  const nV = m.pos.length / 3, nF = m.idx.length / 3;
  const buf = meshToBin(m);
  writeFileSync(`public/mesh/${name}.bin`, Buffer.from(buf));
  console.log(`${name}: ${nV} vertices, ${nF} triangles, ${(buf.byteLength / 1024).toFixed(0)} KB`);
}
