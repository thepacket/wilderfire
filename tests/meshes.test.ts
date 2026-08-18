import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseMeshBin, subdivide, taubinSmooth, faceWeights, buildSampler, defaultMesh, meshKey, prepareMesh, MESH_PRIMITIVES, flameMeshKeys, parseObj, meshToBin } from '../src/core/meshes';
import { flameToXML } from '../src/core/flameXML';
import { compileFlame } from '../src/gpu/codegen';
import { importFlameText } from '../src/core/flameXML';
import { GREY } from './helpers';

const bin = (name: string) => { const b = readFileSync(resolve(process.cwd(), 'public/mesh', name + '.bin')); return parseMeshBin(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer); };

describe('mesh primitives (obj_mesh_primitive_wf)', () => {
  it('ships all 26 JWildfire primitives as parseable binaries with de-duplicated vertices', () => {
    for (const n of MESH_PRIMITIVES) {
      const m = bin(n);
      expect(m.pos.length % 3, n).toBe(0);
      expect(m.idx.length % 3, n).toBe(0);
      expect(m.idx.length, n).toBeGreaterThan(0);
      for (const i of m.idx) expect(i, n).toBeLessThan(m.pos.length / 3);
    }
    const ball = bin('ball');
    expect(ball.pos.length / 3).toBe(642); // SimpleMesh.addVertex de-dupes the 3840 OBJ corners at 1e-4
    expect(ball.idx.length / 3).toBe(1280);
    let r = 0; for (let i = 0; i < ball.pos.length; i += 3) r = Math.max(r, Math.hypot(ball.pos[i], ball.pos[i + 1], ball.pos[i + 2]));
    expect(r).toBeCloseTo(0.5, 5); // JWildfire's ball has radius 0.5
    expect(bin('box').idx.length / 3).toBe(12); // 6 quads → 12 triangles
  });

  it('subdivide quadruples the faces (shared midpoints de-duplicated); Taubin smoothing keeps the vertex count', () => {
    const cube = defaultMesh();
    const s = subdivide(cube);
    expect(s.idx.length).toBe(cube.idx.length * 4);
    expect(s.pos.length / 3).toBe(8 + 12 + 6); // corners + edge midpoints + face-diagonal midpoints
    const before = Float32Array.from(s.pos);
    taubinSmooth(s, 12, 0.42, -0.45);
    expect(s.pos.length).toBe(before.length);
    let moved = 0; for (let i = 0; i < s.pos.length; i++) if (s.pos[i] !== before[i]) moved++;
    expect(moved).toBeGreaterThan(0);
  });

  it('level 0 samples faces proportionally to area (distributeFaces quantisation), subdivided meshes uniformly', () => {
    const cube = defaultMesh(); // equal-area triangles → weight 1 each
    expect(Array.from(faceWeights(cube))).toEqual(new Array(12).fill(1));
    const cone = bin('cone');
    const w = faceWeights(cone);
    expect(Math.max(...w)).toBeGreaterThan(1);
    const smp = buildSampler(cone, w);
    expect(smp.faces).toBe(cone.idx.length / 3);
    expect(smp.cdf[smp.faces - 1]).toBeCloseTo(1.0001, 6);
    for (let i = 1; i < smp.faces; i++) expect(smp.cdf[i]).toBeGreaterThanOrEqual(smp.cdf[i - 1]);
    expect(smp.tris.length).toBe(smp.faces * 9);
    const p = prepareMesh(cube, 2, 12, 0.42, -0.45);
    expect(p.faces).toBe(12 * 16);
    expect(p.cdf[0]).toBeCloseTo(1 / 192, 6); // uniform
  });

  it('meshKey follows OBJMeshUtil.getMeshname (smoothing params only matter with subdivision)', () => {
    expect(meshKey(0, 0, 12, 0.42, -0.45)).toBe('ball#0');
    expect(meshKey(0, 0, 3, 0.1, -0.2)).toBe('ball#0');
    expect(meshKey(5, 2, 12, 0.42, -0.45)).toBe('box#2#12#0.42#-0.45');
    expect(meshKey(99, 0, 12, 0.42, -0.45)).toBe('default#0');
  });

  it('a flame with obj_mesh_primitive_wf compiles with the mesh binding and reports its keys', () => {
    const xml = '<flame name="m" size="64 64" scale="10"><xform weight="1" obj_mesh_primitive_wf="1" obj_mesh_primitive_wf_primitive="4" obj_mesh_primitive_wf_subdiv_level="1" coefs="1 0 0 1 0 0"/></flame>';
    const { flame, unknown } = importFlameText(xml, GREY);
    expect(unknown).toEqual([]);
    expect(flameMeshKeys(flame)).toEqual(['torus#1#12#0.42#-0.45']);
    const c = compileFlame(flame, 1024);
    expect(c.usesMesh).toBe(true);
    expect(c.wgsl).toContain('@binding(12) var<storage, read> mesh');
    expect(c.wgsl).toContain('meshPick(');
    const out = new Float32Array(c.dataSize);
    c.writeData(flame, out); // meshes not loaded in vitest → 0 faces, no throw
    expect(compileFlame(importFlameText('<flame name="n" size="64 64" scale="10"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY).flame, 1024).usesMesh).toBe(false);
  });

  it('parseObj reads v/f lines like OBJMeshUtil (quads split, negative indices, 1e-4 vertex de-dup) and round-trips through the binary', () => {
    const obj = 'o q\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.00001 0 0\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1 4/4/1\nf -5 -4 -3\ng x\n';
    const m = parseObj(obj);
    expect(m.pos.length / 3).toBe(4); // the fifth vertex de-dupes onto the first
    expect(Array.from(m.idx)).toEqual([0, 1, 2, 0, 2, 3, 0, 1, 2]);
    const back = parseMeshBin(meshToBin(m));
    expect(Array.from(back.pos)).toEqual(Array.from(m.pos));
    expect(Array.from(back.idx)).toEqual(Array.from(m.idx));
  });

  it('obj_mesh_wf: the hex "ressource" file name imports as a basename, exports as hex again, and keys the mesh (default cube when empty)', () => {
    const hex = Array.from(new TextEncoder().encode('D:\\Pictures\\obj files\\toy horse.obj'), (b) => b.toString(16).toUpperCase().padStart(2, '0')).join('');
    const xml = `<flame name="m" size="64 64" scale="10"><xform weight="1" obj_mesh_wf="1" obj_mesh_wf_scale_x="2" obj_mesh_wf_obj_filename="${hex}" obj_mesh_wf_colormap_filename="" coefs="1 0 0 1 0 0"/>` +
      '<xform weight="1" obj_mesh_wf="1" obj_mesh_wf_obj_filename="" coefs="1 0 0 1 0 0"/></flame>';
    const { flame, unknown } = importFlameText(xml, GREY);
    expect(unknown).toEqual([]);
    const v0 = flame.layers[0].xforms[0].variations[0], v1 = flame.layers[0].xforms[1].variations[0];
    expect(v0.name).toBe('obj_mesh_wf');
    expect(v0.params.scale_x).toBe(2);
    expect(v0.res).toEqual({ obj_filename: 'toy horse.obj' });
    expect(v1.res).toBeUndefined();
    expect(flameMeshKeys(flame)).toEqual(['obj:toy horse.obj@0#0', 'default#0']);
    const out = flameToXML(flame);
    expect(out).toContain(`obj_mesh_wf_obj_filename="${Array.from(new TextEncoder().encode('toy horse.obj'), (b) => b.toString(16).toUpperCase().padStart(2, '0')).join('')}"`);
    const again = importFlameText(out, GREY).flame;
    expect(again.layers[0].xforms[0].variations[0].res).toEqual({ obj_filename: 'toy horse.obj' });
    const c = compileFlame(flame, 1024);
    expect(c.usesMesh).toBe(true);
    c.writeData(flame, new Float32Array(c.dataSize));
  });
});
