// Emits the oracle test spec: a fixed grid of input points plus, for every
// variation we know (hand-written + generated), two parameter/weight sets.
//
//   node scripts/jwf-port/oracle-spec.ts
//
// Output: scripts/jwf-port/oracle-spec.txt (read by Oracle.java) and
//         scripts/jwf-port/oracle-spec.json (read by the browser harness).

import { writeFileSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { HAND_VARIATIONS as VARIATIONS } from '../../src/core/variations.ts';
import { JWF_VARIATIONS as JWF_VERIFIED } from '../../src/core/variations.jwf.ts';
import { JWF_VARIATIONS_UNVERIFIED } from '../../src/core/variations.jwf.unverified.ts';
const JWF_VARIATIONS = { ...JWF_VERIFIED, ...JWF_VARIATIONS_UNVERIFIED };

const here = dirname(fileURLToPath(import.meta.url));

// int-ness of params comes from the JWildfire dump
const dump = new Map<string, { params: { name: string; int: boolean }[]; priority: number }>();
for (const l of readFileSync(join(here, 'data', 'jwf-variations.jsonl'), 'utf8').trim().split('\n')) {
  const v = JSON.parse(l);
  dump.set(v.name, { params: v.params, priority: v.priority });
}

// ---- points ----
// 3D points: z varies deterministically so z-reading variations are exercised
// (2D variations ignore it and must leave pz at 0).
const points: number[][] = [];
const N = 11;
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  points.push([-1.6 + 3.2 * i / (N - 1) + 0.0137, -1.6 + 3.2 * j / (N - 1) - 0.0091, 0.45 * Math.sin(1.7 * i + 0.9 * j)]);
}
for (const p of [[0.05, 0.02, 0.3], [-0.03, 0.07, -0.2], [0.5, 0.5, 0.5], [-0.7071, 0.7071, 0], [2.5, -1.2, 0.8], [-3.1, 2.4, -0.6], [0.001, -0.002, 0.01], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]) points.push(p);

const AFFINE = [0.8, 0.3, 0.1, -0.2, 0.9, -0.15]; // a b c d e f

interface Set_ { weight: number; params: Record<string, number> }
interface Entry { name: string; priority: number; source: 'hand' | 'jwf' | 'both'; sets: Set_[] }
const entries: Entry[] = [];

const names = new Set<string>([...Object.keys(VARIATIONS), ...Object.keys(JWF_VARIATIONS)]);
for (const name of [...names].sort()) {
  const hand = VARIATIONS[name];
  const jwf = JWF_VARIATIONS[name];
  const d = dump.get(name);
  if (!d) continue; // not a JWildfire variation → nothing to compare against
  const paramDefs = (jwf ?? hand).params ?? [];
  const isInt = (pn: string) => d.params.find((p) => p.name === pn)?.int ?? false;
  // Set 0: defaults. Set 1: float params perturbed (ints unchanged). Set 2: int
  // params perturbed (0/1 toggled as booleans, others stepped) — informational
  // only, since JWildfire's CPU and GPU code often disagree on degenerate ints.
  const setA: Set_ = { weight: 0.7, params: {} };
  const setB: Set_ = { weight: 1.3, params: {} };
  const setC: Set_ = { weight: 0.9, params: {} };
  for (const pd of paramDefs) {
    setA.params[pd.name] = pd.def;
    if (isInt(pd.name)) {
      setB.params[pd.name] = pd.def;
      let alt = pd.def === 0 || pd.def === 1 ? 1 - pd.def : (pd.def > 0 ? pd.def + 1 : pd.def - 1);
      setC.params[pd.name] = alt;
    } else {
      setB.params[pd.name] = pd.def * 1.37 + 0.21;
      setC.params[pd.name] = pd.def;
    }
  }
  entries.push({ name, priority: jwf?.priority ?? d.priority, source: hand && jwf ? 'both' : hand ? 'hand' : 'jwf', sets: [setA, setB, setC] });
}

// text format for Java
let txt = `POINTS ${points.length}\n`;
for (const [x, y, z] of points) txt += `${x} ${y} ${z}\n`;
txt += `AFFINE ${AFFINE.join(' ')}\n`;
for (const e of entries) {
  txt += `VAR ${e.name} ${e.priority}\n`;
  for (const s of e.sets) {
    const ps = Object.entries(s.params);
    txt += `SET ${s.weight} ${ps.length}\n`;
    for (const [k, v] of ps) txt += `${k} ${v}\n`;
  }
}
writeFileSync(join(here, 'oracle-spec.txt'), txt);
writeFileSync(join(here, 'oracle-spec.json'), JSON.stringify({ points, affine: AFFINE, entries }));
console.log(`${entries.length} variations, ${points.length} points → oracle-spec.{txt,json}`);
