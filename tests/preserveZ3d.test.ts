// preserve_z on 3D variations that carry JWildfire's clause anyway (dc_carpet3D, whirligig): the engine adds
// w·z for them like it does for 2D variations. Found on a real flame — `_pdofR`, dc_carpet3D under a 60° pitch —
// which matched JWildfire with preserve_z off (corr 1.00) and diverged with it on (0.09) until this line existed.
import { describe, it, expect } from 'vitest';
import { defaultFlame } from '../src/core/flame';
import { VARIATIONS, defaultParams } from '../src/core/variations';
import { compileFlame } from '../src/gpu/codegen';

const pal = Array.from({ length: 256 }, (_, i) => [i / 255, 0.5, 1 - i / 255] as [number, number, number]);
const PZ = /if \(P\.flags & 1u\) != 0u \{ pz_ \+= /g;
const kernelWith = (name: string) => {
  const f = defaultFlame(pal);
  f.preserveZ = true;
  f.layers[0].xforms = [f.layers[0].xforms[0]];
  f.layers[0].xforms[0].variations = [{ name, weight: 0.5, params: defaultParams(name) }];
  return compileFlame(f, 1024).wgsl;
};

describe('preserve_z clause on 3D variations', () => {
  it('dc_carpet3D and whirligig get the w·z line (their Java carries the clause)', () => {
    for (const n of ['dc_carpet3D', 'whirligig']) {
      expect(VARIATIONS[n].flags, n).toContain('z');
      expect(kernelWith(n).match(PZ)?.length ?? 0, n).toBe(1);
    }
  });
  it('an ordinary 3D variation does not', () => {
    expect(kernelWith('linear3D').match(PZ)?.length ?? 0).toBe(0);
  });
  it('a 2D variation does', () => {
    expect(kernelWith('spherical').match(PZ)?.length ?? 0).toBe(1);
  });
});
