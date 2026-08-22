import { describe, it, expect } from 'vitest';
import { defaultFlame } from '../src/core/flame';
import { applyEdits, editsFromUserText } from '../src/ai/context';

const pal = Array.from({ length: 256 }, (_, i) => [i / 255, 0.5, 1 - i / 255] as [number, number, number]);
const flame = () => { const f = defaultFlame(pal); while (f.layers[0].xforms.length < 4) f.layers[0].xforms.push(JSON.parse(JSON.stringify(f.layers[0].xforms[0]))); f.layers[0].final = JSON.parse(JSON.stringify(f.layers[0].xforms[0])); return f; };

describe('edit commands as models write them', () => {
  it('whole affine arrays, layers.N paths, underscored commands, addxform with fields', () => {
    const block = `# activate the final
set layers.0.final.weight 0.3
# differentiate T4
set layers.0.xforms.3.affine [0.966,0.259,0,-0.259,0.966,0]
set T1.affine [-0.234,0.06,0,-0.06,-0.234,0]
add_xform layers.0 weight=0.08 color=0.5 colorSpeed=0.3 affine=[0.6,0,0,0,0.6,0] variations=blur:1{}
set layers.0.xforms.0.colorSpeed 0.3`;
    const r = applyEdits(flame(), block, 0);
    expect(r.errors).toEqual([]);
    expect(r.applied).toBe(5);
    const ly = r.flame.layers[0];
    expect(ly.final!.weight).toBeCloseTo(0.3, 9);
    expect(ly.xforms[3].affine).toEqual([0.966, 0.259, 0, -0.259, 0.966, 0]);
    expect(ly.xforms[0].affine).toEqual([-0.234, 0.06, 0, -0.06, -0.234, 0]);
    expect(ly.xforms[0].colorSpeed).toBeCloseTo(0.3, 9);
    const added = ly.xforms[4];
    expect(added.weight).toBeCloseTo(0.08, 9); expect(added.color).toBeCloseTo(0.5, 9); expect(added.colorSpeed).toBeCloseTo(0.3, 9);
    expect(added.affine).toEqual([0.6, 0, 0, 0, 0.6, 0]);
    expect(added.variations.map((v) => [v.name, v.weight])).toEqual([['blur', 1]]);
  });
  it('reports what it cannot do instead of silently skipping', () => {
    const r = applyEdits(flame(), 'set T9.weight 1\naddxform variations=nosuchvar:1\nfrobnicate T1', 0);
    expect(r.applied).toBe(0);
    expect(r.errors.length).toBe(3);
  });
  it('recognises pasted edit commands in a chat message, with or without the fence', () => {
    expect(editsFromUserText('```edits\nset brightness 3\n```')).toBe('set brightness 3\n');
    expect(editsFromUserText('# calmer\nset brightness 3\nadd_xform weight=0.1')).toBe('# calmer\nset brightness 3\nadd_xform weight=0.1');
    expect(editsFromUserText('```\n# note\nset brightness 3\nset T1.affine [1,0,0,0,1,0]\n```')).toBe('# note\nset brightness 3\nset T1.affine [1,0,0,0,1,0]');
    expect(editsFromUserText('please set the brightness to 3')).toBeNull();
    expect(editsFromUserText('What does julian do?')).toBeNull();
  });
});
