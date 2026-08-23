import { describe, it, expect } from 'vitest';
import { applyMutation, MUTATION_TYPES, addRandomSymmetry, addRandomWeightingField, classicPalette, similarGradient, coverage, diffCoverage, pixelize } from '../src/core/mutations';
import { randomFlameInStyle } from '../src/core/randomStyles';
import { flameToXML, importFlameText } from '../src/core/flameXML';
import { compileFlame } from '../src/gpu/codegen';
import { GREY } from './helpers';

describe('MutaGen mutations', () => {
  it('every type keeps the flame valid, round-trippable and compilable (on flames of several styles)', () => {
    const styles = ['bubbles', 'julians', 'cross', 'spherical3d', 'solid_stunning', 'subflame'];
    for (const t of MUTATION_TYPES) for (const st of styles) for (let k = 0; k < 3; k++) {
      const f = randomFlameInStyle(st, GREY);
      const applied = applyMutation(f, t.id, 1);
      expect(applied, t.id).not.toBe('all');
      for (const ly of f.layers) {
        expect(ly.xforms.length, `${t.id} on ${st}`).toBeGreaterThan(0);
        for (const x of [...ly.xforms, ly.final, ...ly.moreFinals].filter(Boolean)) {
          for (const c of [...x!.affine, ...x!.post, ...(x!.yz ?? []), ...(x!.zx ?? []), ...(x!.yzPost ?? []), ...(x!.zxPost ?? [])]) expect(Number.isFinite(c), `${t.id} on ${st}: affine`).toBe(true);
          if (x!.xaos) expect(x!.xaos.length, `${t.id} on ${st}: xaos`).toBe(ly.xforms.length);
        }
      }
      const back = importFlameText(flameToXML(f), GREY).flame;
      expect(back.layers[0].xforms.length).toBe(f.layers[0].xforms.length);
      expect(compileFlame(f, 1024).wgsl.includes('${'), `${t.id} on ${st}: template`).toBe(false);
    }
  }, 60000);
  it('"all" draws from JWildfire\'s weighted list and reports the concrete type', () => {
    const seen = new Set<string>();
    for (let k = 0; k < 60; k++) seen.add(applyMutation(randomFlameInStyle('julians', GREY), 'all'));
    expect(seen.size).toBeGreaterThan(4);
    expect(seen.has('all')).toBe(false);
  });
  it('color_type resolves UNSET like JWildfire (DIFFUSION on transforms, NONE on finals) and paints TARGET colours', () => {
    let targets = 0;
    for (let k = 0; k < 40; k++) {
      const f = randomFlameInStyle('cross', GREY); // has a final
      applyMutation(f, 'color_type');
      for (const x of f.layers[0].xforms) { expect(x.colorType === undefined || ['NONE', 'DISTANCE', 'TARGET', 'TARGETG'].includes(x.colorType)).toBe(true); if (x.colorType === 'TARGET') { targets++; expect(x.targetColor?.length).toBe(3); } }
      const fin = f.layers[0].final!;
      if (fin.colorType === undefined) expect(fin.colorSpeed).toBeGreaterThan(0); // a recolouring final
    }
    expect(targets).toBeGreaterThan(0);
  });
});

describe('random symmetry / weighting-field generators', () => {
  it('symmetry kinds set the post symmetry JWildfire\'s way (sparse: about a third)', () => {
    const f = randomFlameInStyle('bubbles', GREY);
    addRandomSymmetry(f, 'point'); expect(f.postSymmetry?.type).toBe('POINT'); expect(f.postSymmetry!.order).toBeGreaterThanOrEqual(2); expect(f.postSymmetry!.order).toBeLessThanOrEqual(7);
    addRandomSymmetry(f, 'xaxis'); expect(f.postSymmetry?.type).toBe('X_AXIS'); expect(Math.abs(f.postSymmetry!.rotation)).toBeLessThanOrEqual(30);
    addRandomSymmetry(f, 'yaxis'); expect(f.postSymmetry?.type).toBe('Y_AXIS'); expect(f.postSymmetry!.rotation).toBe(0);
    addRandomSymmetry(f, 'none'); expect(f.postSymmetry).toBeUndefined();
    let n = 0; for (let k = 0; k < 300; k++) { addRandomSymmetry(f, 'sparse'); if (f.postSymmetry) n++; }
    expect(n).toBeGreaterThan(30); expect(n).toBeLessThan(130); // 0.34 × 0.75 ≈ 0.26 of the draws
  });
  it('weighting fields land on the heavier transforms with JWildfire\'s parameter ranges', () => {
    for (let k = 0; k < 20; k++) {
      const f = randomFlameInStyle('galaxies', GREY);
      addRandomWeightingField(f, 'fractal');
      const with_ = f.layers[0].xforms.filter((x) => x.wfield);
      expect(with_.length).toBeGreaterThan(0);
      for (const x of with_) {
        expect(x.wfield!.type.endsWith('FRACTAL_NOISE')).toBe(true);
        expect(x.wfield!.octaves).toBeGreaterThanOrEqual(2); expect(x.wfield!.octaves).toBeLessThanOrEqual(5);
        expect(x.wfield!.frequency).toBeGreaterThanOrEqual(0.75);
        for (const p of x.wfield!.params) expect(p.paramName === 'amount' || p.paramName in (x.variations.find((v) => v.name === p.varName)?.params ?? {})).toBe(true);
      }
      addRandomWeightingField(f, 'none'); expect(f.layers[0].xforms.some((x) => x.wfield)).toBe(false);
    }
  });
});

describe('gradients and coverage measures', () => {
  it('classicPalette spreads key colours over 256 entries, faded or stepped', () => {
    const p = classicPalette([[255, 0, 0], [0, 0, 255]], true, true);
    expect(p).toHaveLength(256); expect(p[0]).toEqual([1, 0, 0]); expect(p[255][2]).toBeGreaterThan(0.99); expect(p[128][0]).toBeCloseTo(0.5, 1);
    const q = classicPalette([[255, 0, 0], [0, 0, 255]], false, true);
    expect(q[100]).toEqual([1, 0, 0]); expect(q[200]).toEqual([0, 0, 1]);
  });
  it('similarGradient returns a full gradient built from the input\'s colours', () => {
    const p = similarGradient(GREY);
    expect(p).toHaveLength(256);
    for (const c of p) for (const v of c) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1); }
  });
  it('coverage counts clearly lit pixels (Sobel edges when filtered) and diffCoverage the changed 5-px blocks', () => {
    const w = 20, h = 10;
    const px = new Uint8ClampedArray(w * h * 4);
    for (let x = 0; x < 10; x++) for (let y = 0; y < h; y++) { const i = (y * w + x) * 4; px[i] = 200; px[i + 1] = 200; px[i + 2] = 200; px[i + 3] = 255; }
    expect(coverage(px, w, h, [0, 0, 0], false)).toBeCloseTo(0.5, 5);
    const edges = coverage(px, w, h, [0, 0, 0], true);
    expect(edges).toBeGreaterThan(0.05); expect(edges).toBeLessThan(0.3); // only the vertical boundary lights up
    const ref = pixelize(new Uint8ClampedArray(w * h * 4), w, h);
    expect(diffCoverage(px, ref, w, h)).toBeCloseTo(0.5, 5);
  });
});
