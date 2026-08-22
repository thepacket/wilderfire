import { describe, it, expect } from 'vitest';
import { defaultFlame, defaultSolidRender, normalizeFlame } from '../src/core/flame';
import { flameToXML, importFlameText } from '../src/core/flameXML';
import { flameReflMaps } from '../src/core/reflMaps';
import { SOLID_TONEMAP_WGSL } from '../src/gpu/solid.wgsl';

const pal = Array.from({ length: 256 }, (_, i) => [i / 255, 0.5, 1 - i / 255] as [number, number, number]);

describe('solid reflection maps', () => {
  it('the material keeps its reflection map through .flame export/import, path stripped to the file name', () => {
    const f = defaultFlame(pal);
    f.solid = defaultSolidRender(true);
    f.solid.materials[0].reflMapFilename = 'C:\\Users\\someone\\Pictures\\sky.jpg';
    f.solid.materials[0].reflMapIntensity = 0.8;
    f.solid.materials[0].reflMapping = 'SPHERICAL';
    const xml = flameToXML(f);
    expect(xml).toContain('sld_render_material_refl_map_filename0="C:\\Users\\someone\\Pictures\\sky.jpg"');
    const back = importFlameText(xml, pal).flame;
    expect(back.solid?.materials[0].reflMapFilename).toBe('sky.jpg');
    expect(back.solid?.materials[0].reflMapIntensity).toBeCloseTo(0.8, 6);
    expect(back.solid?.materials[0].reflMapping).toBe('SPHERICAL');
    // JSON (library / autosave) keeps it too
    expect(normalizeFlame(JSON.parse(JSON.stringify(back)), pal).solid?.materials[0].reflMapFilename).toBe('sky.jpg');
  });
  it('lists the images a flame needs: solid only, intensity > 0, unique, in material order', () => {
    const f = defaultFlame(pal);
    expect(flameReflMaps(f)).toEqual([]);
    f.solid = defaultSolidRender(true);
    f.solid.materials = [{ ...f.solid.materials[0], reflMapFilename: 'a.png' }, { ...f.solid.materials[0], reflMapFilename: 'b.png', reflMapIntensity: 0 }, { ...f.solid.materials[0], reflMapFilename: 'a.png' }, { ...f.solid.materials[0] }];
    expect(flameReflMaps(f)).toEqual(['a.png']);
    f.solid.enabled = false;
    expect(flameReflMaps(f)).toEqual([]);
  });
  it('the solid tonemap samples the reflection texture array with JWildfire\'s blerp', () => {
    expect(SOLID_TONEMAP_WGSL).toContain('@binding(8) var reflTex: texture_2d_array<f32>');
    expect(SOLID_TONEMAP_WGSL).toContain('fn reflColor(');
    expect(SOLID_TONEMAP_WGSL).toContain('return mix(mix(lu, ru, u), mix(lb, rb, u), v);');
    expect(SOLID_TONEMAP_WGSL).toContain('raw += reflColor(m.reflLayer, u, v) * (vis * ri);');
  });
});
