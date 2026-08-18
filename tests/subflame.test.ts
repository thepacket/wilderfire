import { describe, it, expect, vi } from 'vitest';
import { compileFlame, parseSubFlame } from '../src/gpu/codegen';
import { importFlameText, flameToXML } from '../src/core/flameXML';
import { DFLT_SUBFLAME_XML } from '../src/core/variations';
import { flameSignature } from '../src/core/flame';
import { GREY } from './helpers';

const hex = (s: string) => Array.from(new TextEncoder().encode(s), (b) => b.toString(16).toUpperCase().padStart(2, '0')).join('');

describe('subflame_wf (nested flame compiled into the kernel)', () => {
  it('the default sub-flame parses (3 xforms, no finals) and finals without a colour type become DIFFUSION', () => {
    const d = parseSubFlame(undefined, GREY)!;
    expect(d.layer.xforms.length).toBe(3);
    expect(d.layer.final).toBeNull();
    const withFinal = parseSubFlame('<flame name="s" size="10 10" scale="10"><xform weight="1" linear="1" coefs="1 0 0 1 0 0"/><finalxform color="0.5" symmetry="0" linear="1" coefs="1 0 0 1 0 0"/></flame>', GREY)!;
    expect(withFinal.layer.final?.colorSpeed).toBeCloseTo(0.5, 9); // (1 − symmetry)/2, kept because the sub-flame final recolours
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {}); // the broken XML is meant to be rejected
    expect(parseSubFlame('<nonsense', GREY)).toBeNull();
    warn.mockRestore();
  });

  it('imports the hex flame resource untouched, keeps the default out of the model, exports it back and compiles the nested kernel', () => {
    const inner = '<flame name="inner" size="10 10" scale="10"><xform weight="1" color="0.3" spherical="1" coefs="0.5 0 0 0.5 0.2 0" opacity="0.5"/><xform weight="1" linear="1" coefs="0.5 0 0 0.5 -0.2 0" chaos="1 0"/><finalxform bubble="1" coefs="1 0 0 1 0 0"/></flame>';
    const xml = `<flame name="o" size="64 64" scale="10"><xform weight="1" subflame_wf="1" subflame_wf_color_mode="0" subflame_wf_scale="2" subflame_wf_flame="${hex(inner)}" coefs="1 0 0 1 0 0"/>` +
      `<xform weight="1" subflame_wf="0.7" subflame_wf_flame="${hex(DFLT_SUBFLAME_XML)}" coefs="1 0 0 1 0 0"/></flame>`;
    const { flame, unknown } = importFlameText(xml, GREY);
    expect(unknown).toEqual([]);
    const v0 = flame.layers[0].xforms[0].variations[0], v1 = flame.layers[0].xforms[1].variations[0];
    expect(v0.res?.flame).toBe(inner);
    expect(v1.res).toBeUndefined(); // the default is not stored
    expect(v0.params.color_mode).toBe(0);
    const out = flameToXML(flame);
    expect(out).toContain(`subflame_wf_flame="${hex(inner)}"`);
    expect(out).toContain(`subflame_wf_flame="${hex(DFLT_SUBFLAME_XML)}"`); // the default is written for JWildfire
    const c = compileFlame(flame, 1024);
    expect(c.wgsl).toContain('fn subflame0(');
    expect(c.wgsl).toContain('fn subflame1(');
    expect(c.wgsl).toContain('fn subflameAny(');
    expect(c.wgsl).toContain('applySF0_0'); // the inner final
    expect(c.wgsl).toContain('applyS1_2');  // the default sub-flame's third xform
    const data = new Float32Array(c.dataSize);
    c.writeData(flame, data);
    // the sub-flame index sits in the hidden slot after the 12 params of the first instance (block header 72 + weight)
    expect(data.slice(0, c.dataSize).filter((v) => v === 1e9).length).toBe(0);
    // a different sub-flame is a different structure
    const f2 = importFlameText(xml.replace(hex(inner), hex(inner.replace('spherical="1"', 'spherical="1" bubble="1"'))), GREY).flame;
    expect(flameSignature(f2)).not.toBe(flameSignature(flame));
  });

  it('a sub-flame containing subflame_wf does not recurse', () => {
    const inner = `<flame name="i" size="10 10" scale="10"><xform weight="1" subflame_wf="1" linear="0.5" coefs="1 0 0 1 0 0"/></flame>`;
    const xml = `<flame name="o" size="64 64" scale="10"><xform weight="1" subflame_wf="1" subflame_wf_flame="${hex(inner)}" coefs="1 0 0 1 0 0"/></flame>`;
    const { flame } = importFlameText(xml, GREY);
    const c = compileFlame(flame, 1024);
    expect((c.wgsl.match(/fn subflame\d+\(/g) ?? []).length).toBe(1);
  });
});
