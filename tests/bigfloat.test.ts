import { describe, it, expect } from 'vitest';
import { fromDecimal, toDecimal, fromNumber, toNumber, fxAdd, fxMul, bitsForZoom, referenceOrbit } from '../src/core/bigfloat';
import { compileFormulaDS, DS_WGSL } from '../src/core/formula';
import { defaultEscape, escapeMoveCentre, escapeSetCentre, escapeTier, escapeSignature } from '../src/core/escape';
import { GREY } from './helpers';

describe('bigfloat (BigInt fixed point)', () => {
  it('decimal strings round-trip and match f64 for ordinary values', () => {
    const P = 128;
    for (const s of ['0', '1', '-0.75', '0.1318259042053119', '-1.234e-5', '.5', '3e2']) {
      const x = fromDecimal(s, P);
      expect(toNumber(x)).toBeCloseTo(parseFloat(s), 12);
    }
    const deep = '-0.743643887037158704752191506114774';
    expect(toDecimal(fromDecimal(deep, 200), 33)).toBe(deep);
    expect(toNumber(fromNumber(-0.7436438870371587, P))).toBe(-0.7436438870371587);
    expect(toNumber(fxAdd(fromDecimal('1', P), fromDecimal('1e-30', P)))).toBe(1);
    expect(toDecimal(fxAdd(fromDecimal('1', P), fromDecimal('1e-30', P)), 32)).toBe('1.000000000000000000000000000001');
    expect(toNumber(fxMul(fromDecimal('1.5', P), fromDecimal('-2', P)))).toBe(-3);
    expect(bitsForZoom(1)).toBe(64);
    expect(bitsForZoom(1e30)).toBeGreaterThanOrEqual(100 + 24);
  });

  it('the reference orbit of the Mandelbrot map matches a double iteration while it stays shallow', () => {
    const P = 96;
    const c: [ReturnType<typeof fromNumber>, ReturnType<typeof fromNumber>] = [fromNumber(-1, P), fromNumber(0.05, P)];
    const z0 = [fromNumber(0, P), fromNumber(0, P)] as typeof c;
    const orb = referenceOrbit(z0, c, 2, 50, 128);
    let x = 0, y = 0;
    for (let i = 1; i <= orb.n; i++) {
      const nx = x * x - y * y - 1, ny = 2 * x * y + 0.05; x = nx; y = ny;
      expect(orb.data[i * 4] + orb.data[i * 4 + 1]).toBeCloseTo(x, 6);
      expect(orb.data[i * 4 + 2] + orb.data[i * 4 + 3]).toBeCloseTo(y, 6);
    }
    expect(orb.escaped).toBe(false); // −1+0.05i is inside the period-2 bulb
    const esc = referenceOrbit(z0, [fromNumber(0.5, P), fromNumber(0.5, P)] as typeof c, 2, 500, 128);
    expect(esc.escaped).toBe(true);
    expect(esc.n).toBeLessThan(20);
  });
});

describe('deep-zoom escape model', () => {
  it('centre moves stay exact past f64 and tiers switch with the zoom', () => {
    const e = defaultEscape(GREY);
    expect(escapeTier(e)).toBe('f32');
    e.zoom = 1e6; expect(escapeTier(e)).toBe('ds');
    e.zoom = 1e12; expect(escapeTier(e)).toBe('perturb');
    e.formula = 'burningship'; expect(escapeTier(e)).toBe('ds'); // no perturbation path → ds
    e.formula = 'mandelbrot';
    escapeSetCentre(e, '-0.75', '0.1');
    expect(e.centerHi).toBeDefined();
    escapeMoveCentre(e, 1e-25, -1e-25);
    expect(e.centerHi![0]).toMatch(/^-0\.7499999999999999999999999/);
    expect(e.centerHi![1]).toMatch(/^0\.0999999999999999999999999/);
    e.zoom = 10; escapeMoveCentre(e, 0.25, 0); // shallow again → plain numbers
    expect(e.centerHi).toBeUndefined();
    expect(e.centerX).toBeCloseTo(-0.5, 12);
    // the tier is part of the shader signature
    const a = { ...defaultEscape(GREY), zoom: 1 }, b = { ...defaultEscape(GREY), zoom: 1e6 };
    expect(escapeSignature(a)).not.toBe(escapeSignature(b));
  });
  it('the DS compiler mirrors the f32 one', () => {
    expect(compileFormulaDS('z^2 + c')).toBe('dc_add(dc_pow(z, dc_f(2.0, 0.0)), c)');
    expect(compileFormulaDS('sin(z) * p1 - 3')).toBe('dc_sub(dc_mul(dc_c(csin(dc_to(z))), p1), dc_f(3.0, 0.0))');
    expect(() => compileFormulaDS('foo(z)')).toThrow(/unknown function/);
    for (const fn of ['dc_add', 'dc_mul', 'dc_div', 'dc_pow', 'dc_sqr', 'ds_mul', 'ds_add', 'ds_div']) expect(DS_WGSL).toContain(`fn ${fn}(`);
  });
});
