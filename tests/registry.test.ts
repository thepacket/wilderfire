// Invariants of the generated variation registry (src/core/variations.jwf.ts) against the
// port pipeline's records (scripts/jwf-port/verified.json, data/unportable.json) plus a
// codegen smoke test: every preset and every fixture flame compiles to a kernel whose WGSL
// carries no unexpanded template or transpiler error markers.
import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { JWF_VARIATIONS } from '../src/core/variations.jwf';
import { UNPORTABLE } from '../src/core/variations.unportable';
import { VARIATIONS, VARIATION_NAMES, CLASSIC_VARIATIONS, HAND_VARIATIONS, PREFER_HAND, defaultParams } from '../src/core/variations';
import { compileFlame } from '../src/gpu/codegen';
import { importFlameText } from '../src/core/flameXML';
import { PRESETS } from '../src/core/presets';
import { GREY } from './helpers';

const root = process.cwd();
const verified = JSON.parse(readFileSync(resolve(root, 'scripts/jwf-port/verified.json'), 'utf8')) as { jwf: string[]; failed?: Record<string, unknown> };
const unportable = JSON.parse(readFileSync(resolve(root, 'scripts/jwf-port/data/unportable.json'), 'utf8')) as { categories: Record<string, string>; variations: Record<string, string> };
const jwfNames = Object.keys(JWF_VARIATIONS);
const A = (i: number) => `A${i}`;

describe('registry: shape', () => {
  it('holds the expected number of ports and verified ports (bump these when the pipeline grows)', () => {
    expect(jwfNames.length).toBeGreaterThanOrEqual(930);              // variations.jwf.ts holds the verified ports…
    expect(jwfNames.every((n) => JWF_VARIATIONS[n].verified)).toBe(true); // …only (the rest live in variations.jwf.unverified.ts)
    expect(VARIATION_NAMES.length).toBeGreaterThanOrEqual(932);      // hand-written ∪ ports
  });

  it('every entry emits WGSL for its default params with no unexpanded template, error marker or C leftovers', () => {
    const bad: string[] = [];
    for (const name of jwfNames) {
      const def = JWF_VARIATIONS[name];
      const p = (def.params ?? []).map((_, i) => `P${i}`);
      const codes = [def.code('W', p, A)];
      if (def.preCode) codes.push(def.preCode('W', p, A));
      for (const c of codes) {
        if (/\$\{|\/\*ERROR|varpar->|__amount_|\bfabsf\(|\bsqrtf\(|make_float2\(/.test(c)) bad.push(name);
      }
      if (def.funcs && /\$\{|\/\*ERROR/.test(def.funcs)) bad.push(name + ':funcs');
    }
    expect(bad).toEqual([]);
  });

  it('funcNames match the module-scope items in funcs, and every snippet reference to a helper is declared', () => {
    const bad: string[] = [];
    for (const name of jwfNames) {
      const def = JWF_VARIATIONS[name];
      for (const fn of def.funcNames ?? []) {
        if (!new RegExp(`(^|\\n)(struct|fn|const|var<private>) ${fn}\\b`).test(def.funcs ?? '')) bad.push(`${name}: ${fn} not in funcs`);
      }
    }
    expect(bad).toEqual([]);
  });

  it('priorities are JWildfire\'s (-1 pre, 0, 1 post; prepost pairs carry preCode) and int params are flagged', () => {
    for (const name of jwfNames) {
      const def = JWF_VARIATIONS[name];
      expect([-1, 0, 1, 2], name).toContain(def.priority);
      if (def.preCode) expect(def.priority, name).toBe(2);
      if (def.priority === 2) expect(def.preCode, name).toBeDefined();
      for (const p of def.params ?? []) {
        expect(typeof p.def, `${name}.${p.name}`).toBe('number');
        expect(Number.isFinite(p.def), `${name}.${p.name}`).toBe(true);
        if (p.int) expect(Number.isInteger(p.def), `${name}.${p.name}`).toBe(true);
      }
    }
  });
});

describe('registry: pipeline records agree', () => {
  it('verified.json ⊆ registry, and every registry entry marked verified is in verified.json or the generator\'s FORCE list', () => {
    const inRegistry = new Set(jwfNames);
    const missing = verified.jwf.filter((n) => !inRegistry.has(n) && !n.includes('~'));
    expect(missing).toEqual([]);
    // registry entries marked verified but absent from verified.json are the FORCE_VERIFIED ones (gen.ts): keep that list short
    const forced = jwfNames.filter((n) => JWF_VARIATIONS[n].verified && !verified.jwf.includes(n));
    expect(forced.length).toBeLessThanOrEqual(25);
  });

  it('unportable.json names never appear in the registry and always carry a known category', () => {
    for (const [name, cat] of Object.entries(unportable.variations)) {
      expect(unportable.categories, name).toHaveProperty(cat);
      expect(JWF_VARIATIONS, name).not.toHaveProperty(name);
      expect(typeof UNPORTABLE[name], name).toBe('string'); // the importer's short reason (gen.ts SHORT[cat])
    }
    expect(Object.keys(UNPORTABLE).length).toBe(Object.keys(unportable.variations).length);
  });

  it('g-atan2.json entries (CPU angles from js.glsl G.atan2, an approximation) use the G_atan2 helper and nothing else does', () => {
    const gAtan2 = JSON.parse(readFileSync(resolve(root, 'scripts/jwf-port/data/g-atan2.json'), 'utf8')) as { variations: string[] };
    const ps = Array.from({ length: 64 }, (_, i) => `P${i}`);
    const text = (v: typeof JWF_VARIATIONS[string]) => v.code(A(0), ps, A) + (v.funcs ?? '') + (v.preCode ? v.preCode(A(0), ps, A) : '');
    for (const name of gAtan2.variations) {
      expect(JWF_VARIATIONS, name).toHaveProperty(name);
      const t = text(JWF_VARIATIONS[name]);
      expect(t.includes('G_atan2('), name).toBe(true);
      expect(/(?<![\w.])atan2j?\(/.test(t.replace(/fn atan2j\([^\n]*/, '')), `${name}: exact atan2 left`).toBe(false); // (the atan2j helper's own body excepted)
      expect(JWF_VARIATIONS[name].funcs ?? '', name).toContain('fn G_atan2(');
    }
    const uses = new Set(gAtan2.variations);
    for (const name of jwfNames) if (!uses.has(name)) expect(text(JWF_VARIATIONS[name]).includes('G_atan2('), name).toBe(false);
  });

  it('the app registry prefers verified ports over hand-written entries except PREFER_HAND', () => {
    for (const name of Object.keys(HAND_VARIATIONS)) {
      const jwf = JWF_VARIATIONS[name];
      if (!jwf?.verified) continue;
      expect(VARIATIONS[name] === (name in PREFER_HAND ? HAND_VARIATIONS[name] : jwf), name).toBe(true);
    }
    for (const n of CLASSIC_VARIATIONS) expect(VARIATIONS, n).toHaveProperty(n);
    expect(Object.keys(defaultParams('julian')).sort()).toEqual(['dist', 'power']);
  });
});

describe('codegen: every preset and fixture flame compiles', () => {
  const fixtureDir = resolve(root, 'scripts/jwf-port/testflames');
  const fixtures = readdirSync(fixtureDir).filter((f) => f.endsWith('.flame') && !f.startsWith('_'));
  const check = (wgsl: string, label: string) => {
    expect(wgsl.length, label).toBeGreaterThan(1000);
    // (`NaN`/`undefined` would come from a JS number leaking into a literal — comments are stripped first)
    expect(wgsl.replace(/\/\/[^\n]*/g, ''), label).not.toMatch(/\$\{|\/\*ERROR|\bundefined\b|\bNaN\b|\bnull\b/);
    expect(wgsl, label).toContain('@compute');
  };

  it.each(PRESETS.map((p) => [p.name] as const))('preset %s', (name) => {
    const flame = PRESETS.find((p) => p.name === name)!.make();
    check(compileFlame(flame, 1024).wgsl, name);
  });

  it.each(fixtures.map((f) => [f] as const))('fixture %s', (file) => {
    const { flame, unknown } = importFlameText(readFileSync(resolve(fixtureDir, file), 'utf8'), GREY);
    // a fixture may use a deliberately-unported variation (the importer drops it with a reason) — never an unexplained one
    expect(unknown.filter((n) => !(n in UNPORTABLE)), file).toEqual([]);
    check(compileFlame(flame, 1024).wgsl, file);
  });
});
