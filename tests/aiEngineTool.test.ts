import { describe, it, expect } from 'vitest';
import { runTool, TOOL_DEFS } from '../src/ai/tools';
import { DEFAULT_CONTEXT } from '../src/ai/context';
import { App } from '../src/ui/common';
import { defaultFlame } from '../src/core/flame';
import { GREY } from './helpers';

describe('engine tools', () => {
  const state = { mode: 'draft', qualityCap: 1000, stopAfterS: 30, speed: 'balanced', oversample: 1, dePreview: 'balanced', previewHold: 10, adaptiveBudget: true, paused: false };
  const calls: unknown[] = [];
  const app = { engine: { get: () => state, set: (p: Record<string, unknown>) => { calls.push(p); return { changed: Object.keys(p), state: { ...state, ...p } }; } } };
  const env = { app, ctx: DEFAULT_CONTEXT, screenshot: () => null, confirm: () => true } as unknown as Parameters<typeof runTool>[2];
  it('are offered to the model', () => {
    expect(TOOL_DEFS.map((t) => t.function.name)).toEqual(expect.arrayContaining(['get_engine', 'set_engine']));
  });
  it('get_engine reports the state', async () => {
    const r = await runTool('get_engine', '{}', env);
    expect(JSON.parse(r.text)).toEqual(state);
  });
  it('set_engine passes typed fields through and reports what changed', async () => {
    const r = await runTool('set_engine', JSON.stringify({ mode: 'Final', qualityCap: 4000, paused: true, bogus: 1, speed: 42 }), env);
    expect(calls[0]).toEqual({ mode: 'final', qualityCap: 4000, paused: true });
    expect(r.text).toMatch(/^Changed mode, qualityCap, paused\./);
  });
  it('says so when the engine bridge is missing', async () => {
    const r = await runTool('set_engine', '{"mode":"final"}', { ...env, app: {} } as typeof env);
    expect(r.text).toMatch(/not available/);
  });
});

describe('assistant tools: import, animation, library updates, theme', () => {
  // a real App with a stub renderer (the tools only need setFlame / the active layer / the bridges)
  const app = new App();
  app.renderer = { setFlame: () => {} } as unknown as App['renderer'];
  app.flame = defaultFlame(GREY);
  const env = { app, ctx: DEFAULT_CONTEXT, screenshot: () => null, confirm: () => true } as unknown as Parameters<typeof runTool>[2];
  it('every tool the system prompt names has a definition', () => {
    const names = new Set(TOOL_DEFS.map((t) => t.function.name));
    for (const n of ['import_flame', 'library_update', 'library_delete', 'get_animation', 'animate', 'animation_control', 'set_theme', 'export_png']) expect(names.has(n), n).toBe(true);
  });
  it('import_flame loads XML into the editor and reports a pack', async () => {
    const xml = '<flames><flame name="one" size="64 64" scale="10"><xform weight="1" color="0.3" linear="1" coefs="1 0 0 1 0 0"/></flame><flame name="two" size="64 64" scale="10"><xform weight="1" color="0.3" spherical="1" coefs="1 0 0 1 0 0"/></flame></flames>';
    const r = await runTool('import_flame', JSON.stringify({ text: xml }), env);
    expect(r.text).toContain('Loaded "one"');
    expect(r.text).toContain('first of 2');
    expect(app.flame.name).toBe('one');
    expect((await runTool('import_flame', JSON.stringify({ text: '   ' }), env)).text).toMatch(/empty/);
  });
  it('animate sets, replaces and removes motion curves through the app bridge; get_animation lists them', async () => {
    let curves: import('../src/core/motion').MotionCurve[] = [];
    app.getCurves = () => curves; app.setCurves = (c) => { curves = c; };
    expect((await runTool('animate', JSON.stringify({ path: 'nope.path', points: [[0, 1]] }), env)).text).toMatch(/not a numeric parameter/);
    const r1 = await runTool('animate', JSON.stringify({ path: 'zoom', points: [[2, 3], [0, 1]], interp: 'linear' }), env);
    expect(r1.text).toContain('zoom: 2 points over 0–2 s (linear)');
    expect(curves).toEqual([{ path: 'zoom', points: [{ t: 0, v: 1 }, { t: 2, v: 3 }], interp: 'linear' }]);
    await runTool('animate', JSON.stringify({ path: 'zoom', points: [[0, 1], [1, 2], [4, 1]] }), env);
    expect(curves[0].points.length).toBe(3); expect(curves[0].interp).toBe('linear');
    expect((await runTool('get_animation', '{}', env)).text).toContain('zoom [linear]: [0, 1] [1, 2] [4, 1]');
    expect((await runTool('animate', JSON.stringify({ path: 'zoom', remove: true }), env)).text).toContain('removed');
    expect(curves).toEqual([]);
    expect((await runTool('get_animation', '{}', env)).text).toMatch(/No animation/);
  });
  it('set_theme and animation_control go through the app bridges', async () => {
    let theme: 'dark' | 'light' = 'dark';
    app.theme = { get: () => theme, set: (t) => { theme = t; } };
    expect((await runTool('set_theme', JSON.stringify({ theme: 'light' }), env)).text).toBe('Theme: light.');
    expect(theme).toBe('light');
    expect((await runTool('set_theme', JSON.stringify({ theme: 'blue' }), env)).text).toMatch(/Error/);
    const calls: string[] = [];
    app.anim = { addKey: () => calls.push('key'), play: () => calls.push('play'), stop: () => calls.push('stop'), keyCount: () => calls.filter((c) => c === 'key').length };
    await runTool('animation_control', JSON.stringify({ action: 'keyframe' }), env);
    await runTool('animation_control', JSON.stringify({ action: 'play' }), env);
    await runTool('animation_control', JSON.stringify({ action: 'stop' }), env);
    expect(calls).toEqual(['key', 'play', 'stop']);
  });
});
