import { describe, it, expect } from 'vitest';
import { runTool, TOOL_DEFS } from '../src/ai/tools';
import { DEFAULT_CONTEXT } from '../src/ai/context';

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
