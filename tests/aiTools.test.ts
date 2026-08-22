import { describe, it, expect, vi } from 'vitest';
import { streamChat } from '../src/ai/openrouter';
import { runTool, TOOL_DEFS } from '../src/ai/tools';
import { DEFAULT_CONTEXT } from '../src/ai/context';

/** An SSE body from delta objects, the way OpenAI-compatible servers stream them. */
function sseBody(deltas: object[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  const lines = deltas.map((d) => `data: ${JSON.stringify({ choices: [{ index: 0, delta: d }] })}\n\n`).concat(['data: [DONE]\n\n']);
  return new ReadableStream({ start(c) { for (const l of lines) c.enqueue(enc.encode(l)); c.close(); } });
}

describe('streamChat tool calls', () => {
  it('assembles tool calls streamed as deltas (id/name once, arguments in pieces) alongside text', async () => {
    const args = JSON.stringify({ edits: 'set brightness 4.5' });
    const deltas = [
      { content: 'Sure — ' },
      { tool_calls: [{ index: 0, id: 'call_9', type: 'function', function: { name: 'apply_', arguments: '' } }] },
      { tool_calls: [{ index: 0, function: { name: 'edits', arguments: args.slice(0, 10) } }] },
      { tool_calls: [{ index: 0, function: { arguments: args.slice(10) } }] },
      { tool_calls: [{ index: 1, id: 'call_10', type: 'function', function: { name: 'screenshot', arguments: '{}' } }] },
    ];
    let sentBody: any = null;
    vi.stubGlobal('fetch', vi.fn(async (_url: string, init: RequestInit) => {
      sentBody = JSON.parse(String(init.body));
      return new Response(sseBody(deltas), { status: 200 });
    }));
    const text: string[] = [];
    const r = await streamChat({ apiKey: 'k', model: 'm', messages: [{ role: 'user', content: 'hi' }], tools: TOOL_DEFS, onDelta: (d) => text.push(d) });
    expect(r.text).toBe('Sure — ');
    expect(text).toEqual(['Sure — ']);
    expect(r.toolCalls).toEqual([
      { id: 'call_9', name: 'apply_edits', arguments: args },
      { id: 'call_10', name: 'screenshot', arguments: '{}' },
    ]);
    expect(sentBody.tools).toHaveLength(TOOL_DEFS.length);
    expect(sentBody.tool_choice).toBe('auto');
    vi.unstubAllGlobals();
  });
  it('sends no tools field when none are given', async () => {
    let sentBody: any = null;
    vi.stubGlobal('fetch', vi.fn(async (_u: string, init: RequestInit) => { sentBody = JSON.parse(String(init.body)); return new Response(sseBody([{ content: 'ok' }])); }));
    const r = await streamChat({ apiKey: 'k', model: 'm', messages: [{ role: 'user', content: 'hi' }] });
    expect(r).toEqual({ text: 'ok', toolCalls: [] });
    expect('tools' in sentBody).toBe(false);
    vi.unstubAllGlobals();
  });
});

describe('tools', () => {
  const env = { app: {} as any, ctx: DEFAULT_CONTEXT, screenshot: () => null, confirm: () => true };
  it('variation_lookup finds names and lists their parameters', async () => {
    const r = await runTool('variation_lookup', JSON.stringify({ query: 'julian', limit: 5 }), env);
    expect(r.text).toMatch(/julian — params \(defaults\): power=3, dist=1/);
    expect(r.image).toBeUndefined();
  });
  it('rejects bad argument JSON and unknown tools without throwing', async () => {
    expect((await runTool('variation_lookup', '{not json', env)).text).toMatch(/not valid JSON/);
    expect((await runTool('no_such_tool', '{}', env)).text).toMatch(/Unknown tool/);
  });
  it('every tool definition has a name, a description and an object schema', () => {
    for (const t of TOOL_DEFS) {
      expect(t.type).toBe('function');
      expect(t.function.name).toMatch(/^[a-z_]+$/);
      expect(t.function.description.length).toBeGreaterThan(20);
      expect((t.function.parameters as any).type).toBe('object');
    }
  });
});
