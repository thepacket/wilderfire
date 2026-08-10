// Minimal OpenRouter chat client (OpenAI-compatible API), streaming via SSE.
// Runs entirely in the browser with the user's own API key.

export type ChatPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ChatPart[];
}

export interface StreamOptions {
  apiKey: string;
  model: string;
  messages: ChatMessage[];
  onDelta?: (text: string) => void;
  signal?: AbortSignal;
}

export const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

export const SUGGESTED_MODELS = [
  'anthropic/claude-sonnet-4.5',
  'anthropic/claude-opus-4.1',
  'anthropic/claude-3.5-haiku',
  'openai/gpt-5',
  'openai/gpt-4o-mini',
  'google/gemini-2.5-pro',
  'google/gemini-2.5-flash',
  'meta-llama/llama-3.3-70b-instruct',
  'deepseek/deepseek-chat-v3.1',
];

export async function streamChat(opts: StreamOptions): Promise<string> {
  const res = await fetch(OPENROUTER_URL, {
    method: 'POST',
    signal: opts.signal,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${opts.apiKey}`,
      'X-Title': 'WilderFire',
    },
    body: JSON.stringify({
      model: opts.model,
      messages: opts.messages,
      stream: true,
    }),
  });

  if (!res.ok) {
    let detail = '';
    try {
      const j = await res.json();
      detail = j?.error?.message ?? JSON.stringify(j);
    } catch { detail = await res.text().catch(() => ''); }
    throw new Error(`OpenRouter error ${res.status}: ${detail}`);
  }
  if (!res.body) throw new Error('No response body from OpenRouter.');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let full = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      try {
        const j = JSON.parse(payload);
        const delta: string = j?.choices?.[0]?.delta?.content ?? '';
        if (delta) {
          full += delta;
          opts.onDelta?.(delta);
        }
      } catch { /* keep-alive or partial line — ignore */ }
    }
  }
  return full;
}
