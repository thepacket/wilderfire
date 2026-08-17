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
  /** OpenRouter key; may be empty for a local endpoint */
  apiKey: string;
  /** Any OpenAI-compatible base URL (…/v1) instead of OpenRouter — Ollama, LM Studio, llama.cpp, vLLM */
  baseUrl?: string;
  model: string;
  messages: ChatMessage[];
  onDelta?: (text: string) => void;
  signal?: AbortSignal;
}

export const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';

/** Offline fallback for the model picker (the live list comes from fetchModels()).
 *  IDs verified against https://openrouter.ai/api/v1/models on 2026-08-16. */
export const SUGGESTED_MODELS = [
  'anthropic/claude-sonnet-5',
  'anthropic/claude-opus-5',
  'anthropic/claude-fable-5',
  'anthropic/claude-haiku-4.5',
  'openai/gpt-5',
  'google/gemini-2.5-pro',
  'google/gemini-2.5-flash',
  'deepseek/deepseek-chat-v3.1',
];

export interface ORModel {
  id: string;
  name: string;
  provider: string;       // id prefix, e.g. "anthropic"
  context: number;        // tokens
  promptPerM: number;     // USD per 1M input tokens
  completionPerM: number; // USD per 1M output tokens
  vision: boolean;        // accepts image input
  created: number;        // unix seconds
}

const MODELS_URL = 'https://openrouter.ai/api/v1/models';
const LS_MODELS = 'wilderfire.openrouter.models';
const MODELS_TTL_MS = 24 * 3600 * 1000;

/** Live model catalogue (public endpoint, no key), cached in localStorage for a day.
 *  Excludes `:batch` variants (not usable for interactive chat). */
export async function fetchModels(opts: { force?: boolean } = {}): Promise<ORModel[]> {
  if (!opts.force) {
    try {
      const raw = localStorage.getItem(LS_MODELS);
      if (raw) {
        const c = JSON.parse(raw) as { at: number; models: ORModel[] };
        if (Date.now() - c.at < MODELS_TTL_MS && Array.isArray(c.models) && c.models.length) return c.models;
      }
    } catch { /* ignore cache errors */ }
  }
  const res = await fetch(MODELS_URL);
  if (!res.ok) throw new Error(`OpenRouter models: HTTP ${res.status}`);
  const j = await res.json() as { data: any[] };
  const models: ORModel[] = (j.data ?? [])
    .filter((m) => typeof m?.id === 'string' && !m.id.endsWith(':batch'))
    .map((m) => ({
      id: m.id,
      name: String(m.name ?? m.id),
      provider: String(m.id).split('/')[0],
      context: Number(m.context_length ?? 0),
      promptPerM: Number(m.pricing?.prompt ?? 0) * 1e6,
      completionPerM: Number(m.pricing?.completion ?? 0) * 1e6,
      vision: Array.isArray(m.architecture?.input_modalities) && m.architecture.input_modalities.includes('image'),
      created: Number(m.created ?? 0),
    }));
  try { localStorage.setItem(LS_MODELS, JSON.stringify({ at: Date.now(), models })); } catch { /* quota */ }
  return models;
}

/** `…/v1/chat/completions` for a local base URL (with or without a trailing slash or `/v1`). */
export const chatUrl = (baseUrl?: string) => baseUrl ? baseUrl.replace(/\/+$/, '').replace(/\/chat\/completions$/, '') + '/chat/completions' : OPENROUTER_URL;

/** Model ids offered by a local OpenAI-compatible server (`GET …/v1/models`). */
export async function fetchLocalModels(baseUrl: string): Promise<string[]> {
  const res = await fetch(baseUrl.replace(/\/+$/, '') + '/models');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const j = await res.json() as { data?: { id?: string }[] };
  return (j.data ?? []).map((m) => String(m.id ?? '')).filter(Boolean);
}

export async function streamChat(opts: StreamOptions): Promise<string> {
  const res = await fetch(chatUrl(opts.baseUrl), {
    method: 'POST',
    signal: opts.signal,
    headers: {
      'Content-Type': 'application/json',
      ...(opts.apiKey ? { Authorization: `Bearer ${opts.apiKey}` } : {}),
      ...(opts.baseUrl ? {} : { 'X-Title': 'WilderFire' }),
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
