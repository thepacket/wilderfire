// A small Markdown renderer for the assistant's replies, built as DOM nodes (never innerHTML on model text):
// headings, paragraphs, bullet / numbered lists, blockquotes, fenced and inline code, **bold**, *italic*,
// ~~strike~~, links (http/https only, opened in a new tab), horizontal rules. Anything else stays literal.
import { el } from './common';

const LINK = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/;
const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\s][^*]*\*)|(_[^_\s][^_]*_)|(~~[^~]+~~)|(\[[^\]]+\]\(https?:\/\/[^\s)]+\))/;

function inline(text: string, into: HTMLElement): void {
  let rest = text;
  while (rest) {
    const m = INLINE.exec(rest);
    if (!m) { into.append(rest); return; }
    if (m.index > 0) into.append(rest.slice(0, m.index));
    const tok = m[0];
    if (tok.startsWith('`')) into.append(el('code', '', tok.slice(1, -1)));
    else if (tok.startsWith('**') || tok.startsWith('__')) { const b = el('strong'); inline(tok.slice(2, -2), b); into.append(b); }
    else if (tok.startsWith('~~')) { const s = el('s'); inline(tok.slice(2, -2), s); into.append(s); }
    else if (tok.startsWith('[')) { const lm = LINK.exec(tok)!; const a = el('a', '', lm[1]); a.href = lm[2]; a.target = '_blank'; a.rel = 'noopener noreferrer'; into.append(a); }
    else { const i = el('em'); inline(tok.slice(1, -1), i); into.append(i); }
    rest = rest.slice(m.index + tok.length);
  }
}

/** Render `md` into a fresh fragment. */
export function renderMarkdown(md: string): DocumentFragment {
  const frag = document.createDocumentFragment();
  const lines = md.replace(/\r\n?/g, '\n').split('\n');
  let i = 0;
  let para: string[] = [];
  const flush = () => { if (para.length) { const p = el('p'); inline(para.join(' '), p); frag.append(p); para = []; } };
  while (i < lines.length) {
    const line = lines[i];
    const fence = /^\s*```(\w*)\s*$/.exec(line);
    if (fence) {
      flush();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++; // closing fence (or end)
      const pre = el('pre');
      const code = el('code', fence[1] ? `lang-${fence[1]}` : '', body.join('\n'));
      pre.append(code); frag.append(pre);
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) { flush(); const hd = el(`h${Math.min(6, h[1].length + 2)}` as 'h3'); inline(h[2].trim(), hd); frag.append(hd); i++; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { flush(); frag.append(el('hr')); i++; continue; }
    const ul = /^\s*[-*+]\s+(.*)$/.exec(line), ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (ul || ol) {
      flush();
      const list = el(ol ? 'ol' : 'ul');
      const re = ol ? /^\s*\d+[.)]\s+(.*)$/ : /^\s*[-*+]\s+(.*)$/;
      while (i < lines.length) {
        const m = re.exec(lines[i]);
        if (!m) break;
        const li = el('li');
        let text = m[1];
        i++;
        // continuation lines (indented, not a new item / blank)
        while (i < lines.length && /^\s+\S/.test(lines[i]) && !re.test(lines[i]) && !/^\s*[-*+]\s+/.test(lines[i]) && !/^\s*\d+[.)]\s+/.test(lines[i])) text += ' ' + lines[i++].trim();
        inline(text, li); list.append(li);
      }
      frag.append(list);
      continue;
    }
    const q = /^\s*>\s?(.*)$/.exec(line);
    if (q) {
      flush();
      const bq = el('blockquote'); const parts: string[] = [];
      while (i < lines.length) { const qm = /^\s*>\s?(.*)$/.exec(lines[i]); if (!qm) break; parts.push(qm[1]); i++; }
      inline(parts.join(' '), bq); frag.append(bq);
      continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line.trim());
    i++;
  }
  flush();
  return frag;
}

/** Replace `target`'s content with the rendered markdown (keeps it cheap to call on every streamed delta). */
export function setMarkdown(target: HTMLElement, md: string): void {
  target.replaceChildren(renderMarkdown(md));
  target.classList.add('md');
}
