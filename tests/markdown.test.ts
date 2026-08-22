import { describe, it, expect } from 'vitest';
import { renderMarkdown } from '../src/ui/markdown';

const html = (md: string) => { const d = document.createElement('div'); d.append(renderMarkdown(md)); return d.innerHTML; };

describe('assistant markdown', () => {
  it('paragraphs, emphasis, inline code, headings', () => {
    expect(html('Hello **bold** and *it* with `set T1.weight 0.8`.\n\nNext para.')).toBe('<p>Hello <strong>bold</strong> and <em>it</em> with <code>set T1.weight 0.8</code>.</p><p>Next para.</p>');
    expect(html('## Title\ntext')).toBe('<h4>Title</h4><p>text</p>');
  });
  it('lists and fenced code', () => {
    expect(html('- one\n- two **b**\n\n1. first\n2. second')).toBe('<ul><li>one</li><li>two <strong>b</strong></li></ul><ol><li>first</li><li>second</li></ol>');
    expect(html('```edits\nset brightness 2\n```')).toBe('<pre><code class="lang-edits">set brightness 2</code></pre>');
  });
  it('never injects HTML and only links http(s)', () => {
    expect(html('<img src=x onerror=alert(1)> & <b>no</b>')).toBe('<p>&lt;img src=x onerror=alert(1)&gt; &amp; &lt;b&gt;no&lt;/b&gt;</p>');
    expect(html('[ok](https://example.com/a) [bad](javascript:alert(1))')).toBe('<p><a href="https://example.com/a" target="_blank" rel="noopener noreferrer">ok</a> [bad](javascript:alert(1))</p>');
  });
  it('blockquotes and rules', () => {
    expect(html('> quoted\n> more\n\n---\nend')).toBe('<blockquote>quoted more</blockquote><hr><p>end</p>');
  });
});
