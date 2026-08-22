import { describe, it, expect } from 'vitest';
import { dataUrlToBlob, thumbDataUrl, thumbSrc } from '../src/core/libraryStore';

describe('library thumbnails as Blobs', () => {
  it('decodes a base64 data URL into a Blob of the right type and bytes', async () => {
    const bytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 1, 2, 250]);
    const url = 'data:image/jpeg;base64,' + btoa(String.fromCharCode(...bytes));
    const b = dataUrlToBlob(url)!;
    expect(b).toBeInstanceOf(Blob);
    expect(b.type).toBe('image/jpeg');
    expect(new Uint8Array(await b.arrayBuffer())).toEqual(bytes);
  });
  it('handles percent-encoded data URLs and rejects other strings', async () => {
    const b = dataUrlToBlob('data:text/plain,hi%20there')!;
    expect(await b.text()).toBe('hi there');
    expect(dataUrlToBlob('https://example.com/x.jpg')).toBeNull();
    expect(dataUrlToBlob('')).toBeNull();
  });
  it('keeps strings as they are on the way to JSON and to <img>', async () => {
    const url = 'data:image/jpeg;base64,AAEC';
    expect(await thumbDataUrl(url)).toBe(url);
    expect(thumbSrc(url)).toBe(url);
  });
});
