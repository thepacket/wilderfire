// Saving files: a real "Save as…" dialog where the browser has the File System
// Access API (Chrome/Edge), falling back to a plain download elsewhere (Safari,
// Firefox). Long jobs (hi-res tiles, video encode) should call pickSave() inside
// the click handler — the dialog needs a user gesture — and write when done.

export interface SaveOpts {
  suggestedName: string;
  /** e.g. 'PNG image' — shown in the dialog's type menu */
  description: string;
  mime: string;
  /** e.g. '.png' */
  ext: string;
}

export interface SaveTarget {
  /** Where the bytes will go — 'file' = user-picked path, 'download' = browser download folder. */
  kind: 'file' | 'download';
  write(blob: Blob): Promise<void>;
}

type FSHandle = { createWritable(): Promise<{ write(b: Blob): Promise<void>; close(): Promise<void> }> };
type Picker = (o: { suggestedName: string; types: { description: string; accept: Record<string, string[]> }[] }) => Promise<FSHandle>;

export const hasSaveDialog = () => typeof (window as any).showSaveFilePicker === 'function' && window.isSecureContext;

function downloadTarget(name: string): SaveTarget {
  return {
    kind: 'download',
    async write(blob) {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    },
  };
}

/** Ask where to save. Returns null if the user cancelled the dialog. */
export async function pickSave(opts: SaveOpts): Promise<SaveTarget | null> {
  if (!hasSaveDialog()) return downloadTarget(opts.suggestedName);
  try {
    const picker = (window as any).showSaveFilePicker as Picker;
    const handle = await picker({
      suggestedName: opts.suggestedName,
      types: [{ description: opts.description, accept: { [opts.mime]: [opts.ext] } }],
    });
    return {
      kind: 'file',
      async write(blob) {
        const w = await handle.createWritable();
        await w.write(blob);
        await w.close();
      },
    };
  } catch (e) {
    if ((e as DOMException)?.name === 'AbortError') return null; // user cancelled
    // Not allowed (no user gesture, sandboxed iframe, …) → plain download.
    return downloadTarget(opts.suggestedName);
  }
}

/** One-shot: pick + write. Resolves true if saved, false if cancelled. */
export async function saveBlob(blob: Blob, opts: SaveOpts): Promise<boolean> {
  const t = await pickSave(opts);
  if (!t) return false;
  await t.write(blob);
  return true;
}

export const saveText = (text: string, opts: SaveOpts) => saveBlob(new Blob([text], { type: opts.mime }), opts);
