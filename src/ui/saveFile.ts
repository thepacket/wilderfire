import { toast } from './common';
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
  /** the file name chosen (the dialog's, or the suggested one for a download) */
  name: string;
  write(blob: Blob): Promise<void>;
}

/** The message to show once `blob` went to `target`: "Saved x.png (1.2 MB)" or "Downloaded x.png (…)". */
export const savedMessage = (target: SaveTarget, blob: Blob): string => {
  const size = blob.size >= 1e6 ? `${(blob.size / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(blob.size / 1e3))} KB`;
  return `${target.kind === 'file' ? 'Saved' : 'Downloaded'} ${target.name} (${size})`;
};

type FSHandle = { name?: string; createWritable(): Promise<{ write(b: Blob): Promise<void>; close(): Promise<void> }> };
type Picker = (o: { suggestedName: string; types: { description: string; accept: Record<string, string[]> }[] }) => Promise<FSHandle>;

export const hasSaveDialog = () => typeof (window as any).showSaveFilePicker === 'function' && window.isSecureContext;

function downloadTarget(name: string): SaveTarget {
  return {
    kind: 'download',
    name,
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
      name: handle.name || opts.suggestedName,
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
  try { await t.write(blob); }
  catch (e) { toast(`⚠ Could not save ${t.name}: ${(e as Error).message}`, 'error'); throw e; }
  toast(savedMessage(t, blob));
  return true;
}

export const saveText = (text: string, opts: SaveOpts) => saveBlob(new Blob([text], { type: opts.mime }), opts);

export interface DirTarget {
  /** 'dir' = user-picked folder (File System Access API), 'download' = one browser download per file. */
  kind: 'dir' | 'download';
  /** Folder name for display ('' for downloads). */
  name: string;
  write(fileName: string, blob: Blob): Promise<void>;
}

type DirHandle = { name: string; getFileHandle(name: string, o: { create: boolean }): Promise<FSHandle> };

export const hasDirDialog = () => typeof (window as any).showDirectoryPicker === 'function' && window.isSecureContext;

/** Ask for a folder to write several files into (batch export). Falls back to per-file
 *  downloads where there is no directory picker. Returns null if the user cancelled. */
export async function pickDirectory(): Promise<DirTarget | null> {
  const downloads: DirTarget = { kind: 'download', name: '', write: (name, blob) => downloadTarget(name).write(blob) };
  if (!hasDirDialog()) return downloads;
  try {
    const dir = await ((window as any).showDirectoryPicker as (o: { mode: string }) => Promise<DirHandle>)({ mode: 'readwrite' });
    return {
      kind: 'dir',
      name: dir.name,
      async write(fileName, blob) {
        const h = await dir.getFileHandle(fileName, { create: true });
        const w = await h.createWritable();
        await w.write(blob);
        await w.close();
      },
    };
  } catch (e) {
    if ((e as DOMException)?.name === 'AbortError') return null;
    return downloads;
  }
}
