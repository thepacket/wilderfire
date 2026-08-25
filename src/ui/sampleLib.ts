// The built-in flames, shaped as library entries so the library dialog's grid, search, gallery and
// similarity work on them unchanged: WilderFire's own presets (core/presets.ts). They used to be a
// "Tests" list in the header, together with the JWildfire sample flames; the samples were dropped from
// the collection (the .flame files stay in public/flames — the compare harness and the first-visit
// flame use them).
//
// They are never written to the flame store — the user's library stays exactly what the user put in it —
// so a preset has no date, cannot be deleted, favourited or tagged. Only their thumbnails are cached
// (libraryStore's sampleThumbs), which is why the pictures are rendered once and not on every visit.
import { PRESETS } from '../core/presets';
import { sampleThumbAll, sampleThumbPrune, type LibEntry } from '../core/libraryStore';

export const SAMPLE_COUNT = PRESETS.length;
export const isSample = (e: LibEntry): boolean => e.id.startsWith('sample:');

/** All built-in flames, with any cached thumbnail attached. */
export async function sampleEntries(): Promise<LibEntry[]> {
  const thumbs = await sampleThumbAll().catch(() => new Map<string, Blob>());
  const entries = PRESETS.map((p, i) => ({
    id: `sample:p:${i}`,
    name: p.name,
    date: 0,
    flame: p.make(),
    thumb: thumbs.get(`sample:p:${i}`) ?? '',
    source: 'WilderFire preset',
  }));
  void sampleThumbPrune(entries.map((e) => e.id)).catch(() => { /* a stale picture is harmless */ });
  return entries;
}
