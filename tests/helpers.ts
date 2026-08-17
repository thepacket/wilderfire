import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { PRESETS } from '../src/core/presets';
import { defaultFlame, type Flame, type RGB } from '../src/core/flame';

export const GREY: RGB[] = Array.from({ length: 256 }, (_, i) => [i / 255, i / 255, i / 255]);
export const preset = (name: string): Flame => {
  const p = PRESETS.find((p) => p.name.toLowerCase().includes(name.toLowerCase()));
  if (!p) throw new Error(`no preset ${name}`);
  return p.make();
};
export const blank = () => defaultFlame(GREY);
export const fixture = (file: string) => readFileSync(resolve(process.cwd(), 'scripts/jwf-port/testflames', file), 'utf8');
export const close = (a: number, b: number, eps = 1e-6) => Math.abs(a - b) <= eps;
