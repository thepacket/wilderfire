// Built-in flames. These are NOT the preset list any more (that is
// src/core/samples.ts: the JWildfire sample flames); they serve as the
// synchronous startup fallback (Ember Swirl) and as test fixtures (Clockwork).
import type { Flame } from './flame';
import { normalizeFlame } from './flame';
import { paletteFromPreset } from './palette';

interface PresetSpec { name: string; palette: string; data: any; }

const P: PresetSpec[] = [
  {
    name: 'Ember Swirl',
    palette: 'Ember',
    data: {
      brightness: 4, gamma: 4, zoom: 0.9,
      xforms: [
        {
          affine: [0.62, -0.4, 0.25, 0.4, 0.62, 0.1],
          weight: 1, color: 0.0, colorSpeed: 0.6,
          variations: [{ name: 'swirl', weight: 0.9 }, { name: 'linear', weight: 0.25 }],
        },
        {
          affine: [-0.55, 0.28, -0.6, -0.28, -0.55, 0.2],
          weight: 0.8, color: 0.55, colorSpeed: 0.5,
          variations: [{ name: 'spherical', weight: 1.0 }],
        },
        {
          affine: [0.35, 0, 0.8, 0, 0.35, -0.55],
          weight: 0.5, color: 0.95, colorSpeed: 0.7,
          variations: [{ name: 'sinusoidal', weight: 0.8 }, { name: 'bubble', weight: 0.3 }],
        },
      ],
    },
  },
  {
    name: 'Clockwork',
    palette: 'Solar',
    data: {
      brightness: 3.75, gamma: 3.75, zoom: .8,
      paletteStops: [[0, .05, .03, 0], [.35, .6, .25, .05], [.65, .95, .65, .15], [1, 1, .95, .75]],
      xforms: [
        { affine: [0.62, 0, 0.45, 0, 0.62, 0], weight: 1, color: 0, colorSpeed: .5,
          variations: [{ name: 'ngon', weight: .9, params: { power: 2, sides: 6, corners: 1, circle: .55 } }] },
        { affine: [-0.5, 0.5, -0.4, -0.5, -0.5, 0.35], weight: .8, color: .6, colorSpeed: .55,
          variations: [{ name: 'linear', weight: .6 }, { name: 'rings', weight: .35 }] },
        { affine: [-1, 0, 0, 0, 1, 0], weight: .7, color: .9, colorSpeed: 0,
          variations: [{ name: 'linear', weight: 1 }] },
      ],
    },
  },
];

export const PRESETS: { name: string; make: () => Flame }[] = P.map((spec) => ({
  name: spec.name,
  make: () => {
    const f = normalizeFlame(spec.data, paletteFromPreset(spec.palette));
    f.name = spec.name;
    return f;
  },
}));
