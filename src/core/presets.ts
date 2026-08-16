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
    name: 'Deep Julia',
    palette: 'Ocean',
    data: {
      brightness: 3.75, gamma: 3.75, zoom: 0.8,
      xforms: [
        {
          affine: [0.9, 0.05, 0.3, -0.05, 0.9, 0.0],
          weight: 1, color: 0.1, colorSpeed: 0.55,
          variations: [{ name: 'julian', weight: 1, params: { power: 3, dist: 1 } }],
        },
        {
          affine: [0.5, -0.5, -0.5, 0.5, 0.5, 0.3],
          weight: 0.7, color: 0.7, colorSpeed: 0.5,
          variations: [{ name: 'spherical', weight: 0.9 }, { name: 'diamond', weight: 0.2 }],
        },
      ],
    },
  },
  {
    name: 'Night Bloom',
    palette: 'Violet',
    data: {
      brightness: 4.25, gamma: 4.25, zoom: 0.85,
      xforms: [
        {
          affine: [0.75, -0.35, 0.0, 0.35, 0.75, 0.0],
          weight: 1, color: 0.05, colorSpeed: 0.5,
          variations: [{ name: 'spiral', weight: 0.7 }, { name: 'linear', weight: 0.4 }],
        },
        {
          affine: [-0.6, 0.0, 0.7, 0.0, -0.6, -0.3],
          weight: 0.85, color: 0.5, colorSpeed: 0.6,
          variations: [{ name: 'julia', weight: 0.95 }],
        },
        {
          affine: [0.42, 0.3, -0.5, -0.3, 0.42, 0.5],
          weight: 0.45, color: 0.9, colorSpeed: 0.65,
          variations: [{ name: 'curl', weight: 0.9, params: { c1: 0.4, c2: 0.2 } }],
        },
      ],
    },
  },
  {
    name: 'Aurora Fan',
    palette: 'Aurora',
    data: {
      brightness: 4, gamma: 4, zoom: 0.75,
      xforms: [
        {
          affine: [0.8, 0.15, 0.1, -0.15, 0.8, 0.15],
          weight: 1, color: 0.0, colorSpeed: 0.5,
          variations: [{ name: 'pdj', weight: 0.8, params: { a: 1.1, b: 2.3, c: 0.9, d: 1.8 } }],
        },
        {
          affine: [0.5, -0.55, -0.35, 0.55, 0.5, -0.25],
          weight: 0.75, color: 0.6, colorSpeed: 0.55,
          variations: [{ name: 'hyperbolic', weight: 0.7 }, { name: 'linear', weight: 0.35 }],
        },
        {
          affine: [0.3, 0.0, 0.6, 0.0, 0.3, 0.6],
          weight: 0.4, color: 1.0, colorSpeed: 0.7,
          variations: [{ name: 'gaussian_blur', weight: 0.25 }],
        },
      ],
    },
  },
  {
    name: 'Solar Disc',
    palette: 'Solar',
    data: {
      brightness: 3.75, gamma: 3.75, zoom: 0.95,
      xforms: [
        {
          affine: [0.7, -0.5, 0.0, 0.5, 0.7, 0.0],
          weight: 1, color: 0.1, colorSpeed: 0.5,
          variations: [{ name: 'disc', weight: 0.9 }, { name: 'linear', weight: 0.2 }],
        },
        {
          affine: [-0.45, 0.32, 0.55, -0.32, -0.45, -0.4],
          weight: 0.8, color: 0.65, colorSpeed: 0.6,
          variations: [{ name: 'horseshoe', weight: 0.7 }, { name: 'sinusoidal', weight: 0.3 }],
        },
      ],
    },
  },
];

// Feature-showcase presets (xaos, direct-color, pre-stages, layers, symmetry).
const P2: PresetSpec[] = [
  {
    name: 'Neon Orbit',
    palette: 'Ember',
    data: {
      brightness: 4.25, gamma: 4, zoom: 0.8,
      paletteStops: [[0, .02, 0, .12], [.3, 0, .35, 1], [.6, 0, 1, .8], [.85, 1, .9, .25], [1, 1, 1, 1]],
      xforms: [
        { affine: [-0.38, -0.65, 0.6, 0.65, -0.38, 0], weight: 1, color: 0, colorSpeed: .55, xaos: [0, 1, 1],
          variations: [{ name: 'julian', weight: 1, params: { power: 3, dist: 1 } }] },
        { affine: [0.35, 0.61, -0.5, -0.61, 0.35, 0.3], weight: .8, color: .5, colorSpeed: .5, xaos: [1, 0, 1],
          variations: [{ name: 'spherical', weight: .8 }, { name: 'swirl', weight: .3 }] },
        { affine: [0.6, 0, 0, 0, 0.6, -0.6], weight: .6, color: 1, colorSpeed: .6, xaos: [1, 1, 0],
          variations: [{ name: 'sinusoidal', weight: .9 }] },
      ],
    },
  },
  {
    name: 'Golden Nautilus',
    palette: 'Ember',
    data: {
      brightness: 4, gamma: 4, zoom: .75,
      paletteStops: [[0, .02, .01, 0], [.4, .45, .28, .05], [.7, 1, .75, .2], [1, 1, .98, .9]],
      xforms: [
        { affine: [0.68, -0.32, 0.15, 0.32, 0.68, 0.05], weight: 1, color: 0, colorSpeed: 0,
          variations: [{ name: 'dc_radial', weight: .85, params: { scale: .55, offset: .1 } }, { name: 'linear', weight: .35 }] },
        { affine: [-0.6, 0.25, -0.7, -0.25, -0.6, 0.3], weight: 1, color: .5, colorSpeed: .3,
          variations: [{ name: 'curl', weight: 1, params: { c1: .45, c2: .25 } }, { name: 'linear', weight: .2 }] },
        { affine: [0.45, 0.45, 0.5, -0.45, 0.45, -0.5], weight: .6, color: .9, colorSpeed: .4,
          variations: [{ name: 'horseshoe', weight: .6 }, { name: 'sinusoidal', weight: .3 }] },
      ],
    },
  },
  {
    name: 'Silk Veil',
    palette: 'Violet',
    data: {
      brightness: 4.25, gamma: 4, zoom: .8,
      paletteStops: [[0, .03, 0, .08], [.35, .4, .1, .5], [.65, .9, .4, .6], [1, 1, .95, .9]],
      xforms: [
        { affine: [0.6, -0.6, 0.2, 0.6, 0.6, 0], weight: 1, color: .05, colorSpeed: .55,
          preVariations: [{ name: 'linear', weight: 1 }, { name: 'gaussian_blur', weight: .06 }],
          variations: [{ name: 'julian', weight: 1, params: { power: 4, dist: 1 } }] },
        { affine: [-0.5, 0, 0.55, 0, -0.5, -0.3], weight: .7, color: .8, colorSpeed: .5,
          variations: [{ name: 'spherical', weight: .95 }] },
      ],
    },
  },
  {
    name: 'Aurora Veils',
    palette: 'Aurora',
    data: {
      brightness: 4.5, gamma: 4.25, zoom: .75,
      layers: [
        { weight: 1, visible: true,
          paletteStops: [[0, 0, .05, .03], [.4, .05, .5, .25], [.7, .3, .95, .5], [1, .9, 1, .9]],
          xforms: [
            { affine: [0.85, .1, 0, -.1, .85, .15], weight: 1, color: .1, colorSpeed: .45,
              variations: [{ name: 'pdj', weight: .85, params: { a: 1.1, b: 2.1, c: .8, d: 1.9 } }] },
            { affine: [0.5, -.5, -.3, .5, .5, -.3], weight: .7, color: .7, colorSpeed: .5,
              variations: [{ name: 'hyperbolic', weight: .7 }, { name: 'linear', weight: .3 }] },
          ] },
        { weight: .4, visible: true,
          paletteStops: [[0, .1, 0, .25], [.6, .5, .2, .9], [1, .95, .85, 1]],
          xforms: [
            { affine: [0.7, 0, 0, 0, 0.7, 0], weight: 1, color: .5, colorSpeed: .5,
              variations: [{ name: 'gaussian_blur', weight: .35 }, { name: 'bubble', weight: .5 }] },
          ] },
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

export const PRESETS: { name: string; make: () => Flame }[] = [...P, ...P2].map((spec) => ({
  name: spec.name,
  make: () => {
    const f = normalizeFlame(spec.data, paletteFromPreset(spec.palette));
    f.name = spec.name;
    return f;
  },
}));
