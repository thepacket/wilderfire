// Random flame generator — curated to keep results contractive and interesting.
import type { Flame, XForm, Affine } from './flame';
import { defaultFlame } from './flame';
import { defaultParams } from './variations';
import { randomPalette } from './palette';

export const SAFE_VARIATIONS = [
  'linear', 'sinusoidal', 'spherical', 'swirl', 'horseshoe', 'polar', 'disc',
  'spiral', 'hyperbolic', 'diamond', 'julia', 'fisheye', 'eyefish', 'bubble',
  'pdj', 'curl', 'julian', 'juliascope', 'ex', 'heart', 'handkerchief',
  'waves', 'blob', 'ngon', 'cylinder', 'tangent', 'popcorn', 'rings', 'fan2',
];

const rr = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
const pick = <T,>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];

function randomAffine(): Affine {
  const theta = rr(0, Math.PI * 2);
  const s = rr(0.35, 0.9);
  const skew = rr(-0.25, 0.25);
  const c = Math.cos(theta) * s, sn = Math.sin(theta) * s;
  return [
    c, -sn + skew, rr(-1, 1),
    sn, c + skew * 0.5, rr(-1, 1),
  ];
}

function randomXForm(colorIdx: number): XForm {
  const nVars = Math.random() < 0.55 ? 1 : 2;
  const chosen = new Set<string>();
  while (chosen.size < nVars) chosen.add(pick(SAFE_VARIATIONS));
  const names = [...chosen];
  const variations = names.map((name, i) => ({
    name,
    weight: i === 0 ? rr(0.6, 1.1) : rr(0.15, 0.5),
    params: jitterParams(name),
  }));
  return {
    affine: randomAffine(),
    post: [1, 0, 0, 0, 1, 0],
    weight: rr(0.35, 1),
    color: colorIdx,
    colorSpeed: rr(0.35, 0.75),
    opacity: 1,
    variations,
  };
}

function jitterParams(name: string): Record<string, number> {
  const params = defaultParams(name);
  for (const k of Object.keys(params)) {
    params[k] = params[k] * rr(0.7, 1.4);
    if (name === 'julian' || name === 'juliascope') {
      if (k === 'power') params[k] = Math.max(2, Math.round(rr(2, 6))) * (Math.random() < 0.3 ? -1 : 1);
      if (k === 'dist') params[k] = rr(0.5, 1.5);
    }
    if (name === 'ngon' && k === 'sides') params[k] = Math.max(3, Math.round(rr(3, 8)));
    if (name === 'blob' && k === 'waves') params[k] = Math.max(2, Math.round(rr(2, 8)));
  }
  return params;
}

export function randomFlame(): Flame {
  const n = 2 + Math.floor(Math.random() * 3); // 2..4 xforms
  const f = defaultFlame(randomPalette());
  f.name = 'random-' + Math.random().toString(36).slice(2, 7);
  const ly = f.layers[0];
  ly.xforms = [];
  for (let i = 0; i < n; i++) {
    ly.xforms.push(randomXForm(n === 1 ? 0 : i / (n - 1)));
  }
  // Occasionally add a gentle final transform for extra structure.
  if (Math.random() < 0.25) {
    ly.final = {
      affine: [1, 0, 0, 0, 1, 0],
      post: [1, 0, 0, 0, 1, 0],
      weight: 1,
      color: Math.random(),
      colorSpeed: 0.2,
      opacity: 1,
      variations: [{
        name: pick(['spherical', 'julia', 'bubble', 'eyefish', 'polar']),
        weight: rr(0.7, 1.1),
        params: {},
      }],
    };
  }
  f.brightness = rr(2.8, 4);
  f.gamma = rr(3.2, 4);
  f.gammaThreshold = 0.04;
  f.zoom = rr(0.7, 1.1);
  return f;
}
