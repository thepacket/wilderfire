// The preset list: all 33 sample flames bundled with JWildfire
// (https://github.com/thargor6/JWildfire, resources/flames/, © Andreas Maschke,
// LGPL 2.1+), loaded through our .flame importer and verified against headless
// JWildfire renders (scripts/jwf-port compare harness). Names are JWildfire's
// own file ids.

export interface SampleFlame {
  file: string;  // under /flames/
  name: string;  // display name
}

export const JWF_SAMPLES: SampleFlame[] = [
  { file: 'TINA0001.flame', name: 'TINA0001' },
  { file: 'TINA0002.flame', name: 'TINA0002' },
  { file: 'TINA0003.flame', name: 'TINA0003' },
  { file: 'TINA0004.flame', name: 'TINA0004' },
  { file: 'TINA0005.flame', name: 'TINA0005' },
  { file: 'TINA0006.flame', name: 'TINA0006' },
  { file: 'TINA0007.flame', name: 'TINA0007' },
  { file: 'TINA0008.flame', name: 'TINA0008' },
  { file: 'TINA0009.flame', name: 'TINA0009' },
  { file: 'TINA0010.flame', name: 'TINA0010' },
  { file: 'TINA0011.flame', name: 'TINA0011' },
  { file: 'TINA0012.flame', name: 'TINA0012' },
  { file: 'TINA0013.flame', name: 'TINA0013' },
  { file: 'TINA0014.flame', name: 'TINA0014' },
  { file: 'TINA0015.flame', name: 'TINA0015' },
  { file: 'TINA0016.flame', name: 'TINA0016' },
  { file: 'TINA0017.flame', name: 'TINA0017' },
  { file: 'TINA0018.flame', name: 'TINA0018' },
  { file: 'TINA0019.flame', name: 'TINA0019' },
  { file: 'TINA0020.flame', name: 'TINA0020' },
  { file: 'TINA0021.flame', name: 'TINA0021' },
  { file: 'TINA0022.flame', name: 'TINA0022' },
  { file: 'TINA0023.flame', name: 'TINA0023' },
  { file: 'TINA0024.flame', name: 'TINA0024' },
  { file: 'TINA0025.flame', name: 'TINA0025' },
  { file: 'TINA0026.flame', name: 'TINA0026' },
  { file: 'TINA0027.flame', name: 'TINA0027' },
  { file: 'TINA0028.flame', name: 'TINA0028' },
  { file: 'TINA0029.flame', name: 'TINA0029' },
  { file: 'TINA0030.flame', name: 'TINA0030' },
  { file: 'TINA0031.flame', name: 'TINA0031' },
  { file: 'TINA0032.flame', name: 'TINA0032' },
  { file: 'TINA0033.flame', name: 'TINA0033' },
];
