// Sample flames bundled from the JWildfire repository
// (https://github.com/thargor6/JWildfire, resources/flames/, © Andreas Maschke,
// LGPL 2.1+). Loaded through our .flame importer; JWildfire's 3D-flavored
// variations are alias-mapped to their z=0 2D equivalents, so these render
// faithfully in structure though not pixel-identically to JWildfire.

export interface SampleFlame {
  file: string;  // under /flames/
  name: string;  // display name
}

export const JWF_SAMPLES: SampleFlame[] = [
  { file: 'TINA0002.flame', name: 'Golden Silk' },
  { file: 'TINA0004.flame', name: 'Ember Rings' },
  { file: 'TINA0005.flame', name: 'Teal Feathers' },
  { file: 'TINA0008.flame', name: 'Coral Bloom' },
  { file: 'TINA0009.flame', name: 'Storm Eye' },
  { file: 'TINA0010.flame', name: 'Frost Coral' },
  { file: 'TINA0011.flame', name: 'Green Lattice' },
  { file: 'TINA0016.flame', name: 'Twin Galaxies' },
  { file: 'TINA0018.flame', name: 'Nebula Ring' },
  { file: 'TINA0019.flame', name: 'Magenta Weave' },
  { file: 'TINA0022.flame', name: 'Violet Orb' },
  { file: 'TINA0025.flame', name: 'Orchid Swirl' },
  { file: 'TINA0029.flame', name: 'Fire Clouds' },
  { file: 'TINA0031.flame', name: 'Rose Burst' },
  { file: 'TINA0032.flame', name: 'Gilded Splash' },
  { file: 'TINA0033.flame', name: 'Deep Current' },
];
