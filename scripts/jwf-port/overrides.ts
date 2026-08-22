// Hand corrections applied before transpiling. JWildfire's GPU snippets are
// not always in sync with its CPU (Java) implementation, which is what .flame
// files are authored against; the oracle harness (README.md) flags the
// divergent ones and these overrides bring the WGSL back to CPU semantics.
// Snippets stay in the CUDA dialect so they go through the same transpiler.

export interface Override {
  gpuCode?: string; gpuFunctions?: string; note: string;
  /** text appended to the original snippet */
  append?: string;
  /** JWildfire CPU rejection-samples up to N times before hiding the point; the GPU
   *  snippet samples once. Wrap the snippet in a retry loop that breaks on success. */
  retry?: number;
  /** Java setParameter() clamps these params (Tools.limitValue); the GPU snippet reads
   *  them raw. Reads of `__<var>_<param>` are wrapped in the same clamp. */
  clampParams?: Record<string, [number, number]>;
  /** textual patches (from → to) applied to the snippet / the helper functions before transpiling */
  patch?: [string | RegExp, string][];
  patchFuncs?: [string | RegExp, string][];
}

export const OVERRIDES: Record<string, Override> = {
  ovoid3d: {
    note: 'CPU uses x²+y²+z²; GPU used the 2D r².',
    gpuCode: `float T = __x*__x + __y*__y + __z*__z + epsilon;
float r = __ovoid3d / T;
__px += __x * r * __ovoid3d_x;
__py += __y * r * __ovoid3d_y;
__pz += __z * r * __ovoid3d_z;`,
  },
  linearT3D: {
    note: 'GPU used powY for the z component; CPU uses powZ.',
    gpuCode: `__px += (__x < 0.f ? -1.f : 1.f) * powf(fabsf(__x), __linearT3D_powX) * __linearT3D;
__py += (__y < 0.f ? -1.f : 1.f) * powf(fabsf(__y), __linearT3D_powY) * __linearT3D;
__pz += (__z < 0.f ? -1.f : 1.f) * powf(fabsf(__z), __linearT3D_powZ) * __linearT3D;`,
  },
  brick: {
    note: 'CPU uses a floored modulo for the row parity; GPU used fmod (wrong sign for negative rows).',
    gpuCode: `float br_sx = fmaxf(fabsf(__brick_scale_x), 1e-4f);
float br_sy = fmaxf(fabsf(__brick_scale_y), 1e-4f);
float row = floorf(__y / br_sy);
float offset = (mod(row, 2.0f) > 0.5f) ? br_sx * 0.5f : 0.0f;
float nx = __x - (floorf((__x + offset) / br_sx) * br_sx + br_sx * 0.5f - offset);
float ny = __y - (floorf(__y / br_sy) * br_sy + br_sy * 0.5f);
__px += __brick * nx;
__py += __brick * ny;`,
  },
  chainmail: {
    note: 'CPU normalizes the row modulo into [0,2); GPU used fmod.',
    gpuCode: `float cm_cx = __x * __chainmail_scale;
float cm_cy = __y * __chainmail_scale;
float cm_row = floorf(cm_cy);
float cm_offset = mod(cm_row, 2.0f) < 0.5f ? 0.5f : 0.0f;
float cm_lx = (cm_cx + cm_offset) - floorf(cm_cx + cm_offset) - 0.5f;
float cm_ly = cm_cy - floorf(cm_cy) - 0.5f;
float cm_r = sqrtf(cm_lx * cm_lx + cm_ly * cm_ly);
float cm_s = cm_r < __chainmail_ring_ratio ? __chainmail_ring_ratio / (cm_r + 1e-6f) : 1.0f;
__px += __chainmail * (cm_lx * cm_s) / __chainmail_scale;
__py += __chainmail * (cm_ly * cm_s) / __chainmail_scale;`,
  },
  pre_blur: {
    note: 'CPU: 6-entry gaussian ring buffer advanced with `& 5` (only entries 0,1 refresh; 2..5 are per-instance constants from init()) → r = w·(u0+u1+c−3), c ≈ 2; JWildfire\'s GPU snippet draws six fresh uniforms (a wider blur). Ported like pre_blur3D: two fresh uniforms plus the constants\' mean.',
    patch: [['(RANDFLOAT()+RANDFLOAT()+RANDFLOAT()+RANDFLOAT()+RANDFLOAT()+RANDFLOAT()-3.f)', '(RANDFLOAT()+RANDFLOAT()-1.f)']],
  },
  ouroboros: {
    note: 'CPU wraps the radius r; GPU wrapped x.',
    gpuCode: `float or_radius = fmaxf(fabsf(__ouroboros_radius), 0.01f);
float or_r = sqrtf(__x * __x + __y * __y);
float or_theta = atan2f(__y, __x);
float or_wrapped_r = or_radius * (or_r / or_radius - floorf(or_r / or_radius));
float or_nt = or_theta + __ouroboros_twist * or_r;
__px += __ouroboros * or_wrapped_r * cosf(or_nt);
__py += __ouroboros * or_wrapped_r * sinf(or_nt);`,
  },
  rays3: {
    note: 'CPU uses 1/t²; GPU wrote 1.0f/t*t which parses as (1/t)*t.',
    gpuCode: `float t = __x*__x + __y*__y;
float u = 1.0f / sqrtf(cosf(sinf(t*t + 1.e-6f) * sinf(1.0f / (t*t) + 1.e-6f)));
__px = (__rays3 / 10.0f) * u * cosf(t) * t / __x;
__py = (__rays3 / 10.0f) * u * tan(t) * t / __y;`,
  },
  waves22: {
    note: 'CPU raises the signed sine to an integer power (even powers are positive); GPU applied the sign afterwards.',
    gpuCode: `float x0 = __x;
float y0 = __y;
float sinx;
float siny;
int px = (int)__waves22_powerx;
int py = (int)__waves22_powery;
if (__waves22_modex < 0.5f){
  sinx = sinf(y0 * __waves22_freqx);
} else {
  sinx = 0.5f * (1.0f + sinf(y0 * __waves22_freqx));
}
float offsetx = powf(sinx, (float)px) * __waves22_scalex;
if (__waves22_modey < 0.5f){
  siny = sinf(x0 * __waves22_freqy);
} else {
  siny = 0.5f * (1.0f + sinf(x0 * __waves22_freqy));
}
float offsety = powf(siny, (float)py) * __waves22_scaley;
__px += __waves22 * (x0 + offsetx);
__py += __waves22 * (y0 + offsety);`,
  },
  perspective: {
    note: 'CPU init() uses angle*PI/2 (turns of a right angle); the GPU snippet used the raw angle.',
    gpuCode: `float _vsin = sinf(__perspective_angle * PI / 2.f);
float _vfcos = __perspective_dist * cosf(__perspective_angle * PI / 2.f);
float _d = __perspective_dist - __y * _vsin;
if (_d != 0.f) {
  float _t = 1.f / _d;
  __px += __perspective * __perspective_dist * __x * _t;
  __py += __perspective * _vfcos * __y * _t;
}`,
  },
  onion: {
    note: 'GPU snippet dropped the z output (CPU: pVarTP.z += z1 + pAffineTP.z).',
    append: '\n__pz += z1 + __z;',
  },
  // --- JWildfire GPU snippet syntax/copy-paste bugs (semantics unchanged) ---
  rings3: {
    note: "GPU snippet is rings2's code with a missing ';' and rings2's param names.",
    gpuCode: `float l = sqrtf(__x * __x + __y * __y);
float _dx = __rings3_val * __rings3_val ADD_EPSILON;
float c = 2.f * (_dx - _dx * _dx);
if (_dx == 0.f || l == 0.f) { return; }
float k = (int) ((l / _dx + 1.f) / 2.f);
float r = __rings3 * (2.f - _dx * (k * 2.f / l + 1.f) - __rings3_n * (k * c - 1.f) / l);
__px += r * __x;
__py += r * __y;`,
  },
  sym_ng13: {
    note: 'GPU snippet declares Mathc Tx[6] but writes Tx[6] and Tx[7] (out of bounds).',
    // (patched textually in gen.ts: Tx[6] → Tx[8])
  },
  julia3D: {
    note: 'CPU raises r2d + (z/|power|)² to the power; the GPU used the unscaled z.',
    patch: [['powf(__r2 + __z*__z, cn)', 'powf(__r2 + _z*_z, cn)']],
  },
  post_circlecrop: {
    note: 'GPU snippet added to __px/__py where the CPU assigns, never hid the cropped points, and tested zero as a bool instead of == 1.',
    gpuCode: `float x0 = __post_circlecrop_x;
float y0 = __post_circlecrop_y;
float cr = __post_circlecrop_radius;
float ca = fmaxf(-1.0f, fminf(__post_circlecrop_scatter_area, 1.f));
float vv = __post_circlecrop;
__px -= x0;
__py -= y0;
float rad = sqrtf(__px * __px + __py * __py);
float ang = atan2f(__py, __px);
float rdc = cr + (RANDFLOAT() * 0.5f * ca);
bool esc = rad > cr;
bool cr0 = lroundf(__post_circlecrop_zero) == 1;
float s = sinf(ang);
float c = cosf(ang);
__doHide = false;
if (cr0 && esc) {
  __px = 0.f; __py = 0.f;
  __doHide = true;
} else if (cr0 && !esc) {
  __px = vv * __px + x0; __py = vv * __py + y0;
} else if (!cr0 && esc) {
  __px = vv * rdc * c + x0; __py = vv * rdc * s + y0;
} else {
  __px = vv * __px + x0; __py = vv * __py + y0;
}`,
  },
  post_mirror_wf: {
    note: 'GPU snippet ignored xscale/yscale (CPU scales both axes on every mirror) and the |amount| > EPSILON guard.',
    gpuCode: `if (fabsf(__post_mirror_wf) > 1e-8f) {
  if (__post_mirror_wf_xaxis > 0.f && RANDFLOAT() < 0.5f) {
    __px = __post_mirror_wf_xscale * (-__px - __post_mirror_wf_xshift);
    __py = __post_mirror_wf_yscale * __py;
    __pal = fmodf(__pal + __post_mirror_wf_xcolorshift, 1.0f);
  }
  if (__post_mirror_wf_yaxis > 0.f && RANDFLOAT() < 0.5f) {
    __px = __post_mirror_wf_xscale * __px;
    __py = __post_mirror_wf_yscale * (-__py - __post_mirror_wf_yshift);
    __pal = fmodf(__pal + __post_mirror_wf_ycolorshift, 1.0f);
  }
  if (__post_mirror_wf_zaxis > 0.f && RANDFLOAT() < 0.5f) {
    __pz = -__pz - __post_mirror_wf_zshift;
    __pal = fmodf(__pal + __post_mirror_wf_zcolorshift, 1.0f);
  }
}`,
  },
  checkerboard_wf: {
    note: 'CPU puts the checker "sides" on checker boundaries (random(_max_checks + 1) is an int); the GPU snippet multiplied by a continuous random, smearing the sides across the board (seen on a solid collection flame, _sh8: with post_curl3D amplifying the z offset, corr 0.87).',
    patch: [['x = RANDFLOAT()*(_max_checks + 1) *  __checkerboard_wf_checker_size ;', 'x = (int)(RANDFLOAT()*(_max_checks + 1)) *  __checkerboard_wf_checker_size ;'],
      ['y = RANDFLOAT()*(_max_checks + 1) *  __checkerboard_wf_checker_size ;', 'y = (int)(RANDFLOAT()*(_max_checks + 1)) *  __checkerboard_wf_checker_size ;']],
  },
  // random(Integer.MAX_VALUE)·2π/p on the CPU picks one of p rotations exactly (only k mod p matters; k·2π/p is
  // evaluated in double); the GPU snippets round RANDFLOAT·0x7fff and multiply in f32 — an argument of tens of
  // thousands of radians whose sine the GPU evaluates with coarse error, so the rotation jitters and the tiling
  // smears into faint extra coverage (solid fixture _mesh10, hypertile3D2 p=7 q=5: 20 % more covered pixels).
  hypertile3D1: { note: 'see hypertile3D2', patch: [['float a = lroundf(RANDFLOAT() * 0x00007fff) * pa;', 'float a = (float)((int)(RANDFLOAT() * lroundf(__hypertile3D1_p))) * pa;']] },
  hypertile3D2: { note: 'random(MAX_INT)·2π/p → (int)(rnd·p)·2π/p (see the comment above)', patch: [['float a = lroundf(RANDFLOAT()*0x00007fff) * pa;', 'float a = (float)((int)(RANDFLOAT() * lroundf(__hypertile3D2_p))) * pa;']] },
  hypertile3D2b: {
    note: 'see hypertile3D2; here pa = b·π/p, so the period in k is 2p/b — that many discrete angles when it is an integer, a uniform angle otherwise',
    patch: [['float a = lroundf(RANDFLOAT() * 0x00007fff) * pa;',
      'float h2b_m = 2.f * lroundf(__hypertile3D2b_p) / __hypertile3D2b_b; float a = (fabsf(h2b_m - lroundf(h2b_m)) < 1e-6f) ? (float)((int)(RANDFLOAT() * h2b_m)) * pa : RANDFLOAT() * 2.f * PI;']],
  },
  hypertile2: { note: 'see hypertile3D2', patch: [['float rpa = pa * lroundf(RANDFLOAT() * 0x00007fff);', 'float rpa = pa * (float)((int)(RANDFLOAT() * lroundf(__hypertile2_p)));']] },
  phoenix_julia: {
    note: 'see hypertile3D2; power is a double here: an integer power gives |power| discrete angles, a fractional one makes k·2π/power equidistributed, i.e. a uniform angle',
    patch: [['float a = atan2f(preY, preX) * _invN + lroundf(0x00007fff * RANDFLOAT()) * _inv2PI_N;',
      'float pj_n = fabsf(__phoenix_julia_power); float a = atan2f(preY, preX) * _invN + ((fabsf(pj_n - lroundf(pj_n)) < 1e-6f) ? (float)((int)(RANDFLOAT() * pj_n)) * _inv2PI_N : RANDFLOAT() * 2.0f * PI);']],
  },
  post_point_symmetry_wf: {
    note: 'CPU picks the symmetry index uniformly (random(order)); the GPU rounded rnd·(order−1), halving the weight of the first and last copies.',
    patch: [['int idx = lroundf(RANDFLOAT() * (order-1));', 'int idx = (int)(RANDFLOAT() * (float)order); if (idx >= order) idx = order - 1;']],
  },
  ...Object.fromEntries(['cut_glypho', 'cut_fingerprint'].map((n) => [n, {
    note: 'CPU returns right after hiding (point stays at 0,0); the GPU snippet fell through and wrote the position anyway.',
    patch: [[new RegExp(`__px = __${n} \\* \\(x-px_center\\);(\\s*)__py = __${n} \\* \\(y-py_center\\);`), `if (!__doHide) { __px = __${n} * (x-px_center);$1__py = __${n} * (y-py_center); }`]],
  } satisfies Override])),
  xtrb: {
    note: 'GPU init computed S2ac = S2/c/6 where the CPU uses S2/(a+c)/6.',
    patch: [['S2ac = S2 / (c) / 6.0;', 'S2ac = S2 / (a + c) / 6.0;']],
  },
  // --- sin(x)*43758 shader hashes on integer cell ids: evaluated in double-float (HSIN2, see cwgsl.ts)
  //     so the cell pattern matches the f64 Java instead of f32 noise ---
  worley: {
    note: 'cell hash sin(cx·127.1+cy·311.7)·43758.5453 in double-float so the sites match the Java.',
    patch: [['sinf(cell_x * 127.1f + cell_y * 311.7f) * 43758.5453f', 'HSIN2(cell_x, 127.1, cell_y, 311.7, 0.0, 0.0, 43758.5453)'],
      ['sinf(cell_x * 269.5f + cell_y * 183.3f) * 43758.5453f', 'HSIN2(cell_x, 269.5, cell_y, 183.3, 0.0, 0.0, 43758.5453)']],
  },
  voronoi_fold: {
    note: 'cell hash in double-float (see worley).',
    patch: [['sinf(cell_x * 127.1f + cell_y * 311.7f) * 43758.5453f', 'HSIN2(cell_x, 127.1, cell_y, 311.7, 0.0, 0.0, 43758.5453)'],
      ['sinf(cell_x * 269.5f + cell_y * 183.3f) * 43758.5453f', 'HSIN2(cell_x, 269.5, cell_y, 183.3, 0.0, 0.0, 43758.5453)']],
  },
  r_circleblur: {
    note: 'circle hashes on the rounded cell (bx, by) in double-float (see worley).',
    patch: [['sinf(bx * 127.1 + by * 311.7 +  __r_circleblur_seed ) * 43758.5453', 'HSIN2(bx, 127.1, by, 311.7, __r_circleblur_seed, 0.0, 43758.5453)'],
      ['sinf(bx * 269.5 + by * 183.3 +  __r_circleblur_seed ) * 43758.5453', 'HSIN2(bx, 269.5, by, 183.3, __r_circleblur_seed, 0.0, 43758.5453)'],
      ['sinf(by * 12.9898 + bx * 78.233 +  __r_circleblur_seed ) * 43758.5453', 'HSIN2(by, 12.9898, bx, 78.233, __r_circleblur_seed, 0.0, 43758.5453)']],
  },
  waves4: {
    note: 'row hash in double-float (see worley).',
    patch: [['sinf(ax * 12.9898 + ax * 78.233 + 1.0 + y0 * 0.001 * __waves4_yfact) * 43758.5453', 'HSIN2(ax, 12.9898, ax, 78.233, 1.0, y0 * 0.001 * __waves4_yfact, 43758.5453)']],
  },
  waves42: {
    note: 'row hash in double-float (see worley).',
    patch: [['sinf(ax * 12.9898f + ax * 78.233f + 1.0f + y0 * 0.001f * __waves42_yfact) * 43758.5453f', 'HSIN2(ax, 12.9898, ax, 78.233, 1.0, y0 * 0.001 * __waves42_yfact, 43758.5453)']],
  },
  cut_truchetweaving: {
    note: 'tile hash in double-float (see worley).',
    patchFuncs: [['fract(sinf(id.x*324.23+id.y*5604.342)*87654.53)', 'fract(HSIN2(id.x, 324.23, id.y, 5604.342, 0.0, 0.0, 87654.53))']],
  },
  cut_wood: {
    note: 'noise-cell hash in double-float (see worley).',
    patchFuncs: [['fract(sinf(dot(make_float2(st.x,st.y),make_float2(12.9898,78.233)))*43758.5453123)', 'fract(HSIN2(st.x, 12.9898, st.y, 78.233, 0.0, 0.0, 43758.5453123))'],
      ['fract(sinf(seed + dot(make_float2(st.x,st.y), make_float2(12.9898,78.233)))* 43758.5453123)', 'fract(HSIN2(st.x, 12.9898, st.y, 78.233, seed, 0.0, 43758.5453123))']],
  },
  chrysanthemum: {
    note: 'CPU scales the radius by 0.1; the GPU snippet dropped it.',
    patch: [['r *= __chrysanthemum;', 'r *= __chrysanthemum * 0.1;']],
  },
  pixel_flow: {
    note: 'CPU fade = fLen·r⁴ (GPU dropped fLen), hash() divides by Integer.MAX_VALUE (GPU: 2³²), and the block/seed products are int arithmetic.',
    patch: [['float fade = 1.0 * r01', 'float fade = fLen * r01'], [/__pixel_flow_seed/g, '((int)__pixel_flow_seed)']],
    patchFuncs: [['return (float) a/ exp2f(32.0);', 'return (float) a / 2147483647.0f;']],
  },
  // --- dc_* shader-art family: GPU helper ≠ Java helper ---
  dc_hoshi: {
    note: 'Java hsv() computes (t1+h)/3 (GPU: t1+h/3) and rotate() by +3.14159/(…) (GPU: −PI/(…)).',
    patchFuncs: [[/t1\s*=\s*t1\s*\+\s*h\/3\.0;/, 't1 = (t1 + h) / 3.0;'], ['-PI/(0.10', '3.14159/(0.10']],
  },
  dc_gmandelbroot: {
    note: 'Java hsv() computes (t1+h)/3 (GPU: t1+h/3); Java hides black (0,0,0) colours.',
    patchFuncs: [[/t1\s*=\s*t1\s*\+\s*h\/3\.0;/, 't1 = (t1 + h) / 3.0;']],
    append: '\nif (color.x == 0.0f && color.y == 0.0f && color.z == 0.0f) __doHide = true;',
  },
  dc_turbulence: {
    note: 'GPU passed zoom where the Java uses level (loop count) — the argument was mislabelled.',
    patch: [['dc_turbulence_getRGBColor(uv,__dc_turbulence_time,__dc_turbulence_zoom)', 'dc_turbulence_getRGBColor(uv,__dc_turbulence_time,__dc_turbulence_level)']],
  },
  dc_mandbrot: {
    note: 'Java rotates with p.times(rot) (row vector · matrix); the GPU used times(&rot,p) (matrix · column) — opposite rotation.',
    patchFuncs: [['p = times(&rot,p);', 'p = make_float2(cs*p.x + sn*p.y, -sn*p.x + cs*p.y);']],
  },
  // 3D solid samplers: CPU tries up to 50 random points inside the SDF before
  // hiding; the GPU snippet tries once (mostly hidden, wrong density).
  ...Object.fromEntries([
    'bbox3D', 'cappedcone3D', 'cappedtorus3D', 'capsule3D', 'cone3D', 'cylinder3D', 'ellipsoid3D', 'hexprism3D',
    'octahedron3D', 'ocappedcone3D', 'ocylinder3D', 'octogonprism3D', 'oroundcone3D', 'pyramid3D', 'rhombus3D',
    'solidangle3D', 'triprism3D',
  ].map((n) => [n, { note: 'CPU rejection-samples up to 50 times before hiding; GPU sampled once.', retry: 50 } satisfies Override])),
};
