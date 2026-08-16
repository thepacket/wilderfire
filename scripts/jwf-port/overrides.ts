// Hand corrections applied before transpiling. JWildfire's GPU snippets are
// not always in sync with its CPU (Java) implementation, which is what .flame
// files are authored against; the oracle harness (README.md) flags the
// divergent ones and these overrides bring the WGSL back to CPU semantics.
// Snippets stay in the CUDA dialect so they go through the same transpiler.

export interface Override { gpuCode?: string; gpuFunctions?: string; note: string; /** text appended to the original snippet */ append?: string }

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
};
