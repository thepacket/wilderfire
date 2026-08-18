// JWildfire solid rendering on the GPU (RasterFloatIntForSolidRendering / NormalsCalculator /
// LogDensityFilter.addSolidColors / GammaCorrectionFilter, ported literally).
//
// Instead of accumulating density, every raster cell keeps its NEAREST point (largest camera-space z):
//   zkey[cell]  — atomic u32, the depth as an order-preserving integer (0 = empty)
//   zpay[cell]  — 5 words: untransformed x, y, z (f32 bits; the normals are taken from these,
//                 in world space, exactly like JWildfire's originX/Y/ZBuf), colour rg | b + material (f16 pairs)
//   nrm[cell]   — packed normal (3 × 10-bit snorm + 'has normal' bit), refreshed by the post pass
// The kernel does atomicMax on the key and, when it raised it, writes the payload. Two record-setting
// threads on one cell in the same instant can leave the payload one step behind the key; the post pass
// re-derives the key from the payload's position, so such a cell only stays inconsistent for one pass.
// The tonemap shades each cell (ambient + per-light diffuse/Phong on the material) and applies
// JWildfire's spatial filter in raster cells (noiseFilterSizeHalve = N/2 − 1, as in LogDensityFilter),
// then composites over the background with alpha = coverage^(1 + 1/gamma).

export const SOLID_PAY_WORDS = 5;
export const SOLID_MAX_LIGHTS = 4;
export const SOLID_MAX_MATS = 8;
/** solid tonemap filter buffer: JWildfire kernel of up to 25×25 raster cells */
export const SOLID_FILT_FLOATS = 640;

/** Bindings + splat helper spliced into the iteration kernel when the flame renders solid. */
export const SOLID_KERNEL_WGSL = `
@group(0) @binding(7) var<storage, read_write> zkey: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> zpay: array<u32>;

// order-preserving u32 of a float (any finite/inf value maps to > 0; 0 = empty cell)
fn zkeyOf(z: f32) -> u32 {
  let b = bitcast<u32>(z);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn solidSplat(cell: u32, z: f32, origin: vec3f, col: vec3f, mat: f32) {
  let zk = zkeyOf(z);
  let old = atomicMax(&zkey[cell], zk);
  if (zk > old) {
    let b = cell * ${SOLID_PAY_WORDS}u;
    zpay[b] = bitcast<u32>(origin.x);
    zpay[b + 1u] = bitcast<u32>(origin.y);
    zpay[b + 2u] = bitcast<u32>(origin.z);
    zpay[b + 3u] = pack2x16float(col.xy);
    zpay[b + 4u] = pack2x16float(vec2f(col.z, mat));
  }
}
`;

/** Post pass (compute, one thread per raster cell): key repair + normals from the origin neighbourhood. */
export const SOLID_POST_WGSL = `
struct SPP {
  width: u32,   // raster cells (output × oversample)
  height: u32,
  ppu: f32,     // raster cells per world unit (JWildfire view.bws): zr = depth · ppu
  _p1: u32,
  m2: vec4f,    // camera matrix row z (+ camPos.z in w): depth = m2·origin + w
};
@group(0) @binding(0) var<uniform> Q: SPP;
@group(0) @binding(1) var<storage, read_write> zkey: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> zpay: array<u32>;
@group(0) @binding(3) var<storage, read_write> nrm: array<u32>;
@group(0) @binding(4) var<storage, read_write> zr: array<f32>; // depth in raster units (JWildfire zBuf); ZBUF_ZMIN when empty

const ZBUF_ZMIN: f32 = -3.0e38; // JWildfire: -Float.MAX_VALUE (any real depth is above this)

fn zkeyOf(z: f32) -> u32 {
  let b = bitcast<u32>(z);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

fn originAt(cell: u32) -> vec3f {
  let b = cell * ${SOLID_PAY_WORDS}u;
  return vec3f(bitcast<f32>(zpay[b]), bitcast<f32>(zpay[b + 1u]), bitcast<f32>(zpay[b + 2u]));
}

fn packNormal(n: vec3f) -> u32 {
  let q = vec3u(clamp((n * 0.5 + 0.5) * 1023.0 + 0.5, vec3f(0.0), vec3f(1023.0)));
  return q.x | (q.y << 10u) | (q.z << 20u) | 0x80000000u;
}

// NormalsCalculator.NNEIGHBOURS_COARSE, in order (a, b pairs of cell offsets)
const NPAIRS = array<vec4i, 16>(
  vec4i(0, 1, 1, 0), vec4i(1, 0, 0, -1), vec4i(0, -1, -1, 0), vec4i(-1, 0, 0, 1),
  vec4i(-1, 1, 1, 1), vec4i(1, 1, 1, -1), vec4i(1, -1, -1, -1), vec4i(-1, -1, -1, 1),
  vec4i(0, 1, 1, 1), vec4i(1, 1, 1, 0), vec4i(1, 0, 1, -1), vec4i(1, -1, 0, -1),
  vec4i(0, -1, -1, -1), vec4i(-1, -1, -1, 0), vec4i(-1, 0, -1, 1), vec4i(-1, 1, 0, 1));

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  let W = i32(Q.width);
  let H = i32(Q.height);
  if (x >= W || y >= H) { return; }
  let cell = u32(y * W + x);
  let key = atomicLoad(&zkey[cell]);
  if (key == 0u) { nrm[cell] = 0u; zr[cell] = ZBUF_ZMIN; return; }
  let o = originAt(cell);
  // repair: a payload written behind a concurrent, higher key — the key must describe the payload
  let depth = dot(Q.m2.xyz, o) + Q.m2.w;
  let zk = zkeyOf(depth);
  if (zk != key) { atomicStore(&zkey[cell], zk); }
  zr[cell] = depth * Q.ppu;
  // NormalsCalculator.refreshNormalAtLocation with MAX_NORMALS_SAMPLES = 8 (the final refreshAllNormals)
  var n = vec3f(0.0);
  var samples = 0;
  for (var k = 0; k < 16; k = k + 1) {
    let pr = NPAIRS[k];
    let ax = x + pr.x; let ay = y + pr.y; let bx = x + pr.z; let by = y + pr.w;
    if (ax < 0 || ax >= W || ay < 0 || ay >= H || bx < 0 || bx >= W || by < 0 || by >= H) { continue; }
    let ca = u32(ay * W + ax);
    if (atomicLoad(&zkey[ca]) != 0u) {
      let a = o - originAt(ca);
      let cb = u32(by * W + bx);
      if (atomicLoad(&zkey[cb]) != 0u) {
        let b = o - originAt(cb);
        samples = samples + 1;
        n += vec3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
      }
    }
    if (samples >= 8) { break; }
  }
  var out = 0u;
  if (samples > 0) {
    let r = length(n);
    if (r > 1e-8) { out = packNormal(n / r); }
  }
  nrm[cell] = out;
}
`;

/** Solid tonemap: fullscreen fragment over the output pixels. */
export const SOLID_TONEMAP_WGSL = `
struct SP {
  width: u32,     // output pixels
  height: u32,
  os: u32,        // oversample: raster = (width·os) × (height·os)
  filterN: u32,   // JWildfire noiseFilterSize in raster cells (<= 1 = no spatial filter)
  gamma: f32,     // flame gamma
  transparent: f32,
  nLights: u32,
  nMats: u32,
  bg: vec4f,
  ao: vec4f,      // x: AO enabled, y: aoIntensity (0..4), z: aoAffectDiffuse
};
struct Light { dir: vec4f, col: vec4f };          // dir.xyz = direction (LightViewCalculator), dir.w = intensity
struct Mat { a: vec4f, b: vec4f, c: vec4f };      // a: diffuse, ambient, phong, phongSize; b: phong rgb, diffFunc; c: reflMapIntensity
struct SolidLights { lights: array<Light, ${SOLID_MAX_LIGHTS}>, mats: array<Mat, ${SOLID_MAX_MATS}> };

@group(0) @binding(0) var<uniform> S: SP;
@group(0) @binding(1) var<storage, read> zkey: array<u32>;
@group(0) @binding(2) var<storage, read> zpay: array<u32>;
@group(0) @binding(3) var<storage, read> nrm: array<u32>;
@group(0) @binding(4) var<storage, read> sfilt: array<f32>;
@group(0) @binding(5) var<uniform> L: SolidLights;
@group(0) @binding(6) var<storage, read> aoBuf: array<f32>; // AOCalculator result per cell (zeros when AO is off)

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(pos[vi], 0.0, 1.0);
}

fn unpackNormal(p: u32) -> vec3f {
  return vec3f(f32(p & 1023u), f32((p >> 10u) & 1023u), f32((p >> 20u) & 1023u)) / 1023.0 * 2.0 - 1.0;
}

// LightDiffFuncPreset
fn diffFunc(kind: u32, cosa: f32) -> f32 {
  switch kind {
    case 1u: { return cosa * cosa; }
    case 2u: { return cosa * 0.5 + 0.5; }
    case 3u: { let h = cosa * 0.5 + 0.5; return h * h; }
    default: { return cosa; }
  }
}

struct MatI { diffuse: f32, ambient: f32, phong: f32, phongSize: f32, phongCol: vec3f, f0: u32, f1: u32, fmix: f32, valid: bool };

// SolidRenderSettings.getInterpolatedMaterial (incl. morphMaterial's quirk: the morphed diffuse is the refl-map intensity blend)
fn materialAt(idx: f32) -> MatI {
  var m: MatI;
  m.valid = false;
  if (idx < 0.0 || S.nMats == 0u) { return m; }
  var fi = i32(idx);
  let n = i32(S.nMats);
  var to = fi + 1;
  var scl = idx - floor(idx);
  if (fi >= n) { fi = n - 1; to = fi; scl = 0.0; }
  if (to >= n) { to = 0; }
  if (scl < 0.01) { to = fi; scl = 0.0; }
  else if (scl > 0.99) { fi = to; scl = 0.0; }
  let A = L.mats[fi];
  let B = L.mats[to];
  m.valid = true;
  m.f0 = u32(A.b.w + 0.5); m.f1 = u32(B.b.w + 0.5); m.fmix = scl;
  if (fi == to) {
    m.diffuse = A.a.x; m.ambient = A.a.y; m.phong = A.a.z; m.phongSize = A.a.w; m.phongCol = A.b.xyz;
  } else {
    m.diffuse = mix(A.c.x, B.c.x, scl);
    m.ambient = mix(A.a.y, B.a.y, scl); m.phong = mix(A.a.z, B.a.z, scl); m.phongSize = mix(A.a.w, B.a.w, scl);
    m.phongCol = mix(A.b.xyz, B.b.xyz, scl);
  }
  return m;
}

fn diffResp(m: MatI, cosa: f32) -> f32 {
  let d0 = diffFunc(m.f0, cosa);
  if (m.fmix <= 0.0) { return d0; }
  return mix(d0, diffFunc(m.f1, cosa), m.fmix);
}

// LogDensityFilter.addSolidColors without shadows/AO (visibility 1). Returns rgb (0..1 scale) and whether the cell had a normal.
fn shadeCell(cell: u32, out: ptr<function, vec3f>) -> bool {
  if (zkey[cell] == 0u) { return false; }
  let np = nrm[cell];
  if ((np & 0x80000000u) == 0u) { return false; }
  let normal = unpackNormal(np);
  let b = cell * ${SOLID_PAY_WORDS}u;
  let rg = unpack2x16float(zpay[b + 3u]);
  let bm = unpack2x16float(zpay[b + 4u]);
  let obj = vec3f(rg, bm.x);
  let m = materialAt(bm.y);
  var raw: vec3f;
  if (!m.valid) {
    // no material for this index: the background lit by the lights' visibility (1 each)
    raw = S.bg.xyz * clamp(f32(S.nLights), 0.0, 1.0);
  } else {
    // SSAO darkens the ambient term and, by aoAffectDiffuse, the diffuse term
    var ambient = m.ambient;
    var diffuse = m.diffuse;
    if (S.ao.x > 0.5) {
      let ao = aoBuf[cell];
      ambient = max(0.0, ambient - ao * S.ao.y);
      diffuse = max(0.0, diffuse - ao * S.ao.y * S.ao.z);
    }
    raw = obj * ambient;
    for (var i = 0u; i < S.nLights; i = i + 1u) {
      let lt = L.lights[i];
      let ld = lt.dir.xyz;
      let cosa = dot(ld, normal);
      if (cosa > 1e-8) {
        raw += (lt.col.xyz + obj * ambient / 3.0) * (diffResp(m, cosa) * diffuse * lt.dir.w);
      }
      if (m.phong > 1e-8) {
        let r = ld - 2.0 * dot(ld, normal) * normal;
        let vr = r.z; // viewDir (0, 0, 1) · r
        if (vr < 1e-8) {
          raw += m.phongCol * (pow(diffResp(m, -vr), m.phongSize) * m.phong * lt.dir.w);
        }
      }
    }
  }
  *out = raw;
  return true;
}

@fragment
fn fs(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let transparent = S.transparent > 0.5;
  let bgOut = select(vec4f(S.bg.xyz, 1.0), vec4f(0.0), transparent);
  let x = i32(fragPos.x);
  let y = i32(fragPos.y);
  if (x >= i32(S.width) || y >= i32(S.height)) { return bgOut; }
  let os = i32(S.os);
  let W = i32(S.width) * os;
  let H = i32(S.height) * os;
  var sum = vec3f(0.0);
  var inten = 0.0;
  let N = i32(S.filterN);
  var c: vec3f;
  if (N <= 1) {
    // no spatial filter: mean of the covered cells of the os×os block; intensity = covered fraction
    var cnt = 0;
    for (var j = 0; j < os; j = j + 1) {
      for (var i = 0; i < os; i = i + 1) {
        if (shadeCell(u32((y * os + j) * W + x * os + i), &c)) { sum += c; cnt = cnt + 1; }
      }
    }
    if (cnt > 0) { sum = sum / f32(cnt); inten = f32(cnt) / f32(os * os); }
  } else {
    let half = N / 2 - 1; // LogDensityFilter: noiseFilterSizeHalve = size/2 − 1
    for (var i = 0; i < N; i = i + 1) {
      let cy = y * os + i - half;
      if (cy < 0 || cy >= H) { continue; }
      for (var j = 0; j < N; j = j + 1) {
        let cx = x * os + j - half;
        if (cx < 0 || cx >= W) { continue; }
        if (shadeCell(u32(cy * W + cx), &c)) {
          let f = sfilt[u32(i * N + j)] / f32(os * os);
          sum += c * f;
          inten += f;
        }
      }
    }
  }
  if (inten <= 0.0) { return bgOut; }
  // GammaCorrectionFilter (solid branch): alpha = intensity^(1 + 1/gamma); colour = round(solid·255) + (invAlpha·bg) >> 8
  let alpha = pow(inten, 1.0 + 1.0 / max(S.gamma, 0.1));
  let alphaI = clamp(floor(alpha * 255.0 + 0.5), 0.0, 255.0);
  let solid255 = floor(sum * 255.0 + 0.5);
  if (transparent) {
    let a = alphaI / 255.0;
    return vec4f(clamp(solid255 / 255.0 / max(a, 1e-6), vec3f(0.0), vec3f(1.0)), a);
  }
  let bg255 = floor(S.bg.xyz * 255.0 + 0.5);
  let outc = clamp(solid255 + floor((255.0 - alphaI) * bg255 / 256.0), vec3f(0.0), vec3f(255.0));
  return vec4f(outc / 255.0, 1.0);
}
`;

/** AOCalculator (screen-space ambient occlusion on the z-buffer): raw pass + optional gaussian smoothing pass. */
export const SOLID_AO_WGSL = `
struct AOP {
  width: u32,
  height: u32,
  radiusSamples: u32,
  azimuthSamples: u32,
  sphereRadius: f32,   // aoSearchRadius · imgSize / 500 (raster cells)
  radiusStep: f32,     // sphereRadius / radiusSamples
  azimuthStep: f32,    // 2π / azimuthSamples
  falloff: f32,
  blurN: u32,          // smoothing kernel size (0 = raw result stays)
  seed: u32,
  _p0: u32,
  _p1: u32,
};
const ZBUF_ZMIN: f32 = -3.0e38; // JWildfire: -Float.MAX_VALUE (any real depth is above this)
@group(0) @binding(0) var<uniform> A: AOP;
@group(0) @binding(1) var<storage, read> zr: array<f32>;
@group(0) @binding(2) var<storage, read> nrm: array<u32>;
@group(0) @binding(3) var<storage, read_write> aoRaw: array<f32>;
@group(0) @binding(4) var<storage, read_write> aoOut: array<f32>;
@group(0) @binding(5) var<storage, read> bfilt: array<f32>;
@group(0) @binding(6) var<storage, read_write> aoTmp: array<f32>;

fn unpackNormal(p: u32) -> vec3f {
  return vec3f(f32(p & 1023u), f32((p >> 10u) & 1023u), f32((p >> 20u) & 1023u)) / 1023.0 * 2.0 - 1.0;
}
fn pcg(v: u32) -> u32 {
  let st = v * 747796405u + 2891336453u;
  let w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd01(h: ptr<function, u32>) -> f32 { *h = pcg(*h); return f32(*h) * 2.3283064365386963e-10; }
// Tools.FTOI + clamp to the raster (AOCalculator.getZ)
fn zAt(x: f32, y: f32) -> f32 {
  let xi = clamp(i32(select(x - 0.5, x + 0.5, x > 0.0)), 0, i32(A.width) - 1);
  let yi = clamp(i32(select(y - 0.5, y + 0.5, y > 0.0)), 0, i32(A.height) - 1);
  return zr[u32(yi) * A.width + u32(xi)];
}

@compute @workgroup_size(16, 16)
fn aoRawPass(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x; let y = gid.y;
  if (x >= A.width || y >= A.height) { return; }
  let cell = y * A.width + x;
  var acc = 0.0;
  let z0 = zr[cell];
  let np = nrm[cell];
  if (z0 != ZBUF_ZMIN && (np & 0x80000000u) != 0u) {
    let n = unpackNormal(np);
    var h = pcg(cell ^ A.seed);
    let rJitter = A.radiusStep / 10.0;
    var angle = (0.5 - rnd01(&h)) * A.azimuthStep / 4.0;
    let x0 = f32(x); let y0 = f32(y);
    for (var k = 0u; k < A.azimuthSamples; k = k + 1u) {
      let dx = cos(angle); let dy = sin(angle);
      var r0 = A.radiusStep;
      var prevH = 0.0;
      for (var l = 0u; l < A.radiusSamples; l = l + 1u) {
        let r = r0 + (0.5 - rnd01(&h)) * rJitter;
        let px = x0 + r * dx + (0.5 - rnd01(&h)) * rJitter;
        let py = y0 + r * dy + (0.5 - rnd01(&h)) * rJitter;
        let z = zAt(px, py);
        let hh = atan2(z - z0, r) + 0.001;
        // (JWildfire normalises the absolute sample position here, not the offset — kept as is)
        let RR = sqrt(px * px + py * py);
        let Rx = px / RR; let Ry = py / RR;
        let tt = n.x * Rx + Ry * n.y;
        let tx = -(Ry * n.x * n.y - Rx * n.y * n.y - Rx * n.z * n.z) / tt;
        let ty = (Ry * n.x * n.x + Ry * n.z * n.z - n.x * Rx * n.y) / tt;
        let tz = (-Ry * n.y * n.z - n.x * Rx * n.z) / tt;
        let ta = atan2(tz, sqrt(tx * tx + ty * ty));
        let ao = sin(hh) - sin(ta);
        if (ao > prevH) {
          prevH = ao;
          let dist = r / A.sphereRadius;
          acc += ao * exp(-dist * dist * A.falloff) * 0.5;
        }
        r0 += A.radiusStep;
      }
      angle += A.azimuthStep;
    }
  }
  aoRaw[cell] = acc;
  if (A.blurN == 0u) { aoOut[cell] = acc; }
}

// SmoothAOBufferThread: gaussian FilterHolder (os 1) over the raw buffer, × 0.1. The kernel is a pure gaussian
// exp(−2t²) normalised over the N×N window, so it factorises exactly into two 1-D passes (bfilt holds the
// normalised 1-D weights): H aoRaw → aoTmp, V aoTmp → aoOut.
@compute @workgroup_size(16, 16)
fn aoBlurH(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x); let y = i32(gid.y);
  let W = i32(A.width); let H = i32(A.height);
  if (x >= W || y >= H) { return; }
  let N = i32(A.blurN);
  let c = N / 2;
  var v = 0.0;
  for (var k = 0; k < N; k = k + 1) {
    let px = x + k - c;
    if (px < 0 || px >= W) { continue; }
    v += aoRaw[u32(y * W + px)] * bfilt[u32(k)];
  }
  aoTmp[u32(y * W + x)] = v;
}
@compute @workgroup_size(16, 16)
fn aoBlurV(@builtin(global_invocation_id) gid: vec3u) {
  let x = i32(gid.x); let y = i32(gid.y);
  let W = i32(A.width); let H = i32(A.height);
  if (x >= W || y >= H) { return; }
  let N = i32(A.blurN);
  let c = N / 2;
  var v = 0.0;
  for (var l = 0; l < N; l = l + 1) {
    let py = y + l - c;
    if (py < 0 || py >= H) { continue; }
    v += aoTmp[u32(py * W + x)] * bfilt[u32(l)];
  }
  aoOut[u32(y * W + x)] = v * 0.1;
}
`;
