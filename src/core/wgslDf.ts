// Extracted from the generated registry (scripts/jwf-port/cwgsl.ts HELPER_FUNCS): double-float (two f32) arithmetic and
// the sign-preserving sin hash `hsin_` for hand-written ports that must reproduce JWildfire's f64
// `frac(sin(a·x + b·y + c) · K)` cell hashes. `df_zero` is set by the kernel prelude.
export const DF_HASH_FUNCS = `fn op_(v: f32) -> f32 { return bitcast<f32>(bitcast<u32>(v) + df_zero); }

fn df_ts(a0: f32, b0: f32) -> vec2f { let a = op_(a0); let b = op_(b0); let s = op_(a + b); let bb = op_(s - a); return vec2f(s, op_(a - op_(s - bb)) + op_(b - bb)); }

fn df_qts(a0: f32, b0: f32) -> vec2f { let a = op_(a0); let b = op_(b0); let s = op_(a + b); return vec2f(s, b - op_(s - a)); }

fn df_add(x: vec2f, y: vec2f) -> vec2f { var s = df_ts(x.x, y.x); let t = df_ts(x.y, y.y); s.y = op_(s.y + t.x); s = df_qts(s.x, s.y); s.y = op_(s.y + t.y); return df_qts(s.x, s.y); }

fn df_mul(x: vec2f, y: vec2f) -> vec2f { let p = op_(x.x * y.x); var e = fma(x.x, y.x, -p); e = op_(e + op_(x.x * y.y + x.y * y.x)); return df_qts(p, e); }

fn df_mulf(x: vec2f, b: f32) -> vec2f { let p = op_(x.x * b); var e = fma(x.x, b, -p); e = op_(e + x.y * b); return df_qts(p, e); }

fn df_sin(x: vec2f) -> vec2f {
  let x2 = df_mul(x, x);
  var s = vec2f(-7.647163609812713e-13, -1.2200710471178288e-20);
  s = df_add(df_mul(s, x2), vec2f(1.6059044372074283e-10, -5.352526511562726e-18));
  s = df_add(df_mul(s, x2), vec2f(-2.5052107943679403e-08, -4.4176230446483665e-16));
  s = df_add(df_mul(s, x2), vec2f(2.7557318844628753e-06, 3.793571224297229e-14));
  s = df_add(df_mul(s, x2), vec2f(-0.00019841270113829523, 2.725596874933456e-12));
  s = df_add(df_mul(s, x2), vec2f(0.008333333767950535, -4.34617203337595e-10));
  s = df_add(df_mul(s, x2), vec2f(-0.1666666716337204, 4.967053879312289e-09));
  s = df_add(df_mul(s, x2), vec2f(1.0, 0.0));
  return df_mul(s, x);
}

fn df_cos(x: vec2f) -> vec2f {
  let x2 = df_mul(x, x);
  var s = vec2f(4.7794772561329454e-14, 7.62544404448643e-22);
  s = df_add(df_mul(s, x2), vec2f(-1.147074536050896e-11, -2.372207689231238e-19));
  s = df_add(df_mul(s, x2), vec2f(2.0876755879584152e-09, 1.1082839809204342e-16));
  s = df_add(df_mul(s, x2), vec2f(-2.755731998149713e-07, 7.575112209051195e-15));
  s = df_add(df_mul(s, x2), vec2f(2.4801587642286904e-05, -3.40699609366682e-13));
  s = df_add(df_mul(s, x2), vec2f(-0.0013888889225199819, 3.3631094437103215e-11));
  s = df_add(df_mul(s, x2), vec2f(0.0416666679084301, -1.2417634698280722e-09));
  s = df_add(df_mul(s, x2), vec2f(-0.5, 0.0));
  s = df_add(df_mul(s, x2), vec2f(1.0, 0.0));
  return s;
}

fn hsin_(x: vec2f, k: vec2f) -> f32 {
  let n = round(x.x * 0.15915494309189535);
  var r = df_add(x, df_mulf(vec2f(-6.2831854820251465, 1.7484555314695172e-07), n));
  let q = round(r.x * 0.6366197723675814);
  r = df_add(r, df_mulf(vec2f(-1.5707963705062866, 4.371138828673793e-08), q));
  let qi = i32(q) & 3;
  var s: vec2f;
  if (qi == 0) { s = df_sin(r); } else if (qi == 1) { s = df_cos(r); } else if (qi == 2) { s = -df_sin(r); } else { s = -df_cos(r); }
  let v = df_mul(s, k);
  // sign-preserving fraction of v.x + v.y; the low part can carry across the integer v.x rounded to
  var f = op_(v.x - trunc(v.x)) + v.y;
  let pos = v.x > 0.0 || (v.x == 0.0 && v.y > 0.0);
  if (pos && f < 0.0) { f += 1.0; } else if (!pos && f > 0.0) { f -= 1.0; }
  if (f >= 1.0) { f -= 1.0; } else if (f <= -1.0) { f += 1.0; }
  return f;
}`;
