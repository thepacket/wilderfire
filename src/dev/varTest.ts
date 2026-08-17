// Dev harness: evaluates every variation's WGSL on the GPU over the oracle
// point grid and diffs against JWildfire's Java results (scripts/jwf-port).
//
//   await window.wilderfire.varTest()               // all variations
//   await window.wilderfire.varTest({ only: ['julian', 'curl'] })
//   window.wilderfire.varTestReport                  // last report
//
// Requires scripts/jwf-port/oracle-spec.json and oracle-out.jsonl (see
// scripts/jwf-port/README.md) — served by the vite dev server.

import { HAND_VARIATIONS as VARIATIONS, type VariationDef } from '../core/variations';
import { JWF_VARIATIONS as JWF_VERIFIED, type JwfVariationDef } from '../core/variations.jwf';
import { JWF_VARIATIONS_UNVERIFIED } from '../core/variations.jwf.unverified';

/** Every port, verified or not — the harness re-tests all of them. */
const JWF_VARIATIONS: Record<string, JwfVariationDef> = { ...JWF_VERIFIED, ...JWF_VARIATIONS_UNVERIFIED };

export interface VarTestResult {
  name: string;
  source: 'hand' | 'jwf';
  status: 'pass' | 'fail' | 'compile-error' | 'oracle-error' | 'oracle-missing' | 'runtime-error';
  random: boolean;
  /** fraction of compared points within tolerance */
  passFrac: number;
  maxErr: number;
  /** pass fraction per parameter set */
  perSet?: number[];
  worst?: { pt: number[]; ours: number[]; theirs: number[]; set?: number };
  /** which criteria failed how often per set (diagnostics): e.g. {1: {mean: 12, z: 40}} */
  why?: Record<number, Record<string, number>>;
  /** a few failing points per set with the criterion, ours and theirs (diagnostics) */
  samples?: { set: number; pt: number[]; ours: number[]; theirs: number[]; why: string[] }[];
  msg?: string;
  flags?: string[];
}

interface Spec { points: number[][]; affine: number[]; entries: { name: string; priority: number; source: string; sets: { weight: number; params: Record<string, number> }[] }[] }
interface OracleRow { name: string; set: number; random: boolean; samples: number; out: (number[] | null)[]; error?: string; paramError?: string }

const PRELUDE = `const PI: f32 = 3.14159265358979;
fn rnd(state: ptr<function, u32>) -> f32 {
  var x = *state;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  *state = x;
  return f32(x) * 2.3283064365386963e-10;
}
fn mmod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }
`;

/** Stateful variations (per-thread `jwx_` state, e.g. attractors) are evaluated as one
 *  sequential trajectory per point — S steps in one thread — like a JWildfire render. */
function buildShader(def: VariationDef, funcs: string, priority: number, seq = false): string {
  const nP = def.params?.length ?? 0;
  const w = 'xd[0]';
  const p = Array.from({ length: nP }, (_, k) => `xd[${1 + k}]`);
  const base = 1 + nP; // affine at base..base+5, samples at base+6, seed at base+7
  const A = (k: number) => `xd[${base + k}]`;
  const snippet = def.code(w, p, A);
  return `${PRELUDE}
var<private> pal: array<vec4f, 256>; // palette stand-in for direct-colour variations
var<private> df_zero: u32 = 0u; // runtime 0 (opaque) for the double-float helpers
${funcs}
@group(0) @binding(0) var<storage, read> inp: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4f>;
@group(0) @binding(2) var<storage, read> xd: array<f32>;

// the variation body lives in its own function (like the kernel's xform functions) so an early
// return in the snippet ends the variation, not the shader; returns (x, y, z, colour+hide)
fn snip_(pin: vec4f, rs: ptr<function, u32>) -> vec4f {
  var c: f32 = 0.5;
  var hide: bool = false;
  let cp = &c;
  let hd = &hide;
  var rgbo = vec4f(0.0);
  let rgb = &rgbo;
  let PALB_: u32 = 0u;
  var t0 = pin.xy;
  var t = t0;
  var z_ = pin.z;
  var pz_ = ${priority === 1 || priority === 2 ? 'z_' : '0.0'};
  var r2 = max(dot(t, t), 1e-12);
  var r = sqrt(r2);
  var th = atan2(t.x, t.y);
  var ph = atan2(t.y, t.x);
  var v = ${priority === 1 || priority === 2 ? 't' : 'vec2f(0.0, 0.0)'}; // post (and the forward half of a prepost pair) mutate the output point
  ${snippet}
  // color + hide packed: c in [0,1], +10 when hidden
  // pre-priority: input + accumulator (JWildfire keeps what a pre step adds into pVarT for the main sum)
  return vec4f(${priority === -1 ? '(t + v)' : 'v'}, ${priority === -1 ? '(z_ + pz_)' : 'pz_'}, c + select(0.0, 10.0, hide));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  df_zero = bitcast<u32>(xd[${base + 6}]) >> 31u;
  let S = u32(xd[${base + 6}]);
  ${seq ? `let pi = gid.x; if (pi >= arrayLength(&inp)) { return; }
  var rsv: u32 = ((pi + 1u) * 2654435761u) ^ u32(xd[${base + 7}]);
  if (rsv == 0u) { rsv = 1u; }
  _ = rnd(&rsv); _ = rnd(&rsv);
  // warm-up: the oracle runs one instance over all points in order, so point pi's samples are
  // steps [pi*S, (pi+1)*S) of one trajectory — replay the earlier steps first (state only)
  // (capped: unbounded warm-up on heavy stateful bodies can hang the GPU for seconds per dispatch)
  for (var s_ = 0u; s_ < min(pi * S, 4096u); s_++) { _ = snip_(inp[pi], &rsv); }
  for (var s_ = 0u; s_ < S; s_++) { let i = pi * S + s_;` : `let i = gid.x;
  if (i >= arrayLength(&inp) * S) { return; }
  let pi = i / S;
  var rsv: u32 = ((i + 1u) * 2654435761u) ^ u32(xd[${base + 7}]);
  if (rsv == 0u) { rsv = 1u; }
  _ = rnd(&rsv); _ = rnd(&rsv);`}
  outp[i] = snip_(inp[pi], &rsv);
${seq ? '  }' : ''}
}
`;
}

async function compile(device: GPUDevice, code: string): Promise<{ pipeline: GPUComputePipeline } | { error: string }> {
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === 'error');
  if (errs.length) {
    await device.popErrorScope();
    return { error: errs.slice(0, 3).map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | ') + '\n' + contextLines(code, errs[0].lineNum) };
  }
  try {
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const e = await device.popErrorScope();
    if (e) return { error: e.message };
    return { pipeline };
  } catch (err) {
    await device.popErrorScope().catch(() => null);
    return { error: String((err as Error).message ?? err) };
  }
}
function contextLines(code: string, line: number): string {
  const ls = code.split('\n');
  return ls.slice(Math.max(0, line - 2), line + 1).map((l, i) => `${Math.max(0, line - 2) + i + 1}: ${l}`).join('\n');
}

async function run(device: GPUDevice, pipeline: GPUComputePipeline, points: number[][], xd: Float32Array<ArrayBuffer>, samples: number, seq = false): Promise<Float32Array> {
  const n = points.length * samples;
  const threads = seq ? points.length : n;
  const inp = device.createBuffer({ size: points.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inp, 0, new Float32Array(points.flatMap((p) => [p[0], p[1], p[2] ?? 0, 0])));
  const outp = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const xdb = device.createBuffer({ size: xd.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(xdb, 0, xd);
  const staging = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: inp } }, { binding: 1, resource: { buffer: outp } }, { binding: 2, resource: { buffer: xdb } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(threads / 64));
  pass.end();
  enc.copyBufferToBuffer(outp, 0, staging, 0, n * 16);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const res = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  inp.destroy(); outp.destroy(); xdb.destroy(); staging.destroy();
  return res;
}

/** Debug helper: the test shader for one variation. */
export function shaderFor(name: string, source: 'hand' | 'jwf' = 'jwf'): string {
  const def = source === 'hand' ? VARIATIONS[name] : JWF_VARIATIONS[name];
  const jdef = source === 'jwf' ? (def as JwfVariationDef) : null;
  return buildShader(def, jdef?.funcs ?? '', jdef?.priority ?? 0);
}

export async function runVarTest(device: GPUDevice, opts: { only?: string[]; verbose?: boolean; tol?: number; save?: boolean } = {}): Promise<VarTestResult[]> {
  const spec: Spec = await (await fetch('/scripts/jwf-port/oracle-spec.json')).json();
  const oracleText = await (await fetch('/scripts/jwf-port/oracle-out.jsonl')).text();
  const oracle = new Map<string, OracleRow[]>();
  for (const line of oracleText.split('\n')) {
    if (!line.trim()) continue;
    const row: OracleRow = JSON.parse(line);
    oracle.set(row.name, [...(oracle.get(row.name) ?? []), row]);
  }
  const tol = opts.tol ?? 2e-3;
  const results: VarTestResult[] = [];
  const entries = spec.entries.filter((e) => !opts.only || opts.only.includes(e.name));
  let done = 0;
  for (const e of entries) {
    const rows = oracle.get(e.name);
    // `name~inv`: a prepost port's inverse snippet, tested as a pre-priority variation
    const inv = e.name.endsWith('~inv');
    const baseName = inv ? e.name.slice(0, -4) : e.name;
    const sources: ('hand' | 'jwf')[] = [];
    if (VARIATIONS[baseName] && !inv) sources.push('hand');
    if (JWF_VARIATIONS[baseName]) sources.push('jwf');
    for (const source of sources) {
      const def0: VariationDef | JwfVariationDef = source === 'hand' ? VARIATIONS[baseName] : JWF_VARIATIONS[baseName];
      const jdef = source === 'jwf' ? (def0 as JwfVariationDef) : null;
      if (inv && !jdef?.preCode) continue;
      const def: VariationDef | JwfVariationDef = inv ? { ...def0, code: jdef!.preCode! } : def0;
      const priority = inv ? -1 : (jdef?.priority ?? e.priority);
      const res: VarTestResult = { name: e.name, source, status: 'pass', random: false, passFrac: 1, maxErr: 0, flags: jdef?.flags };
      results.push(res);
      if (!rows) { res.status = 'oracle-missing'; continue; }
      if (rows.every((r) => r.error && !r.out)) { res.status = 'oracle-error'; res.msg = rows[0].error; continue; }
      let shader: string;
      try {
        shader = buildShader(def, jdef?.funcs ?? '', priority, !!(jdef?.flags?.includes('state') || jdef?.flags?.includes('stateful')));
      } catch (err) { res.status = 'compile-error'; res.msg = 'build: ' + String(err); continue; }
      const c = await compile(device, shader);
      if ('error' in c) { res.status = 'compile-error'; res.msg = c.error; continue; }
      let total = 0, ok = 0, maxErr = 0;
      const perSet: number[] = [];
      for (const row of rows) {
        let sTotal = 0, sOk = 0;
        const set = e.sets[row.set];
        if (!set || !row.out) continue;
        res.random = res.random || row.random;
        // random variations: REPS replicas of the oracle's 256 samples — the pooled stats are tighter and the
        // spread of the per-replica stds estimates the sampling noise of a 256-sample std (rare-event
        // Bernoulli tails make that noise far larger than the Gaussian 1/sqrt(2n))
        const REPS = 4, ORACLE_N = 256;
        const samples = row.random ? ORACLE_N * REPS : 1;
        const nP = def.params?.length ?? 0;
        const base = 1 + nP;
        const xd = new Float32Array(Math.max(16, base + 8));
        xd[0] = set.weight;
        (def.params ?? []).forEach((pd, k) => { xd[1 + k] = set.params[pd.name] ?? pd.def; });
        for (let k = 0; k < 6; k++) xd[base + k] = spec.affine[k];
        xd[base + 6] = samples;
        xd[base + 7] = 7919 * (row.set + 1);
        let out: Float32Array;
        try { out = await run(device, c.pipeline, spec.points, xd, samples, !!(jdef?.flags?.includes('state') || jdef?.flags?.includes('stateful'))); }
        catch (err) { res.status = 'runtime-error'; res.msg = String(err); break; }
        for (let pi = 0; pi < spec.points.length; pi++) {
          const theirs = row.out[pi];
          if (!theirs) continue;
          // ours: stats over samples
          let sx = 0, sy = 0, sz = 0, sxx = 0, syy = 0, szz = 0, sc = 0, hides = 0, valid = 0;
          const rep = Array.from({ length: REPS }, () => ({ n: 0, sx: 0, sy: 0, sxx: 0, syy: 0 }));
          for (let s = 0; s < samples; s++) {
            const o = (pi * samples + s) * 4;
            const x = out[o], y = out[o + 1], z = out[o + 2];
            if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
            const ch = out[o + 3]; const hid = ch >= 5 ? 1 : 0; const col = ch - 10 * hid;
            valid++; sx += x; sy += y; sz += z; sxx += x * x; syy += y * y; szz += z * z; sc += col; hides += hid;
            const rp = rep[Math.floor(s / ORACLE_N)]; rp.n++; rp.sx += x; rp.sy += y; rp.sxx += x * x; rp.syy += y * y;
          }
          // spread of the per-replica std (of a 256-sample replica) — the sampling noise of theirs' std
          const repStd = rep.filter((r) => r.n > 1).map((r) => Math.hypot(Math.sqrt(Math.max(0, r.sxx / r.n - (r.sx / r.n) ** 2)), Math.sqrt(Math.max(0, r.syy / r.n - (r.sy / r.n) ** 2))));
          const repMean = repStd.reduce((a, b) => a + b, 0) / Math.max(1, repStd.length);
          const sdRep = repStd.length > 1 ? Math.sqrt(repStd.reduce((a, b) => a + (b - repMean) ** 2, 0) / (repStd.length - 1)) : 0;
          const tMag = Math.hypot(theirs[0], theirs[1]);
          if (tMag > 1e5) continue; // both blow up; skip
          total++; sTotal++;
          if (valid === 0) { updWorst(res, spec.points[pi], [NaN, NaN], theirs, Infinity, row.set); maxErr = Infinity; continue; }
          const mx = sx / valid, my = sy / valid, mz = sz / valid, mc = sc / valid, mh = hides / valid;
          const tz = theirs[4] ?? 0; // deterministic rows: [x,y,c,hide,z]; random: [mx,my,c,hide,stdx,stdy,mz,stdz]
          let err: number;
          if (!row.random) {
            const scale = Math.max(1, tMag, Math.abs(tz));
            err = Math.max(Math.abs(mx - theirs[0]), Math.abs(my - theirs[1]), Math.abs(mz - tz)) / scale;
            err = Math.max(err, Math.abs(mc - theirs[2]) * 0.5, Math.abs(mh - theirs[3]));
            if (err <= tol) { ok++; sOk++; } else noteFail(res, row.set, spec.points[pi], [mx, my, mc, mh, mz], theirs, ['det']);
          } else {
            const stdx = Math.sqrt(Math.max(0, sxx / valid - mx * mx)), stdy = Math.sqrt(Math.max(0, syy / valid - my * my));
            const tstdx = theirs[4] ?? 0, tstdy = theirs[5] ?? 0;
            // means within a few standard errors; stds within a factor
            const sem = Math.max(Math.hypot(tstdx, tstdy), Math.hypot(stdx, stdy)) * Math.sqrt(1 / ORACLE_N + 1 / samples);
            const scale = Math.max(1, tMag);
            const meanErr = Math.hypot(mx - theirs[0], my - theirs[1]);
            const stdErr = Math.abs(Math.hypot(stdx, stdy) - Math.hypot(tstdx, tstdy)) / Math.max(0.05, Math.hypot(tstdx, tstdy));
            const okMean = meanErr <= 4.5 * sem + 0.02 * scale;
            // within 35 %, or within the measured sampling noise of a 256-sample std (theirs) + ours (pooled)
            const okStd = stdErr <= 0.35 || Math.abs(Math.hypot(stdx, stdy) - Math.hypot(tstdx, tstdy)) <= 4.5 * sdRep * Math.sqrt(1 + 1 / REPS) + 0.005 * scale;
            const okC = Math.abs(mc - theirs[2]) <= 0.1 || tstdx === 0 && tstdy === 0 && Math.abs(mc - theirs[2]) <= 0.02;
            const okH = Math.abs(mh - theirs[3]) <= 0.15;
            const tmz = theirs[6] ?? 0, tstdz = theirs[7] ?? 0;
            const stdz = Math.sqrt(Math.max(0, szz / valid - mz * mz));
            const okZ = Math.abs(mz - tmz) <= 4.5 * Math.max(tstdz, stdz) * Math.sqrt(1 / ORACLE_N + 1 / samples) + 0.02 * scale;
            err = Math.max(meanErr / (4.5 * sem + 0.02 * scale), stdErr / 0.35) * tol; // normalized so tol is the pass line
            if (okMean && okStd && okC && okH && okZ) { ok++; sOk++; }
            else noteFail(res, row.set, spec.points[pi], [mx, my, mc, mh, stdx, stdy, mz, stdz], theirs, [!okMean && 'mean', !okStd && 'std', !okC && 'color', !okH && 'hide', !okZ && 'z'].filter(Boolean) as string[]);
          }
          if (err > maxErr) { maxErr = err; updWorst(res, spec.points[pi], [mx, my, mc, mh, mz], theirs, err, row.set); }
        }
        perSet[row.set] = sTotal ? sOk / sTotal : 1;
      }
      res.perSet = perSet;
      if (res.status === 'pass' || res.status === 'fail') {
        // gate on sets 0 and 1 (defaults, floats perturbed); set 2 (ints perturbed) is informational
        const gating = perSet.slice(0, 2).filter((f) => f !== undefined);
        res.passFrac = gating.length ? gating.reduce((a, b) => a + b, 0) / gating.length : (total ? ok / total : 1);
        res.maxErr = maxErr;
        res.status = total === 0 ? 'pass' : (res.passFrac >= 0.97 ? 'pass' : 'fail');
        if (total === 0) res.msg = 'no comparable points';
        else if (res.status === 'pass' && perSet[2] !== undefined && perSet[2] < 0.97) res.msg = `int-set ${(perSet[2] * 100).toFixed(0)}%`;
      }
    }
    done++;
    if (opts.verbose && done % 25 === 0) console.log(`varTest ${done}/${entries.length}`);
  }
  (window as any).wilderfire.varTestReport = results;
  summarize(results);
  if (!opts.only && opts.save !== false) await saveVerified(results);
  return results;
}

/** Every name the oracle spec knows (for batched runs). */
export async function allVarNames(): Promise<string[]> {
  const spec: Spec = await (await fetch('/scripts/jwf-port/oracle-spec.json')).json();
  return spec.entries.map((e) => e.name);
}
/** Persist verdicts to scripts/jwf-port/verified.json via the dev-server sink (vite.config.ts). */
export async function saveVerified(results: VarTestResult[]): Promise<void> {
  const fmt = (x: VarTestResult) => x.status + (x.perSet ? ' sets ' + x.perSet.map((f) => (f * 100).toFixed(0)).join('/') : '') + (x.random ? ' random' : '') + (x.flags?.length ? ' [' + x.flags.join(',') + ']' : '');
  const body = {
    _note: 'Generated by window.wilderfire.varTest() (src/dev/varTest.ts) against the JWildfire Java oracle — see scripts/jwf-port/README.md. "jwf" lists generated variations whose WGSL matches JWildfire on the oracle grid (only these enter the app registry); "failed" explains the rest.',
    jwf: results.filter((x) => x.source === 'jwf' && x.status === 'pass').map((x) => x.name).sort(),
    hand: results.filter((x) => x.source === 'hand' && x.status === 'pass').map((x) => x.name).sort(),
    failed: Object.fromEntries(results.filter((x) => x.source === 'jwf' && x.status !== 'pass').map((x) => [x.name, fmt(x)])),
    handFailed: Object.fromEntries(results.filter((x) => x.source === 'hand' && x.status !== 'pass').map((x) => [x.name, fmt(x)])),
  };
  try {
    const r = await fetch('/__jwf/verified', { method: 'POST', body: JSON.stringify(body) });
    console.log('verified.json:', await r.text());
  } catch (err) {
    console.warn('could not save verified.json (dev server sink missing?)', err);
  }
}
function noteFail(res: VarTestResult, set: number, pt: number[], ours: number[], theirs: number[], why: string[]) {
  res.why ??= {}; res.why[set] ??= {};
  for (const w of why) res.why[set][w] = (res.why[set][w] ?? 0) + 1;
  res.samples ??= [];
  if (res.samples.filter((s) => s.set === set).length < 3) res.samples.push({ set, pt, ours: ours.map(r4), theirs: theirs.map(r4), why });
}
function updWorst(res: VarTestResult, pt: number[], ours: number[], theirs: number[], err: number, set: number) {
  if (!res.worst || err >= res.maxErr) res.worst = { pt, ours: ours.map(r4), theirs: theirs.map(r4), set };
}
const r4 = (x: number) => Math.round(x * 1e4) / 1e4;

function summarize(results: VarTestResult[]) {
  const by = (s: VarTestResult['status'], src?: string) => results.filter((r) => r.status === s && (!src || r.source === src));
  const line = (src: 'hand' | 'jwf') => `${src}: pass ${by('pass', src).length} fail ${by('fail', src).length} compile-error ${by('compile-error', src).length} runtime-error ${by('runtime-error', src).length} oracle-error ${by('oracle-error', src).length} oracle-missing ${by('oracle-missing', src).length}`;
  console.log('varTest summary\n' + line('hand') + '\n' + line('jwf'));
  const fails = results.filter((r) => r.status === 'fail');
  if (fails.length) console.log('FAIL:', fails.map((r) => `${r.name}(${r.source} ${(r.passFrac * 100).toFixed(0)}%)`).join(' '));
  const ce = results.filter((r) => r.status === 'compile-error');
  if (ce.length) console.log('COMPILE-ERROR:', ce.map((r) => `${r.name}(${r.source})`).join(' '));
}
