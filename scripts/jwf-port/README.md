# JWildfire variation port pipeline

WilderFire's variation registry ([src/core/variations.ts](../../src/core/variations.ts))
is composed of a hand-written flam3 set plus **JWildfire ports** generated here:
JWildfire's variations are Java classes, but most of them also carry a CUDA-C
snippet for JWildfire's own GPU renderer (`SupportsGPU.getGPUCode()`), and that
dialect is close enough to WGSL to transpile mechanically. Every port is then
verified numerically against JWildfire's *Java* implementation (the one `.flame`
files are authored against), and only verified ports enter the app registry.

```
JWildfire source ──Dump.java──▶ data/jwf-variations.jsonl ──gen.ts──▶ src/core/variations.jwf.ts (verified)
   (Java, LGPL)                  data/kernel-lib.cu   (cwgsl.ts)       src/core/variations.jwf.unverified.ts
                                                                          │
JWildfire source ──Oracle.java─▶ oracle-out.jsonl ◀── oracle-spec.ts ◀────┘
                                        │
              browser: window.wilderfire.varTest() ──▶ verified.json ──▶ (feeds gen.ts on the next run)
```

## Files

| file | role |
|---|---|
| `cwgsl.ts` | CUDA-C → WGSL transpiler (tokenizer, C-subset parser, typed emitter: int/float coercions, pointers/arrays/structs, overloads, macros, `switch`, helper library incl. the double-float `hsin_` hash). |
| `gen.ts` | Reads the dump + kernel helpers, transpiles every variation, applies `overrides.ts`, writes the two registry files and prints a summary. `--report` also writes `report.json`. |
| `overrides.ts` | Hand corrections (in CUDA dialect) for variations whose JWildfire GPU code diverges from its CPU code (`perspective`, `brick`, `chainmail`, `ouroboros`, `rays3`, `waves22`, `dc_hoshi`, `post_circlecrop`, …): full `gpuCode`, textual `patch`/`patchFuncs`, plus the `retry` (rejection-sampling) and `clampParams` mechanisms and the `HSIN2` double-float hash rewrites. |
| `data/param-clamps.json`, `data/param-ints.json`, `extract-clamps.py` | Per-parameter clamps and int casts from JWildfire's Java `setParameter()` (`Tools.limitValue` & co., `(int)`/`FTOI`), applied by `gen.ts` to every param read so out-of-range and fractional values behave like the CPU (int params are read as ints). Regenerate with `python3 extract-clamps.py <jwf>/src/org/jwildfire/create/tina/variation`. |
| `data/dc-base.json` | Which `dc_*` classes inherit / copy `DC_BaseFunc.transform` (sampling `rnd−0.5`) — drives the DC family fix in `gen.ts`. |
| `data/jwf-variations.jsonl` | Metadata + GPU code for all 1026 JWildfire variations (name, params/defaults/int-ness, priority, types, GPU code, helper functions). Produced by `Dump.java`; checked in so regeneration does not need Java. |
| `data/kernel-lib.cu` | Helper library extracted from JWildfire's `Flam4_3dKernal_TemplateJWF.cu` (Complex/Mat2/Jacobi/noise/misc); transpiled on demand. |
| `java2cu.ts`, `data/jwf-java-ports.jsonl` | Java → CUDA-dialect pre-processor for the variations that have *no* GPU snippet in JWildfire: extracts fields, params (from `setParameter`), `init()`, helper methods and `transform()` from the Java class, rewrites the Java idioms (`pAffineTP.x`→`__x`, `pVarTP.x`→`__px`, `pAmount`→`__amount_`, `pContext.random()`, `Math.*`/MathLib statics, `sinAndCos`, `this.` fields, locals shadowing fields, `random(Integer.MAX_VALUE)`→31-bit `RANDINT`), replays setParameter-derived fields, turns per-instance state (attractors) into per-thread `varpar->` state initialised on the first call, and copies helper-used params/fields into state. GLSL-style Java (js.glsl `vec2`/`vec3`/`vec4` objects, `G.*` statics, `new mat2(…)` in `.times()`) goes through a small expression parser that turns method chains into vector arithmetic (`crop_*`, `glsl_*`, `truchetflow`); the abstract `GLSLFunc` parent's fields/params are merged in. Output feeds `gen.ts` exactly like a dump entry. Needs the JWildfire source tree (`--jwf`). |
| `Dump.java` | Dumps the catalogue by reflection against a compiled JWildfire tree. |
| `Oracle.java` | Headless JWildfire oracle: evaluates each variation's Java `transform()` on the spec grid; deterministic → exact values, random → per-point mean/std/hide-fraction over 256 samples. |
| `oracle-spec.ts` | Emits the shared test spec (130-point grid × 3 parameter sets: defaults, floats perturbed, ints perturbed) for every known variation. |
| `verified.json` | Verdicts written by the browser harness; `jwf` = ports that match JWildfire (these enter the app registry). |
| `testflames/` | JWildfire random-generator flames (`GenFlames` via JWildfire's own generators) + `yflip.flame` (orientation check). Loaded by `window.wilderfire.flameTest()`. |
| `../../src/dev/varTest.ts` | Browser harness: compiles each variation's WGSL, evaluates the spec grid on the GPU, diffs against the oracle, POSTs `verified.json` via the dev-server sink in `vite.config.ts`. |
| `../../src/dev/flameTest.ts` | Browser harness: imports every fixture flame, reports unsupported variations, compiles the kernel. |
| `../../src/dev/flameCompare.ts`, `RenderOne.java`, `Compare.java` | Whole-image comparison against headless JWildfire: `await window.wilderfire.flameCompare()` renders the fixtures, the bundled JWildfire samples and the authored presets offscreen (512 px wide, quality 100) into `compare-out/` (PNG + the exact .flame XML, gitignored); `java … Compare <repo>/compare-out` renders the same XML with JWildfire (cached as `<id>.jwf.png`) and prints per-flame metrics — mean luma of both, ratio, coverage, 16×16-block MAE, luma-histogram intersection, block correlation — with flags. Numbers only; no pixels are judged by eye. `RenderOne.java` renders one .flame to PNG. |

## Regenerating

Without Java (metadata + GPU code are checked in):

```bash
node scripts/jwf-port/gen.ts
```

Full loop (needs a JDK; JWildfire is LGPL-2.1):

```bash
# 1. get + compile JWildfire (sparse checkout keeps it small)
git clone --depth 1 --filter=blob:none --sparse https://github.com/thargor6/JWildfire.git jwf
cd jwf && git sparse-checkout set src lib resources
find src -name '*.java' > sources.txt
javac -nowarn -encoding ISO-8859-1 -source 8 -target 8 -proc:none -d out -cp "$(ls lib/*.jar | tr '\n' ':')" @sources.txt
# (Java ≥ 20: comment out the two Thread.suspend()/resume() calls in
#  src/org/jwildfire/create/tina/swing/TinaInteractiveRendererController.java)

# 2. dump the catalogue and run the oracle
CP="out:src:resources:$(ls lib/*.jar | tr '\n' ':')"
javac -d tools/out -cp "$CP" /path/to/wilderfire/scripts/jwf-port/Dump.java /path/to/wilderfire/scripts/jwf-port/Oracle.java
java -Djava.awt.headless=true -cp "tools/out:$CP" Dump   /path/to/wilderfire/scripts/jwf-port/data/jwf-variations.jsonl
cd /path/to/wilderfire
node scripts/jwf-port/gen.ts            # transpile
node scripts/jwf-port/oracle-spec.ts    # spec for everything we know
java -Djava.awt.headless=true -cp "jwf/tools/out:$CP" Oracle scripts/jwf-port/oracle-spec.txt scripts/jwf-port/oracle-out.jsonl

# 3. verify in the browser (dev server running)
#    await window.wilderfire.varTest()   → writes scripts/jwf-port/verified.json
node scripts/jwf-port/gen.ts            # re-emit with the new verdicts
```

## Status

Of JWildfire's 1026 variations, 786 carry a GPU snippet and 85 more are ported
straight from their Java `transform()` by `java2cu.ts` (871 transpile);
**856 verify numerically against the Java oracle in 3D**, 13 more are verified
by inspection where the per-point oracle cannot compare (`FORCE_VERIFIED` in
`gen.ts`, each with its reason: `arch`/`rays`/`starfractal` heavy-tailed
statistics, the chaotic attractors `hopalong`/`macmillan`/`threeply`/
`gumowski_mira`/`gingerbread_man` and `post_point_crop` order-dependent state,
`minkQM` f32 boundary artefact of the test grid, `circular`/`circular2` hash of
the continuous input point, `pre_flatten`), so 869 ship in `variations.jwf.ts`;
2 stay in `variations.jwf.unverified.ts` because the Java itself is
order-dependent (`dc_circuits` accumulates a member `S` across points,
`dc_gnarly` updates only 2 of its 6 gaussian summands — `& 5` — so its blur
depends on the render's init randoms); the remaining 155 have neither a GPU
snippet nor a Java transform the pre-processor handles yet (resources/images/
sub-flames/text, `DrawFunc`/`DynamicArray`/`Complex`/BigInteger helpers, JWildfire
`XYZPoint`/`Point` objects, `custom_wf`). Together with the 70 hand-written
flam3 entries the app registry has 871 variations.

Systematic GPU≠CPU families fixed at generator level: the 17 `*3D` solid
samplers (CPU rejection-samples up to 50 times before hiding — `retry`
override), Java `setParameter()` clamps and int casts (`data/param-clamps.json`,
`data/param-ints.json` from `extract-clamps.py`, applied to every param read;
int-typed params are read as *ints* so `n / 2`, `xx * yy + seed` and friends
keep Java's integer arithmetic; `lroundf(flag) > 0` tests on double params
become plain comparisons like the Java), the `DC_BaseFunc` shader-art family
(CPU samples `rnd−0.5` and sets z = greyscale(colour); the GPU sampled `2·rnd−1`
and left z = 0.5 — `data/dc-base.json` says which classes inherit or copy the
base transform), mistyped param names, a few C-isms (`++i` in conditions,
chained assignment, `case a: case b:` groups, `while ((i++ < n) && …)`), and the
`fract_*` buddhabrot family whose helpers carry state through a
`struct VarPar *varpar` pointer (flattened onto per-thread `jwx_` globals; the
params those helpers read are copied into state at the top of the snippet).
Individual GPU≠CPU snippet bugs are patched in `overrides.ts` (`patch` /
`patchFuncs` textual edits, or a full `gpuCode`), each with a note naming the
Java line it restores — e.g. `dc_hoshi` hsv/rotate, `dc_turbulence` passing
zoom for level, `dc_mandbrot` rotating the wrong way, `pixel_flow` dropping
`fLen` and dividing its hash by 2³², `post_circlecrop` adding instead of
assigning, `post_mirror_wf` ignoring the scales, `chrysanthemum` missing the
0.1, `julia3D` powering the unscaled z, `xtrb` S2ac, `cut_glypho`/`cut_fingerprint`
writing the position after hiding.

**Shader hashes.** `sin(a·127.1 + b·311.7 [+ seed])·43758.5453` (and the
`fract(sin(dot(…))·K)` spelling) is f32 noise on the GPU: the fraction of a
number near 4·10⁴ has 8 bits left, and the argument itself is off by ~1e-5. When
`a`, `b` are integer cell ids (worley, voronoi_fold, r_circleblur, waves4/42,
cut_wood, cut_truchetweaving) the *pattern* — which cell is cut, where the
circle sits — depends on the exact value, so the transpiler has a `HSIN2(a1, c1,
a2, c2, add_a, add_b, K)` builtin that evaluates the whole thing in double-float
(two-f32, Dekker/Knuth error-free transforms + a Taylor sin/cos after 2π and
quadrant reduction, `hsin_` in `cwgsl.ts`) and returns the sign-preserving
fraction to ~1e-6; the overrides rewrite the hash lines to it. Two GPU facts it
relies on: `fma()` is fused (tested), and the shader compiler *reassociates*
floating point (Metal fast-math folds TwoSum's error term to zero even across
buffer loads), so every intermediate goes through `op_()` — an integer add of the
runtime zero `df_zero` that the kernel entry point sets opaquely
(`bitcast<u32>(P.ppu) >> 31u`) and the optimiser cannot see through.
Scalar `smoothstep` goes through `smoothstepc` (WGSL rejects equal constant
edges) and `distance` has a typed builtin. Direct RGB colour (`__useRgb`,
`__colorR/G/B`) and palette reads (`read_imageStepMode`, `numColors`) are
supported: the kernel passes an `rgb` out-pointer and each layer's palette base
(`PALB_`) into every transform function. The oracle uses a Mersenne-Twister RNG
because JWildfire's Marsaglia generator degenerates for some seeds. Run
`await window.wilderfire.varTestAll()` in the dev console for the current verdicts
(each failing result carries `why` — which criterion failed per set — and a
few `samples`) and `await window.wilderfire.flameTest()` to import + compile
every fixture flame.

**Harness modes.** Snippets run inside their own WGSL function (`snip_`, like the
kernel's xform functions), and a snippet-level `return;` is transpiled into a
`loop { … break; }` wrapper (nested returns set `ret_`) so it ends the variation
in both the harness and the kernel. Stateful variations (`state`/`stateful`
flags: attractors, `mandelbrot`, `post_point_crop`) are evaluated as one
sequential trajectory per point — the oracle runs one Java instance over all
points in order, so point *i*'s samples are steps [i·256, (i+1)·256) of one
trajectory; the harness replays the earlier steps as warm-up (state only,
capped at 4096 steps — an unbounded warm-up on a heavy body can hang the GPU)
and then records — `mandelbrot` verifies numerically that way. Run the sweep
with `await window.wilderfire.varTestAll()` (batches of 100, then saves
`verified.json`) rather than one 870-variation `varTest()`. Scalar `atan2` goes
through `atan2j` because the Metal GPU returns NaN for atan2(0, 0) where Java
gives ±0/±π (and fast-math then folds NaN·0 to 0).

**Harness statistics.** Random variations are compared as per-point mean/std
over the oracle's 256 samples; the harness draws 4 replicas of 256 and uses the
spread of the per-replica std as the sampling noise of a 256-sample std (a
rare-event Bernoulli tail — a `tile_hlp` column that shifts with p = 0.007 —
makes that noise far larger than the Gaussian 1/√2n), so a std difference within
that noise passes. Heavy-tailed variations (1/cos, tan) still cannot be judged
per point and are the `FORCE_VERIFIED` cases above.

## Image comparison (2026-08-17)

Whole-image metrics vs headless JWildfire at 512 px / quality 100 (see the
`flameCompare` row above): all **16 bundled JWildfire samples** and **8 of the
10 authored presets** match to a luma ratio of 0.98–1.02, block MAE < 3 and
block correlation ≥ 0.98. The two others differ by design: *Golden Nautilus*
uses WilderFire's own `dc_radial` (not a JWildfire variation) and *Clockwork*
relies on flam3's weighted `rings` (`PREFER_HAND`; JWildfire's ignores the
weight). Two engine bugs this surfaced and fixed: (1) **DE rounding** — JWildfire's
`DeCalculator` stores the estimated density as an int (`(int)(sumA + 0.5)`),
so an isolated stray sample spread over the DE kernel rounds to *nothing*;
we kept the float, and every dark region grew a speckle haze around stray
hits (coverage 0.94 vs 0.44 on Phoenix_0) — now rounded the same way; (2)
**fuse** — 65k GPU walkers × a few hundred steps each per export made the
20-step transient visible on slowly contracting flames; re-seeds now fuse
200 (100 live) and those iterations are excluded from the sample count.
Of the 44 random-generator fixtures, 6 reference variations we do not have
(`barycentroid`, `colordomain`, `glsl_apollonian`, `glynns3subfl`,
`post_brush_stroke_wf`, `cut_triskel`) and 7 still differ (`Bokeh_1`,
`Cross_1`, `EDisc_1`, `Galaxies_0`, `Julians_0` — hopalong's single long
trajectory vs our 65k short ones — `Orchids_0/1`, `Rays_0`); the rest match.

## Semantics worth knowing

* **Snippet scope** (`variations.ts` header): `t` (input point, mutable), `r2 r th=atan2(x,y) ph=atan2(y,x)`, `v` (output accumulator), `rs` rng, `cp` palette-coordinate pointer, `hd` hide-flag pointer, `A(i)` affine coefficients. JWildfire's `__phi` is our `th` and `__theta` our `ph`.
* **Priority.** JWildfire pre-variations (`pre_blur`, priority −1) mutate the input point in place and post-variations (`post_curl`, +1) mutate the accumulated output; they are *not* weighted sums. Codegen runs an xform's variation list in priority order (pre → normal → post) inside one mutable stage. WilderFire's own pre/post *stages* (weighted sums) still exist alongside.
* **3D.** Points carry z: `__z` reads the input depth and `__pz` writes go to the output; codegen exposes `z_`/`pz_` in every stage and applies `preserve_z` (JWildfire semantics: 2D variations pass z through scaled by weight). The oracle grid is 3D and the harness diffs z as well as x/y. Direct RGB colour output, weighting fields, resource-backed variations (images/text) and `custom_wf` are not ported.
* **Precision.** WGSL is f32, JWildfire is f64. Hash-style variations (`sin(x)*43758.5453`) on integer cell ids go through the double-float `HSIN2` builtin (see Status); hashes of the continuous input point (`circular`) are identical in distribution only.
* **JWildfire GPU ≠ CPU.** The oracle caught several JWildfire GPU snippets that disagree with the Java (see `overrides.ts`); overrides restore CPU semantics.
* **Duplicate instances.** JWildfire writes a second `bubble` on one xform as `bubble#1#="…"` (invalid XML); the importer's lenient pre-parser normalises this and attribute names with spaces/punctuation.
* **Camera.** JWildfire's effective pixels-per-unit is `scale × cam_zoom`; the importer folds `cam_zoom` in. flam3/JWildfire raster +y points *down*; the kernel and overlay follow that convention. The 3D camera reproduces JWildfire's matrix (yaw → pitch → bank, then perspective `1/(1 − cam_persp·z + cam_pos_z)`); `cam_pitch`/`cam_yaw`/`cam_roll` in the XML are radians and `cam_roll` is the bank axis. `testflames/cam3d.flame` is the reference composition.
* **Tonemap.** `src/gpu/codegen.ts` TONEMAP_WGSL reproduces JWildfire's `LogScaleCalculator` (k1 = 2·contrast·brightness, k2 = 1/(contrast·area·quality), low-density glow), `RenderColor`'s 200/256 palette pre-scale over `whiteLevel`, `GammaCorrectionFilter` (colour + bg·(1−alpha), colour already alpha-scaled), `DeCalculator` (estimator radius = de_radius·9 px, similar-density gather with the erf/deCurve test) and `LogDensityFilter`'s sharpening-kernel rule (Mitchell colours, gaussian-0.75 intensity). `testflames/synth.flame` (single `blur` xform, known 1/r density) is the numeric check: at quality 300 both engines give 229/197 (r = 0.5/0.99, gamma 4, brightness 4). `aff3d.flame` / `dof.flame` are the 3D-affine, dimish-z and DOF references.
* **Weight semantics.** JWildfire's `rings` ignores its weight while flam3/Apophysis apply it; `PREFER_HAND` in `variations.ts` keeps the flam3 behaviour for such cases even when the port verifies.
