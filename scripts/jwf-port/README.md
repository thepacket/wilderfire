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
| `cwgsl.ts` | CUDA-C → WGSL transpiler (tokenizer, C-subset parser, typed emitter: int/float coercions, pointers/arrays/structs, overloads, macros, `switch`, helper library incl. the double-float `hsin_` hash). Helper functions that draw randoms get the RNG state as a trailing `rs` parameter (callers pass theirs on, settled by the emitter's fix-point passes); `jpostinc(&i)` is C's `i++` inside an expression. |
| `gen.ts` | Reads the dump + kernel helpers, transpiles every variation, applies `overrides.ts`, writes the two registry files (+ `variations.unportable.ts`) and prints a summary. `--report` also writes `report.json`. `EXTRA_STATE` restores per-instance Java state that a JWildfire GPU snippet declared as locals (`recurrenceplot`); the `fix` table hoists JWildfire's per-thread constant tables to module scope (`dc_perlin`: 2050-int permutation + 1024×3 gradients). |
| `overrides.ts` | Hand corrections (in CUDA dialect) for variations whose JWildfire GPU code diverges from its CPU code (`perspective`, `brick`, `chainmail`, `ouroboros`, `rays3`, `waves22`, `dc_hoshi`, `post_circlecrop`, …): full `gpuCode`, textual `patch`/`patchFuncs`, plus the `retry` (rejection-sampling) and `clampParams` mechanisms and the `HSIN2` double-float hash rewrites. |
| `data/param-clamps.json`, `data/param-ints.json`, `extract-clamps.py` | Per-parameter clamps and int casts from JWildfire's Java `setParameter()` (`Tools.limitValue` & co., `(int)`/`FTOI`), applied by `gen.ts` to every param read so out-of-range and fractional values behave like the CPU (int params are read as ints). Regenerate with `python3 extract-clamps.py <jwf>/src/org/jwildfire/create/tina/variation`. |
| `data/dc-base.json` | Which `dc_*` classes inherit / copy `DC_BaseFunc.transform` (sampling `rnd−0.5`) — drives the DC family fix in `gen.ts`. |
| `data/jwf-variations.jsonl` | Metadata + GPU code for all 1026 JWildfire variations (name, params/defaults/int-ness, priority, types, GPU code, helper functions). Produced by `Dump.java`; checked in so regeneration does not need Java. |
| `data/kernel-lib.cu` | Helper library extracted from JWildfire's `Flam4_3dKernal_TemplateJWF.cu` (Complex/Mat2/Jacobi/noise/misc); transpiled on demand. |
| `java2cu.ts`, `data/jwf-java-ports.jsonl` | Java → CUDA-dialect pre-processor for the variations that have *no* GPU snippet in JWildfire: extracts fields, params (from `setParameter`), `init()`, helper methods and `transform()` from the Java class, rewrites the Java idioms (`pAffineTP.x`→`__x`, `pVarTP.x`→`__px`, `pAmount`→`__amount_`, `pContext.random()`, `Math.*`/MathLib statics, `sinAndCos`, `this.` fields, locals shadowing fields, `random(Integer.MAX_VALUE)`→31-bit `RANDINT`), replays setParameter-derived fields, turns per-instance state (attractors) into per-thread `varpar->` state initialised on the first call, and copies helper-used params/fields into state. GLSL-style Java (js.glsl `vec2`/`vec3`/`vec4` objects, `G.*` statics, `new mat2(…)` in `.times()`) goes through a small expression parser that turns method chains into vector arithmetic (`crop_*`, `glsl_*`, `truchetflow`); the abstract `GLSLFunc` parent's fields/params are merged in. Plain-data inner classes (`Point`, `Double2`, `RandXYData`, `SinCosPair`, …) become WGSL structs with maker/setter functions; JWildfire's `XYZPoint`, `DoubleWrapperWF`, `MarsagliaRandomGenerator` and `java.util.Random` are built-in structs (`jxyz_`, `jdw_`, `jmrg_`, `jrand_` — the last two are exact 32-bit / 48-bit-LCG ports so per-cell seeded randoms match the Java); js.glsl `mat2`/`mat3` travel as `float4` / a `mat3_` struct (`.times()` resolves matrix·vector by tracked operand kinds); helpers taking `pAffineTP`/`pVarTP` are inlined at their call sites; array/object helper params become pointers; `pXForm.getXYCoeff*()` reads the affine; static-final constants are inlined and constant tables hoisted to module scope; `long` maps to `int` (seeds/hashes); pre-priority ports refresh `__r2/__r/__phi/__theta` after rewriting the input like JWildfire's own pre snippets. JWildfire *prepost* variations (`prepost_blob`/`mobius`/`circlize`/`affine`: `invtransform()` runs as a pre step on the affine point, `transform()` as a post step on the output) get two snippets — `preCode` (the inverse + precalc refresh) and `gpuCode` — that the codegen runs at priority −2 and 2 of the same stage; the oracle tests the inverse as its own `name~inv` entry (Oracle.java calls `invtransform`) and `gen.ts` requires both verdicts. Output feeds `gen.ts` exactly like a dump entry. Needs the JWildfire source tree (`--jwf`). |
| `data/g-atan2.json` | The variations whose CPU code takes its angles from js.glsl `G.atan2` — JWildfire's GLSL-helper class implements it as a rational approximation, not `Math.atan2` (see *Direct colour scale and G.atan2*). `gen.ts` rewrites every scalar `atan2f`/`atan2` in those entries' snippets to the `G_atan2` kernel helper (`cwgsl.ts`); `java2cu.ts` maps `G.atan2` / `G.Kscope` there directly. Regenerate by grepping the Java tree (the note inside says how). |
| `data/fastnoise.cu` | FastNoise (JWildfire's Flam4 template CUDA port of Auburn's FastNoise_Java; all noise types) + `wfieldValue()` for weighting fields; `gen.ts` transpiles it to `src/gpu/wfield.wgsl.ts`. |
| `data/unportable.json` | The JWildfire variations WilderFire deliberately does not implement, by category (see *What is not ported*); `gen.ts` checks it against the registry and emits `src/core/variations.unportable.ts` for the importer. |
| `bindings.ts` | The snippet environment (magic `__x/__px/…`, weight, params, per-thread state → WGSL) shared by `gen.ts` and the regression tests. |
| `../../tests/transpiler.test.ts`, `../../tests/java2cu.test.ts`, `../../tests/registry.test.ts` | Regression tests (`npm test`, run by the build): crafted CUDA snippets through `cwgsl.ts` and synthetic JWildfire-style Java classes (`tests/fixtures/java2cu/`, not JWildfire code) through `java2cu.ts` → CUDA → WGSL, with the output pinned as vitest snapshots (`tests/__snapshots__/`; review the diff and `npx vitest run -u` after an intended change); registry invariants against `verified.json`/`unportable.json`; and a codegen smoke test compiling every preset and fixture flame. |
| `Dump.java` | Dumps the catalogue by reflection against a compiled JWildfire tree. |
| `Oracle.java` | Headless JWildfire oracle: evaluates each variation's Java `transform()` on the spec grid; deterministic → exact values, random → per-point mean/std/hide-fraction over 256 samples. |
| `oracle-spec.ts` | Emits the shared test spec (130-point grid × 3 parameter sets: defaults, floats perturbed, ints perturbed) for every known variation. |
| `verified.json` | Verdicts written by the browser harness; `jwf` = ports that match JWildfire (these enter the app registry). |
| `testflames/` | JWildfire random-generator flames (`GenFlames` via JWildfire's own generators) + `yflip.flame` (orientation check). Loaded by `window.wilderfire.flameTest()`. |
| `../../src/dev/varTest.ts` | Browser harness: compiles each variation's WGSL, evaluates the spec grid on the GPU, diffs against the oracle, POSTs `verified.json` via the dev-server sink in `vite.config.ts`. |
| `../../src/dev/flameTest.ts` | Browser harness: imports every fixture flame, reports unsupported variations, compiles the kernel. |
| `../../src/dev/renderCheck.ts`, `render-baseline.json` | **Render-regression check** (`await window.wilderfire.renderCheck()`, ~8 s): renders the fixtures, the JWildfire samples and the presets offscreen (256 px, 200 spp), reduces each image to a signature (mean luma, covered fraction, 8×8 block means, 16-bin luma histogram) and compares it with the checked-in baseline — a per-flame failure with the numbers when a change alters what WilderFire renders (engine semantics, tonemap, a port, the importer). Tolerances sit ~4× above the run-to-run noise (block MAE 0.02 vs 0.005 measured), so it catches a wrong variation / broken tonemap / dropped layer, not 1 % shifts. After an *intended* change: `renderCheck({ update: true })` re-records the baseline through the dev-server sink. Needs WebGPU, so it is a manual pre-push check, not CI. **The baseline is environment-sensitive at the coverage threshold** (2/255 luma — faint stray speckle): a browser/Dawn/driver update can flip those pixels and fail a *block* of borderline flames on coverage alone while mean/blocks/histogram still pass (seen 2026-08-21: ~10 flames uniformly −0.03 coverage on byte-identical code, run-to-run spread ≤ 0.014). That signature — coverage-only failures, all in the same direction, on code that matches the recording commit — means *re-record*, not debug. A genuine code regression shows in mean/blocks/histogram too, or on flames outside the borderline set. |
| `mkfix.mjs`, `extract2.mjs`, `probes/*.java` | Fixture tools: `node scripts/jwf-port/mkfix.mjs <id> '<xform attrs>' [scale] [white\|gray\|graymin] [min]` writes a two-xform fixture around one variation (a corpus header, or `min` for a plain one — keep `saturation` at 1: JWildfire's saturation shift turns greys red); `node scripts/jwf-port/extract2.mjs '<regex>' <id> [skip]` picks the first corpus flame matching a regex with no unportable variation; the probes (compile against the JWildfire tree: `javac -d out -cp "$CP" probes/MProbe.java`) print a variation's private tables by reflection — mandala primitives / mandala2 grids (`MProbe`), snowflake points (`SProbe`), sunvoroni edges/regions/triangles (`VProbe`), klein_group generators (`KProbe`), java.awt.Color maps (`CProbe`) — the way to tell a table bug from a kernel bug. |
| `../../src/dev/flameCompare.ts`, `RenderOne.java`, `Compare.java` | Whole-image comparison against headless JWildfire: `await window.wilderfire.flameCompare()` renders the fixtures, the bundled JWildfire samples and the authored presets offscreen (512 px wide, quality 100) into `compare-out/` (PNG + the exact .flame XML, gitignored); `java … Compare <repo>/compare-out` renders the same XML with JWildfire (cached as `<id>.jwf.png`) and prints per-flame metrics — mean luma of both, ratio, coverage, 16×16-block MAE, luma-histogram intersection, block correlation, per-channel R/G/B mean ratios (flag `channel` when one strays > 0.2 from the luma ratio: a colour-scale error cancelling a pattern error hid behind a 1.04 luma ratio once) — with flags. Numbers only; no pixels are judged by eye. `RenderOne.java` renders one .flame to PNG. |

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
# (put JWildfire's `resources` dir on $CP too — Compare/Oracle need variation_costs.txt for weighting-field param modulations)
cd /path/to/wilderfire
node scripts/jwf-port/gen.ts            # transpile
node scripts/jwf-port/oracle-spec.ts    # spec for everything we know
java -Djava.awt.headless=true -cp "jwf/tools/out:$CP" Oracle scripts/jwf-port/oracle-spec.txt scripts/jwf-port/oracle-out.jsonl

# 3. verify in the browser (dev server running)
#    await window.wilderfire.varTest()   → writes scripts/jwf-port/verified.json
node scripts/jwf-port/gen.ts            # re-emit with the new verdicts
```

## Status

Of JWildfire's 1026 variations, 800 carry a usable GPU snippet (14 more than
before: `REFERENCE` ressources — links to the author's page — no longer count
as resources) and 140 more are ported straight from their Java `transform()`
by `java2cu.ts` (940 transpile); **920 verify numerically against the Java
oracle in 3D**, 18 more are verified by inspection where the per-point oracle
cannot compare (`FORCE_VERIFIED` in `gen.ts`, each with its reason:
`arch`/`rays`/`starfractal` heavy-tailed statistics, the chaotic attractors
`hopalong`/`macmillan`/`threeply`/`gumowski_mira`/`gingerbread_man`, the
`curliecue2` walk, `post_point_crop` and `recurrenceplot` order-dependent
state, `minkQM` f32 boundary artefact of the test grid, `circular`/`circular2`
hash of the continuous input point, `iconattractor_js` presetId table,
`pre_blur`/`pre_blur3D` `& 5` ring buffers, `pre_flatten`), so 938 ship in
`variations.jwf.ts`; 2 stay in `variations.jwf.unverified.ts` because the Java
itself is order-dependent (`dc_circuits` accumulates a member `S` across
points, `dc_gnarly` updates only 2 of its 6 gaussian summands — `& 5` — so its
blur depends on the render's init randoms). Together with the 70 hand-written
flam3 entries the app registry has 940 variations.

### What is not ported (50)

`data/unportable.json` is the definitive list — every JWildfire variation is
either in the registry or in that file with a category, and `gen.ts` writes it
to `src/core/variations.unportable.ts` so the `.flame` importer can say *why* a
variation was skipped. Categories:

| category | count | why |
|---|---|---|
| user-code | 21 | compiles user-supplied code or a formula at run time (`custom_wf`, `dc_code`, `glsl_code`, `c_var`, `ducks`, `fract_formula_*`, the `yplot2d_wf`… plot family, `colordomain`); the WebGPU kernel has no run-time compiler |
| external-content | 25 | renders external content that would have to be uploaded to the GPU: sub-flames (`ringsubflame`, `glynns3subfl`), images (`post_bumpmap_wf`, `displacemap_wf`, `colormap_wf`, `kaleidoimg`, `plane_wf`, `wangtiles`), meshes (`terrain3D`, `metaballs3d_wf`, `knots3D`; `sattractor3D` IS ported — its formulas run through a small safe evaluator, src/core/formula.ts, and the tube is built on the CPU, src/core/sattractor.ts), `svg_wf`, `text_wf`, L-systems, brushes (`obj_mesh_wf` IS ported — the user loads the OBJ file into the browser's mesh store; `subflame_wf` IS ported — the sub-flame is compiled into the kernel) |
| point-set | 1 | `taprats` (needs the csk.taprats tiling library, which the sparse JWildfire tree does not carry); `neuron3D` builds a seed-shuffled 512-entry Perlin permutation table per instance. **Ported through the point-set mechanism (see below):** `dragon_js`, `sunflower`, `scrambly`, `dla_wf`, `snowflake_wf`, `brownian_js`, `htree_js`, `koch_js`, `tree_js`, `hilbert_js`, `klein_group`, `grid3d_wf`, `maurer_lines` (lines render mode), `szubieta`, `gpattern`, `curliecue`, `gosperisland_js`, `rsquares_js`, `arctruchet`, `triantruchet`, `meeple`, `mandala`, `mandala2`, `nsudoku`, `natural_foam`, `geometricPrimitives`, `point_mirror_symmetry`, `neuron3D`, `sunvoroni` |
| engine | 2 | needs an engine feature WilderFire lacks: a variation instantiating another (`sphtiling3v2`), `post_dcztransl` (no Java class) |
| resource-params | 1 | `dc_triantess` keeps its colours as byte-array ressources |

The last `not-yet` batch (2026-08-17: `exp_multi`, `quad`, `ringtile`, `pre_wave3D_wf`, `boxfold`, `drunken_tiles`, `cactusGlobe`, `dc_triTile`) went through `java2cu`: JWildfire's `Complex` class is a built-in struct library (`jcx_`, a line-by-line port of `org.jwildfire.base.mathlib.Complex` incl. `per_fix`/save state), helpers returning small `double[]` become `vec3`/`vec4` (pre-patch), an object parameter that is only handed on to another helper's pointer parameter becomes a pointer too, null-only guards are dropped, `Integer.hashCode`/`Double.isNaN`/`isInfinite` map, the affine point passed to a helper travels as a value copy (`jxyz_make`), and `setParameter` branches with a local (`double v = …; field = v;`) resolve to the field. The oracle/harness now report *input + accumulator* for pre-priority rows (JWildfire keeps what a pre step adds into `pVarT` for the main sum — `ringtile`'s inverse writes it).

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

### Point-set variations (2026-08-22)

JWildfire's `DrawFunc` family, the `_js` turtle fractals and a few simulations build a list of primitives on the CPU
in `init()` and sample one per point. `src/core/pointSets.ts` does the same: a builder per variation (registered by
name, cached per parameter set) writes 12-float records — point / line / triangle / n-gon / dot — through a
`PointSetWriter`, the renderer packs every set the flame uses into one storage buffer (binding 13 `pset`,
synchronously in `setFlame`, like the mesh buffer), each instance's two hidden slots carry its record base and count
(codegen data hook, flag `pset`), and WGSL `psetSample()` reproduces `DrawFunc.getPrimitive` (uniform by index),
`plotBlur`, `plotLine` (± thickness), the uniform triangle and `nBlur`'s `randXY` polygon sampling exactly;
`psetLineOrDot()` is the turtle family's `pLine`/`pDot` (a line with probability show_lines/(show_lines+show_points),
else a dot of radius point_thickness/100 around the first end). JWildfire's `Turtle`, its
`MarsagliaRandomGenerator` (seeded clouds) and `java.util.Random` (the 48-bit LCG, for `brownian_js`) are ported so
seeded sets are the same points. `klein_group` is the odd one: no table, but its Möbius generators (the seven recipes
— Grandma, Maskit, modified Maskit, Jorgensen, Riley, modified Riley, Maskit-Leys — with JEP's principal-branch
complex sqrt) are computed on the CPU through a `VariationDef.derive` hook into 32 hidden slots, and the
"previous matrix" memory for `avoid_reversal` is a per-thread private.

Verdicts (Compare against headless JWildfire on corpus flames, 512 px / quality 100; fixtures
`testflames/_ps_*.flame`): `sunflower` 1.00 / corr 1.00, `scrambly` 0.98 / 1.00, `dla_wf` 0.99 / 1.00,
`dragon_js` 1.02 / 0.94 (sparse), `htree_js` 1.00 / 0.99, `koch_js` 1.00 / 1.00, `hilbert_js` 1.01 / 1.00,
`tree_js` 1.00 / 1.00 (TreeFunc adds its output twice — kept), `brownian_js` 1.01 / 0.97, `klein_group` 1.02 / 1.00
(Jorgensen, final xform) and 1.04 / 1.00 (modified Riley); `grid3d_wf` 1.03 / 0.98 — not a point set after all: a
per-cell *memoised* random spread (one shared `java.util.Random(123)` drawn in first-visit order, so not reproducible
even in JWildfire) became a per-cell hash with the same distribution, the cube faces / c1..c6 colours / rotation are
exact and, as in JWildfire, the amount is ignored; `snowflake_wf` **exact** (1.00 / blkMAE 0.0 / corr 1.00 with a
`linear` partner — Reiter's hexagonal automaton ported cell for cell: same Marsaglia background noise, freeze /
diffusion passes, stretched canvas, row→x column→y mapping and per-cell jitter draws; the point list equals
`SnowflakeWFFunc._points` to float precision and a unit test pins five of the 1622 points; the corpus fixture itself
stays at corr 0.53 because its other xform carries `curliecue2`, whose transform ignores the input and advances one
global trajectory by a step per call — JWildfire's few long-lived threads trace the whole curlicue, our short-lived
walkers only its first steps: the known short-walker limit, recorded in verified.json); `maurer_lines` 0.97 / 0.5 /
1.00 (rhodonea) — ported as an ordinary kernel variation for the LINES render mode with all 19 parametric curves,
show_points / show_curve, line thickness and diff_mode; the spline / circle / ellipse / sine / Bézier render modes
(drawn as the line), cosets, meta modes, filters and the direct-colour measures are not ported — the corpus never sets
them. The generator matrices of every recipe are unit-tested
against values probed from `KleinGroupFunc.init()` (`tests/pointSets.test.ts`). Quirks kept: JWildfire reads
the brownian canvas sequentially from one shared list (we pick uniformly — same distribution); `tree_js_size` in old
files is an attribute the current `TreeFunc` no longer has (ignored by both).

**2026-08-23 — the rest of the family (14 more).** Compare on synthetic two-xform fixtures (`mkfix`: one xform = the
variation, one linear) unless noted: `szubieta` 0.99 / 1.00 (both patterns), `gpattern` 0.98 / 1.00 (the side-count
list is a string *resource*: `VariationDef.res` now reaches point-set builders and `derive`), `curliecue` 0.99 / 1.00
(points + lines, n-gons, 4-fold symmetry), `gosperisland_js` 0.99 / 0.99 (level capped at 6: 6·7^level segments),
`rsquares_js` 1.00 / 1.00, `arctruchet` 0.99 / 1.00 (seeded quarter-turn table), `triantruchet` 0.99 / 1.00 (seeded
and string-driven), `meeple` 0.99 / 1.00, `natural_foam` 1.00 / 1.00 (bubble table; 3D), `geometricPrimitives`
1.02 / 1.00 (the `frac(sin(…)·43758.5453)` cell hash runs through the double-float `hsin_` helper, now exported for
hand ports in `src/core/wgslDf.ts`; a negative hash with `shape_mix_enable` hits no case in the Java switch and
plots the origin — kept), `nsudoku` 0.97 / 0.99 (JWildfire's board comes from an *unseeded* Random, ours from
java.util.Random(1000); the 1296-row transform machinery is exact), `mandala` 1.00 / blkMAE 0.7 / corr 1.00 on a
grey-ramp palette and 1.03–1.05 / 1.00 with real palettes (the rotator-map orbits, step counts and java.awt.Color
HSB/RGB colour maps are unit-tested against Java reflection probes), `mandala2` (step grid and colours exact by the
same probes; its image is a lattice of single points, so the two renderers' treatment of isolated dots dominates the
Compare numbers — ≈1.5 for both its direct-colour and its index-colour modes once direct colours sit on JWildfire's scale), `point_mirror_symmetry` (exact without the crop, 0.42–0.74 with it: JWildfire keeps
one ring of the last `buffer_size` valid points per render thread and feeds cropped points from it; ours is a
per-walker ring of at most 32, a different sampling of the same set — approximate by design).

`neuron3D` 1.01 / blkMAE 0.5 / corr 1.00 and 0.97 / 1.4 / 1.00 (second seed, zoom 6, texture 0.25): the seed-shuffled
Perlin permutation is a 512-entry table in the pset buffer; the cell type comes from `java.util.Random(cellHash)` whose
48-bit seed (`ix·73856093 ^ iy·19349663 ^ iz·83492791 ^ seed` as Java longs) is rebuilt with a 16-bit-limb multiply
(`n3_mul48`) so it matches for any zoom; connector hashes go through the ordinary 32-bit `jrand_make`; the noise
arguments `connHash + 1.2` are split into integer and fractional parts before they reach f32. `sunvoroni` (JWildfire:
megamu.mesh.Voronoi on a quickhull jar + csk.taprats Triangulate): a Bowyer–Watson insertion into megamu's far triangle
(±8000) gives the same Delaunay faces, circumcentres by megamu's own formula, the cell outlines as lines and the cells
ear-clipped exactly as `Triangulate.Process`; the 50-point case is unit-tested against a Java reflection probe (edge
set, region vertex sets, triangle count). What the port cannot know is QuickHull's face order, which sets each region's
starting vertex and orientation (the probe shows 29 counter-clockwise / 21 clockwise regions) — so the ear-clipping
decomposition and the area-signed colours of mode 0 (`fmod(1 + area, 1)`: a or 1 − |a|) differ per cell: colour by
index (mode 1) 0.99 / corr 0.96, area colours 1.03 / 0.89. `taprats` stays out (library not in the tree).

**Engine leftovers, second look (2026-08-23).** The old list was mostly stale or empty: `receiveOnlyShadows`, light
motion curves, `smooth_gradient` and gradient maps have no corpus flame setting them (every
`*_receive_only_shadows` in 5623 flames is 0), `dc_carpet3D` (f8313ea) and `_sh8` / `_mesh10` (6b12c8b) were already
fixed. The four real ones are in:
* **Colour types TARGET / TARGETG** (70 uses): `TransformationTargetColorStep` — unless a direct-colour variation
  set the RGB, the point's carried colour (the palette entry of its index, or the RGB a DISTANCE/TARGET step gave it)
  is lerped towards `targetcolor` by t = (symmetry + 1)/2; the index is kept (c1 = 1). TARGETG targets the palette
  entry at the xform's `color` and blends the index like DIFFUSION. Plot-colour tier 0.5 in the kernel (a later
  gradient step replaces it), xd slots 66..68. Butterfly (TARGET) 0.97 / blkMAE 1.3 / corr 1.00; Fake Widow Spider
  (19 xforms mixing both) 0.84 / 0.95. Not modelled: a point carrying a TARGET colour through a *later iteration*
  into the next TARGET xform (the carried RGB is re-derived from the index each iteration).
* **DOF blur shapes** (21 flames with cam_dof > 0): `DOFBlurShapeType` runs a blur variation on the jitter with
  weight dr = 0.1·cam_dof·dist·scale·fade and `cam_dof_param1..6` as its parameters, the offset rotated by
  `cam_dof_rotate`: starblur, xheart_blur_wf (|dr|), flower (|dr|·2, random input), taurus (|dr|·0.25, random
  angles·fade), cloverleaf_wf (random input·fade), sineblur (no rotation), square (RECT, x × width). The variation's
  snippet is wrapped into a kernel function `dofShape_` like the oracle harness does; NBLUR, SNOWFLAKE, CANNABISCURVE,
  PERLIN_NOISE, BRUSH_STROKE and SUBFLAME fall back to the bubble disc. Mini Hearts (HEART) 1.03 / 1.7 / 1.00, Warper
  (FLOWER) 0.97 / 2.9 / 0.99, Snow (STARBLUR) 0.88 / 0.5 / 0.99, A snowball fight (NBLUR → bubble) 1.05 / 0.94.
* **Channel mixer** (`mixer_mode` RGB / BRIGHTNESS / FULL, 18 flames): `LogDensityFilter` hands the nine curves the
  pixel's *average* raw colour (rp.red·100/count — hence the 0..25600 axis) and puts the result ×count back before
  log scaling. 9 LUTs of 257 entries (our `evalCurve` is already JWildfire's SPLINE envelope) in the filter buffer,
  applied in tonemap pass A after density estimation. FULL 1.00 / 3.6 / 0.99, RGB 0.99 / 0.5 / 1.00, BRIGHTNESS
  1.03 / 4.0 / 1.00.
* **background_image** (61 flames): `calculateBGColor` stretches the picture to the full image and samples it
  bilinearly — a linear texture sample at the texel-centre coordinate in tonemap pass B (bindings 4/5, a 1×1 fallback
  while absent). The file name is kept (basename) and the picture comes from the browser's image store (Render →
  Background → ⬆ image / chooser), like reflection maps. Compare with JWildfire loading the same PNG: 1.04 / 1.6 /
  1.00.

**Short-walker variations (2026-08-23).** A JWildfire variation whose `transform` ignores its input and advances one
global state per call draws, per render thread, the first Q steps of that state's trajectory, Q = the thread's iteration
count; our 65k walkers each live a few hundred iterations, so the generated port only ever drew the first steps. Of the
registry's stateful entries only `curliecue2` is of that kind (the attractors cover their invariant set from any start;
`recurrenceplot`'s `oldx/oldy`, `dc_gnarly`'s six-deep random ring and `point_mirror_symmetry`'s ring are per-walker
memory that works as such). It is now a point set: the trajectory (x += 0.001·cos φ, φ += θ, θ += 2π·s, s =
java.util.Random(seed).nextDouble()) is tabulated for 2^20 steps (8 MB) and each point samples a uniform step — Q for a
512 px / quality-100 render on 8 threads is ≈ 2.5 M iterations × the xform's share, so 2^20 is the Compare regime (a
longer JWildfire render draws a longer curve: inherent). The Sanctuary Xmas 2022 snowflake flame went from corr 0.53
to **1.00 / blkMAE 0.5 / corr 1.00**; the koch isolation from 0.70 to 1.00 / 1.1 / 1.00. `PREFER_HAND` keeps the hand
entry over the generated one. The other "inherent" limit, f32 arithmetic, stays: WebGPU has no f64; the places where
it matters (cell hashes, double-float sin) already run in two-float arithmetic.

Two lessons. (1) **Do colour maths on the CPU.** The first mandala port computed Java's HSBtoRGB / RGBtoHSB in WGSL;
the functions returned Java's exact values in a stand-alone compute shader yet produced wrong hues inside the full
kernel (fast-math lowering of the `switch`/`log` chain, presumably) — a grey-ramp palette made the shift
measurable (our index mean 0.72 vs 0.45). The hue / packed RGB now live in the records and the kernel only copies.
(2) **JWildfire's `saturation` ≠ 1 turns greys red**: `applyModSaturation` shifts HSL saturation and an achromatic
pixel's hue is 0, so a grey palette renders red-tinted in JWildfire; keep fixtures at saturation 1.

Leftover found with the RGB-direct controls (resolved 2026-08-23, see *Direct colour scale and G.atan2* below): under a
minimal header the already-verified direct-colour variations `dc_truchet` / `dc_menger` rendered at 0.82 / 0.92 of
JWildfire's luma — fixtures `_ps_rgbctl`, `_ps_rgbctl2`, now 0.99 / 0.99.

Found on the way: **`gamma="0.0"`** (20 of 5433 corpus flames, JWildfire V4.10/V5.50 files) is not "default" in
JWildfire — `GammaCorrectionFilter` keeps the exponent 0, i.e. `pow(intensity, 0) = 1`: a flat image at full alpha
above the threshold. The importer used to reject it and keep gamma 4 (renders at 0.5–0.6 of JWildfire's luma); now
gamma 0 is kept, both tonemaps use exponent 0 for it (`gpow` keeps 0⁰ = 1 under fast-math), the JSON normaliser
preserves it. `_ps_brownian_js` went from 0.51 / blkMAE 69 / corr 0.83 to 1.01 / 3.9 / 0.97.

### Direct colour scale and G.atan2 (2026-08-23)

Two engine findings from the `_ps_rgbctl` leftover, both root causes rather than tonemap tuning:

* **Direct RGB is on a different scale from the palette in JWildfire.** `RGBPalette.createRenderPalette` stores every
  palette entry as `RenderColor = rgb · 200/256` (0..199.2) and `TransformationGradientColorStep` plots that; a
  direct-colour variation (`DC_BaseFunc` and the `GLSL*`/`mandala`/`maurer_lines`/`colormap_wf`/… families — all of
  them write 0..255 ints: `dbl2int`, `Color.getRed()`, pixel channels) puts its raw value into `redColor` and
  `plotPoint` uses it unchanged. So a direct white is 255 where a palette white is 199.2: direct colours are ×1.28
  brighter. Our histogram held both on the palette's scale (`rgbo` 1.0 = palette white) and the tonemap's
  ×199.2/whiteLevel then applied to both. Now the plot multiplies a direct colour by 256/200 (codegen, both the density
  and the solid splat; before the colour modifiers, whose gamma/contrast formulas work on JWildfire's absolute values),
  and `subflame_wf`'s pass-through hands a sub-palette colour over as ·200/256 — JWildfire copies `q.redColor`
  verbatim, so a CM_FLAME (-2) sub point carries the RenderColor-scale value (`_sub5`, a `_sub3` variant with
  `color_mode="-2"`: 1.14 vs `_sub3`'s 1.15 — identical in both renderers). TARGET / DISTANCE plot colours come from
  `targetRenderColor` / the palette (RenderColor scale) and keep `w = 0.5`, i.e. unscaled. `_ps_rgbctl` 0.82 → 0.99,
  `_ps_rgbctl2` 0.92 → 0.99, and per-channel means of the PNG pairs now agree to ±0.1.
* **js.glsl `G.atan2` is an approximation.** JWildfire's GLSL-helper class `js.glsl.G` implements `atan2` as the
  rational approximation `π/4 − π/4·(x−|y|)/(x+|y|)` (x ≥ 0) / `3π/4 − π/4·(x+|y|)/(|y|−x)` (x < 0), signed by y —
  up to 0.07 rad off — and `G.Kscope` uses it too; `G.atan(n, d)` is `atan(n/d)` (no quadrant). The CPU render the
  corpus is authored against uses these, but JWildfire's own GPU snippets call the exact `atan2f`, and java2cu used
  to map `G.atan2` there as well. For `dc_ducks` (50 chaotic iterations of `(log|e|, atan2 − 6.2)`) the exact atan2
  produces a different colour pattern: per channel, our red/green means were 22.6/22.9 vs JWildfire's 19.1/18.8 with
  a histogram peak in the wrong bin (f32 vs f64 was ruled out by an emulation: same bin counts). The 28 verified
  variations whose Java uses `G.atan2`/`G.Kscope` (`data/g-atan2.json`: `cut_*`, `dc_*`, `glsl_*`, `crop_stars`,
  `post_crop_stars`) now go through a `G_atan2` WGSL helper; the oracle keeps all 28 at pass (26 at 1.000). Compare:
  `_ps_ducks` (before, scale fix alone) 1.39 / blkMAE 8.8 → **0.99 / 0.4 / corr 1.00** with the per-channel
  histograms matching bin for bin; `Machine_1` (corpus, dc_ducks + yin_yang) 1.04 / 2.4 / 0.95 → 0.99 / 0.5 / 1.00;
  `_ps_hoshi`, `_ps_kaleid`, `_ps_cutweb` 0.99 / corr 1.00. Lesson: when a luma ratio looks "close enough", check
  the channels — the 0.78 scale error and the pattern error were cancelling in `Machine_1`.

### The carried colour (2026-08-23)

JWildfire's point carries an **RGB** across iterations (`XYZPoint.redColor/greenColor/blueColor`, initially 0 = black on
the RenderColor scale), and an xform's colour step only ever edits that: DIFFUSION/CYCLIC replace it with the palette
entry of the (blended) index (`GradientColorStep`), TARGET/TARGETG lerp *from it* towards the target, DISTANCE replaces
it, **NONE has no colour step at all** (the point keeps what it carried — finals default to NONE), and a direct-colour
variation overwrites it (the `rgbColor` flag that makes the steps skip is reset per transform, so in the next iteration
a TARGET/DISTANCE step applies to the direct values). The index travels separately and is only read by gradient steps
(`c1 = 1, c2 = 0` for every type but DIFFUSION/TARGETG). Finals work on a copy (`q`): they read the carried colour but
do not change it. Ours rebuilt the colour from the index every iteration, so TARGET always lerped from
`palette[index]`, NONE plotted `palette[index]`, and a direct colour never reached a later TARGET.

Now `colorType` has a real `NONE` (importer: explicit NONE and finals without DIFFUSION; legacy JSON finals with
colorSpeed 0 normalise to it; the exporter writes `color_type="DIFFUSION"` on a recolouring final; the speed slider
turns a NONE final into DIFFUSION above 0 and back), and when a normal xform is TARGET/TARGETG/DISTANCE/NONE or writes
a direct colour (`usesCarry`) the kernel keeps the point's RGB in a per-walker buffer (binding 14 `crgb`, vec4: rgb +
tier — 0 = palette[index], 0.5 = palette-scale RGB, 1 = direct this iteration, 0.8 = direct carried over, 0..255 scale).
DIFFUSION/CYCLIC reset it to tier 0 before their variations; TARGETG pins a tier-0 colour to the palette entry *before*
it blends the index (the carried colour is palette[old index] — this was also wrong for TARGETG finals); the plot looks
the palette up the way `GradientColorStep` does (`(int)(c·254 + 0.5)`, it used flam3's `c·255.99`). The signature
carries each xform's colour type (a TARGET → DISTANCE edit did not regenerate the kernel before).

Compare (old → new): `_ct2` blkMAE 3.9 → 0.4, `_ct3` 1.17 → 0.99, `_ct4` 13.4 → 5.5 (corr 0.75 → 0.89), `_ct4n`
(NONE) 37.6 → 8.3, `_sh4` 1.12 → 1.00, `_sh6` 1.09 → 0.97, `_ps_sunflower` 0.96 → 1.00, synthetic chains `_carry1`
(dc_truchet → NONE → TARGET) 1.16 → 0.99, `_carry2` (DISTANCE → NONE → TARGET) 1.19 → 0.99, `_carry3/4/5`
(DIFFUSION → TARGET → TARGET, + NONE + TARGETG) 0.99 / corr 1.00; across fixtures + samples 14 flames moved, all
towards JWildfire (TINA0019 2.4 → 0.4, Machine_0 1.1 → 0.3) bar noise. Left: `_ct_target2` (Fake Widow Spider, 17
`line` xforms under xaos) 0.82 with a pre-existing coverage gap (0.14 vs 0.18: thin random-point lines, the
isolated-dot class) though its structure improved (3.3 → 2.1). `Bokeh_0`'s `channel` flag (R 1.22, G 1.03, B 1.17) is
not a colour error: the picture is green on black, so the red/blue means are tiny (4.0 vs 3.1, 8.8 vs 8.3) and a few
hundred extra faint-fringe pixels (GLOBAL_SHARPENING + DOF, the DE-fringe class) move them by 20 % while green moves
1 % — read the channel ratios against the channel means. `sunvoroni`'s region order stays approximate on purpose: no
corpus flame uses the variation and an exact order means porting QuickHull3D itself.

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
Of the 44 random-generator fixtures, 3 reference variations we do not have
(`colordomain` in `Rays_0`, `post_brush_stroke_wf` in `Duckies_1`/`Painterly_0`)
and 3 differ for an inherent reason (`Julians_0` — hopalong's single long
trajectory vs our 65k short ones — `Cross_0`, `Splits_1`: non-contracting
walks whose JWildfire points drift out of view over millions of steps while
our short trajectories stay near the origin); **the other 38 match**
(`Bokeh_1` closest at 0.80 / corr 0.84 — a three-layer flame).

### Solid rendering — stage 5: post-process DOF + bokeh (2026-08-21)

JWildfire never jitters solid plots: `FlameRendererView` calls `applyOnlyCamera` and defers the whole
blur to `PostDOFCalculator`, a scatter pass over the finished tonemapped image. Ported as
`SOLID_PDOF_WGSL` (+ blit): when the flame is solid and `cam_dof > 0`, the solid tonemap renders into
the rgba16float mid texture (**clamped to 0..1 before scattering — JWildfire's addSample sees finished
0..255 ints, not HDR**), then a compute pass scatters every pixel as a kernel-weighted disc into a
fixed-point buffer (energy-normalised per disc, like PostDOFBuffer.addSamples), and a blit presents it.
`dofDist` is recomputed per pixel from the z-buffer's stored world position with the view's own formula —
**scaled by `cam_dof · image-diagonal / 1000`** (`RasterFloatIntForSolidRendering.calcDOF`'s `dofAmount`;
missing this made discs world-sized: blur far too wide and glints ~6× too strong). Glints
(`post_bokeh_*`, now on `SolidRender.postBokeh`): probability `intensity·1000/diagonal` per pixel with
luma ≥ activation, colour × rnd(0.2..0.4)·radius²·brightness, radius × size·(1+rnd·size); the disc's
kernel argument keeps the *pre-glint* `support/plainRadius` scale, and SINEPOW15 dies at r ≈ 0.9, so the
LUT (256 samples over 8·support, negatives clamped like JWildfire's `> EPSILON` skip) covers it. The
scatter's rnd is seeded per pixel with a fixed constant so live-preview glints don't flicker.
**Compare verdict:** `_pdof0/1/2` (Solid_0 + DOF; glints off / default / ×10) all at **ratio 1.00,
blkMAE ≤ 0.7, corr 1.00**; real flame `_pdofS` 0.99/0.98. (`_pdofT` (sattractor3D) is at **ratio 1.00 / corr 0.99** since the sattractor3D port — not post-DOF. `_pdofR` was at corr 0.14 and
is now **0.99**: its dc_carpet3D is a `z`-flagged variation whose Java still carries the preserve-z
clause, which the port had compiled to `false` and the engine's 2D-only rule skipped — with
`preserve_z="1"` under a 60° pitch every point sat at the wrong z. `Z_PRESERVE_TOO` in codegen adds the
line for dc_carpet3D and whirligig; tests/preserveZ3d.test.ts. Isolated by variants of the flame:
preserve_z off → 1.00, DOF off alone → 0.09.)
Survey correction: `post_bokeh_*` only ever fires with solid + `cam_dof > 0` — 0 flames in the
community corpus, 19 in the user's own collection; the "2 %" was JWildfire writing its default 0.005.
Tracked fixture `Solid_5` (baseline 98). Leftover from the solid plan now: `receiveOnlyShadows`,
light motion curves. **Reflection maps** (2026-08-22): `MaterialSettings.reflMapFilename` → the image is kept in
the browser's image store (`src/core/reflMaps.ts`, IndexedDB `images`, name = file name, the .flame's path is
stripped like obj_mesh_wf's), resampled to 512² into one layer of a `texture_2d_array` bound at the solid tonemap's
binding 8; `shadeCell` adds, per light, `reflColor(uv) · visibility · intensity` with `reflect((0,0,1), n)` mapped by
`UVPairD.sphericalBlinnNewellLatitudeMapping` (atan of the quotient) or `sphericalOpenGlMapping`, the SSAO reduction,
and `getColorFromMap`'s quirk (four texels, black outside, blerp weighted by the *global* u, v). Not multiplied by the
light's intensity — that cost a 3.6× miss on the first try (flat-grey map: 2 lights ÷ (0.35 + 0.2)). A blended
material (fractional index) has no map, as `morphMaterial` never copies the file name. Verified with a synthetic
256×128 gradient image and a flat grey one on a dimmed `_solid1` (`_refl0` control 1.00; `_refl1` Blinn–Newell,
`_refl2` spherical, `_refl3` grey: ratio 1.00 / blkMAE 0.4 / corr 1.00; JWildfire reads the absolute path written
in the fixture, WilderFire the same image stored under its basename). `_sh8`/`_mesh10` resolved 2026-08-22 — neither was geometry: `_sh8` was `checkerboard_wf`'s GPU snippet (int cast, above), `_mesh10` was `hypertile3D2` (p=7, q=5): JWildfire's CPU picks one of p rotations exactly (`random(MAX_INT)·2π/p`, only k mod p matters), its GPU snippets round `RANDFLOAT·0x7fff` and multiply in f32 — tens of thousands of radians whose sine the GPU evaluates coarsely, so the rotation jittered and the tiling smeared into 20 % more covered pixels (in the density render too, faintly; every point counts in a z-buffer). `overrides.ts` rewrites the angle as `(int)(rnd·p)·2π/p` for hypertile2/3D1/3D2/3D2b (period 2p/b there) and phoenix_julia (uniform angle for a fractional power); `_mesh10` 1.25 / 0.90 → **1.00 / 1.00**, oracle 100 % on all six.

### Attributes the importer used to drop (2026-08-21)

A flame can import without a single reported error and still render differently, because
the importer reports unknown *variations* but silently ignores unknown *attributes*.
`node scripts/jwf-port/attr-survey.mjs <dir>` counts, over a corpus of real flames, how many
of them switch an unported feature ON — the number that decides what is worth porting.
Over 5623 flames from ~150 community packs:

| JWildfire attribute | flames | ported as |
|---|---|---|
| `bg_transparency` | 32% | `Flame.bgTransparency`; presets the PNG export's alpha checkbox |
| `oversample` | 24% | `Flame.oversample`, applied to the renderer when a flame loads |
| `saturation` | 23% | HSL saturation shift of (value − 1) on the finished pixel, **after** the background is composited in (`GammaCorrectionFilter.applyModSaturation`); clamped at −1 |
| `post_symmetry_*` | 11% | plotted points are duplicated in the kernel (`symApply`): X/Y axis mirror about the centre at half `distance`, with the two copies counter-rotated by `rotation`; POINT plots the original plus `order` rotated copies — JWildfire's own i = 0 copy repeats the original, so the centre is plotted twice and we match that. **The tonemap divides k1 by the copy count** (see below) |
| `filter_kernel="MITCHELL_SINEPOW"` | 6% | adaptive filtering: per pixel, SINEPOW10 at 1.5×(filter + 0.25) where the density is below `filter_low_density`, SINEPOW10 at 1× where the Scharr edge response is below `filter_sharpness`, else Mitchell-smooth at 0.75× |
| `fg_opacity` | 4% | scales **alpha only** by 1 − atan(3·(v − 1))/1.25; the colour keeps its unscaled log-scale factor, so the background composites in more or less |

**Compare verdict (2026-08-21, Brokat_0 variants at 384 px / quality 300).** Isolate-and-diff
fixtures `_cmp_*` (gitignored) against headless JWildfire, control `_cmp_base` at ratio 0.98 /
blkMAE 4.7 / corr 0.98:

| fixture | ratio | blkMAE | hist | corr | |
|---|---|---|---|---|---|
| `_cmp_sat05` / `_cmp_sat14` | 0.98 / 0.97 | 4.9 / 7.9 | 0.95 / 0.94 | 0.98 / 0.96 | matches the control ✓ |
| `_cmp_adapt` (MITCHELL_SINEPOW) | 1.00 | 2.7 | 0.97 | 0.99 | better than the control ✓ |
| `_cmp_fg05` | 0.99 | 2.0 | 0.98 | 1.00 | better than the control ✓ |
| `_cmp_psymX` / `_cmp_psymY` | 1.17 | 23.7 | 0.66 | 0.94 | **too bright** |
| `_cmp_psymP` (order 3) | 1.29 | 42.4 | 0.42 | 0.97 | **too bright** |

**Post-symmetry brightness — root-caused and FIXED (2026-08-21, the raster-count experiment).**
The raw-raster probe (`scratchpad/diff/RasterProbe.java`, reflection into `FlameRenderer.raster`
→ `rawCount`; our side reads `offHist` back — it now carries COPY_SRC for exactly this) showed
both engines double their plot counts *exactly* (JW 14.34M → 28.67M, ours ratio 2.000), yet
JWildfire's luma rose only +7% while ours rose +28% — so the divergence was in the tonemap's
response, not the plotting. The cause sits in plain sight in `LogScaleCalculator`'s constructor:
**k1 is divided by the symmetry copy count** (POINT → `/order`, X/Y_AXIS → `/2` — note `/order`
even though order+1 points are plotted). In the log curve's linear regime the doubled density and
the halved k1 cancel, leaving only the redistribution gain. The `bg_glow` term is built from k2
and is deliberately NOT divided. Fix: `writeUniforms` divides the brightness uniform (k1's only
input; both its shader consumers — `logScaled` and `adaptSelect`'s intensity — want the divided
value). After the fix: `_cmp_psym0/X/Y/P` at ratio 0.98–0.99 (control 0.97), psymP blkMAE
42.4 → 2.1, hist ∩ 0.42 → 0.98. Tracked fixtures `Psym_0` (POINT order 3 off-centre) and
`Psym_1` (X_AXIS + rotation) are in the render baseline (97 flames).

Findings worth keeping: `filter_type` (GLOBAL_SHARPENING / GLOBAL_SMOOTHING / ADAPTIVE, on 51%
of flames) does **not** change the render — at render time it only drives a debug overlay gated on
`filter_indicator`, which is 0 everywhere in the corpus; its real effect is choosing a kernel at
import time when `filter_kernel` is absent, and the kernel is what we already read. `balancing_red/green/blue`
(4%) and `color_oversample` (9%) are dead attributes too: the renderer never reads them
(`balancing_*` only drives a gradient-editor button, `color_oversample` only a render-file header).
Still unported: `ai_post_denoiser` (18%, OptiX/OIDN — not reproducible), `post_bokeh` (2%),
`background_image` (1%, absolute Windows paths), `mixer_mode` (0.3%), non-BUBBLE `cam_dof_shape` (0.4%).

Two traps met on the way, both worth remembering: `mod` is a **reserved keyword** in WGSL, and a
backtick inside a WGSL comment closes the TypeScript template literal the shader lives in. Also,
adding a histogram read to the tonemap's `fsB` changed its derived bind-group layout — the cached
`bgB`/`bgBExport` groups had to gain binding 1 and be invalidated when either histogram is
reallocated, or the whole filter pass silently produces a transparent image.

### Engine semantics found by the isolate-and-diff loop (2026-08-17)

Each of these was found by making XML variants of a differing fixture
(`flameCompare({sets: [], files: […]})` + `MANIFEST=manifest.part.json … Compare`)
until one attribute explained the difference, then ported from the JWildfire source:

| JWildfire behaviour | fixture | fix |
|---|---|---|
| A **final xform does not recolour** unless `color_type="DIFFUSION"` (finals default to `ColorType.NONE`; normal xforms to DIFFUSION) | Cross_1, Orchids | importer sets colorSpeed 0 for finals without an explicit DIFFUSION/TARGETG; exporter writes `color_type="DIFFUSION"` when a final has speed; new finals get speed 0 |
| **Several final xforms** per layer, applied in sequence | Orchids_0/1, Bokeh_1 | `Layer.moreFinals` (imported, rendered as chained `applyF` steps, exported, shown read-only as "Final 2…" with swap/remove) |
| Per-xform **colour modifiers** `mod_gamma/contrast/saturation/hue` (+ speeds), carried per point and applied to the plotted RGB (`transformPlotColor`, incl. its clamping HSL converter and the 0..199.2 RenderColor scale) | Galaxies_0/1 | `XForm.colorMods`, per-point `mods` buffer (bound only when a flame uses them), `applyColorMods` in the kernel |
| A point restarts only on **NaN/∞** (never on magnitude), and only at the next `validateState()` (every 1000 iterations, so ~500 counted iterations are lost) + 20 fuse | EDisc_1 (z grew 15× per step under preserve_z, overflowed f32 and poisoned x/y through the identity 3D affines' 0·z) | no magnitude limit on x/y; z kept finite (±1e18); restart waits 21+U(0,1000) |
| **Layer weight** multiplies the plotted colour; every layer iterates equally often | Bokeh_1 | equal thread split, colour × weight (`xd[8+li]`), colour accumulation dithered so small weights/opacities stay unbiased |
| **Per-instance priority** `<var>_fx_priority`: a normal variation forced to pre runs as input ← input + w·f(input), forced to post as output ← output + w·f(output) (`EnforcedPre/PostVariationTransformationStep`) | Ghosts_1, Brokat_0 (+ Bubbles, Layers, Spirals, Painterly, Bokeh) | `VarInstance.priority`, imported/exported, emitted by codegen with borrowed t/v |
| `crackle` jitters its cells with `NoiseTools.simplexNoise3D` on the CPU but FastNoise `singleSimplex` in its own GPU code | Bokeh_1 | the table simplex from `dc_perlin` (hoisted once, shared) replaces `singleSimplex` |

### Weighting fields (2026-08-17)

JWildfire's per-transform **weighting fields** (`wfield_*` attributes) are supported: a FastNoise value
(cellular / cubic / perlin / simplex / value, plain or fractal FBM/Billow/RigidMulti, white noise) at the
affine result or the incoming position, which scales every variation amount of the transform
(`wfield_var_amount_intensity`), up to three named variation params or amounts
(`wfield_var_paramN_*`; int params rounded like `Tools.FTOI`), the colour (×0.1) and jitters the output
(×0.1, x/y/z each from a permuted lookup) — exactly `TransformationInitWeightingFieldStep`,
`Variation.transform/executeTransform`, `TransformationApplyWeightMapJitter/ToColorStep`. The noise is the
Flam4 template's own CUDA port of FastNoise (`data/fastnoise.cu`, extracted with every noise type enabled,
enums as `#define`s, plus `wfieldValue()`), transpiled by `gen.ts` into `src/gpu/wfield.wgsl.ts` and
included in the kernel only when a flame uses a field. `IMAGE_MAP` fields need an image and are ignored.
Verified against headless JWildfire on the `wfield_*` fixtures (all match to ≤1.01 / corr 1.00) —
**note: `Compare.java` must run with JWildfire's `resources` dir on the classpath**, otherwise
`variation_costs.txt` is missing and JWildfire silently drops the param modulations
(`isValidVariationForWeightingFields`). Model: `XForm.wfield` (`WeightingField`); the transform editor
has a collapsible "Weighting field" section; keyframe morphs interpolate the numeric knobs of same-type fields.

### Solid rendering — stage 1 (2026-08-18)

JWildfire's **solid rendering** (`sld_render_enabled="1"`; 29 % of the 1 615 flames in four public
collections use it) is ported from `RasterFloatIntForSolidRendering` / `NormalsCalculator` /
`LogDensityFilter.addSolidColors` / `GammaCorrectionFilter` (`src/gpu/solid.wgsl.ts`):

* **Kernel** — no density: `solidSplat` does `atomicMax` on an order-preserving key of the camera-space
  depth (`cz`, before the perspective divide; JWildfire's `prj.z`) and, when it raised it, writes the
  payload (untransformed x/y/z as f32 — JWildfire's `originX/Y/ZBuf`, world space —, palette colour ×
  layer weight × 200/256 as f16 pairs, material). OPAQUE draw mode drops a point with probability
  1−opacity, hidden points never plot, colour modifiers do not apply, and — a `FlameRenderer` constructor
  detail that cost an evening — **antialiasing is switched off for solid flames** (with it on we covered
  28 % more cells than JWildfire). Materials: `p.material` blends per transform like the colour
  (`material1/2`), starts at `random()`, and the plotted value is the pre-final one; the per-point state
  is only allocated when a transform has a non-zero material/speed (`usesMaterials`).
* **Post pass** (compute, per cell): re-derives the key from the payload (a payload can land one step
  behind a concurrent, higher key — self-heals every pass) and computes the normal from up to 8 of the
  16 `NNEIGHBOURS_COARSE` pairs (cross products of origin differences, `refreshAllNormals` order), packed
  as 3×10-bit snorm.
* **Tonemap** (`fs`): per output pixel JWildfire's kernel *in raster cells* — `FilterHolder` sizes
  `int(2·os·support·r)+1`, weights normalised to os², and `noiseFilterSizeHalve = N/2 − 1` (the kernel is
  applied off-centre by one cell exactly like `LogDensityFilter`) — or the os×os block mean without a
  filter; a cell contributes when it has a normal: `raw = obj·ambient + Σ_lights (light + obj·ambient/3)·
  f(cosa)·diffuse·I + phong·f(−r.z)^size·I` with `lightDir = aᵀ·(0,0,−1)` from `LightViewCalculator`,
  `getInterpolatedMaterial` incl. `morphMaterial`'s quirk (the morphed diffuse is the refl-map blend);
  then `alpha = coverage^(1 + 1/gamma)`, `round(solid·255) + ((255−alphaInt)·bg) >> 8`.
* Verified on 8 solid flames from the collections (AO/shadows switched off in the copies, since those are
  not ported yet) + the two authored fixtures `Solid_0/1`: 9 match at luma ratio 1.00–1.03, block MAE
  ≤ 1.8, hist ≥ 0.97, corr ≥ 0.99; the tenth uses a JWildfire **background gradient** (`background_type=
  GRADIENT_2X2_C`, not modelled). One flame's remaining 0.98 was traced to the point distribution itself
  (its density render differs the same way).
* Found on the way: the transpiler expanded `sincosf(expr, &s, &c)` into `sin(expr)`/`cos(expr)`, so an
  argument that draws a random (`julia3D`: `atan2 + 2π·(int)(rnd·n)`, `blur3D`, `circleblur`,
  `farblur`) drew *two* — the (cos, sin) pair was off the unit circle. `cwgsl.ts` now hoists the angle;
  those four re-verified, `TINA0019` moved closer to JWildfire, baseline updated.
* Not yet: the reflection map, JWildfire's post-process DOF for solid flames (`PostDOFCalculator`),
  `receiveOnlyShadows` (`plane_wf`/`obj_mesh_*`), light motion curves. Their attributes round-trip untouched.

### Solid rendering — stage 2: ambient occlusion (2026-08-18)

`AOCalculator` ported literally as two compute passes over the raster (`SOLID_AO_WGSL`): the post pass now
also writes the depth in raster units (`zr = depth · ppu`, JWildfire's `zBuf = prj.z · bws`; `ZBUF_ZMIN`
for empty cells, which therefore never occlude); `aoRawPass` walks `azimuthSamples` directions ×
`radiusSamples` radii up to `sphere_radius = aoSearchRadius · imgSize/500` (imgSize = the FULL raster's
diagonal, so hi-res tiles keep the whole image's radius), jittered (`r_jitter = step/10`, per-cell hash
instead of the shared Marsaglia stream), horizon `atan2(z − z0, r) + 0.001` against the tangent-plane
angle — including JWildfire's normalisation of the *absolute* sample position instead of the offset —,
monotone `prevH` gate, `exp(−dist²·falloff) · 0.5`. `SmoothAOBufferThread` (gaussian `FilterHolder`,
os 1, radius `aoBlurRadius · imgSize/500`, only when ≥ 0.42 cells, result × 0.1 — without smoothing the
raw sum is used unscaled) is a pure gaussian, so it runs as two 1-D passes with the exact same weights
(`gaussianFilter1D`; unit test asserts the outer product equals the N×N kernel). Shading: `ambient −=
ao·aoIntensity`, `diffuse −= ao·aoIntensity·aoAffectDiffuse` (`addSolidColors`). Live view refreshes the AO
passes every third presented frame while accumulating (a 3016×3408 raster: 28 ms per full solid present).
Verified on the same 8 collection flames with their own AO settings restored + `Solid_2`: 7 at ratio
1.00–1.03 / blkMAE ≤ 2 / corr 1.00 (AO visibly darkens both engines alike: `_solid5` 188 → 184,
`_solid8` 99 → 94), the lacy `_ao2` at 0.95 (its z-buffer holes differ by the point-distribution noise
already seen without AO). UI: Solid → Ambient occlusion knobs.

### Solid rendering — stage 3: shadow maps (2026-08-18)

`ShadowCalculator` + `LightViewCalculator.project` ported: every drawn point (camera-visible or not, like
JWildfire's `plotShadowMapPoint` before the `insideView` test) splats its light-space depth `a₂·q` into a
`shadowmapSize²` map per casting light with `atomicMax` on the ordered key (`SOLID_KERNEL_WGSL
shadowSplat`; maps + the 16 bounds words share one storage buffer to stay under the 8-storage-buffer
default — the device also asks for up to 10 for the mods+material+solid combination). The map's extent
is JWildfire's light-space bounding box of the flame's *first 40960 samples* (+3 % safety): the kernel runs
in **mode 1** after every reseed, where each walker contributes exactly ONE bounds sample (its first plotted
point after the fuse, flagged in bit 31 of its rng word — ~65k samples; taking every early sample instead
let rare far points stretch the map 3× and darken everything), then **mode 2** splats with the frozen
bounds. `accPass` = `accelerateShadows` per cell (`step(map − bias, lightZ)`, `ZBUF_ZMIN` outside the map
or on empty cells), `smoothPass` = `calcSmoothShadowIntensity` for SMOOTH (radius `FTOI(r·6·imgSize/1000)`
≤ 128, stride `r/8+1`, gaussian weights, `(1 + Σ acc·w)/Σ w` — the leading 1 is divided too, as in
JWildfire); shading takes `clamp(vis + 1 − shadowIntensity)` per casting light, `avgVisibility` over ALL
lights for the ambient term and the non-casting lights (`addSolidColors`). Hi-res tiles of one export share
the maps (light space is view independent); the live view refreshes the lookups with the AO cadence. Map
size capped at 4096² here (one collection flame asks for 9600²).
Verified: `_sh2`, `_sh3` (FAST) and the authored `Solid_3` (SMOOTH, two casting lights + AO) at ratio
0.99–1.00 / blkMAE ≤ 1.5 / corr 1.00, `_sh7` (SMOOTH) 1.00. The remaining collection flames with shadows
differ exactly as much as they already did with shadows OFF (`_sh4` 0.93 both ways, `_sh8` 0.97 → 0.91:
its point cloud sits ~0.5 light-units off JWildfire's — a `checkerboard_wf`/`truchet` geometry item, not a
shadow one — resolved 2026-08-22: JWildfire's own GPU snippet for `checkerboard_wf` multiplies the checker *sides* by a continuous random where the CPU uses `random(_max_checks + 1)`, an int, so the sides sit on checker boundaries; `overrides.ts` patches the cast back and `_sh8` is at ratio 0.96 / corr 0.99; `_sh5` uses the 9600² map). A JWildfire-side probe (`scratchpad/diff/ShProbe.java`: reflection
into `ShadowCalculator` — map fill, bounds, per-cell `lz − map` percentiles) confirmed our maps have the
same sample density and depth distribution as JWildfire's.

### Parity stretch (2026-08-18)

* **All 18 `FilterKernelType` kernels** (`src/gpu/filters.ts`: SinePow5/10/15 — `acos(4·x^p − 1)/π`, the collections'
  favourite —, plain Mitchell (b = c = ⅓), B-spline, Bell, Blackman, Box, Catmull-Rom, Hamming, Hanning, Hermite,
  Lanczos2/3, Quadratic, Triangle; `MITCHELL_SINEPOW` = JWildfire's adaptive mode, treated as Mitchell-smooth) with
  their spatial supports and the sharpening set (colour kernel + gaussian 0.75 for the intensity). The flame keeps
  JWildfire's kernel name; old `'mitchell'`/`'gaussian'` values normalise.
* **Hand-written ports** of the three portable "point-set" variations: `inversion` (InversionFunc: the seven parametric
  shapes incl. Rhodonea's period/gcd logic and `getMaxCurvePoint`, p-norms, ring modes, pass-through, shape drawing,
  the four direct-colour measures with clamp/wrap, `hide_uninverted`; preserve-z handled inside the snippet because the
  uninverted branch adds `zin` unweighted), `mobius3D_with_inverse` (quaternion Möbius map or its inverse per point,
  det²-normalised), `pre_stabilize` (Rick Sidwell's glitch repellent: on the instance's first call and with probability
  p/1000 the input jumps to one of n `java.util.Random(seed)` points, colours 0.5, 0.25, 0.75, 0.125, …). All three pass
  the oracle harness (`varTest`); the harness now includes hand entries' `funcs`/priority and stands in `P.flags` and
  `wstart_`. The kernel gained a per-walker **age** (bits 28..30 of the rng word, saturating at 7): `wstart_` is true
  for a walker's first iterations after a (re)start — JWildfire's "first call" of a variation instance, which its ~8
  long-lived walkers see once and our 65k short-lived ones must see each (`pre_stabilize` without it left a haze of
  never-reset walkers).
* **Zero-amount variations are applied by JWildfire** (`XForm.transformPoint` never checks the amount): the importer
  now keeps `name="0.0"` instances of pre/post-priority, hide, direct-colour and stateful variations (`pre_stabilize="0"`,
  `post_mirror_wf="0"`, `post_axis_symmetry_wf="0"` are common in the collections) and still drops plain sums.
  `Bokeh_1` — the worst-matching random-generator fixture — went from 0.80 / hist 0.89 to 0.91 / 0.97 because of it.
* Importer clamp lifted: `brightness` up to 1000 (JWildfire files carry 50 and 150; ours stopped at 6).
* **Background gradients** (`BGColorType` GRADIENT_2X2 / GRADIENT_2X2_C: `background_ul/ur/ll/lr/cc`): `bgAt()` in both
  tonemaps ports `LogDensityFilter.calculateBGColor` (bilinear on rounded 0..255 corners; the centre variant is four
  bilinear quadrants of size W/2−1 meeting at `cc`), spanning the full image so hi-res tiles agree; only 36 of the 864
  files carrying the attribute have non-uniform colours — an all-equal gradient imports as the single colour.
  `_solid7`/`_sh1` (the flame that was 0.74 for that reason) → 0.99, density variant 0.99.
* **Colour types CYCLIC and DISTANCE** (`XForm.colorType`; `TransformationInitStep`/`TransformationDistanceColorStep`):
  CYCLIC adds the transform's symmetry to the colour index (mod 1); DISTANCE keeps the index and plots the palette entry at
  `color + |Δposition|·(symmetry+1)` (index `·254 + 0.5`, mod 256) — a plot colour that a following DIFFUSION/CYCLIC
  gradient step replaces, unlike a direct-colour variation's (`rgbo.w` 0.5 vs 1). NONE stays "no recolouring". Verified
  on 4 DISTANCE flames (three at 1.00–1.01; `rhodonea-2` differs for its own reasons) and one CYCLIC (1.00).
* **`obj_mesh_wf`** (hand-written, shares the sampler/snippet with `obj_mesh_primitive_wf`): JWildfire's `OBJMeshWFFunc`
  samples the OBJ file named by its `obj_filename` "ressource" and — when the name is empty or the file cannot be loaded —
  `OBJMeshUtil.createDfltMesh()`, the ±1 cube. 129 of the 151 instances in the collections have an empty file name (people
  used it as a cube primitive) and the other 22 name files on the author's disk (`D:\Pictures\obj files\…`) that no
  one else has, so JWildfire itself renders the cube for them. WilderFire keeps the file's *basename* in
  `VarInstance.res.obj_filename` (`<var>_<name>` attributes are hex-encoded UTF-8; the exporter writes them back), looks it
  up in the IndexedDB mesh store (`libraryStore.ts` store `meshes`, filled from the transform editor's "mesh file" row —
  `⬆ .obj` parses v/f lines with the same reader as `mesh2bin.ts`, `parseObj` in `src/core/meshes.ts`) and falls back to
  the cube. Verified: `_obj1` (cube, no other variations) 1.00, `_obj3` 0.99, `_obj5` (JWildfire's `diamond.obj` loaded
  into the store vs JWildfire reading the file) 1.01/corr 1.00. `_obj2` (0.86) is NOT a mesh issue: the same 0.86 with
  `obj_mesh_primitive_wf`, 0.90 with `blur3D`, 0.88 as a density render and 1.01 with a flat camera — a 3D scene whose z
  is carried through a 2D variation (`waves2`, preserve_z) plus a z-noise variation on one xform and a `juliascope` xform,
  viewed at pitch/yaw ≈ 0.8/1.1; kept as `_zchain.flame` (open engine item, not understood). `_obj4` uses `parplot2d_wf`.
* **`subflame_wf`** (hand entry + codegen): JWildfire's `SubFlameWFFunc` runs a nested flame's chaos game one step per
  call (`subflameIter`: next xform from the current one's weight row, hidden/opaque draw modes skipped up to 1000 times,
  finals applied, the point scaled/rotated/offset, `z += colorscale_z·colour`; `prefuseIter`: 42 unplotted steps from a
  fresh point; the amount is ignored — `pVarTP += q`). WilderFire compiles the sub-flame (the instance's `flame`
  resource, hex XML in `subflame_wf_flame`, JWildfire's `DFLT_FLAME_XML` when absent — kept out of the model, written
  back on export) into the kernel: `parseSubFlame` (first layer, UNSET finals recolour as DIFFUSION like the Java,
  nested subflame_wf dropped), per-sub `applyS<k>_<i>`/`applySF<k>_<j>` xform functions with the sub-flame's own
  preserve_z, weight rows + xform blocks + a 256-RGB palette after the outer layers, `var<private>` walker state
  (point, colour, current xform; re-fused per dispatch), `sub<k>_step`/`subflame<k>`/`subflameAny` dispatcher; the
  hidden slot is the sub index. Colour modes: −1 off, 0 the sub colour index, 1..4 a channel of the sub palette colour
  (JWildfire's `redColor/255` on its 200/256 RenderColor scale), −2 the palette colour as a direct RGB colour; a
  direct-colour sub point passes its RGB through in every mode ≠ −1. Sequences (`flame_is_sequence`) accepted, ignored.
  The transform editor shows a "sub-flame" row (⬆ .flame / default). Verified: `_sub1` (the importer's example) 1.05 /
  corr 0.99, `_sub3only` (a collection sub-flame alone) 0.99, `_sub3` (its full flame) 1.12 / corr 0.98 after the two
  engine fixes below (0.62 / 0.31 before); `Sub_0` fixture 1.08 / corr 1.00 (baseline 95). `_sub2`'s sub-flame uses
  `wangtiles` (image), `_sub4` mesh primitives inside the sub-flame (mesh keys of sub-flames are not loaded — open).
* **Two engine fixes found through them (`_zchain`/`_obj2` 0.86–0.90 → 0.98–0.99, `_sub3`):**
  1. *Enforced-priority preserve-z*: a normal 2D variation forced to pre/post (`<var>_fx_priority="±1"`,
     `EnforcedPre/PostVariationTransformationStep`) still runs its Java `if (isPreserveZCoordinate) pVarTP.z += amount·pAffineTP.z`
     — with pAffineTP the copy of the point it rewrites, so z grows by amount·z; the codegen never added the clause for
     forced variations (466 files carry a forced post, 313 a forced pre in the collections). Also modelled:
     a pre-definition function forced to 0 has no effect (its "affine" argument is a copy) and is dropped, a
     post-definition one forced to 0 rewrites the output at its place in the normal list.
  2. *Pre/post-priority functions with the preserve-z clause* (`PREPOST_PRESERVE_Z` in codegen: post_circlecrop and the
     post_crop family, post_trig, post_c_symmetry/var, pre_recip, pre_c_symmetry/var, ringtile — the Java classes with
     `getPriority() ≠ 0` and `isPreserveZCoordinate`): they add amount·z of the affine point to the output z too;
     `pre_recip="1.0"` in `_sub3` was doubling z's growth factor per step in JWildfire and not in ours.
  Open: a synthetic `_sub3bub` (bubble instead of the subflame, flat camera) still differs wildly (JWildfire's image is
  almost empty, ours full) — not investigated.
* Verified: `inversion` flames `_pv1` 0.99, `_pv2` 1.12, `_pv3` 0.95, `_pv4nc` 0.97 (`_pv4` itself has `curliecue2`, a
  sequential-state variation); `mobius3D_with_inverse` `_pv5` 0.99, `_pv6` 1.09, `_pv7` 1.00; `pre_stabilize` flames are
  attractor scenes whose look depends on the long single trajectory (JWildfire's points diffuse for ~10 k steps
  between resets; ours live ~600) — the same inherent class as `Julians_0`.

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
