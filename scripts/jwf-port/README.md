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
| `cwgsl.ts` | CUDA-C → WGSL transpiler (tokenizer, C-subset parser, typed emitter: int/float coercions, pointers/arrays/structs, overloads, macros, `switch`, helper library). |
| `gen.ts` | Reads the dump + kernel helpers, transpiles every variation, applies `overrides.ts`, writes the two registry files and prints a summary. `--report` also writes `report.json`. |
| `overrides.ts` | Hand corrections (in CUDA dialect) for variations whose JWildfire GPU code diverges from its CPU code (`perspective`, `brick`, `chainmail`, `ouroboros`, `rays3`, `waves22`, …). |
| `data/jwf-variations.jsonl` | Metadata + GPU code for all 1026 JWildfire variations (name, params/defaults/int-ness, priority, types, GPU code, helper functions). Produced by `Dump.java`; checked in so regeneration does not need Java. |
| `data/kernel-lib.cu` | Helper library extracted from JWildfire's `Flam4_3dKernal_TemplateJWF.cu` (Complex/Mat2/Jacobi/noise/misc); transpiled on demand. |
| `Dump.java` | Dumps the catalogue by reflection against a compiled JWildfire tree. |
| `Oracle.java` | Headless JWildfire oracle: evaluates each variation's Java `transform()` on the spec grid; deterministic → exact values, random → per-point mean/std/hide-fraction over 256 samples. |
| `oracle-spec.ts` | Emits the shared test spec (130-point grid × 3 parameter sets: defaults, floats perturbed, ints perturbed) for every known variation. |
| `verified.json` | Verdicts written by the browser harness; `jwf` = ports that match JWildfire (these enter the app registry). |
| `testflames/` | JWildfire random-generator flames (`GenFlames` via JWildfire's own generators) + `yflip.flame` (orientation check). Loaded by `window.wilderfire.flameTest()`. |
| `../../src/dev/varTest.ts` | Browser harness: compiles each variation's WGSL, evaluates the spec grid on the GPU, diffs against the oracle, POSTs `verified.json` via the dev-server sink in `vite.config.ts`. |
| `../../src/dev/flameTest.ts` | Browser harness: imports every fixture flame, reports unsupported variations, compiles the kernel. |

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

## Semantics worth knowing

* **Snippet scope** (`variations.ts` header): `t` (input point, mutable), `r2 r th=atan2(x,y) ph=atan2(y,x)`, `v` (output accumulator), `rs` rng, `cp` palette-coordinate pointer, `hd` hide-flag pointer, `A(i)` affine coefficients. JWildfire's `__phi` is our `th` and `__theta` our `ph`.
* **Priority.** JWildfire pre-variations (`pre_blur`, priority −1) mutate the input point in place and post-variations (`post_curl`, +1) mutate the accumulated output; they are *not* weighted sums. Codegen runs an xform's variation list in priority order (pre → normal → post) inside one mutable stage. WilderFire's own pre/post *stages* (weighted sums) still exist alongside.
* **2D scope.** `__z` reads as 0 and `__pz` writes are dropped (flag `z`); 3D solids that depend on z fail the oracle and stay unverified. Direct RGB colour output, weighting fields, resource-backed variations (images/text) and `custom_wf` are not ported.
* **Precision.** WGSL is f32, JWildfire is f64. Hash-style variations (`sin(x)*43758.5453`) are inherently divergent and stay unverified.
* **JWildfire GPU ≠ CPU.** The oracle caught several JWildfire GPU snippets that disagree with the Java (see `overrides.ts`); overrides restore CPU semantics.
* **Duplicate instances.** JWildfire writes a second `bubble` on one xform as `bubble#1#="…"` (invalid XML); the importer's lenient pre-parser normalises this and attribute names with spaces/punctuation.
* **Camera.** JWildfire's effective pixels-per-unit is `scale × cam_zoom`; the importer folds `cam_zoom` in. flam3/JWildfire raster +y points *down*; the kernel and overlay follow that convention.
