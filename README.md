# WilderFire 🔥

A GPU-native fractal flame editor for the browser — a WebGPU reimagining of
[JWildfire](https://github.com/thargor6/JWildfire) / Apophysis / flam3.
Everything runs client-side; nothing ever touches a server.
Live at [wilderfire.fly.dev](https://wilderfire.fly.dev).

Free software under the **LGPL-2.1-or-later** · © 2026 Andre Paquette ·
**pull requests are not accepted** (issues and forks are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md)).

![WilderFire — a julian/spherical flame in the editor, with the transform panel, triangle overlay and AI assistant](docs/hero.jpg)

## Why WebGPU (and no wasm)

The fractal flame algorithm is a chaos game: billions of tiny independent
iterations accumulating into a histogram. That workload is embarrassingly
parallel, so WilderFire compiles **each flame into its own WGSL compute
shader** (variations inlined, parameters in a storage buffer so slider tweaks
never recompile) and runs 65k persistent chaos-game walkers per frame with
atomic histogram accumulation, followed by a flam3-style log-density /
gamma / vibrancy tonemap pass. On a typical discrete GPU this reaches
**hundreds of millions to billions of iterations per second** — orders of
magnitude beyond what a wasm/CPU port could do, which is why there is no wasm
in the stack.

## Features

- **748 variations** — 70 hand-written flam3 classics plus **744 JWildfire
  variations ported mechanically** from JWildfire's own GPU snippets by a
  CUDA→WGSL transpiler and **numerically verified in 3D against headless
  JWildfire** (see [`scripts/jwf-port/README.md`](scripts/jwf-port/README.md)).
  Parametric controls for every one, JWildfire-style **pre / normal / post
  priority**, and **direct-colour `dc_*` variations** — both the flam3 kind
  that steer the palette coordinate and JWildfire's shader-art kind
  (`dc_hexagons`, `dc_menger`, `dc_voronoise`, …) that paint RGB directly
- **Variation picker** — searchable popover with type chips (2D, 3D, blur,
  DC, pre, post, crop, …) and pinned classics, replacing a 700-entry dropdown
- **3D** — points carry z, every variation stage can read/write depth,
  per-transform **yz / zx affine planes** (+ post), a JWildfire-compatible
  camera (pitch / yaw / bank, perspective, position, `preserve_z`),
  **depth of field** (focus point or focus plane, area, fade) and
  **dimish-z** depth fade; all of it round-trips through `.flame`
- **Layers** (up to 8) — each with its own transforms, final transform,
  gradient, density weight, and visibility, blended in one histogram; walker
  threads are partitioned across layers on the GPU
- **Xaos** (flam3 "chaos") — per-pair transition weight matrix between
  transforms, evaluated natively in the GPU kernel via per-source CDF rows
- **Triangle editor** — drag O/X/Y handles directly on the canvas; pan by
  dragging, zoom with the wheel (raster +y down, like flam3 / Apophysis /
  JWildfire, so imported flames are not mirrored)
- Transform editor: weights, palette color / color-speed / opacity, affine +
  post-affine matrices, optional **final transform**
- **JWildfire-exact tonemap** — the same log-density curve (contrast,
  brightness, white level, low-density glow, gamma threshold, vibrancy),
  the same **density estimation** (`de_radius` / `de_curve` similar-density
  gather; the live preview caps its radius, exports use it in full),
  **spatial filter** (Mitchell / Gaussian) and antialias jitter, verified
  pixel-for-pixel against headless JWildfire on synthetic flames; two-pass
  tonemap so filtering is free; plus **2× oversampling**. Presets and the
  randomizer use JWildfire's brightness 4 / gamma 4 baseline
- **Undo/redo** with slider-gesture coalescing (Ctrl/Cmd+Z, Shift+Ctrl/Cmd+Z)
- **Animation** — capture keyframes, morph between them (structure-merging
  interpolation that keeps the GPU kernel hot within each segment;
  **rotation-aware affine morphing** so spins rotate instead of collapsing),
  live looping playback with scrubbing, per-keyframe easing
  (linear/smooth/in/out), timeline save/load, and **WebM (VP9) or MP4 (H.264)
  export** rendered client-side via WebCodecs
- **Motion curves** — animate any numeric parameter (camera, DOF, tone,
  transform weight/color/affine, any variation weight or parameter) with
  (time, value) keys and Catmull-Rom / linear / smooth / step interpolation,
  layered on top of the keyframe morphs; **drag points on the graph**
  (double-click adds, Alt-click removes), editable point table, persisted
  with the timeline **and written into `.flame` files in JWildfire's
  `*Curve_*` format** (JWildfire animates them; its curves import back)
- **Pre/post variation stages** per transform — weighted variation lists
  evaluated before and after the main sum (include `linear 1` in a stage for a
  pass-through); `pre_*`/`post_*` variation names in imported `.flame` files
  map onto these stages automatically
- **Editor niceties** — reorder transforms (xaos-aware) and layers, copy/paste
  transforms across layers and flames, rotational/mirror **symmetry
  generators**, arrow-key nudging, **collapsible side panes** (`[` / `]`)
- **Session autosave + flame library** — the working flame and animation
  timeline persist across reloads; a thumbnail library (localStorage) stores
  up to 48 flames
- **Mutation grid** (MutaGen-style) — 3×3 explorer of random mutations
  rendered offscreen; click to adopt and keep exploring
- Progressive refinement with quality cap, speed presets, pause/re-render,
  and a **preview hold** (Engine → Preview hold): after each edit the last
  image stays on screen until the new one has accumulated N samples per
  pixel (a bounded burst of extra passes gets it there within a frame), so
  dragging a triangle never shows the sparse first frames
- Gradient presets, IQ-cosine **random palettes**, hue rotation, a
  **draggable stop editor**, and **.ugr / .map gradient import**
- **26 presets**: 10 authored WilderFire presets (showcasing xaos, layers,
  pre-stages, direct color, symmetry) plus 16 sample flames bundled from the
  [JWildfire repository](https://github.com/thargor6/JWildfire)
  (`resources/flames/`, © Andreas Maschke, LGPL 2.1+, loaded through the
  `.flame` importer), and a curated **randomizer**
- PNG export, **hi-res tiled export** (2-4× screen resolution, optional
  transparent background), flame **JSON save/load**, and **.flame XML
  import/export** compatible with flam3 / Apophysis / JWildfire (coefs, chaos,
  color_speed / symmetry, all three palette encodings, `<layer>` blocks,
  3D camera, DOF, dimish-z, tonemap/filter/DE settings, motion curves,
  `cam_zoom` folded into zoom; unsupported variations are skipped and
  reported). Exports open a real **Save as… dialog** where the browser
  supports it (Chrome/Edge), falling back to a download elsewhere
- **Dark & light themes**
- **AI assistant** via [OpenRouter.ai](https://openrouter.ai) — bring your own
  key and pick any model from a **live, searchable model picker** (provider
  chips, context length, price, vision support, custom IDs); the model sees
  the current flame JSON **and a screenshot of the current render** (vision
  models can look at the result and iterate — including an optional
  **auto-refine loop** that re-captures the render and self-critiques for up
  to 3 extra rounds), editing the flame live. **Context controls** let you
  trim what is sent (flame as compact summary / full JSON / nothing, palette
  as 8 stops / full, variation catalogue in-use / all / none, screenshot,
  conversation memory) and what comes back (**edit commands** — a few lines
  like `set T2.weight 0.8`, `addvar T1 julian 0.7 power=3` — or complete
  flame JSON, or text only), with a live per-turn token estimate; the
  defaults are ~1.6k tokens/turn instead of ~19k. The key is stored in
  `localStorage` and requests go straight from your browser to OpenRouter.

## Keyboard shortcuts

| key | action |
|---|---|
| `Ctrl/Cmd+Z`, `Shift+Ctrl/Cmd+Z` | undo / redo |
| arrow keys (`Shift` = coarse) | nudge the selected transform |
| `[` / `]` | collapse / expand the left / right pane |

## Run it

```bash
npm install
npm run dev      # dev server
npm test         # vitest: motion curves, flame model, .flame round-trips, AI edits, tonemap constants
npm run build    # typecheck + tests + production build in dist/
```

Requires a browser with WebGPU (Chrome/Edge 113+, Safari 18+, Firefox behind
`dom.webgpu.enabled`).

Deploy: the site is static (`dist/`); `fly.toml` + `Dockerfile` serve it via
nginx on Fly.io with `fly deploy`.

## Architecture

```
src/
  core/       flame model (flame.ts), variation registry (variations.ts:
              hand-written WGSL + generated variations.jwf.ts ports),
              palettes, presets, randomizer, .flame XML I/O (flameXML.ts),
              keyframe morphing (animate.ts), motion curves (motion.ts)
  gpu/        codegen.ts  — flame → WGSL compute kernel + data layout
              renderer.ts — WebGPU pipelines, atomic histogram, tonemap, camera
  ui/         panels (transforms, render, gradient, anim, AI), triangle
              overlay, variation + model pickers, library, mutation grid
  ai/         OpenRouter streaming chat client + model catalogue
  dev/        browser harnesses (variation oracle diff, fixture-flame compile)
tests/        vitest unit tests (run in CI on every push)
scripts/
  jwf-port/   CUDA→WGSL transpiler, JWildfire oracle, generator (README inside)
```

The flame → shader compiler emits a fixed data layout (CDF of xform weights,
then per-xform blocks: affine, post, color, then variation weights/params), so
numeric edits are a single `writeBuffer` and only *structural* edits (adding
variations/transforms) trigger a pipeline rebuild. Motion curves and keyframe
morphs only touch numbers, so playback never recompiles within a segment.

## Licence and copyright

Copyright © 2026 Andre Paquette. WilderFire is free software, released under
the [GNU Lesser General Public License v2.1 or later](LICENSE). Large parts
of it (the ported variations, the GPU helper library, the sample and fixture
flames) are derived from [JWildfire](https://github.com/thargor6/JWildfire)
(© Andreas Maschke and contributors, LGPL-2.1-or-later), which is why the
whole project carries that licence rather than a permissive one. The full
list of third-party material is in [NOTICE.md](NOTICE.md).

## Contributing

**Pull requests are not accepted** and are closed automatically. Bug reports,
fidelity gaps (a `.flame` that renders differently than in flam3 / Apophysis /
JWildfire) and feature ideas are welcome as issues; forks are welcome under
the licence. Details in [CONTRIBUTING.md](CONTRIBUTING.md).
