# WilderFire 🔥

A GPU-native fractal flame editor for the browser — a WebGPU reimagining of
[JWildfire](https://github.com/thargor6/JWildfire) / Apophysis / flam3.
Everything runs client-side; nothing ever touches a server.

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

- **69 variations** (linear, spherical, swirl, julia/julian/juliascope, pdj,
  curl, ngon, elliptic, escher, cpow, waves2, blur, …) with parametric
  controls, including **direct-color `dc_*` variations** that paint the
  palette coordinate from geometry
- **Layers** (up to 8, JWildfire-style) — each with its own transforms, final
  transform, gradient, density weight, and visibility, blended in one
  histogram; walker threads are partitioned across layers on the GPU
- **Xaos** (flam3 "chaos") — per-pair transition weight matrix between
  transforms, evaluated natively in the GPU kernel via per-source CDF rows
- **JWildfire-style triangle editor** — drag O/X/Y handles directly on the
  canvas; pan by dragging, zoom with the wheel
- Transform editor: weights, palette color / color-speed / opacity, affine +
  post-affine matrices, optional **final transform**
- **Density-estimation filtering** — density-adaptive gaussian smoothing in
  the tonemap pass (Off / Subtle / Medium / Strong)
- **Undo/redo** with slider-gesture coalescing (Ctrl/Cmd+Z, Shift+Ctrl/Cmd+Z)
- **Animation** — capture keyframes, morph between them (structure-merging
  interpolation that keeps the GPU kernel hot within each segment;
  **rotation-aware affine morphing** so spins rotate instead of collapsing),
  live looping playback with scrubbing, per-keyframe easing
  (linear/smooth/in/out), timeline save/load, and **WebM (VP9) or MP4 (H.264)
  export** rendered client-side via WebCodecs
- **Pre/post variation stages** per transform — weighted variation lists
  evaluated before and after the main sum (include `linear 1` in a stage for a
  pass-through); `pre_*`/`post_*` variation names in imported `.flame` files
  map onto these stages automatically
- **2× oversampling** (supersampled histogram, box-downsampled in the tonemap)
  for anti-aliased edges, plus DE filter strengths up to radius 7
- **Editor niceties** — reorder transforms (xaos-aware) and layers, copy/paste
  transforms across layers and flames, rotational/mirror **symmetry
  generators**, arrow-key nudging
- **Session autosave + flame library** — the working flame and animation
  timeline persist across reloads; a thumbnail library (localStorage) stores
  up to 48 flames
- **Mutation grid** (MutaGen-style) — 3×3 explorer of random mutations
  rendered offscreen; click to adopt and keep exploring
- Progressive refinement with quality cap, speed presets, pause/re-render
- Gradient presets, IQ-cosine **random palettes**, hue rotation, a
  **draggable stop editor**, and **.ugr / .map gradient import**
- **26 presets**: 10 authored WilderFire presets (showcasing xaos, layers,
  pre-stages, direct color, symmetry) plus 16 sample flames bundled from the
  [JWildfire repository](https://github.com/thargor6/JWildfire)
  (`resources/flames/`, © Andreas Maschke, LGPL 2.1+ — loaded through the
  `.flame` importer with JWildfire's 3D variation names alias-mapped to their
  2D equivalents), and a curated **randomizer**
- PNG export, **hi-res tiled export** (2-4× screen resolution, optional
  transparent background), flame **JSON save/load**, and **.flame XML
  import/export** compatible with flam3 / Apophysis / JWildfire (coefs, chaos,
  color_speed / symmetry, all three palette encodings, JWildfire `<layer>`
  blocks; unsupported variations are skipped)
- **Dark & light themes**
- **AI assistant** via [OpenRouter.ai](https://openrouter.ai) — bring your own
  key, pick any model; the model sees the current flame JSON **and a
  screenshot of the current render** (vision models can look at the result and
  iterate — including an optional **auto-refine loop** that re-captures the
  render and self-critiques for up to 3 extra rounds), editing the flame live
  via a ```flame fenced block. The key is
  stored in `localStorage` and requests go straight from your browser to
  OpenRouter.

## Run it

```bash
npm install
npm run dev      # dev server
npm run build    # production build in dist/
```

Requires a browser with WebGPU (Chrome/Edge 113+, Safari 18+, Firefox behind
`dom.webgpu.enabled`).

## Architecture

```
src/
  core/       flame model, variation registry (WGSL snippets), palettes,
              presets, randomizer
  gpu/        codegen.ts  — flame → WGSL compute kernel + data layout
              renderer.ts — WebGPU pipelines, atomic histogram, tonemap
  ui/         panels (transforms, render, gradient, AI), triangle overlay
  ai/         OpenRouter streaming chat client
```

The flame → shader compiler emits a fixed data layout (CDF of xform weights,
then per-xform blocks: affine, post, color, then variation weights/params), so
numeric edits are a single `writeBuffer` and only *structural* edits (adding
variations/transforms) trigger a pipeline rebuild.
