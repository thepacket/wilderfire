# Copyright and third-party notices

WilderFire — a WebGPU fractal flame editor for the browser.

Copyright © 2026 Andre Paquette.

WilderFire is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 2.1 of the License, or (at your option)
any later version. See [LICENSE](LICENSE). It is distributed WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

## Why LGPL

Large parts of WilderFire are derived from [JWildfire](https://github.com/thargor6/JWildfire)
(© Andreas Maschke and contributors, LGPL-2.1-or-later). Because those parts
are derivative works, the project as a whole is released under the same
licence rather than a permissive one.

## Third-party material

| what | where | origin | licence |
|---|---|---|---|
| 668 ported variations (WGSL transpiled from JWildfire's CUDA GPU snippets) | `src/core/variations.jwf.ts`, `src/core/variations.jwf.unverified.ts` | JWildfire `org.jwildfire.create.tina.variation.*` | LGPL-2.1-or-later |
| Variation catalogue dump (metadata + GPU code) | `scripts/jwf-port/data/jwf-variations.jsonl` | JWildfire | LGPL-2.1-or-later |
| GPU helper library | `scripts/jwf-port/data/kernel-lib.cu` | JWildfire `Flam4_3dKernal_TemplateJWF.cu` | LGPL-2.1-or-later |
| Hand corrections that restore JWildfire CPU semantics | `scripts/jwf-port/overrides.ts` | derived from JWildfire Java sources | LGPL-2.1-or-later |
| 16 bundled sample flames | `public/flames/*.flame`, `src/core/samples.ts` | JWildfire `resources/flames/` | LGPL-2.1-or-later |
| Test fixture flames | `scripts/jwf-port/testflames/*.flame` | generated with JWildfire's random-flame generators | LGPL-2.1-or-later |
| `Dump.java`, `Oracle.java` | `scripts/jwf-port/` | link against JWildfire classes at build time (JWildfire itself is not redistributed here) | LGPL-2.1-or-later |
| The 70 hand-written "classic" variations | `src/core/variations.ts` | written from the published flam3 / Apophysis formulas (Draves & Reckase, *The Fractal Flame Algorithm*) | original code, LGPL-2.1-or-later as part of WilderFire |
| Tonemap (log-density, gamma threshold, vibrancy) and density-estimation filter | `src/gpu/` | reimplemented from the flam3 paper / algorithm description; no flam3 code copied | original code |
| `mp4-muxer`, `webm-muxer` | npm dependencies (bundled at build) | Vanilagy | MIT |
| `vite`, `typescript`, `@webgpu/types` | dev dependencies (not shipped) | — | MIT / Apache-2.0 / BSD-3-Clause |

The `.flame` file format is the interchange format of flam3 / Apophysis /
JWildfire; WilderFire reads and writes it but contains no code from those
programs' parsers.

The AI assistant talks to [OpenRouter](https://openrouter.ai) with a key the
user supplies; no model, key, or prompt data is bundled.

## Trademarks

"JWildfire", "Apophysis" and "flam3" are the names of independent projects and
are used here only to describe compatibility. WilderFire is not affiliated
with or endorsed by them.
