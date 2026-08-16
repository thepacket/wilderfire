# Contributing

Thanks for your interest in WilderFire. Please read this before opening
anything.

## Pull requests are not accepted

WilderFire is developed by a single maintainer and **does not accept pull
requests**. Any pull request opened against this repository will be closed
automatically. This is not a judgement on your work — it keeps the project's
direction, licensing provenance and review load manageable.

If you want to change WilderFire, **fork it**. The licence (LGPL-2.1-or-later,
see [LICENSE](LICENSE)) lets you modify and redistribute your fork freely as
long as you keep the licence and notices intact. Please rename your fork so it
is not confused with this project.

## Bug reports and ideas

Issues are welcome for:

- **Bugs** — include the browser + version, GPU/OS, what you did, what you
  expected, and, if possible, the flame (`JSON` or `.flame` export) that
  triggers it. Console errors from the dev tools help a lot; WebGPU shader
  compile errors are printed there in full.
- **Fidelity gaps** — a `.flame` that renders differently in WilderFire than
  in flam3 / Apophysis / JWildfire, ideally with both images.
- **Feature ideas** — one idea per issue, with the use case.

Issues may be closed without action; the maintainer decides what gets built.

## Security

There is no server: WilderFire runs entirely in your browser and only talks
to OpenRouter (with a key you supply) when you use the AI panel. If you find
a problem that leaks that key or otherwise exposes user data, open an issue
marked **security** rather than emailing.

## Licence of contributions

Because pull requests are not accepted there is no contributor licence
agreement. Anything you post in an issue (flames, snippets, ideas) is assumed
to be offered under the project licence so it can be used to fix the problem.
