# Contributing

Thanks for your interest in WilderFire. Please read this before opening
anything.

## Pull requests

Pull requests are welcome. Keep each one focused on a single bug or feature,
describe what it changes and how you checked it (a `.flame` that shows the
difference before and after is ideal), and expect a review that may ask for
changes. For a larger idea, open an issue first so the direction can be agreed
before you spend time on it.

If you would rather take WilderFire in your own direction, **fork it**. The
licence (LGPL-2.1-or-later, see [LICENSE](LICENSE)) lets you modify and
redistribute your fork freely as long as you keep the licence and notices
intact. Please rename a fork so it is not confused with this project.

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

There is no contributor licence agreement. By opening a pull request, or by
posting flames, snippets or ideas in an issue, you agree that they are offered
under the project licence (LGPL-2.1-or-later) so they can be used.
