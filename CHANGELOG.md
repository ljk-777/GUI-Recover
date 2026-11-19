# Changelog

## Unreleased

- Added experimental, disabled-by-default pre-action input target checks with one bounded reground, explicit decision/model/action accounting, offline audits, and paired precheck comparisons. Both versions' held-out false rejections are retained; this is not a recommended recovery strategy.
- Added opt-in v3 input appearance guard based on local background contrast, retaining v1/v2 behavior and v2 as default.
- Added theme/focus/background-change regressions, real-browser layout tests, and a separate frozen ShowUI comparison matrix.

## 0.2.0a1 — 2026-11-19 (alpha prerelease)

- Added explicit ShowUI grounding and a separately labelled scripted MiniWoB planner.
- Added click/input fault injection, visual input appearance checks, bounded retries and offline replay audits.
- Added versioned v1/v2 input signals; v2 ignores isolated single-pixel speckles while preserving v1 replay semantics.
- Added partial/wrong input counterexamples and shifted/dark visual robustness fixtures, marked as modified MiniWoB environments.
- Added strict/mixed action parsing, episode evidence, paired comparison validation and explicit model-call accounting.
- Added a CPU diagnostic demo, offline examples, pinned upstream assets and a shared-model study runner.

Model weights, autonomous planning, content-correctness verification, SFT and RL are outside this release. Evaluation details and limits are in docs/RELEASE_VALIDATION.md.
