# Changelog

## Unreleased

- Added an isolated QLoRA grounding smoke profile, offline DOM teacher collection, assistant-only supervision checks, adapter reload acceptance, and independent baseline/adapter evaluation.

## 0.2.0a1 — 2026-11-19 (alpha prerelease)

- Added explicit ShowUI grounding and a separately labelled scripted MiniWoB planner.
- Added click/input fault injection, visual input appearance checks, bounded retries and offline replay audits.
- Added versioned v1/v2 input signals; v2 ignores isolated single-pixel speckles while preserving v1 replay semantics.
- Added partial/wrong input counterexamples and shifted/dark visual robustness fixtures, marked as modified MiniWoB environments.
- Added strict/mixed action parsing, episode evidence, paired comparison validation and explicit model-call accounting.
- Added a CPU diagnostic demo, offline examples, pinned upstream assets and a shared-model study runner.

Model weights, autonomous planning, content-correctness verification, SFT and RL are outside this release. Evaluation details and limits are in docs/RELEASE_VALIDATION.md.
