# Changelog

## 0.2.0rc3 — release candidate

- Added an installed CPU-only tail-selection demo covering single-line and multiline fields, light and dark themes, and negative controls.
- Batched both probe keystrokes and the intermediate native screenshot in one BrowserGym step. Version 2 audits accept either execution path and still verify the original physical keys.
- Added native case-transformation and alphanumeric task variants to the active recovery demo, with reset-only layout deduplication across tasks.

## 0.2.0rc2 — release candidate

- Added a reversible Enter Text tail probe, native selection evidence and CPU-only version 2 trajectory audit. The installed active demo compares glyph checking, unconditional replacement and the probe on paired tasks.
- Added real-browser online recovery studies with frozen ShowUI. The probe-timing study checks the tradeoff between saving probes and preserving exact submission under delayed trailing spaces.

## 0.2.0rc1 — release candidate

- Added installed commands for paired visual text recovery and CPU-only evidence audit. Recovered 18/18 injected visible errors in six MiniWoB layouts and measured the invisible trailing-space failure.
- Added bounded Gymnasium recovery rollouts and an audited CPU offline Q-learning pilot for wait and retry. Recorded the comparison with hand-coded wait-first and the exact reward boundary.

- Added frozen-actor feedback/zoom repair controls, validation guardrails, fresh natural-error evaluation and an action/phase-only detector ablation, retaining native/derived input provenance and all failures.

- Added visual action-outcome verification with reject/wait, train-only frozen-feature classifiers, injection-to-natural-error evaluation, six bounded recovery controls, deterministic screenshot fixtures and full trace/cost auditing. No actor finetuning or default-model promotion.

- Added validation-only history-prompt selection, fresh-scene execution, explicit context-paired Submit balancing and three fixed SFT seed replicates. Target ordering repairs a Submit regression, while balanced retraining fails the preservation gate; all negative outcomes and native paired examples are retained without changing defaults.

- Added an isolated QLoRA grounding smoke profile, offline DOM teacher collection, assistant-only supervision checks, adapter reload acceptance, and independent baseline/adapter evaluation.
- Added a frozen distractor/actually-executed-error dataset, two same-update SFT adapters, six weight/context ablations and paired GUI execution. Conditional repairs improved on a small subset while clean-task performance regressed; adapters remain experimental and are not promoted to defaults. Results and native success/failure examples are preserved in docs/SFT_RECOVERY.md and examples/.

## 0.2.0a1 — 2026-11-19 (alpha prerelease)

- Added explicit ShowUI grounding and a separately labelled scripted MiniWoB planner.
- Added click/input fault injection, visual input appearance checks, bounded retries and offline replay audits.
- Added versioned v1/v2 input signals; v2 ignores isolated single-pixel speckles while preserving v1 replay semantics.
- Added partial/wrong input counterexamples and shifted/dark visual robustness fixtures, marked as modified MiniWoB environments.
- Added strict/mixed action parsing, episode evidence, paired comparison validation and explicit model-call accounting.
- Added a CPU diagnostic demo, offline examples, pinned upstream assets and a shared-model study runner.

Model weights, autonomous planning, content-correctness verification, SFT and RL are outside this release. Evaluation details and limits are in docs/RELEASE_VALIDATION.md.
