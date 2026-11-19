# Offline examples

Open `recovered/replay.html` to inspect one successful dropped-input recovery without installing a model or browser.
`wrong-input/replay.html` is a content-correctness counterexample: visible ink does not prove that the text is correct.

The planner in these examples is explicitly scripted; ShowUI predicts coordinates. They are not autonomous planning or full-benchmark scores.
`provenance.json` records the model revision, experiment code and intervention. Screenshot hashes are in trajectory.jsonl.
Synthetic pages are rendered from MiniWoB++, MIT; the original notice is preserved in MINIWOB_LICENSE.txt. No weights or server login details are included.

`sft-smoke-results.json` records the first 64-pair, 32-update QLoRA smoke, adapter reload acceptance, unchanged 23/28 held-out grounding result, and 4/4 versus 4/4 scripted execution checks.
`sft-grounding/` contains one single-frame unseen-task failure with teacher boxes and both predictions; it is not a recovery episode. The adapter remains a separate local artifact, not part of these examples.

`sft-recovery-results.json` records the frozen follow-up: 225 pairs from 150 scenes, two 128-update adapters, six offline ablations and 72 GUI executions. Conditional corrections are selected by an offline evaluator; this is not autonomous error detection. Small conditional gains coexist with clean-task regressions, and neither adapter is promoted to the default.

- [In-place conditional repair](sft-recovery/history_history/replay.html) and [base retry](sft-recovery/base_plain/replay.html): same executed error prefix, no reset; history adapter succeeds where base retry fails.
- [Normal Submit regression](sft-history-regression/history_history/replay.html) and [base completion](sft-history-regression/base_plain/replay.html): history adapter repeats the input coordinate and fails to finish.

Both pairs are post-evaluation illustrations, not the score denominator. Their native PNGs and logs are byte-identical to the experiment evidence; provenance states the controller and selection limits. Full results include failures and separate terminal-reset retries from in-place recovery. No adapter or base weights are committed here.


`history-prompt-results.json` and `history-balanced-results.json` preserve the follow-up on fresh 8000-series scenes. Prompt ordering restores Submit, but all three balanced retraining seeds fail the original validation gate and remain experimental. The predeclared no-history fallback is reported independently of retraining.

- `history-submit/`: the fixed enter-text seed 8000 across base, legacy, target-last and the fixed primary balanced model, preserving each outcome.
- [Fallback retry](history-fallback-recovery/fallback/replay.html) and [base retry](history-fallback-recovery/base/replay.html): an actual evaluator-selected failure followed by a same-seed restart, not recovery inside the original episode or autonomous detection.
- `history-balanced-regression/`: a retained clean-task counterexample for the fixed primary balanced model.

These examples are post-evaluation illustrations, not the denominators. All PNGs and logs remain unmodified. Both reports retain strict teacher-box scores and GUI outcomes, including boundary differences; no post-hoc margin rescoring. No weights are committed.
