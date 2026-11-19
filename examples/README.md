# Offline examples

Open `recovered/replay.html` to inspect one successful dropped-input recovery without installing a model or browser.
`wrong-input/replay.html` is a content-correctness counterexample: visible ink does not prove that the text is correct.

`dark-recovered/replay.html` shows opt-in v3 recovering a missed input in a modified dark-theme fixture (seed 61).
`dark-grounding-failure/replay.html` retains seed 60: the first proposed input clicks Submit and terminates the task before the guard can help.
`v3-evaluation.json` contains aggregate results for all 50 controlled episodes, including failures; the two dark traces are illustrative samples.

`precheck-blocked-submit/replay.html` shows experimental precheck v2 stopping before an incorrect Submit click (known seed 60); the task still fails.
`precheck-false-rejection/replay.html` retains a new-seed false rejection (seed 83), where the real input can accept a border click but the inferred solid fill excludes it.
`precheck-v1-evaluation.json` and `precheck-v2-evaluation.json` retain all 44 and 52 controlled episodes' aggregate results. Normal-task degradation remains, so this module is disabled by default and not a recommended recovery strategy.

The planner in these examples is explicitly scripted; ShowUI predicts coordinates. They are not autonomous planning or full-benchmark scores.
`provenance.json` records the model revision, experiment code and intervention. Screenshot hashes are in trajectory.jsonl.
Synthetic pages are rendered from MiniWoB++, MIT; the original notice is preserved in MINIWOB_LICENSE.txt. No weights or server login details are included.
