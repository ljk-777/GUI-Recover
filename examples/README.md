# Offline examples

Open `recovered/replay.html` to inspect one successful dropped-input recovery without installing a model or browser.
`wrong-input/replay.html` is a content-correctness counterexample: visible ink does not prove that the text is correct.

The planner in these examples is explicitly scripted; ShowUI predicts coordinates. They are not autonomous planning or full-benchmark scores.
`provenance.json` records the model revision, experiment code and intervention. Screenshot hashes are in trajectory.jsonl.
Synthetic pages are rendered from MiniWoB++, MIT; the original notice is preserved in MINIWOB_LICENSE.txt. No weights or server login details are included.

`sft-smoke-results.json` records the first 64-pair, 32-update QLoRA smoke, adapter reload acceptance, unchanged 23/28 held-out grounding result, and 4/4 versus 4/4 scripted execution checks.
`sft-grounding/` contains one single-frame unseen-task failure with teacher boxes and both predictions; it is not a recovery episode. The adapter remains a separate local artifact, not part of these examples.
