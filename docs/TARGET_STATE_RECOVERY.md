# Target state verification and guarded checkbox recovery

The prior multistep test found false repair triggers on 39 of 48 healthy clicks with the frozen visual detector. The two task families exposed distinct cases: clicking the first sequence button changes internal progress without changing pixels, and checking the wrong checkbox changes pixels. This study checks an explicit visible target state on the checkbox task. On the sequence task the new verifier returns unknown and does not authorize a retry.

The checkbox reader uses the native 498×321 Chromium screenshot and one MiniWoB rendering profile. It matches the requested label against row text using a standard font template. A minimum similarity and a gap to the runner-up are required. The reader checks the checkbox at the matched row before and after the action, and abstains on an unsupported layout, unreadable state, ambiguous label or unexpected changes. It receives screenshots, the goal, the current target and the proposed action through `TransitionView`. The teacher reads DOM state only for offline labels, fixture setup and continuation after the evaluated first step.

Recovery requires verified evidence that the target remains unchecked. If one other box was checked and the click falls in that row, the controller can undo that one change. It reads the updated screenshot before any further click. A target click is allowed once, with a final visual audit. The budget is one undo, one target click, one shared observation after waiting, and zero restarts. The actor and the frozen detector remain fixed. The recovery profile is specific to native checkboxes in this rendering and does not apply to arbitrary websites.

## Protocol

The previous 189 traces informed the design. The experiment takes the first 8 reachable unique button-sequence layouts and 14 checkbox layouts from a separate 43000-series reserve. Exact initial screenshot matches to the previous accepted layouts are excluded before model inference. The smoke uses separate seeds. Correct, blank and wrong-control branches start from a verified identical prefix, alongside a natural ShowUI first click. The inherited frozen head and full-image appearance rule are evaluated on the same observations. Their decisions are recorded without execution. The new policy's permitted repairs are executed. An evaluator validates the first-step outcome, and a scripted oracle continuation checks full task success after a correct or recovered first step. Report trigger, unknown, preserved successes, physical recovery and teacher-assisted completion separately.

The registered glyph score is at least 0.70 and the runner-up margin at least 0.18. The numbered profile config pins the font, detector artifact, prior plan, actor weights and MiniWoB revision.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_target_state_study.py \
  --config configs/target_state.json --profile test \
  --prior-plan runs/crop-control-multistep-001/multistep-plan.json \
  --model models/ShowUI-2B --frozen-detector runs/visual-verification-003-models/frozen \
  --font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \
  --out runs/target-state-001
```
