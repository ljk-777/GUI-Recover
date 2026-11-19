# Pre-RL recovery environment and frozen policy trial

This experiment formulates a small learning problem around an already proposed GUI action. A frozen ShowUI controller supplies click locations, a text input value comes from the task instruction, and a fixed continuation locates Submit for text tasks. A recovery policy chooses how to react from screenshots and its own action. The trial supports click-button, click-dialog, click-tab and enter-text in the deterministic MiniWoB fixture used in earlier studies.

## Interface and reward

Use `RecoveryDecisionEnv(task, grounder)` from `gui_recover.recovery_env`. It exposes Gymnasium reset and step, fixed 321×498×3 RGB observations, task and target strings, the proposed coordinate action, phase, remaining decisions, and a discrete action mask.

| Action | Available phase | Effect |
|---|---|---|
| Continue | Source or review | Execute the frozen source in source phase, or accept the current outcome and run any fixed continuation in review |
| Wait | Review, once | Wait and capture one hover observation at the proposal point |
| Retry plain | Review | Predict on the full current image, execute one repair and run any continuation |
| Retry fixed zoom | Review | Predict from the fixed top-left 60% crop with feedback, map coordinates back, execute one repair and run any continuation |

The source phase has a single valid choice. An episode has at most three decisions, four browser operations and three logical actor requests. The source or a repair can end the task before continuation. A masked decision ends the episode with zero reward and no browser action. An actor parse failure ends its branch with zero reward. The benchmark aborts on browser infrastructure errors and keeps the failed trace.

Reward is sparse. Only a successful native browser termination with `DONE_GLOBAL=true`, binary reward 1, exact raw MiniWoB reward 1, and no error earns +1. The environment returns 0 for partial positive raw rewards, unfinished tasks, failed submissions, timeouts and decision exhaustion. Negative rewards and truth fields stay in evaluation evidence. Truncation and termination remain separate flags. For this finite-budget objective, the terminal or truncated transition closes the trajectory with its recorded return. A policy only receives the observation, and the next rollout starts with reset.

## Frozen comparison and data separation

Train, validation and test have registered nonoverlapping reserves and exact initial screenshot checks. Drop-input and clean versions of a layout stay in one split. The layout preflight runs without actor predictions or outcomes and excludes previously registered sources from the listed archives. The formal test has four layouts per task, plus matched drop-input on the four text layouts. Train and validation trace two fixed policies to provide starter trajectories. The frozen test compares Continue, Plain, Fixed zoom, and an appearance-triggered Fixed zoom policy. It uses one shared frozen inference cache, with logical calls separated from actual generations.

The predeclared candidate for the effect check is appearance-triggered Fixed zoom. To pass, it must add at least one success on clean scenes and one on injected scenes compared with Continue, while preserving Continue successes in each group. Natural source errors and injected text errors are different evaluation cohorts. Results describe a small fixed-controller trial for learning recovery choices.

`iter_learning_transitions(path, split="train")` returns observation, discrete action, reward and next observation with the two end flags. It checks the split index, trace checksums and screenshot checksum/shape. Validation is explicit and the reader rejects test. Policy observations are separate from raw environment evidence. The small dataset checks the plumbing and serves as an initial replay set.

## Reproduce

Use the registered browser and cached ShowUI weights. Engineering smoke seeds are separate and do not enter the trial.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pre_rl_recovery.py \
  --config configs/pre_rl_recovery.json --profile test --model models/ShowUI-2B \
  --reference runs/visual-verification-001-data \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/pre-rl-recovery-001
```
