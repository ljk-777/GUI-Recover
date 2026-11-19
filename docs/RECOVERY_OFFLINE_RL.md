# Offline Q-learning for bounded input recovery

This is the first learning experiment on the recovery decision interface. It fits a small Q table on CPU from actual MiniWoB trajectories. Frozen ShowUI supplies the source click, grounded retry locations and Submit. The policy learns the decision to continue, wait or retry. The learning target and the deployed policy exclude DOM, source labels, seed and the type of intervention. Its state uses the previous visual text signal and whether it has waited.

## When waiting can help

The new controlled delay fixture holds the first INPUT in a queue. A wait releases the input. Submitting early uses the visible field as it is, so it can submit empty text. Retrying before a wait may release the queued input and then append another copy. This makes incomplete delivery and dropped input look the same initially, while their next observations can differ. It simulates delivery in a local browser fixture.

The environment continues to return +1 for exact successful browser termination and zero otherwise. The learning target subtracts 0.02 per wait and 0.05 per retry. This cost is separate from the native task reward. The horizon allows at most one wait and one retry. Fitting first averages terminal outcomes at the post-wait stage, then uses those values in one-step Bellman targets for the initial decisions. It collapses identical repeated wait prefixes within each layout and condition so the three later actions do not triple the importance of waiting. Unseen states use the prior visual rule.

## Protocol

Each registered input layout is tested with normal, dropped and deferred delivery. The training set has three fresh layouts. The validation set has one additional layout and the held-out test has four more. Before inference, a reset-only preflight rejects exact screenshot matches to registered prior data. No cases are selected from test outcomes.

Each train layout/condition executes six branches with the same source proposal: continue, plain retry, fixed zoom, and wait followed by each of those actions. The Q checkpoint and training manifest hashes are saved before validation or test execution. Validation records the learned and wait-first rule outcomes; no fitting step follows validation. The test compares Continue, the previous Visual Fixed Zoom rule, the predeclared Wait First rule, and the frozen learned Q. The Wait First rule uses the same visual signal as the learner and waits once when the text signal is uncertain or missing.

The effect gate requires all clean and dropped case successes from the original Visual Fixed Zoom to be preserved, and at least one additional successful deferred case without a paired deferred regression. The wait-first baseline remains a separate comparison even if the gate passes. This experiment tests only the fixed-controller enter-text task.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_recovery_offline_rl.py \
  --config configs/recovery_offline_rl.json --profile test --model models/ShowUI-2B \
  --previous-splits runs/pre-rl-recovery-001/splits.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/recovery-offline-rl-001
```
