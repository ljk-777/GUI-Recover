# Idempotent replacement for input recovery

In the previous offline Q study, dropped input and deferred input had identical screenshots at the point where a recovery decision was needed. An immediate retry decided to append based on that screenshot. When the queued input arrived before the append, the field contained two copies and Submit failed.

This change adds the opt-in `RecoveryDecisionEnv("enter-text", grounder, input_repair="replace")` profile. It uses the same original target text and grounded field location. It releases any pending fixture action first, checks that the episode remains live, then clicks the grounded point, selects existing text and types the original value. Repeating the operation at the supported field yields the same text. Reward, actor prompt, visual decisions, wait budget and retry budget are shared with the existing environment. The new operation is scoped to the single editable field task. A successful coordinate prediction remains a dependency; the implementation does not certify arbitrary page targets.

## Frozen comparison

The test takes six fresh Enter Text layouts from a registered reserve and excludes prior initial screenshots from the supplied corpora and two prior split registries. On each layout with normal, dropped and delayed delivery it executes three paired strategies:

- **Legacy Visual** and **Replace Visual** share the same visual trigger and fixed zoom grounding. The two strategies differ in the input operation performed once a retry is selected.
- **Learned Wait** uses the Q table frozen in the previous experiment and the legacy executor. Its checkpoint is checksum-locked.

Separate partial and wrong input stress branches force the same retry in each executor. They check execution once a location is known. The visual trigger infers that text is present and does not verify the text itself, so these stress results cannot be used to claim automatic detection of wrong content. All branches are actual browser executions and complete with the same actor-guided Submit and exact MiniWoB termination rule.

The predeclared gate requires all baseline normal and dropped successes to be preserved, an improvement over Legacy Visual on delayed input, no delayed loss against Learned Wait, no explicit wait on delayed replacement, and no regression in either forced-retry stress condition. The report separates number of waits, edits, logical requests and cost-adjusted return. Cost constants are unchanged from the previous experiment. The delay and release schedule come from the local fixture and do not provide a real latency model.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_idempotent_input_repair.py \
  --config configs/idempotent_input_repair.json --profile test \
  --model models/ShowUI-2B --q-policy examples/recovery-offline-rl-policy.json \
  --prior-pre-rl runs/pre-rl-recovery-001/splits.json \
  --prior-rl runs/recovery-offline-rl-001/splits.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/idempotent-input-repair-001
```
