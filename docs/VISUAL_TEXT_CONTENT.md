# Visual text comparison for automatic input recovery

The previous executor could replace a field with the original target text after recovery was selected. Its trigger detected text presence and missed nonempty mistakes. This follow-up adds a rendered-text comparator for the native MiniWoB Enter Text profile.

An independent local Chromium process renders each requested string into four fixed reference controls. Those positions cover the native pixel phases at a 1.5× scale. Reference generation reads the visible goals recorded during reset-only preflight and runs before any action or outcome. The reference process contains only its own controls. It never opens the task. The bank records the browser version and is checksummed before and after evaluation.

The verifier receives the before/after screenshots, requested input and prior proposed coordinates. It finds the same initially blank white field as the original visual guard, then compares the cropped visible ink with the requested glyph mask. It allows small translations and leaves character widths unchanged. A best overlap of at least 0.995 is a visual match; overlap at or below 0.965 indicates different visible text. The gap, a clipped crop, an unsupported geometry and a missing reference cause abstention. It uses the profile for short ASCII letter strings on the tested Chromium viewport.

Visible mismatch triggers one grounded replacement of the original goal text. An empty field can wait once to let deferred text arrive; the second empty observation triggers replacement. An uncertain observation can wait once and then proceeds without a forced edit. A screenshot cannot in general establish exact string identity. The formal experiment explicitly includes an invisible trailing space.

## Frozen protocol

Six fresh layouts are selected by reset and screenshot comparison, excluding registered prior layouts. Each runs in seven conditions: normal, drop, deferred, partial, appended exclamation mark, first-letter case change, and appended trailing space. Both policies execute the same frozen ShowUI actor, idempotent replacement, waiting budget and exact reward. The presence-only baseline waits once when it sees no text and leaves nonempty fields alone. The new policy compares the screenshot with its prepared templates. Native outcomes are saved for evaluation, and the actor gets the same screenshot and target query as the baseline.

The effect gate requires preserving paired normal, missing and delayed successes, zero replacements on normal inputs, and a positive paired gain in each of the three visible-error groups. The trailing-space group is a registered negative visibility control. Both paths operate on the same task repeated per layout. The experiments use exact terminal MiniWoB rewards, including failure when a trailing space changes the required value.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_visual_text_content.py \
  --config configs/visual_text_content.json --profile test --model models/ShowUI-2B \
  --prior-pre-rl runs/pre-rl-recovery-001/splits.json \
  --prior-rl runs/recovery-offline-rl-001/splits.json \
  --prior-idempotent runs/idempotent-input-repair-001/preflight.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/visual-text-content-001
```
