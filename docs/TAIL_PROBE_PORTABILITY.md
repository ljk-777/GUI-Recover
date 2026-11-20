# Tail selection across controls and one-step execution

## CPU browser demo

Install the project's base dependencies and Chromium. The demo creates its own local HTML fixtures and requires no model weights or MiniWoB assets.

```bash
python -m pip install -e .
python -m playwright install chromium
gui-recover demo-tail-probe --out runs/cpu-tail-demo
```

The demo covers light and dark input fields and textareas, three fonts, four viewport sizes and multiple pixel scales. The fixture supplies the field rectangle. The detector uses only that rectangle and the native screenshots. It isolates the changed selection region, estimates the background color inside the original crop, and rejects unsupported geometry. Ink connected to the preceding glyph can overhang the left edge of a selected space, so a narrow connected strip is treated as bleed. The demo checks both exact form submission and whether the probe restored the pixel state without changing the field value.

Each of the twelve layouts runs as clean input and as input with an added trailing space. Additional controls include an unfocused field and a selection spanning multiple characters. The outcomes, native screenshots and their hashes are saved in the output directory. This evaluates the selection detector and a fixed recovery rule on synthetic controls with known bounds. ShowUI grounding is measured separately in MiniWoB.

## Registered native latency comparison

The active MiniWoB demo takes one Gymnasium step for the two fixed key presses and the intermediate capture. BrowserGym validates and returns the native task state at the end of that step. The selected and restored frames remain available to the same verifier. The trace identifies both physical keys and records the batched Gym step explicitly. Older two-step traces remain supported.

The registered comparison alternates execution order and uses shared frozen ShowUI grounding. Four prior-unseen MiniWoB layouts each run seven conditions and both implementations. The promotion gate requires the same outcomes and logical model requests, plus at least 30 percent median paired reduction in probe duration. The audit verifies physical key evidence, screenshot hashes, action choices, native rewards and whether each pair has identical visual evidence.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tail_probe_latency.py \
  --config configs/tail_probe_latency.json --profile formal --model models/ShowUI-2B \
  --prior-plan runs/pre-rl-recovery-001/splits.json \
  --prior-plan runs/recovery-offline-rl-001/splits.json \
  --prior-plan runs/idempotent-input-repair-001/preflight.json \
  --prior-plan runs/visual-text-content-002/preflight.json \
  --prior-plan runs/online-visual-recovery-rl-001/preflight.json \
  --prior-plan runs/online-visual-recovery-rl-smoke-001/preflight.json \
  --prior-plan runs/active-text-verification-001/preflight.json \
  --prior-plan runs/active-text-verification-smoke-001/preflight.json \
  --prior-plan runs/learned-active-probe-001/preflight.json \
  --prior-plan runs/learned-active-probe-smoke-001/preflight.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --exclude-run runs/tail-probe-latency-smoke-001 \
  --out runs/tail-probe-latency-001
python scripts/audit_tail_probe_latency.py runs/tail-probe-latency-001
```

Run the engineering profile in a fresh directory with `--profile smoke` and omit `--exclude-run`.

## Transfer to native MiniWoB task variants

The installed active demo can also compare the same controllers on case transformation and randomized alphanumeric goals. The goal grammar derives the requested value from the displayed instruction. ShowUI locates the input and Submit, and MiniWoB provides the exact reward. The randomized alphanumeric task omits the case-flip scenario, since a digit cannot be case-flipped.

```bash
gui-recover active-text-recovery --model models/ShowUI-2B \
  --tasks enter-text-2 enter-text-dynamic --seeds 84000 84004 84005 84002 \
  --layouts-per-task 2 --out runs/native-text-transfer
gui-recover audit-active-text-recovery runs/native-text-transfer
```

MiniWoB task variants can render the same geometry for the same seed. The command treats the supplied seeds as a reserve and excludes duplicate initial screenshots across tasks. The declared reserve includes upper-case, lower-case and alphanumeric goals, identified from reset-only observations.
