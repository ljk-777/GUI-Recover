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

The same release wheel passed the CPU demo on macOS and Linux with Chromium 125. On each platform all 12 clean and all 12 corrupted submissions succeeded, zero clean fields were rewritten, and the detector abstained on all eight negative controls. The unselected screenshots of the 12 clean/corrupted pairs matched exactly. All 24 probes left the field value unchanged and restored the original frame before the repair decision.

The dark textarea example below shows the shared original frame followed by the separate selections. The selected glyph is visible in the clean input. The blank selection reveals the injected trailing space. [Example metadata and frame hashes](../examples/rc3-tail-selection/provenance.json).

| Before either probe | Clean last character | Trailing space |
|---|---|---|
| [Original screenshot](../examples/rc3-tail-selection/before.png) | [Clean selection](../examples/rc3-tail-selection/clean-selected.png) | [Blank selection](../examples/rc3-tail-selection/trailing-selected.png) |

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

| Result on four frozen MiniWoB layouts | Two BrowserGym steps | One BrowserGym step |
|---|---:|---:|
| Exact submissions | 28/28 | 28/28 |
| Median probe duration | 1.926 s | 1.270 s |
| BrowserGym steps across the run | 108 | 96 |
| Logical ShowUI requests | 76 | 76 |

The 12 pairs that used the probe had a median paired latency reduction of 33.8%, passing the registered 30% threshold. All 28 pairs matched in reward, decisions and available visual evidence. All 24 executed probes restored the original frame. The CPU audit recomputed 152 decisions, 204 BrowserGym steps and 256 branch-local screenshots. These timings describe the probe on the evaluated machine. Both implementations use two physical key presses per probe.

## Transfer to native MiniWoB task variants

The installed active demo can also compare the same controllers on case transformation and randomized alphanumeric goals. The goal grammar derives the requested value from the displayed instruction. ShowUI locates the input and Submit, and MiniWoB provides the exact reward. A matrix containing the alphanumeric task omits the case-flip fault across both tasks, since a randomized target can consist entirely of digits.

```bash
gui-recover active-text-recovery --model models/ShowUI-2B \
  --tasks enter-text-2 enter-text-dynamic --seeds 84000 84004 84005 84002 \
  --layouts-per-task 2 --out runs/native-text-transfer
gui-recover audit-active-text-recovery runs/native-text-transfer
```

MiniWoB task variants can render the same geometry for the same seed. The command treats the supplied seeds as a reserve and excludes duplicate initial screenshots across tasks. The declared reserve includes upper-case, lower-case and alphanumeric goals, identified from reset-only observations.

The wheel ran 72 browser rollouts across four distinct layouts, six conditions and three controllers. Conditions covered clean, delayed, omitted, partial, wrong and trailing-space input.

| Native task | Glyph check | Unconditional replacement | Active tail probe |
|---|---:|---:|---:|
| `enter-text-2` | 10/12 | 12/12 | 12/12 |
| `enter-text-dynamic` | 10/12 | 12/12 | 12/12 |
| Trailing-space subset across both tasks | 0/4 | 4/4 | 4/4 |
| Rewrites of the four clean inputs | 0 | 4 | 0 |
| All rewrites | 12 | 24 | 16 |
| BrowserGym steps | 68 | 76 | 84 |
| Logical ShowUI requests | 60 | 72 | 64 |

The active controller saved eight rewrites and eight logical requests relative to unconditional replacement, at the cost of eight additional BrowserGym steps. On each of the four layouts, the clean and trailing-space observations matched before selection. All 12 executed probes restored the original frame. The independent CPU audit checked 172 decisions, 228 BrowserGym steps and 268 branch-local screenshots. Frozen ShowUI used 24 actual generations across the shared cache. GPU memory sampled at two-second intervals peaked at 5309 MiB on an RTX 4090 D with 12032 MiB available.

The evaluation covers short targets on three related MiniWoB task families, including the latency study. A fixed controller reads the task instruction and chooses the verification or repair action. ShowUI predicts the control locations. The results measure bounded recovery with injected faults in known task grammars. Wider coverage would require a separate evaluation with natural errors, new applications and multi-step agent plans. [Machine-readable results, layout registry and checks](../examples/rc3-tail-validation.json).
