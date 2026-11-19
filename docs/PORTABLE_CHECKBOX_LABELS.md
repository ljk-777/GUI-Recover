# Portable checkbox label matching

The previous screenshot verifier abstained on two checkbox layouts where the top row led by a wide margin but the absolute glyph score was below its threshold. Pillow also selected different font layout engines on Linux and macOS. The new profile pins BASIC layout, renders a native 15 px reference and a 4× reference downsampled to the screenshot resolution, and smooths pixel coverage in the same way for the template and screenshot. The two scales must choose the same row. The stronger match must score at least 0.87 and lead by at least 0.18; both margins must reach 0.12. Armed recovery actions still require the previous checked-state, attribution and reinspection checks.

The profile is opt-in. It expects the known native MiniWoB checkbox rendering and a goal that refers to a label present on the form. A high similarity score alone cannot establish that an absent near-match string is present. For this reason the study explicitly probes absent lookalike labels offline, on the same untouched frame, without letting the probe trigger an action. Labels are collected by the benchmark fixture exclusively for ground-truth absence checks. The verifier sees the altered task text and the image, without access to that list.

## Study

Development used 38 previously recorded layouts. The fresh comparison reserves 100 seeds in the 44000 series and takes 20 reachable unique checkbox layouts while excluding previously registered layouts. The same before/after frames feed both the Linux legacy and the portable matcher. A correct first click, blank click and wrong checkbox click where available are real browser actions. Only the portable path executes repair. A scripted teacher continuation verifies full task success after a correct or repaired first step. Separate engineering seeds are excluded from the formal result.

Two previously ambiguous natural errors are replayed separately with cached prior actions and exact screenshot checks. These are regression cases from development. Fresh browser traces do not use actor calls, and the visual matcher requires no GPU. The baseline retains its existing RAQM font layout and thresholds. The portable renderer is audited again using the same font and saved screenshots on macOS.

```bash
python scripts/run_portable_checkbox_labels.py \
  --config configs/portable_checkbox_labels.json --profile test \
  --font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \
  --previous-run runs/target-state-001 \
  --earlier-plan runs/crop-control-multistep-001/multistep-plan.json \
  --out runs/portable-labels-001
```
