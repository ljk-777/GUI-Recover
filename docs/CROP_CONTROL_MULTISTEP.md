# Fixed crop controls and normal intermediate clicks

The contextual repair study recovered more natural failures with action-centered zoom and feedback. That experiment could not isolate the benefit of crop placement from the benefit of making the MiniWoB content larger. Its live healthy controls were all INPUT actions, so the detector could also rely on action type. This follow-up tests both explanations with the existing frozen actor and detector thresholds.

## Crop placement

The formal configuration reserves 80 seeds for each of click-button, click-dialog, and click-tab. The reset-only preflight accepts the first 20 unique layouts per task, excluding previously registered source layouts. A frozen ShowUI prediction supplies the first click in each scene. Every natural failure is retained in the conditional denominator, including terminal failures. The diagnostic uses the known first-click error to trigger each repair attempt and allows one additional click with zero restarts.

| Method | Text | Screenshot region |
|---|---|---|
| Plain | Original target | Full screenshot |
| Feedback | Same feedback in all four feedback conditions | Full screenshot |
| Action-centered zoom | Feedback | 60% of width and height centered on the first proposal, clamped at the edges |
| Fixed zoom | Feedback | Same width and height, fixed to the upper left |
| Fixed content crop | Feedback | MiniWoB's standard 160×210 CSS content panel, from the 332×214 CSS viewport |

All crop inputs are enlarged to the original canvas dimensions with bicubic interpolation. The output coordinates are mapped back to the original screenshot before they are executed. Native PNGs, derived inputs and crop coordinates are available separately. The fixed content crop includes more vertical content than the equal-size comparison and therefore changes the relative resize factors. Report actual task success, paired differences and post-evaluation target-center coverage separately. Target geometry participates only in evaluation. Keep the prior repair selection and report all methods.

## Normal clicks while the task is still live

The formal profile reserves 120 seeds per task and accepts 24 layouts per task for click-button-sequence and click-checkboxes. Before inference, filter exact screenshot duplicates, goals with zero required checkbox clicks, duplicate visible control labels and setups without an actionable point on any required control. The test uses the repository's existing deterministic rendering and capture fixture on stock MiniWoB task logic.

Each accepted setup provides a correct first click, a click in the blank right margin, an incorrect control click where available, and a first click from frozen ShowUI. Replays start from the same initial screenshot. The sequence log and checkbox checked state give the first-step labels. The selected checkbox must match the specified first target even if a different requested checkbox would eventually satisfy the full task. The correct control must still be live, and the remaining required actions are executed using fixture coordinates to prove the initial click allows full success. This continuation is an execution check with a privileged controller.

Detector inputs use the screenshot pair, the observed click coordinates, the goal, and the current target. A shared hover captures the hypothetical observation after waiting. Each detector's reported trigger uses the original threshold, with a second decision only when its immediate verdict is unknown. DOM state and fixture coordinates do not provide verifier inputs beyond the executed action itself. Detector comparisons are first-step decisions. They do not execute a recovery or report end-to-end policy success.

Report false triggers on healthy live clicks, detection on blank clicks and wrong controls, natural live first-click results, and unresolved verdicts. A button sequence can accept a click with no observable change to the screenshot. A checkbox can visibly change even when the wrong box is checked. These cases test separate limits of appearance-based detection. Natural and controlled transitions share layouts and should keep separate denominators.

## Run

From the repository root, set the same MiniWoB and browser environment as the contextual repair study and use the cached ShowUI model and detector artifacts.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_crop_control_multistep.py \
  --config configs/crop_control_multistep.json --profile formal \
  --model models/ShowUI-2B --inherited-detectors runs/visual-verification-003-models \
  --prior-development runs/contextual-repair-001-development \
  --reference runs/visual-verification-003-data \
  --reference runs/visual-verification-001-data \
  --reference runs/contextual-repair-001-test/data \
  --out runs/crop-control-multistep-001
```

The smoke profile has separate seeds and counts and is for checking the pipeline. The formal report uses only the committed formal config and its reserved layouts.
