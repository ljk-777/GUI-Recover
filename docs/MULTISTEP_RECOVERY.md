# Multistep recovery

The benchmark measures recovery at the first live checkpoint of native MiniWoB button-sequence, checkbox and short form tasks. All controllers use the same scripted suffix after the checkpoint. Natural first actions come from a frozen vision model. Controlled healthy, missed and wrong actions are paired by layout, and the report retains wrong order, invalid predictions and terminal failures. The audit reports binary task reward and full credit separately. The checkbox task can award a positive binary result for selecting only part of the requested set. Full credit requires the recorded raw reward to equal one.

## Compare three controllers

Install the package and pinned MiniWoB assets. The Linux command uses the local Arial-compatible font reported by fontconfig. A BF16 GPU and local ShowUI checkpoint are required for the ShowUI profile.

```bash
python -m pip install -e '.[showui]'
bash scripts/setup_miniwob.sh
python -m playwright install chromium
python scripts/download_showui.py --out models/ShowUI-2B
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export CUDA_VISIBLE_DEVICES=0

python scripts/run_multistep_recovery.py \
  --config configs/multistep_recovery.json --profile smoke \
  --model models/ShowUI-2B --font "$(fc-match -f '%{file}' Arial)" \
  --out runs/multistep-smoke
python scripts/audit_multistep_recovery.py runs/multistep-smoke --report
```

Open `report.html` inside the audited run. It links every native frame and shows full credit, binary reward, false repairs, natural failure recovery and operation cost. Use `--profile formal` for the frozen larger population. `--prior-evidence` accepts old rollout directories to exclude duplicate native observations. Use a fresh output directory for each run.

## Formal comparison

The ShowUI evaluation ran 540 trajectories across 36 held-out layouts, including 12 layouts from the reserved checkbox family. All methods use the same extra-operation cap.

| Controller | Full credit | Native binary success | Natural misses recovered with full credit | Healthy checkpoints repaired | Extra operations |
|---|---:|---:|---:|---:|---:|
| Continue | 71/180 | 96/180 | 0/13 | 0/59 | 0 |
| Fixed retry | 104/180 | 138/180 | 1/13 | 59/59 | 204 |
| Visual recovery | 167/180 | 167/180 | 12/13 | 0/59 | 255 |

Full-credit outcomes were recomputed from the original terminal rewards after identifying partial credit in the checkbox task. The registered binary comparison also passed both gates. The visual controller spent 51 more operations than fixed retry. It maintained the same zero-repair rate as continuation on healthy checkpoints.

Each arm has a shared cap of three additional operations. The no-recovery arm proceeds to the suffix. The fixed baseline performs one zoomed re-grounding and retry. The visual rule can wait once for text delivery, inspect a reversible tail selection or button focus marker, and take a bounded repair. The checkbox verifier reads pixels and may undo a specific wrong toggle before setting the intended box. The sequence probe briefly exposes button focus with a trusted browser style, checks that the unmarked frame was restored, and can retry when neither button acquired focus. An already recorded click on the wrong sequence button remains a failure under a zero-reset budget.

On the first registered button layout, the correct click, omitted click and wrong click produced the same original PNG. The focus probe distinguished the omitted click and let the controller retry once. The correct click continued with no retry. The wrong click still failed. All three probes restored the original frame. [Original screenshot](../examples/multistep-focus-probe/without-probe.png) · [Correct focus](../examples/multistep-focus-probe/healthy-focus.png) · [Omitted click](../examples/multistep-focus-probe/miss-focus.png) · [Wrong focus](../examples/multistep-focus-probe/wrong-focus.png) · [Provenance](../examples/multistep-focus-probe/provenance.json).

The protocol freezes screenshot hashes and layout splits before acting. Checkbox layouts form the entire transfer task family. DOM geometry builds the controlled actions and common suffix, and privileged state labels are collected for reporting. The recovery controller takes the visible goal and proposal, screenshots and the outputs of its bounded operations. The benchmark reports extra checks of healthy states separately from repairs of healthy states.

## Learn only from the development split

The train sequence explores probe and skip on native form outcomes. Rewards include the same wait, probe and repair costs in all arms. The runner freezes Q estimates before using test scenes and reports zero regression on deferred trailing spaces and a reduction in probe operations as independent promotion criteria.

Export successful repair coordinates and clean anchors from the development evidence. The export verifies the source trajectory and frame checksums and checks that train, validation and test share neither layouts nor exact screenshots. The SFT loop uses assistant-only LoRA on the fixed ShowUI checkpoint, with no test input to the trainer.

```bash
python -m pip install -e '.[sft]'
python scripts/export_multistep_sft.py runs/multistep-smoke runs/multistep-sft-data
python scripts/train_grounding_sft.py --config configs/multistep_sft.json \
  --data runs/multistep-sft-data --model models/ShowUI-2B --out runs/multistep-adapter
python scripts/eval_multistep_sft.py --run runs/multistep-smoke \
  --model models/ShowUI-2B --adapter runs/multistep-adapter --out runs/multistep-adapted
```

The held-out adapter check reuses frozen original proposals and replaces the re-grounding model only. It requires a stable committed worktree for recording training provenance.

The online probe learner completed 64 training episodes. Independent CPU replay reproduced every seeded selection and both Q-value tables. Its frozen policy reached 48/48 on held-out form episodes, including 12/12 on delayed trailing spaces. The visual rule also reached 48/48. Both executed 48 probes, so the learned policy missed the cost-reduction gate.

The ShowUI SFT pilot used 56 training and 28 validation examples, with 40 and 20 verified corrections in those splits. It completed 32 decoder LoRA updates. Validation loss went from 4.150 to 1.830, and the adapter reload matched the saved test loss exactly. On 18 matched browser rollouts the frozen retry model achieved 11 full-credit outcomes and the adapted retry model achieved 10. Both achieved 14 native binary successes. These results keep the visual rule and frozen weights as the evaluation reference.

## Transfer to a second model

Download a separate local Qwen2.5-VL-3B-Instruct snapshot at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. See [third-party terms](../THIRD_PARTY.md). This backend returns integer pixels on its prepared image. The adapter validates the output and maps it into the normalized action interface used by the same evaluator and audit.

```bash
python scripts/run_multistep_recovery.py \
  --config configs/multistep_recovery.json --actor qwen --device cpu \
  --model models/Qwen2.5-VL-3B-Instruct --font "$(fc-match -f '%{file}' Arial)" \
  --transfer-from runs/multistep-smoke --layouts-per-task 1 --out runs/multistep-qwen
python scripts/audit_multistep_recovery.py runs/multistep-qwen --report
```

In the three-layout Qwen compatibility study, the three controllers ran 45 native trajectories and the CPU audit checked 199 screenshots. Full credit was 14/15 with visual recovery, 10/15 with fixed retry and 7/15 with continuation. The respective binary scores were 14/15, 13/15 and 9/15. The visual method repaired zero of six healthy intermediate states and used 20 extra operations. Fixed retry repaired four of those states and used 13 extra operations. All three natural Qwen first actions in this sample were successful, so the comparison measures controlled faults.

## Inspect VisualWebArena recordings

The offline diagnostic checks six native viewport screenshots from VisualWebArena Classifieds tasks 118, 120 and 149. It compares a last-frame answer with a bounded second look when Qwen explicitly reports uncertainty. It reports exact short-answer matches separately from the historical environment reward. The upstream assets stay local. See [AgentRewardBench dataset terms](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench#terms-of-use).

```bash
python scripts/replay_visualwebarena.py --download configs/visualwebarena_replay.json \
  --recordings runs/vwa-recordings --out runs/vwa-cpu-check.json
python scripts/replay_visualwebarena.py --recordings runs/vwa-recordings \
  --qwen models/Qwen2.5-VL-3B-Instruct --device cpu --out runs/vwa-qwen-check.json
```

A native run requires a Classifieds instance you control. The opt-in adapter sends answers to the BrowserGym task evaluator and uses its terminal reward. The registered tasks use the native string judge, so the runner skips the unused BLIP captioner and external judging clients. The first command reports the setup status without loading a model or starting a browser. Configure `VWA_*` and a private NLTK data directory with `punkt_tab` before the live command. [BrowserGym setup](https://github.com/ServiceNow/BrowserGym/tree/main/browsergym/visualwebarena)

```bash
python -m pip install 'browsergym-visualwebarena==0.14.3'
python scripts/run_visualwebarena_live.py --preflight
python scripts/run_visualwebarena_live.py --model models/ShowUI-2B --out runs/visualwebarena-live
```

For CPU Qwen navigation, pass `--actor qwen --device cpu --model models/Qwen2.5-VL-3B-Instruct`.

The offline answer check matched one of three reference answers. The model returned an answer in every case, so an uncertainty gate based on an explicit `UNKNOWN` requested zero additional checks. The recording comparison is independent of the live benchmark.

The server ran the optimized Classifieds site with fixed web and database image digests. The optimized site uses re-encoded AVIF photos. A known-answer wiring check on the live site reached the native evaluator and earned full reward for all three tasks. [Classifieds image source](https://github.com/bgrins/vwa_classifieds_optimized)

Qwen then ran the same three task IDs in each arm on the single GPU with a two-step budget. Each arm executed three actions in total and finished zero tasks. One task hit the step limit and two produced invalid model actions. The feedback arm issued zero recovery messages. This run validates the live connection and action loop and leaves the recovery effect on VisualWebArena unmeasured.

### Evaluate invalid-action recovery

The Qwen adapter recognizes the observed labelled scroll form as one fixed-size scroll. Clicks and text input still require a valid position. The format study compares stopping, repeating the same request, and asking again with guidance about the action schema. A retry uses the same screenshot and action history. Each arm shares the same prediction budget, and a retry consumes one prediction from it. The study allows one retry per episode.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --model models/Qwen2.5-VL-3B-Instruct \
  --study format --task-set transfer --max-steps 6 --out runs/visualwebarena-format-transfer
python scripts/audit_visualwebarena_live.py runs/visualwebarena-format-transfer \
  --out runs/visualwebarena-format-transfer-audit.json
```

Use `--task-set diagnostic` to reproduce the three tasks that exposed the original parsing failures. The transfer set is a separate group of 10 registered Classifieds tasks with local string grading. The CPU audit checks the paired reset screenshots, prediction budgets, executed actions, and terminal reward. Its report tracks how often a retry produced a valid action and how often the original task earned full native reward.

The Qwen CUDA run used six predictions per episode as its cap. The audit verified 39 trajectories and 117 screenshots. All 13 task trios shared identical reset pixels and identical pixels and provider output at the first rejection.

| Transfer strategy | Model calls | Executed actions | Valid retries | Native successes |
| --- | ---: | ---: | ---: | ---: |
| Stop after rejection | 29 | 19 | – | 0/10 |
| Repeat the request | 38 | 19 | 0/9 | 0/10 |
| Retry with format guidance | 43 | 24 | 3/9 | 0/10 |

On the three diagnostic tasks, zero of three plain retries and one of three guided retries yielded a valid action. No diagnostic arm completed a task. The guided transfer arm executed one click and two text inputs immediately after its successful retries. It submitted zero answers across the 10 tasks. This establishes a benefit for generating executable actions under the tested prompt and budget. Task completion on Classifieds remains unresolved.

### Evaluate unchanged-scroll feedback

Pass `--study scroll --task-set diagnostic` to compare the existing click and execution-error feedback with the same controller plus scroll feedback. Both arms use the one-retry format guidance and the same prediction budget. The scroll detector fires when a scroll leaves the screenshot pixel-identical. Its message describes the observed lack of movement. Scroll messages share the controller's two-message limit with click and execution-error feedback.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --model models/Qwen2.5-VL-3B-Instruct \
  --study scroll --task-set diagnostic --max-steps 6 --out runs/visualwebarena-scroll-diagnostic
python scripts/audit_visualwebarena_live.py runs/visualwebarena-scroll-diagnostic \
  --out runs/visualwebarena-scroll-diagnostic-audit.json
```

Two of the three paired tasks reached an identical unchanged viewport after a scroll. The next action was a scroll that moved the viewport on both eligible tasks with scroll feedback, and on one eligible task with the ordinary feedback controller. The third task never reached a scroll. Both arms earned full native reward on zero of three tasks. The audit checked all six trajectories and 20 screenshots in this diagnostic comparison.

[Live format and scroll results with run provenance](../examples/vwa-live-recovery-results.json)

### Check whether the current viewport supports an answer

The answer study keeps the same Qwen checkpoint, navigation prompt, invalid-action retry, and scroll feedback in both arms. The added check asks a separate question about the current screenshot before the first navigation prediction on that image. A short answer is submitted to the native judge. An explicit abstention or malformed response leaves the browser untouched and gives the remaining budget to navigation. A repeated screenshot is checked once per episode. Each check, retry, and navigation prediction counts against the same eight-call cap.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --model models/Qwen2.5-VL-3B-Instruct \
  --study answer --task-set answer-transfer --max-steps 8 --out runs/visualwebarena-answer-transfer
python scripts/audit_visualwebarena_live.py runs/visualwebarena-answer-transfer \
  --out runs/visualwebarena-answer-transfer-audit.json
```

The diagnostic set reuses tasks 118, 128 and 221 from earlier runs. The transfer set contains eight different tasks fixed before running the check. It includes category navigation, gallery inspection, reading details in listing photos, and a quantity question on a listing page. The CPU audit reconstructs which screenshot each check read, whether an answer reached the benchmark, the source of that answer, and the native reward.

On the three diagnostic tasks, the navigation arm submitted zero answers. The added check made four judgments across distinct screenshots and submitted two answers. Both submissions occurred on the initial screenshot and received zero reward. Both arms finished with zero native successes.

[Machine-readable results, method costs and model provenance](../examples/multistep-recovery-results.json)
