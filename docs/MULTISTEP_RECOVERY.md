# Multistep recovery

The benchmark measures recovery at the first live checkpoint of native MiniWoB button-sequence, checkbox and short form tasks. All controllers use the same scripted suffix after the checkpoint. Native task termination decides whether the full task succeeded. Natural first actions come from a frozen vision model. Controlled healthy, missed and wrong actions are paired by layout, and the report retains wrong order, invalid predictions and terminal failures.

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

Open `report.html` inside the audited run. It links every native frame and shows exact outcomes, false repairs, natural failure recovery and the operation cost. Use `--profile formal` for the frozen larger population. `--prior-evidence` accepts old rollout directories to exclude duplicate native observations. Use a fresh output directory for each run.

Each arm has a shared cap of three additional operations. The no-recovery arm proceeds to the suffix. The fixed baseline performs one zoomed re-grounding and retry. The visual rule can wait once for text delivery, inspect a reversible tail selection or button focus marker, and take a bounded repair. The checkbox verifier reads pixels and may undo a specific wrong toggle before setting the intended box. The sequence probe briefly exposes button focus with a trusted browser style, checks that the unmarked frame was restored, and can retry when neither button acquired focus. An already recorded click on the wrong sequence button remains a failure under a zero-reset budget.

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

## Transfer to a second model

Download a separate local Qwen2.5-VL-3B-Instruct snapshot at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. See [third-party terms](../THIRD_PARTY.md). This backend returns integer pixels on its prepared image. The adapter validates the output and maps it into the normalized action interface used by the same evaluator and audit.

```bash
python scripts/run_multistep_recovery.py \
  --config configs/multistep_recovery.json --actor qwen --device cpu \
  --model models/Qwen2.5-VL-3B-Instruct --font "$(fc-match -f '%{file}' Arial)" \
  --transfer-from runs/multistep-smoke --layouts-per-task 1 --out runs/multistep-qwen
python scripts/audit_multistep_recovery.py runs/multistep-qwen --report
```

## Inspect VisualWebArena recordings

The offline diagnostic checks six native viewport screenshots from VisualWebArena Classifieds tasks 118, 120 and 149. It compares a last-frame answer with a bounded second look when Qwen explicitly reports uncertainty. It reports exact short-answer matches separately from the historical environment reward. The upstream assets stay local. See [AgentRewardBench dataset terms](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench#terms-of-use).

```bash
python scripts/replay_visualwebarena.py --download configs/visualwebarena_replay.json \
  --recordings runs/vwa-recordings --out runs/vwa-cpu-check.json
python scripts/replay_visualwebarena.py --recordings runs/vwa-recordings \
  --qwen models/Qwen2.5-VL-3B-Instruct --device cpu --out runs/vwa-qwen-check.json
```

A native run requires an instance you control with the original benchmark site assets. The opt-in live adapter sends answers to the BrowserGym task evaluator and uses its terminal reward. The first command shows the missing setup without loading a model or starting a browser. Configure the `VWA_*` variables from [BrowserGym's setup](https://github.com/ServiceNow/BrowserGym/tree/main/browsergym/visualwebarena) before the second command.

```bash
python -m pip install 'browsergym-visualwebarena==0.14.3'
python scripts/run_visualwebarena_live.py --preflight
python scripts/run_visualwebarena_live.py --model models/ShowUI-2B --out runs/visualwebarena-live
```
