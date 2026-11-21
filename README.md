# GUI-Recover

**A visual recovery and evaluation toolkit for browser agents.** Compare what happens when an agent continues, checks the screen, or retries after a missed click, wrong input, or unsupported answer.

An action can run without having its intended effect. A click can leave the page unchanged, a field can contain the wrong text, and an answer can sound plausible before the agent reaches the relevant listing. GUI-Recover makes those failures reproducible, tests bounded recovery strategies against matched baselines, and saves the evidence for each decision.

[Download v0.2.0rc3](https://github.com/ljk-777/GUI-Recover/releases/tag/v0.2.0rc3) · [Run the CPU demo](#quick-start) · [Review the benchmark protocols](docs/MULTISTEP_RECOVERY.md) · [Browse saved examples](examples/README.md)

<a name="安装与本地验收"></a>

## Quick start

From the repository root, use Python 3.11 and Chromium. The demo builds its own fixtures and runs on CPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m playwright install chromium
gui-recover demo-tail-probe --out runs/cpu-tail-demo
```

**Try an error the first screenshot cannot reveal.** The clean input and the input with a trailing space have identical initial screenshots. A reversible selection exposes either the last glyph or a blank space. The published demo completed 24 exact submissions on each of macOS and Linux, with zero clean-field rewrites in each run. The output contains the selected and restored PNGs, screenshot hashes, and exact submission results. Use a fresh output directory for each run. [Demo protocol and verified outputs](docs/TAIL_PROBE_PORTABILITY.md)

| Clean input selects the last glyph | Corrupted input selects a trailing space |
| --- | --- |
| ![Dark textarea with its final glyph selected](examples/rc3-tail-selection/clean-selected.png) | ![Dark textarea with the trailing blank space selected](examples/rc3-tail-selection/trailing-selected.png) |

[Shared initial screenshot](examples/rc3-tail-selection/before.png) · [Frame provenance](examples/rc3-tail-selection/provenance.json)

For a MiniWoB++ environment smoke test, run `bash scripts/demo_cpu.sh runs/miniwob-smoke`. It sets up the pinned environment and Chromium and uses a privileged diagnostic oracle with an expected 3/3 on `click-test`. Open `runs/miniwob-smoke/click-test-0/replay.html` to inspect the screenshots and executed actions. Ubuntu setup and mirror instructions are in the [server guide](docs/SERVER.md).

## What GUI-Recover measures

- **MiniWoB++ input and action recovery.** BrowserGym resets each task and executes bounded coordinate and keyboard actions. Fault injection simulates missed or offset clicks and dropped, partial, or wrong text. Recovery strategies can recheck screenshots, inspect a localized change, or retry a bounded action. ShowUI supplies visual grounding for the model-backed runs.
- **VisualWebArena answer verification.** For supported Classifieds tasks, the Qwen study can search from a reference image, inspect visible results, read a screenshot crop from the matching listing, and check the candidate in a second model call before submitting. Paired strategies share a model and per-episode budget. The evaluator retains native task scoring.
- **Evidence from each run.** Episodes save proposed and executed actions, raw model outputs, PNGs, rewards, and model-call counts. The summary records the evaluation setup and outcomes. The HTML viewer walks through saved frames, and the CPU auditors reconstruct supported decisions and scores from saved artifacts.

The visual policies use rendered screenshots and task inputs. VisualWebArena's listing adapter can also use hit-tested geometry and destinations for visible cards. The default MiniWoB++ policy receives task text, pixels, and action history. [Evaluation scope and negative results](docs/MULTISTEP_RECOVERY.md) · [Example recovery trace](examples/recovered/replay.html) · [Example wrong-input trace](examples/wrong-input/replay.html)

## Selected results

| Experiment | Paired outcome | Evidence |
| --- | --- | --- |
| MiniWoB++ Enter Text with injected visible text errors | Glyph verification repaired 18/18 visible errors across six layouts. All six clean inputs succeeded with zero rewrites. | [Protocol and outcomes](docs/VISUAL_TEXT_CONTENT.md#正式结果) · [Result data](examples/visual-text-content-results.json) |
| Native MiniWoB++ text task variants on one RTX 4090 D | Active verification scored 24/24, glyph verification scored 20/24, and unconditional replacement scored 24/24. Active verification made eight fewer rewrites and used eight more browser steps than unconditional replacement. | [Native task evaluation](docs/TAIL_PROBE_PORTABILITY.md#transfer-to-native-miniwob-task-variants) · [Result data](examples/rc3-tail-validation.json) |
| Two previously observed VisualWebArena Classifieds development tasks | The evidence strategy earned native full credit on 2/2 versus 0/2 for the answer-check baseline. The candidate used three Qwen calls per task. Both strategies made zero wrong submissions. | [Replay instructions](docs/MULTISTEP_RECOVERY.md#replay-reference-image-retrieval-failures) · [Per-task results](examples/vwa-reference-retrieval-development-results.json) |
| Fresh Classifieds acceptance comparison with seven official tasks and one independent task | Both strategies scored 0/8. The evidence strategy made zero wrong submissions and the baseline made one. The comparison missed the acceptance criteria. | [Protocol and findings](docs/MULTISTEP_RECOVERY.md#run-the-fresh-classifieds-acceptance-comparison) · [Per-task results](examples/vwa-fresh-heldout-results.json) |

These results apply to the listed task grammars, fault conditions, and budgets. [Earlier VisualWebArena holdout audit](examples/vwa-input-heldout-results.json)

<a name="gpu-最小恢复演示"></a>

## Reproduce the MiniWoB++ GPU comparison

After the CPU setup, use one CUDA GPU with BF16 support. The setup scripts pin the MiniWoB++ checkout and verify the separately downloaded ShowUI-2B weights.

```bash
python -m pip install -e '.[showui]'
bash scripts/setup_miniwob.sh
python scripts/download_showui.py --out models/ShowUI-2B
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export CUDA_VISIBLE_DEVICES=0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gui-recover text-recovery --model models/ShowUI-2B --out runs/text-recovery-demo
gui-recover audit-text-recovery runs/text-recovery-demo
```

The default demo runs seven conditions with two strategies on one layout. Inspect `summary.json` for aggregate scores and the episode directories for PNG, JSONL, and HTML. Auditing a saved run requires the base package and CPU. The active comparison adds reversible tail selection.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gui-recover active-text-recovery --model models/ShowUI-2B --out runs/active-tail-demo
gui-recover audit-active-text-recovery runs/active-tail-demo
```

See [single-card setup](docs/SERVER.md), [visual text verification](docs/VISUAL_TEXT_CONTENT.md), and [active verification](docs/ACTIVE_TEXT_VERIFICATION.md) for the full layouts, budgets, and known failure cases.

<details>
<summary>Reproduce the earlier dropped-input diagnostic</summary>

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gui-recover run --policy showui-scripted --model models/ShowUI-2B --tasks enter-text --seeds 40 --max-steps 4 --parser mixed --fault drop-input --recovery input-guard --input-guard-version v2 --out runs/recovery-demo
```

This diagnostic uses a scripted planner and ShowUI grounding. The input guard checks for the appearance of text pixels within a supported field crop. [Input guard protocol and scope](docs/INPUT_GUARD.md)

</details>

## Research and contributions

- **Run and inspect.** Start with [the command guide](docs/SERVER.md) or [release validation](docs/RELEASE_CANDIDATE.md#020rc3-验收). The model-facing interfaces, paired studies, and known failure cases are described in [the multistep report](docs/MULTISTEP_RECOVERY.md).
- **Train and compare.** There are [SFT adapter experiments](docs/SFT_RECOVERY.md), a [Gymnasium recovery decision environment](docs/PRE_RL_RECOVERY.md), and [small offline](docs/RECOVERY_OFFLINE_RL.md) and [online RL studies](docs/ONLINE_VISUAL_RECOVERY_RL.md). The reports include regressions and promotion criteria. The default model-backed demos use frozen model weights.
- **Contribute.** Include a reproducible task and seed, the exact command, and a small trace when opening an issue. See [contribution guidelines](CONTRIBUTING.md), [change history](CHANGELOG.md), and [upstream provenance](docs/UPSTREAM_AUDIT.md).

**中文文档** · [服务器操作单](docs/SERVER.md) · [发布验收](docs/RELEASE_CANDIDATE.md) · [已知限制与反例](docs/RELEASE_VALIDATION.md)

GUI-Recover is licensed under [Apache-2.0](LICENSE) and builds on ShowUI, BrowserGym, MiniWoB++, and VisualWebArena. Model weights, datasets, and upstream assets are governed by their respective terms. [Credits and third-party licenses](THIRD_PARTY.md)
