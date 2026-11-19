# Pre-RL recovery environment and frozen policy trial

This experiment formulates a small learning problem around an already proposed GUI action. A frozen ShowUI controller supplies click locations, a text input value comes from the task instruction, and a fixed continuation locates Submit for text tasks. A recovery policy chooses how to react from screenshots and its own action. The trial supports click-button, click-dialog, click-tab and enter-text in the deterministic MiniWoB fixture used in earlier studies.

## Interface and reward

Use `RecoveryDecisionEnv(task, grounder)` from `gui_recover.recovery_env`. It exposes Gymnasium reset and step, fixed 321×498×3 RGB observations, task and target strings, the proposed coordinate action, phase, remaining decisions, and a discrete action mask.

| Action | Available phase | Effect |
|---|---|---|
| Continue | Source or review | Execute the frozen source in source phase, or accept the current outcome and run any fixed continuation in review |
| Wait | Review, once | Wait and capture one hover observation at the proposal point |
| Retry plain | Review | Predict on the full current image, execute one repair and run any continuation |
| Retry fixed zoom | Review | Predict from the fixed top-left 60% crop with feedback, map coordinates back, execute one repair and run any continuation |

The source phase has a single valid choice. An episode has at most three decisions, four browser operations and three logical actor requests. The source or a repair can end the task before continuation. A masked decision ends the episode with zero reward and no browser action. An actor parse failure ends its branch with zero reward. The benchmark aborts on browser infrastructure errors and keeps the failed trace.

Reward is sparse. Only a successful native browser termination with `DONE_GLOBAL=true`, binary reward 1, exact raw MiniWoB reward 1, and no error earns +1. The environment returns 0 for partial positive raw rewards, unfinished tasks, failed submissions, timeouts and decision exhaustion. Negative rewards and truth fields stay in evaluation evidence. Truncation and termination remain separate flags. For this finite-budget objective, the terminal or truncated transition closes the trajectory with its recorded return. A policy only receives the observation, and the next rollout starts with reset.

## Frozen comparison and data separation

Train, validation and test have registered nonoverlapping reserves and exact initial screenshot checks. Drop-input and clean versions of a layout stay in one split. The layout preflight runs without actor predictions or outcomes and excludes previously registered sources from the listed archives. The formal test has four layouts per task, plus matched drop-input on the four text layouts. Train and validation trace two fixed policies to provide starter trajectories. The frozen test compares Continue, Plain, Fixed zoom, and an appearance-triggered Fixed zoom policy. It uses one shared frozen inference cache, with logical calls separated from actual generations.

The predeclared candidate for the effect check is appearance-triggered Fixed zoom. To pass, it must add at least one success on clean scenes and one on injected scenes compared with Continue, while preserving Continue successes in each group. Natural source errors and injected text errors are different evaluation cohorts. Results describe a small fixed-controller trial for learning recovery choices.

`iter_learning_transitions(path, split="train")` returns observation, discrete action, reward and next observation with the two end flags. It checks the split index, trace checksums and screenshot checksum/shape. Validation is explicit and the reader rejects test. Policy observations are separate from raw environment evidence. The small dataset checks the plumbing and serves as an initial replay set.

## Reproduce

Use the registered browser and cached ShowUI weights. Engineering smoke seeds are separate and do not enter the trial.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pre_rl_recovery.py \
  --config configs/pre_rl_recovery.json --profile test --model models/ShowUI-2B \
  --reference runs/visual-verification-001-data \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/pre-rl-recovery-001
```

## 正式结果

正式源码冻结在 `241465e`。28 个初始布局通过预检，训练、验证和测试分别包含 8、4、16 个布局。成对的漏输入分支随其原布局留在同一分区。最终记录 20 条训练轨迹、10 条验证轨迹和 80 条独立测试轨迹。

| 冻结策略 | 未注入故障成功 /16 | 漏输入成功 /4 | 未注入故障下重试 | 未注入故障下逻辑模型调用 |
|---|---:|---:|---:|---:|
| 继续 | 10 | 0 | 0 | 20 |
| 原图重试 | 10 | 4 | 10 | 30 |
| 固定裁剪重试 | 15 | 4 | 10 | 30 |
| 视觉触发＋固定裁剪 | 15 | 4 | 6 | 26 |

视觉方案通过预设效果门槛。在未注入故障组，逐例增加 5 个成功，保留原本全部 10 个成功。在漏输入组新增 4 个成功。未注入故障任务的成功数分别为按钮 4/4、对话框 3/4、标签页 4/4、文本输入 4/4。视觉触发和无条件固定裁剪在这批任务上成功数相同，视觉触发少重试 4 次。两个重试策略都能修复本批漏输入；固定裁剪的额外收益来自自然点击错误。[成对真实执行示例](../examples/pre-rl-recovery-pair/provenance.json)

## 强化学习前验收

独立审计从浏览器事件重算 110 条轨迹的奖励和结果，核对 198 次浏览器操作、截图校验值、模型坐标、动作掩码、配对前缀和缓存计数。真实浏览器另外验证了一步与两步决策预算耗尽，两者都给零回报，标记截断。正式运行的 198 次逻辑模型请求复用为 71 次真正生成，单卡显存采样峰值 5427 MiB，低于 12032 MiB 配额。两套 Ubuntu 测试和打包检查通过。[结构化结果](../examples/pre-rl-recovery-results.json) · [CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44547348242)

训练读取器验证并载入 34 个训练转移和 14 个验证转移。训练和测试在策略决策时刻使用的当前截图无哈希重叠。训练样本中包含继续和固定裁剪重试；等待和原图重试可以从环境采样，已有测试分别核验等待截断和原图修复。这个小语料适合验收数据链路。效果门槛覆盖冻结策略和本轮四类 MiniWoB 任务。多步状态判断、训练后的泛化和并行采样吞吐需要独立评测。

```python
from gui_recover.recovery_rollouts import iter_learning_transitions

sample = next(iter_learning_transitions("runs/pre-rl-recovery-001", split="train"))
assert sample["observation"]["action_mask"][sample["action"]]
assert sample["observation"]["current"].shape == (321, 498, 3)
```

至此可以交接小规模恢复策略训练。正式结果来自冻结 ShowUI 与固定策略，本轮没有强化学习更新。
