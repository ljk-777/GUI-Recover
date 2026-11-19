# Target state verification and guarded checkbox recovery

The prior multistep test found false repair triggers on 39 of 48 healthy clicks with the frozen visual detector. The two task families exposed distinct cases: clicking the first sequence button changes internal progress without changing pixels, and checking the wrong checkbox changes pixels. This study checks an explicit visible target state on the checkbox task. On the sequence task the new verifier returns unknown and does not authorize a retry.

The checkbox reader uses the native 498×321 Chromium screenshot and one MiniWoB rendering profile. It matches the requested label against row text using a standard font template. A minimum similarity and a gap to the runner-up are required. The reader checks the checkbox at the matched row before and after the action, and abstains on an unsupported layout, unreadable state, ambiguous label or unexpected changes. It receives screenshots, the goal, the current target and the proposed action through `TransitionView`. The teacher reads DOM state only for offline labels, fixture setup and continuation after the evaluated first step.

Recovery requires verified evidence that the target remains unchecked. If one other box was checked and the click falls in that row, the controller can undo that one change. It reads the updated screenshot before any further click. A target click is allowed once, with a final visual audit. The budget is one undo, one target click, one shared observation after waiting, and zero restarts. The actor and the frozen detector remain fixed. The recovery profile is specific to native checkboxes in this rendering and does not apply to arbitrary websites.

## Protocol

The previous 189 traces informed the design. The experiment takes the first 8 reachable unique button-sequence layouts and 14 checkbox layouts from a separate 43000-series reserve. Exact initial screenshot matches to the previous accepted layouts are excluded before model inference. The smoke uses separate seeds. Correct, blank and wrong-control branches start from a verified identical prefix, alongside a natural ShowUI first click. The inherited frozen head and full-image appearance rule are evaluated on the same observations. Their decisions are recorded without execution. The new policy's permitted repairs are executed. An evaluator validates the first-step outcome, and a scripted oracle continuation checks full task success after a correct or recovered first step. Report trigger, unknown, preserved successes, physical recovery and teacher-assisted completion separately.

The registered glyph score is at least 0.70 and the runner-up margin at least 0.18. The numbered profile config pins the font, detector artifact, prior plan, actor weights and MiniWoB revision.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_target_state_study.py \
  --config configs/target_state.json --profile test \
  --prior-plan runs/crop-control-multistep-001/multistep-plan.json \
  --model models/ShowUI-2B --frozen-detector runs/visual-verification-003-models/frozen \
  --font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \
  --out runs/target-state-001
```

## 正式结果

正式实现冻结在 `67882f3`。22 个新布局在首击前登记，未发现重复或不可用布局。85 条实际首击轨迹均保持可操作，模型输出均可解析。旧检测器和新检测器在同一组观察上做判断。

| 分组 | 样本 | 冻结视觉头触发 | 全图变化规则触发 | 目标状态方案触发 | 目标状态方案未知 |
|---|---:|---:|---:|---:|---:|
| 正常受控首击 | 22 | 19 | 22 | 0 | 10 |
| 点空白 | 22 | 19 | 22 | 12 | 10 |
| 点错控件 | 19 | 16 | 19 | 11 | 8 |
| 自然正常首击 | 6 | 6 | 6 | 0 | 6 |
| 自然错误首击 | 16 | 12 | 16 | 12 | 4 |

按钮顺序的 32 条轨迹均为未知，因而保留了 14 个已经正确的首击，2 个自然错误也未被修复。对复选框，14 个正确首击中有 12 个得到目标状态的正面确认，另 2 个因字形匹配置信度不足而暂缓判断。这两个布局在空点和自然失败分支上也保持未知。

| 复选框分支 | 首击已正确 | 恢复后正确 | 修复点击 |
|---|---:|---:|---:|
| 正常受控点击 | 14/14 | 14/14 | 0 |
| 点空白 | 0/14 | 12/14 | 12 |
| 点错控件 | 0/11 | 11/11 | 22 |
| 自然 ShowUI 首击 | 0/14 | 12/14 | 16 |

35 个修复均达到准确的首步目标状态，之后在固定教师续步下完成任务。共执行 50 次物理修复点击，其中 15 次撤销误勾。教师续步用于验证修复后的任务可继续执行。公开示例保留同一自然首击的误勾、撤销和正确勾选画面及动作记录。[示例证据](../examples/target-state-checkbox-recovery/evidence.json)

在冻结 Linux 字形环境下，CPU 重算与正式记录的分数和判断一致。Mac 的字形布局引擎为 BASIC，Linux 使用 RAQM 0.11.0、HarfBuzz 14.5.0 和 FreeType 2.14.3。Mac 的匹配分数或间隔与原记录最大相差 0.0981，这批轨迹的判断和动作证据一致。正式验收仅覆盖已登记的 Linux 渲染条件、原生复选框和已支持的文本形态。稳定截图且无可验证目标状态的点击仍需其他观察信号。

独立审计复核 85 条轨迹、429 张原生截图和 50 次物理修复点击。新验证器与恢复逻辑仅使用 CPU，正式服务器运行的 GPU 显存峰值采样为 5299 MiB，消耗来自冻结 ShowUI 和继承的视觉检测对照。[正式汇总](../examples/target-state-results.json) · [冻结实现 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44539836353)
