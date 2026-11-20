# 学习何时主动探测

主动核验规则在字形匹配时探测末尾字符。这保证了能处理不可见的尾空格，也会给正常输入增加两次按键。本轮用真实 MiniWoB 交互学习在何时省略这一步。

ShowUI 定位和视觉规则保持固定。学习器仅在当前输入看起来匹配、尚未探测时选择继续提交或执行探测。已确认末尾字符有字形时直接提交，确认空白时交给原有重输动作。两个可学习情境为立即看到匹配内容和等待后看到匹配内容，均来自 Agent 自己的动作历史。每回合至多学习一个选择，使用该回合的精确提交奖励减去等待 0.02、探测 0.02 与重输 0.05 的成本，在线更新该情境的动作价值。每个动作优先采样两次，再做 0.15 概率探索。

正式训练使用三个独立布局与正常、延迟、尾空格三类条件，各重复三次。冻结后在两个新布局上比较总是按需探测的主动规则、等过就跳过探测的手工规则和学习策略。七种常规条件之外，额外加入从未参与训练的延迟尾空格挑战。三条策略共享其他决策和精确提交口径，合计 27 个训练回合与 48 条测试轨迹。

准入要求为正常输入零重输、常规组保留主动规则原有的成功、常规组减少探测，以及延迟尾空格挑战不回退。每个布局跨多个条件复用，常规和挑战组分别报告。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_learned_active_probe.py \
  --config configs/learned_active_probe.json --profile formal --model models/ShowUI-2B \
  --accepted-rule runs/active-text-verification-001 --rule-smoke runs/active-text-verification-smoke-001 \
  --prior-plan runs/pre-rl-recovery-001/splits.json \
  --prior-plan runs/recovery-offline-rl-001/splits.json \
  --prior-plan runs/idempotent-input-repair-001/preflight.json \
  --prior-plan runs/visual-text-content-002/preflight.json \
  --prior-plan runs/online-visual-recovery-rl-001/preflight.json \
  --prior-plan runs/online-visual-recovery-rl-smoke-001/preflight.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --exclude-run runs/learned-active-probe-smoke-001 \
  --out runs/learned-active-probe-001
```

工程验收用 `--profile smoke` 和独立输出目录，省略 `--exclude-run`。

## 正式结果

实现冻结于 `25c4bdd`。三个训练布局 77000–77002 的 27 个回合中有 26 个精确成功。学习器在立即看到匹配时选择探测，在等待后看到匹配时选择提交。后者的训练回报估计为直接提交 0.98、探测 0.96。

| 策略 | 常规成功 | 常规探测 | 常规浏览器操作 | 延迟尾空格挑战 | 所有测试重输 |
|---|---:|---:|---:|---:|---:|
| 主动核验规则 | 14/14 | 6 | 54 | 2/2 | 12 |
| 等过就跳过规则 | 14/14 | 4 | 50 | 0/2 | 10 |
| 学习策略 | 14/14 | 4 | 50 | 0/2 | 10 |

学习策略在常规条件少做两次探测和四次键盘操作，零正常重输，与手工跳过规则在 16 个测试分支上的动作完全一致。两个测试布局的普通延迟与延迟尾空格在等待后截图逐对相同。主动规则在后一组通过选区看到空白并重输。学习策略跳过了获取该画面的动作，产生两个成对回退，因此未通过准入门槛。当前发布候选提供经过验收的主动规则。

CPU 审计复算出同一份 Q 表，确认训练与测试决策截图分离，并核对 75 条轨迹、211 个决策、285 次浏览器操作和 354 张分支内截图。全部 34 次探测均复原了画面。训练和测试使用 190 次逻辑定位及 23 次实际生成，设备采样显存峰值为 5309 MiB。

```bash
python scripts/audit_learned_active_probe.py runs/learned-active-probe-001
```

[结构化结果](../examples/learned-active-probe-results.json) · [冻结 Q 表](../examples/learned-active-probe-checkpoint.json) · [挑战组配对轨迹](../examples/learned-active-probe-delayed-tail/provenance.json) · [实现 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44684330024)
