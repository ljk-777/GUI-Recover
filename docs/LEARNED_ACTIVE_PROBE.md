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
