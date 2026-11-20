# 在线强化学习恢复策略

训练在真实 MiniWoB 浏览器中采样。ShowUI 提供原始定位和重试定位，始终使用冻结的底座。学习器在每个训练回合结束后更新有限动作的 Q 值，下一回合使用更新后的数值进行探索。策略输入为截图核对结果、是否已等待和动作掩码。精确任务终止奖励来自浏览器，环境的真实输入值和注入类型仅留在实验记录中。

动作包含继续、等待一次、固定裁剪重输。回报为精确提交奖励减去等待成本 0.02 和重输成本 0.05。学习按可用动作首次探索，随后以 0.2 的概率继续探索。每个布局和故障按预定随机顺序执行两遍。训练有 3 个布局，独立测试有 2 个布局，各自覆盖七种条件。训练结束后保存并锁定 Q 表，测试比较先等的文字存在规则、字形规则与冻结的 Q 策略，共 42 条训练和 42 条测试轨迹。

升为推荐策略的条件包括保留字形规则的所有测试成功、正常输入零重输，以及平均成本调整回报的提升。末尾空格与正确输入可能产生相同截图，策略无法通过这类画面区分它们。成功恢复末尾空格可能来自对整类相同画面的预防性重输，因此报告同时保留正常组的重输成本。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_online_visual_recovery_rl.py \
  --config configs/online_visual_recovery_rl.json --profile formal --model models/ShowUI-2B \
  --prior-plan runs/pre-rl-recovery-001/splits.json \
  --prior-plan runs/recovery-offline-rl-001/splits.json \
  --prior-plan runs/idempotent-input-repair-001/preflight.json \
  --prior-plan runs/visual-text-content-002/preflight.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --exclude-run runs/online-visual-recovery-rl-smoke-001 \
  --out runs/online-visual-recovery-rl-001
```

先用 `--profile smoke`、不传 `--exclude-run`，在新目录验证采样、学习、冻结和评测。工程布局与正式布局隔离。正式训练或评测输出目录每次运行使用新路径。
