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

训练结束后，CPU 可以从训练索引和轨迹独立重算探索动作与 Q 表，核对参考字形、训练索引和冻结检查点的哈希。复算只读取训练轨迹。

```bash
python scripts/replay_online_recovery_q.py runs/online-visual-recovery-rl-001
```

## 首轮正式结果

冻结源码为 `0efc0f2`。训练布局为 71000、71001、71002，测试布局为 72000、72001。42 个在线训练回合中有 36 个获得精确终止奖励。训练轨迹可在 CPU 上重算全部 90 个动作决策和 13 个 Q 表项，复算出的检查点与冻结文件一致。测试策略的 14 个复查决策都命中了训练得到的 Q 表。

| 测试条件 | 文字存在规则 | 字形规则 | 在线 Q |
|---|---:|---:|---:|
| 正常 | 2/2 | 2/2 | 2/2 |
| 延迟输入 | 2/2 | 2/2 | 2/2 |
| 漏输入 | 2/2 | 2/2 | 2/2 |
| 少输 | 0/2 | 2/2 | 2/2 |
| 多输一个 `!` | 0/2 | 2/2 | 2/2 |
| 首字母大小写错误 | 0/2 | 2/2 | 2/2 |
| 末尾多一个空格 | 0/2 | 0/2 | 2/2 |
| **合计成功** | **6/14** | **12/14** | **14/14** |
| 等待次数 | 4 | 4 | 0 |
| 重输次数 | 2 | 8 | 14 |
| 逻辑定位请求 | 30 | 36 | 42 |
| 平均成本调整回报 | 0.416 | 0.823 | 0.950 |

在线 Q 在 14 条测试轨迹的全部 28 个决策上与无条件固定裁剪重输一致。两组末尾空格样本的输入后截图都与正常样本逐字节相同。Q 策略将两种场景都重输，因此新增两次精确提交成功，同时使两个正常输入都发生多余重输。该检查点未通过正常输入零重输门槛，发布候选继续使用原有规则。

两套留出布局被重复用于七种条件。当前平均成本调整回报反映这七种条件的等权频率和给定动作成本。独立审计复核了 84 条轨迹、182 个决策和 246 次浏览器操作，训练、测试决策截图无哈希重叠。训练和评测共 227 次逻辑定位、44 次实际模型生成，RTX 4090 D 设备采样显存峰值 5309 MiB。

[结构化结果](../examples/online-visual-recovery-rl-results.json) · [冻结 Q 表](../examples/online-visual-recovery-rl-checkpoint.json) · [同截图的配对轨迹](../examples/online-visual-recovery-rl-space-pair/provenance.json) · [CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44672468787)
