# Offline Q-learning for bounded input recovery

This is the first learning experiment on the recovery decision interface. It fits a small Q table on CPU from actual MiniWoB trajectories. Frozen ShowUI supplies the source click, grounded retry locations and Submit. The policy learns the decision to continue, wait or retry. The learning target and the deployed policy exclude DOM, source labels, seed and the type of intervention. Its state uses the previous visual text signal and whether it has waited.

## When waiting can help

The new controlled delay fixture holds the first INPUT in a queue. A wait releases the input. Submitting early uses the visible field as it is, so it can submit empty text. Retrying before a wait may release the queued input and then append another copy. This makes incomplete delivery and dropped input look the same initially, while their next observations can differ. It simulates delivery in a local browser fixture.

The environment continues to return +1 for exact successful browser termination and zero otherwise. The learning target subtracts 0.02 per wait and 0.05 per retry. This cost is separate from the native task reward. The horizon allows at most one wait and one retry. Fitting first averages terminal outcomes at the post-wait stage, then uses those values in one-step Bellman targets for the initial decisions. It collapses identical repeated wait prefixes within each layout and condition so the three later actions do not triple the importance of waiting. Unseen states use the prior visual rule.

## Protocol

Each registered input layout is tested with normal, dropped and deferred delivery. The training set has three fresh layouts. The validation set has one additional layout and the held-out test has four more. Before inference, a reset-only preflight rejects exact screenshot matches to registered prior data. No cases are selected from test outcomes.

Each train layout/condition executes six branches with the same source proposal: continue, plain retry, fixed zoom, and wait followed by each of those actions. The Q checkpoint and training manifest hashes are saved before validation or test execution. Validation records the learned and wait-first rule outcomes; no fitting step follows validation. The test compares Continue, the previous Visual Fixed Zoom rule, the predeclared Wait First rule, and the frozen learned Q. The Wait First rule uses the same visual signal as the learner and waits once when the text signal is uncertain or missing.

The effect gate requires all clean and dropped case successes from the original Visual Fixed Zoom to be preserved, and at least one additional successful deferred case without a paired deferred regression. The wait-first baseline remains a separate comparison even if the gate passes. This experiment tests only the fixed-controller enter-text task.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_recovery_offline_rl.py \
  --config configs/recovery_offline_rl.json --profile test --model models/ShowUI-2B \
  --previous-splits runs/pre-rl-recovery-001/splits.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/recovery-offline-rl-001
```

## 正式结果

实现冻结在 `5120cfa`。预检登记了 8 个互不重复的输入布局，训练、验证和测试分别使用 3、1、4 个。每个布局分别执行正常、漏输入和延迟送达。54 条训练轨迹得到 81 条复查决策，去除 18 条重复等待前缀后，余下 63 条状态动作结果参与拟合，产生 14 个非空 Q 表项。[冻结策略检查点](../examples/recovery-offline-rl-policy.json)

| 视觉状态 | 学到的决策 | Q 值 |
|---|---|---:|
| 首击后看到文字 | 继续 | 1.000 |
| 首击后没看到文字 | 等待 | 0.955 |
| 等待后看到文字 | 继续 | 1.000 |
| 等待后仍为空 | 原图重试 | 0.950 |

首击后为空时，立即原图重试和固定裁剪重试的 Q 值均为 0.450。等待后为空时，两种重试的值相同，按预先定义的动作顺序选择原图重试。测试的 20 个复查决策全部命中训练所得的表项。

| 策略 | 正常 /4 | 漏输入 /4 | 延迟输入 /4 | 延迟条件下等待 / 重试 / 逻辑模型请求 |
|---|---:|---:|---:|---:|
| 继续 | 4 | 0 | 0 | 0 / 0 / 8 |
| 原视觉规则 | 4 | 4 | 0 | 0 / 4 / 12 |
| 人工先等规则 | 4 | 4 | 4 | 4 / 0 / 8 |
| 学到的 Q 策略 | 4 | 4 | 4 | 4 / 0 / 8 |

学习策略通过了预设效果门槛。延迟组新增 4 个成功，正常与漏输入组各保留原有的 4 个成功。正常组无额外等待或重试。漏输入组多等一次，按预设成本计算的平均回报为 0.93，原视觉规则为 0.95。延迟组的平均回报由 −0.05 提高到 0.98。学习策略与人工先等规则在这批场景的任务成功数、平均回报及模型调用数相同；两者在漏输入后分别使用原图和固定裁剪。三种条件共享同一批 4 个独立测试布局。

独立审计证实，8 个布局中漏输入和延迟条件的首击后截图逐对相同。旧视觉规则在测试中出现 4 次重复输入导致的提交失败，学习策略等待后见到文字才继续。配对示例记录了原视觉规则的重复输入和学习策略的成功提交。[原生轨迹示例](../examples/recovery-offline-rl-delay/provenance.json)

Q 表从训练索引独立重算，冻结 SHA 在验证和测试完成后保持一致。108 条实际轨迹的 263 个决策和 327 次浏览器操作均通过审计，训练和测试的决策截图无哈希重叠。270 次逻辑定位请求复用为 52 次实际生成。Q 拟合使用 CPU，冻结 ShowUI 推理的 GPU 显存采样峰值为 5309 MiB。两套 Ubuntu 测试和打包通过。[完整汇总](../examples/recovery-offline-rl-results.json) · [冻结代码 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44551846076)
