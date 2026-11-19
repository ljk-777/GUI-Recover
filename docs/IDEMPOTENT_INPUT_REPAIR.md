# Idempotent replacement for input recovery

In the previous offline Q study, dropped input and deferred input had identical screenshots at the point where a recovery decision was needed. An immediate retry decided to append based on that screenshot. When the queued input arrived before the append, the field contained two copies and Submit failed.

This change adds the opt-in `RecoveryDecisionEnv("enter-text", grounder, input_repair="replace")` profile. It uses the same original target text and grounded field location. It releases any pending fixture action first, checks that the episode remains live, then clicks the grounded point, selects existing text and types the original value. Repeating the operation at the supported field yields the same text. Reward, actor prompt, visual decisions, wait budget and retry budget are shared with the existing environment. The new operation is scoped to the single editable field task. A successful coordinate prediction remains a dependency; the implementation does not certify arbitrary page targets.

## Frozen comparison

The test takes six fresh Enter Text layouts from a registered reserve and excludes prior initial screenshots from the supplied corpora and two prior split registries. On each layout with normal, dropped and delayed delivery it executes three paired strategies:

- **Legacy Visual** and **Replace Visual** share the same visual trigger and fixed zoom grounding. The two strategies differ in the input operation performed once a retry is selected.
- **Learned Wait** uses the Q table frozen in the previous experiment and the legacy executor. Its checkpoint is checksum-locked.

Separate partial and wrong input stress branches force the same retry in each executor. They check execution once a location is known. The visual trigger infers that text is present and does not verify the text itself, so these stress results cannot be used to claim automatic detection of wrong content. All branches are actual browser executions and complete with the same actor-guided Submit and exact MiniWoB termination rule.

The predeclared gate requires all baseline normal and dropped successes to be preserved, an improvement over Legacy Visual on delayed input, no delayed loss against Learned Wait, no explicit wait on delayed replacement, and no regression in either forced-retry stress condition. The report separates number of waits, edits, logical requests and cost-adjusted return. Cost constants are unchanged from the previous experiment. The delay and release schedule come from the local fixture and do not provide a real latency model.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_idempotent_input_repair.py \
  --config configs/idempotent_input_repair.json --profile test \
  --model models/ShowUI-2B --q-policy examples/recovery-offline-rl-policy.json \
  --prior-pre-rl runs/pre-rl-recovery-001/splits.json \
  --prior-rl runs/recovery-offline-rl-001/splits.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/idempotent-input-repair-001
```

## 正式结果

冻结实现为 `45e60d4`。六个新布局通过预检，各布局执行相同的配对条件。总计 78 条实际浏览器轨迹。

| 策略 | 正常 /6 | 漏输入 /6 | 延迟输入 /6 | 延迟组显式等待 | 延迟组逻辑模型调用 |
|---|---:|---:|---:|---:|---:|
| 旧视觉触发＋原执行器 | 6 | 6 | 0 | 0 | 18 |
| 冻结学习策略＋原执行器 | 6 | 6 | 6 | 6 | 12 |
| 旧视觉触发＋替换执行器 | 6 | 6 | 6 | 0 | 18 |

替换方案通过预设门槛。延迟组中，两种视觉方案共享完全一致的修复输入和坐标。旧执行器在原文字送达后又追加了一份，造成六次提交失败；替换方案用原始目标文本覆盖，六次提交均成功。冻结学习策略等待看到文字后直接提交，省下六次重试和六次逻辑定位请求。若确实漏输，替换方案比学习策略少等待六次，两者的成功数与定位请求数相同。这里比较的是显式等待和逻辑请求数量，原输入排队由夹具模拟。

按原实验成本计算，延迟组替换平均回报 0.95、学习等待 0.98；漏输入组分别为 0.95 和 0.93。两种方案适用于不同成本偏好。在部分输入、错误输入的强制重试组，新旧执行器均分别恢复 6/6。两组验证的是修复执行，视觉触发仍无法读出文本是否完全正确。[延迟输入三路径回放](../examples/idempotent-input-repair-pair/provenance.json)

独立审计复算了 168 个决策、234 次浏览器操作，核对 24 对相同的修复输入以及原始终局奖励。正常组原有成功全部保留。单元和浏览器测试验证同一字段连续替换两次仍等于原始目标文本。210 次逻辑模型请求共触发 42 次冻结生成，显存采样峰值 5309 MiB。两套 Ubuntu 测试和打包检查通过。[结构化结果](../examples/idempotent-input-repair-results.json) · [冻结实现 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44556083291)
