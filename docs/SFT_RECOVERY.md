# 干扰按钮与真实动作前缀：冻结补充实验

## 问题与范围

首轮 smoke 在 28 条定位查询上为 23/28 对 23/28，未测到收益。其训练没有干扰按钮选择，也没有实际执行失败后的状态。
本轮收集新的场景和原始 ShowUI 的真实错误动作，比较有无上一动作前缀的同预算 SFT。
这是被请求纠正时的定位与固定控制器执行实验：不训练自主规划，也不证明模型能自主发现错误。

## 数据划分

配置 `configs/sft_recovery.json` 在挖掘前固定。旧的 1000/2000/3000 系列和此前手工规则评测 seed 不用于本轮独立验收。

| split | click-button | enter-text/click-test/focus-text | SFT 留出任务 |
|---|---|---|---|
| train | 4000–4047 | 各 4000–4007 | 无 |
| validation | 5000–5011 | 各 5000–5003 | 无 |
| test | 6000–6023 | 各 6000–6005 | click-dialog、click-tab，各 6000–6005 |

正常查询计划为 train 80、validation 28、test 60；实际纠错数量由冻结底座的错误决定，不人为补造失败。训练至少需要 8 条真实纠错，否则停止此 pilot。
enter-text 有两个正常查询：初始输入框，以及实际输入正确文字后的 Submit。后者保留真实上一动作；底座输入失败时，明确标记用于推进正常 Submit 场景的教师动作，不将其伪装成模型成功轨迹。

错误挖掘先实际执行底座的第一次预测。只有同时满足“点不在正确目标框内”和“实际目标后置条件未完成”的动作，才产生纠错样本。
无效输出、正确框内执行失败和不可见目标单独记录在完整 source cohort，不据此宣称有可执行的错误前缀。
DOM 只做离线标签和错误子集选择；模型输入仅为当前截图、目标描述、可选的原始拟执行动作，不包含奖励、正确框、错误标签或执行代码。

- 非终止错误：使用真实执行后的截图，记为 `retry_in_place`。
- 已终止错误：同 seed 重启并核验初始截图一致，记为 `restart_after_terminal`。这是带一次重启的重试，不是原 episode 内恢复。
- 纠错由评测器选择，错误检测能力不在本轮结论范围内。

原图、执行后图、重启图、目标框和原始输出保存在 source trace；图像、trace 和 split 均记录哈希。所有同场景查询归入同一 split，并拒绝相同截图跨 split。
“留出任务”仅指未参与本轮 SFT，不保证未出现在底座预训练中。

## 训练与推理消融

两个 adapter 从同一冻结 ShowUI 底座分别初始化：

| adapter | 图片/正确答案/顺序 | 上一动作输入 | 更新预算 |
|---|---|---|---|
| visual | 完全相同，包含真实错误后的截图 | 无 | 128 updates，累积 2 |
| history | 完全相同 | 仅原始 proposed action | 128 updates，累积 2 |

两组沿用 NF4、BF16 compute、decoder q/v LoRA rank 8、学习率 1e-4、相同 seed；均进行参数变更和独立重载验收。训练读取 train/validation，test 只在两组训练结束后读取，不按测试调整参数。

同一 BF16 底座加载两个 adapter，生成六个固定对照：base/visual/history × 无历史/有历史。相同提示词的重复计算可复用，实际 GPU 调用另行记录。
报告正常定位、纠错定位、各任务、新增命中、退化和无效输出。复制上一坐标作为机制对照，防止将简单重复原点当作视觉纠错。
训练输入相同而历史不同的差异，不与多训练数据、更多更新或精度变化混为一谈。

## 真实 GUI 执行验收

默认执行三组：base_plain、visual_plain、history_history。
正常控制使用每个测试任务的 6000、6001 两个 seed；纠错控制在训练前按任务/原地重试或重启模式轮转选取最多 12 条，不使用 adapter 的测试结果选样本。其余纠错仍全部保留在离线定位分母。
纠错组对三组模型重放完全相同的冻结底座错误动作，核验初始和错误后截图/终止状态，再执行一次候选修复；enter-text 最后仍由相应模型定位 Submit，不用 DOM 点击完成任务。
最多一次同 seed 重启、四次逻辑模型决策预算。固定错误前缀与离线候选的真实调用被复用，分别记录逻辑调用与新的神经网络调用，不据此宣称吞吐优势。
初始/执行后截图不匹配的样本保留并计失败，单独报告 mismatch；运行错误也不删除。
正常控制和纠错执行沿用离线 test 场景，是执行验收，不是额外独立泛化样本。

## 复现

```bash
export CUDA_VISIBLE_DEVICES=0
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers"
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-sft/bin/python scripts/collect_recovery_sft.py --config configs/sft_recovery.json --model models/ShowUI-2B --out runs/sft-recovery-data-001
.venv-sft/bin/python scripts/train_grounding_sft.py --config configs/sft_recovery.json --data runs/sft-recovery-data-001 --model models/ShowUI-2B --context-mode none --out runs/sft-recovery-visual-001
.venv-sft/bin/python scripts/train_grounding_sft.py --config configs/sft_recovery.json --data runs/sft-recovery-data-001 --model models/ShowUI-2B --context-mode previous-action --out runs/sft-recovery-history-001
.venv-sft/bin/python scripts/eval_recovery_sft.py --data runs/sft-recovery-data-001 --model models/ShowUI-2B --visual-adapter runs/sft-recovery-visual-001 --history-adapter runs/sft-recovery-history-001 --out runs/sft-recovery-eval-001
```

每个目录必须全新。GPU 0 是本次用户已分配的卡，忙时不切到其他卡。模型底座和原推理环境保持固定，数据继续使用带原始 MIT 通知的 MiniWoB 合成页面。
原有 `run --adapter` 默认不添加动作历史；本轮历史条件由上述实验执行器显式提供。不启动 RL，也不将 adapter 发布到公共模型仓库。

## 2026-11-19 实测结果

实现、配置、数据挖掘和两组训练均冻结于 `537dddb98434d776c9d38c710af352d68b4e58aa`，运行时 Git 干净。
底座为 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`。
训练后仅补充文档和示例，没有根据 test 修改提示词、训练预算或在线选样规则。
机器可读结果与哈希见 [sft-recovery-results.json](../examples/sft-recovery-results.json)。

### 数据与训练验收

实际执行 150 个底座首动作，其中 93 个达成首动作后置条件，57 个为已执行的定位错误；这不是 93 个完整任务成功。没有预测前过滤的场景。
正常查询 168 条、纠错查询 57 条，共 225 条监督对。按场景划分，无跨 split 场景或截图哈希重合。

| split | 场景 | 正常查询 | 原地纠错 | 终止后重启纠错 | 合计 |
|---|---:|---:|---:|---:|---:|
| train | 72 | 80 | 22 | 6 | 108 |
| validation | 24 | 28 | 9 | 1 | 38 |
| test | 54 | 60 | 17 | 2 | 79 |

两组分别完成 128 次更新，均仅训练 1,089,536 个 decoder q/v LoRA 参数，参数确实发生变化，独立重载后的探针 loss 差均为 0。
visual 验证 loss 为 4.4338 → 0.7140，history 为 3.8084 → 0.7188；不同上下文的 loss 不作直接优劣比较，也不能用 loss 下降代替任务收益。
两组数据、顺序和更新数相同；历史提示较长，因此并非严格等 token/FLOPs 实验。

### 六组离线定位

`plain` 不提供上一动作；`history` 在推理时提供上一动作。正常查询包含 54 个初始目标与 6 个后续 Submit；纠错查询为全部 19 条 test 错误。
下表为目标框命中数，括号内为无效输出数；定位命中不等于完整任务成功。

| 权重 / 推理上下文 | 正常 /60 | 纠错 /19 | 合计 /79 |
|---|---:|---:|---:|
| base_plain | 36 (0) | 1 (0) | 37 (0) |
| base_history | 30 (0) | 0 (1) | 30 (1) |
| visual_plain | 33 (2) | 1 (5) | 34 (7) |
| visual_history | 27 (3) | 2 (9) | 29 (12) |
| history_plain | 38 (0) | 4 (0) | 42 (0) |
| history_history | 32 (0) | 6 (0) | 38 (0) |

正常的干扰按钮选择和后续 Submit 进一步拆开：

| 权重 / 推理上下文 | 初始干扰按钮 /24 | 正常 Submit /6 |
|---|---:|---:|
| base_plain | 3 | 6 |
| base_history | 3 | 0 |
| visual_plain | 4 | 6 |
| visual_history | 4 | 0 |
| history_plain | 6 | 6 |
| history_history | 6 | 0 |

history_history 相对 base_plain，在全部纠错上新增命中 5 条、丢失 0 条，但正常查询新增 6 条、丢失 10 条。
保持历史输入相同，visual_history → history_history 在纠错上新增 5 条、丢失 1 条；保持 history 权重相同，plain → history 则新增 5 条、丢失 3 条。
这支持继续研究历史条件，却不足以证明模型可靠利用了每个历史字段。
复制上一坐标为 0/19；由于错误子集本就要求原点未命中，这只是机制下限检查，不能单独证明视觉推理能力。

明显退化出现在正常 Submit：三个提供历史的 head 均为 0/6，不提供历史均为 6/6。
示例中模型在 Submit 步重复输入框坐标，与历史坐标干扰的假设一致；尚未用新实验验证其原因。
SFT 留出的 click-tab 也从 base 的 5/6 降为 visual 的 1/6、history 的 3/6，不能宣称跨任务泛化改善。
history_plain 的离线结果更好，但本轮没有为其安排在线验收；不能据此宣称已验证某种选择性历史策略。

### 真实 GUI 执行

共 72 次执行：三组 head 各 12 个正常控制、12 个条件纠错。无运行异常、无错误前缀不匹配。
纠错组在训练前按固定规则选定，三组执行完全相同的底座错误动作；不是挑选 adapter 成功样本。

| head | 正常任务 /12 | 原地恢复 /10 | 终止后重启成功 /2 | 条件纠错合计 /12 |
|---|---:|---:|---:|---:|
| base_plain | 11 | 1 | 0 | 1 |
| visual_plain | 8 | 1 | 1 | 2 |
| history_history | 6 | 3 | 0 | 3 |

history_history 在条件纠错上相对 base 新增成功 2 条、丢失 0 条；正常控制新增 0 条、丢失 5 条。
visual_plain 在条件纠错上新增 2 条、丢失 1 条；正常控制新增 0 条、丢失 3 条。
这些是小样本描述，只有一个训练 seed，不能据此宣称统计显著改善。重启成功不算原 episode 内恢复。
正常与纠错执行复用了离线 test 场景和候选预测，不是额外独立样本。错误选择由离线评测器完成，不代表自主检测错误的成功率。

[原地纠错示例](../examples/sft-recovery/history_history/replay.html) 与 [对应 base 重试](../examples/sft-recovery/base_plain/replay.html) 保留同一实际错误前缀。
[正常任务退化示例](../examples/sft-history-regression/history_history/replay.html) 与 [对应 base 成功](../examples/sft-history-regression/base_plain/replay.html) 保留 Submit 反例。
示例在结果后选择，仅用于说明现象；总分母和失败记录完整保留，没有用示例计算成绩。

### 资源、证据与结论

GPU 0 可见配额 12032 MiB；约每 2 秒采样，共 492 点，整轮设备显存采样峰值 5419 MiB。
这是采样值，不是瞬时峰值保证。两组训练各约 103 秒；评测实际新增神经网络调用 318 次（包括 Submit，重复提示复用另行记账）。
评测 PyTorch allocated/reserved 峰值分别为 4,646,364,672 / 5,307,891,712 字节，与设备占用分开记录。

本地审计重新核对 309 张 source PNG、474 组点预测和 189 张在线截图的哈希、提示与分数，以及真实前缀、重载和更新记录。
在线成功在执行时使用核心 MiniWoB 成功规则；本轮轨迹保存奖励和终止标志，未完整保存原始 task_info，离线审计不扩大为对其全部字段的复验。
Linux 85 项测试、本地 74 项单元测试和 11 项 Chromium 集成测试通过；[冻结实现 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44469773939) 的 Ubuntu 22.04、24.04 与打包检查通过。

本轮验收结论是：单卡训练、真实错误收集和同预算纠错消融已可运行；部分纠错改善与正常能力退化同时存在。
两组 adapter 保持实验状态，不替换默认模型，不开启 RL。下一轮可用新的冻结划分研究正确/错误历史混合监督与 Submit 保持；这仍是待验证方案。
发布 Alpha、默认行为和模型底座均保持原版本，训练权重只保留为用户本地实验产物。

对应 Git 阶段：实现提交 `537dddb` 冻结后运行；结果以独立文档提交保存。可通过以下命令检查范围与版本：

```bash
git switch feat/sft-distractors-recovery
git show --stat 537dddb
git diff 537dddb --name-only  # 之后只补文档与脱敏示例
```
