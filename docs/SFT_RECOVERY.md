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
