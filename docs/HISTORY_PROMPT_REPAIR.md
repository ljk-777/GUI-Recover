# 历史提示修复：先冻结权重，再做新场景验收

## 问题与冻结范围

上一轮 history adapter 在部分真实错误上改善，但正常 Submit 定位从 6/6 降为 0/6；无历史的同一权重仍为 6/6。
本轮先检验“历史坐标/提示顺序干扰当前目标”的假设。原始底座、两个既有 adapter、处理器、图片预算和生成预算保持一致；不追加训练。
提示修复若已达到验证门槛，就先验收这一最小改动；平衡正常/错误历史的数据重训作为后续独立实验，不混入本轮效果。

沿用 `sft-recovery-data-001` 的 train/validation；选择器只读取原 validation 的 38 条记录，不读取 test。
验收配置在运行前固定为 `configs/sft_history_holdout.json`：新 test 使用 8000 系列，从未参与前两轮的训练、验证或测试。
其中 click-button 32 场景、enter-text 12 场景，其余四任务各 6 场景，共 68 场景，计划 80 个正常查询。错误数量由冻结底座的实际执行决定。
新 test 的生成复用原开发集原始字节，原 test 的行、PNG 和 trace 不复制；审计拒绝新 test 与全部旧场景或旧样本截图重合。

## 固定提示对照

同一 base 和同一 history adapter 分别运行五种提示：

| profile | 原理 | 信息量 |
|---|---|---|
| plain | 无上一动作 | 纯截图和目标 |
| legacy | 旧的目标、上一动作、Recheck 三行 | 完整上一动作 |
| target-last | 仅把旧三行中的目标移到最后 | 与 legacy 相同 |
| current-target | 明确区分 Historical action context 和 CURRENT target | 完整上一动作，改措辞 |
| action-only | 同 current-target，但去除上一坐标 | 信息删除诊断，不算等信息对照 |

只接受严格验证的原始 action/value/position。任务奖励、错误标签、目标框、DOM 和执行代码不能进入提示。
所有 profile 在没有上一动作时保持原始纯目标提示；legacy 的默认字节与上一轮相同。

## 验证集选择与一次测试

候选是 history 权重的 target-last、current-target、action-only，顺序预先固定。
候选正常查询和 Submit 的命中数必须分别不低于 base_plain、history_plain 两个对照的较高值。
在达标者中按纠错命中、正常命中、较少无效输出、固定候选顺序选择；无候选达标则回退 history_plain，并明确标记提示修复未通过。
该门槛用于防止只优化纠错；不构成正式统计非劣检验。

选择文件记录验证预测、配置、验证 split、adapter 和代码的哈希。新 test 评测前重新计算选择并核对哈希，拒绝人为更换候选。
test 只运行 base_plain、history_plain、history_legacy 和选定 head（去重），不在 test 上重新挑提示。
与原 data 一致：只有真实执行错误、失败后置条件且原点未命中，才生成条件纠错；终止后同 seed 重启单独记账。

真实执行正常控制使用每任务 8000–8003，共 24 条；条件纠错在模型候选运行前，按任务/模式轮转固定最多 16 条。
执行同一错误前缀，最多一次重启、四次逻辑决策。完整保留 task_info、执行代码、错误和截图；无效预测、状态不匹配和运行异常不从分母删除。
错误选择仍由离线评测器完成，不代表自主检测；固定控制器负责目标切换和 INPUT→Submit，不训练自主规划。

## 运行与验收

```bash
export CUDA_VISIBLE_DEVICES=0
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers"
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 首先仅在开发集选择，两个 adapter 不变
.venv-sft/bin/python scripts/eval_history_prompts.py --stage validation --config configs/sft_history_holdout.json --reference-data runs/sft-recovery-data-001 --data runs/sft-recovery-data-001 --model models/ShowUI-2B --visual-adapter runs/sft-recovery-visual-001 --history-adapter runs/sft-recovery-history-001 --out runs/history-prompt-validation-001

# 之后挖掘新的验收场景，复用原开发集字节
.venv-sft/bin/python scripts/collect_recovery_sft.py --config configs/sft_history_holdout.json --reuse-development-from runs/sft-recovery-data-001 --model models/ShowUI-2B --out runs/history-prompt-data-001

# 选择固定后才允许读取新 test
.venv-sft/bin/python scripts/eval_history_prompts.py --stage test --config configs/sft_history_holdout.json --reference-data runs/sft-recovery-data-001 --data runs/history-prompt-data-001 --model models/ShowUI-2B --visual-adapter runs/sft-recovery-visual-001 --history-adapter runs/sft-recovery-history-001 --selection runs/history-prompt-validation-001/selection.json --out runs/history-prompt-test-001
```

新建分支 `feat/target-first-history` 隔离实验。实现和配置先提交，再运行，之后只提交结果文档与脱敏示例。
原 Alpha、默认 run 和既有实验记录保持原版本。只有用户已分配 GPU 0 空闲时运行；训练权重不进入 Git。
