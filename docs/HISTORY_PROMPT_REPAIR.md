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

## 第二阶段：依据原验证集冻结的平衡重训

第一阶段的 5000 系列 validation 显示：三种新提示均将 Submit 从 0/4 恢复至 4/4。
但旧 history 权重正常总分均为 16/28，低于 base 的 17/28，未通过完整门槛；原选择器诚实回退 history_plain。
旧权重的纠错在 legacy 下 1/10，新提示或无历史均为 0/10。第一阶段不宣称修复已全面成功。

因此冻结 `configs/sft_history_balanced.json`，只根据上述 validation 设计重训，不参考 8000 系列成绩。
平衡版本仍只有原来的 72 个训练场景，不挖掘新 test 监督：

- 初始正常目标 72 条，各训练一次无历史上下文。
- 28 条真实纠错，各配对训练无历史和 target-last，共 56 条。
- 8 条正常 Submit，各配对训练两种上下文，并重复四次，共 64 条。

合计 192 条训练记录；重复和上下文配对不冒充新增场景。原始上一动作、图片、答案和 trace 不改写，显式 `history_input` 只控制是否把历史送入模型。
验证仍是原 38 条。训练数据目录故意不含 test；训练器不读取 8000 场景。
三组 seed 固定为 20261119、20261120、20261121，每组 128 次更新，其他 QLoRA 参数沿用上一轮，历史格式固定 target-last。
主 seed 在训练前固定为 20261119，不按验证或 test 挑最好 seed；所有三组结果均保留。
这是训练分布与提示的联合修正，不能将全部差异归因于某个历史字段。

三组的有/无历史正常与 Submit 都要守住原验证门槛，纠错要达到旧历史两种提示的较高值，才标记全组通过。
无论是否通过，都在原先已经冻结的 8000 系列上报告全部三组；不根据该 test 追加训练或修改配方。
在线只验收主 seed 的有/无历史、base_plain 和旧权重 target-last，共四组。其他种子只做离线复核。

```bash
.venv-sft/bin/python scripts/build_balanced_history_data.py --config configs/sft_history_balanced.json --reference-data runs/sft-recovery-data-001 --out runs/history-balanced-data-001
for seed in 20261119 20261120 20261121; do
  .venv-sft/bin/python scripts/train_grounding_sft.py --config configs/sft_history_balanced.json --data runs/history-balanced-data-001 --model models/ShowUI-2B --context-mode record-action --history-profile target-last --seed "$seed" --out "runs/history-balanced-$seed"
done
```

`scripts/eval_balanced_history.py --stage validation` 先写入带配置、checkpoint 和预测哈希的验证门槛文件；`--stage test --validation-eval ...` 核验后才读取 test。
完整参数见该脚本 `--help`。本阶段不启动 RL，不自动将任何 checkpoint 设为默认。

## 2026-11-19 实测与结论

第一阶段冻结于 `ec8b086d64b4c40c3d1fbe4e17a4b0037fd27ecc`；平衡训练阶段冻结于 `e668e06babce47c84311d79085673d794d73b981`。
第二阶段配方在读取 8000 成绩前依据原 validation 固定；三个模型全部训练完成并固定权重后才读取第一阶段新 test 成绩。
训练、验证、选样和种子均未按 test 结果修改。后续提交只补充结果和脱敏示例。
机器可读完整记录：[提示实验](../examples/history-prompt-results.json)、[三种子平衡训练](../examples/history-balanced-results.json)。

### 提示顺序修复了什么

原 validation 上，不更改任何权重，仅重排 legacy 的三行就恢复了正常 Submit。不同表示的对照如下：

| profile | base 正常 /28 | 旧 SFT 正常 /28 | base Submit /4 | 旧 SFT Submit /4 | base 纠错 /10 | 旧 SFT 纠错 /10 |
|---|---:|---:|---:|---:|---:|---:|
| plain | 17 | 16 | 4 | 4 | 0 | 0 |
| legacy | 13 | 12 | 0 | 0 | 0 | 1 |
| target_last | 17 | 16 | 4 | 4 | 0 | 0 |
| current_target | 17 | 16 | 4 | 4 | 0 | 0 |
| action_only | 17 | 16 | 4 | 4 | 0 | 0 |

三种新提示在旧 SFT 权重上正常总分仍为 16/28，低于原模型 17/28，因此完整门槛未通过。
选择器按预设规则回退 `history_plain`，没有因 test 表现更换候选。
修复 Submit 不等于整体模型达标；结果也不证明内部注意力机制或每个历史字段都被正确使用。
第二阶段固定采用信息不变、改动最小的 target-last；action-only 没有被当作等信息对照。

### 新场景离线定位

8000 系列共 68 个新场景，80 条正常查询、30 条实际执行错误后的查询；无与旧 cohort 场景或旧样本截图重合。
30 条纠错中 28 条原地重试、2 条终止后重启。无预测前过滤。正常 Submit 是正常查询的子集。
下表为严格 teacher 框命中；无效输出仍在分母中：

| head | 正常 /80 | Submit /12 | 纠错 /30 | 全部无效输出 /110 |
|---|---:|---:|---:|---:|
| base_plain | 44 | 12 | 2 | 0 |
| history_plain | 46 | 12 | 6 | 0 |
| history_legacy | 36 | 2 | 7 | 0 |
| history_target_last | 46 | 12 | 4 | 0 |
| balanced_20261119_plain | 41 | 11 | 5 | 2 |
| balanced_20261119_target_last | 42 | 12 | 6 | 1 |
| balanced_20261120_plain | 29 | 0 | 4 | 36 |
| balanced_20261120_target_last | 41 | 12 | 6 | 16 |
| balanced_20261121_plain | 31 | 1 | 4 | 34 |
| balanced_20261121_target_last | 42 | 12 | 6 | 14 |

旧权重使用新顺序的 Submit 为 12/12，旧提示仅 2/12；这在新场景上复现了提示干扰。
三个平衡 seed 的 target-last 均恢复 Submit 到 12/12，但正常定位仍低于 base，部分 seed 的无效输出明显增多。
无历史下的平衡模型也未稳定保持正常能力。重复与上下文配对没有解决整体退化，不能将 loss 下降解释为任务提升。

### 实际 GUI 执行

第一阶段 120 次执行，第二阶段 160 次执行，均为相同 24 个正常和 16 个条件纠错场景的受控重复，不是 280 个独立场景。
条件纠错的 16 条在候选预测前按固定规则选择，其中 14 条原地、2 条终止后重启。错误检测依赖离线评测器，目标切换由固定控制器负责。

| 模型 / 提示 | 正常任务 /24 | 原地恢复 /14 | 重启后成功 /2 | 条件纠错合计 /16 |
|---|---:|---:|---:|---:|
| base / plain | 17 | 2 | 0 | 2 |
| 旧 SFT / 无历史（预设回退） | 17 | 6 | 1 | 7 |
| 旧 SFT / legacy | 14 | 5 | 0 | 5 |
| 旧 SFT / target-last | 17 | 5 | 0 | 5 |
| 平衡主 seed / 无历史 | 17 | 6 | 1 | 7 |
| 平衡主 seed / target-last | 18 | 7 | 0 | 7 |

预设无历史回退相对 base 在条件纠错上新增成功 5 条、丢失 0 条；正常总分虽持平，逐例新增 2 条、丢失 2 条，不能称为完全无回归。
平衡主 seed 的 target-last 在本次在线子集达到 18/24 正常、7/16 条件纠错，7 条均为原地成功；相对 base 的正常逐例新增 3 条、丢失 2 条。
这是局部收益，不能覆盖三 seed 验证失败、严格定位下降和输出格式不稳定的结果，也不据此重新选择 seed。
终止后重启成功不计为原 episode 内恢复；其他两个 seed 只有离线复核，没有在线验收。
新鲜 test 仅用于本轮验收，后续继续迭代必须另外冻结独立验收场景。

### 训练和证据验收

三个 seed 都完成 128 次更新、1,089,536 个 decoder q/v LoRA 参数，参数确实变化，独立重载探针 loss 差均为 0。
但三个 seed 均未通过完整 validation 门槛。主 seed 在训练前固定，没有挑成绩最好的 seed。

| seed | validation loss 前 → 后 | 重载 loss 差 | 权重 SHA256 |
|---|---|---:|---|
| 20261119 | 3.8581 → 0.7376 | 0.0 | `4ae35782be305e8f0ab3d26057f31e8742405a807c8a0316f6718432bd782fdb` |
| 20261120 | 3.8581 → 0.7411 | 0.0 | `9b7596fcf483fb7f3d7bb432676219e22c38bb2df6c7d4b2af87c50c75afb3e4` |
| 20261121 | 3.8581 → 0.7247 | 0.0 | `c74e58dd01091a66dc1668b31a226af3efd5fc695e267eca0d081e6b306b0bcc` |

第一阶段审计 337 张 source PNG、710 组预测和 306 张在线截图；
第二阶段审计 1480 组预测和 407 张在线截图，重新计算严格框分数并核对全部 checkpoint、配置、前缀和成功规则。
本轮动作轨迹完整保存 task_info、执行代码与截图；成功审计使用原始 task_info，而非仅奖励与终止位。
两阶段均无运行错误或前缀不匹配。另以真实处理器在 CPU 核对五种训练输入的 assistant-only 掩码及历史开关，最长代表样本 378 tokens，未截断图像。

GPU 0 配额 12032 MiB；第一阶段约每 2 秒采样 348 点，峰值 5419 MiB；
第二阶段 608 点，峰值 5433 MiB。采样峰值不保证覆盖瞬时峰值，PyTorch allocated/reserved 另见 JSON。
服务器 96 项测试通过；[冻结实现 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44476901245) 的 Ubuntu 22.04、24.04 和打包均通过。

少数边界点击出现严格框外而实际 GUI 成功，具体记录见两个 JSON 的 environment_success_outside_strict_teacher_box。
保留两个指标的原始定义，不按观察到的 test 边界事后增加容差或调整成绩。定位命中与环境成功不能互相替代。

### 保留的案例与采用决定

[固定 Submit 场景](../examples/history-submit/provenance.json) 同时保存 base、旧提示、新顺序、平衡主 seed 的原图和轨迹。
[旧 SFT 无历史回退重试](../examples/history-fallback-recovery/fallback/replay.html) 与对应 base 共享真实错误前缀；该示例明确为终止后重启成功，不能当作原 episode 内恢复。
平衡模型的正常退化案例也保留在 examples/history-balanced-regression/。案例在结果后选择，仅解释现象，不用于选样或计分。

本轮采用决定：保留提示修复与可复现评测工具；预设无历史回退作为实验候选保留，平衡重训 checkpoint 不采用为默认。
默认仍是原模型，已发布 Alpha 不改动，不开启 RL。新模型与全部负面结果仅作为可复核的研究产物保留。
后续优先增加真实、多样的正常目标切换训练场景，并独立验证输出格式稳定性；本轮的重复采样不能替代这类数据。

Git 验收可用以下命令检查两个冻结阶段与后续改动范围：

```bash
git switch feat/target-first-history
git show --stat ec8b086
git show --stat e668e06
git diff e668e06 --name-only  # 之后只补文档和脱敏示例
```
