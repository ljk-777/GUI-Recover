# INPUT 执行前的视觉目标检查

**实验模块，未通过正常任务无退化的验收门槛，默认关闭。** 两轮模型验收均发现误拦；本模块用于复现和研究该限制，不作为推荐的恢复策略。

## 目标和机制

v3 事后检查无法撤销已经点到 Submit 而触发的任务失败。可选 `--input-precheck visual` 在 INPUT 执行前，显式调用 `inspect_input_target(screenshot, proposed_position, version="v2")`。函数的缺省版本保留为 v1，CLI 则在启用时默认选择 v2。
它复用 v3 的均匀底色区域识别，要求预测点在识别出的区域内，且区域没有可见文字笔画。信号不使用 DOM、奖励、任务答案、主题名或故障标签。

这是空白矩形的外观检查，不能证明控件可编辑、唯一或确实是任务要求的输入框。无文字按钮或其他空白矩形可能通过；低对比、占位文字、已有值、渐变、不同缩放仍可能返回 unknown。
截图也只反映观测时刻，检查与真实点击之间仍可能发生界面变化。

- 通过：执行原始候选坐标，不吸附或改成区域中心。
- unknown：候选不执行，最多请求一次重新定位。用同一张当前截图和固定改写查询，让 ShowUI 给出新坐标。
- 再次 unknown：以 `input_target_unverified` 停止，不执行候选。

改写查询不含 DOM、环境结果、目标真值坐标或注入信息；成功后保留该查询用于同一 episode 后续 INPUT，仍每次重新调用模型定位。重置 episode 后清除全部状态。
默认关闭；仅供明确标记的 `showui-scripted`、`enter-text` 使用，可独立启用或与事后 input-guard 组合，不与盲目追加 input-retry 混用。
启用时默认使用 `--input-precheck-version v2`。v1 要求点在区域内再缩进两像素，已发现会误拦原本可输入的靠边点；v2 只要求点在区域内，不向区域外扩张。两版共享同一笔画检测裁边、改写查询和预算。

## 预算与记录

被拦截的候选占一次 policy 决策和一次模型调用，不执行环境 step，不写入已执行动作历史，不消耗执行层故障注入。
每个 `predict` 最多调用一次模型；`--max-steps 4` 仍限制最多 4 次决策，因此最多 4 次模型调用。事前定位修复加事后漏输入修复可在同一预算内完成，不能隐藏额外推理。
`input_precheck.regrounds` 记录重新定位请求数，若决策预算已经用完，请求不会变成额外模型调用；真实调用以 `model_calls` 为准。
`pre_execution_check` 保存候选、信号、决策、当前截图文件和检测耗时；episode 单独累计 `precheck_seconds`，包含在总耗时中，不混入模型推理耗时。

```bash
gui-recover run --policy showui-scripted --model models/ShowUI-2B --tasks enter-text --seeds 70 --max-steps 4 --parser mixed --input-guard-version v3 --recovery input-guard --input-precheck visual --out runs/precheck-demo
```

## 冻结验收协议

开发使用合成图像和 Chromium seed 7 的诊断 grounder：首次故意给出 Submit 位置，随后返回输入框位置；另测持续错误和漏输入。该 grounder 为 DOM 特权测试替身，其成功不算 ShowUI 成绩。
模型实验前冻结实现、阈值、改写查询、一次重新定位上限与总决策预算。配置为 `configs/input_precheck.json`：

- 新 seeds 70–74：默认/深色 × 正常/漏输入 × precheck off/visual，共 40 个 episode；每个条件两组都启用相同的 v3 事后检查。
- 已知开发回归 seed 60：深色 × 正常/漏输入 × off/visual，共 4 个 episode，单独报告，不混入新 seed 成绩。

上述首轮冻结版本为 `2365889`、precheck v1。seeds 70–74 暴露靠边点误拦后，只修改候选点是否在区域内的规则并版本化；首轮结果保留。
v2 的独立验收配置为 `configs/input_precheck_v2.json`：新的 seeds 80–84 共 40 个 episode，另将已见 seeds 60、70、73 的 12 个回归 episode 单独报告。不能将 seeds 70–74 再当作修复后的独立验收集。

用 `compare-precheck` 检查 off/visual 配对：除开关和输出目录外，任务、seed、预算、模型、事后恢复、代码与环境均相同；初始截图与第一次原始预测一致。
执行前干预可能改变后续候选坐标，因此不宣称后续注入前缀完全相同，也不放宽原 `compare` 的注入前缀规则。
成功提升同时涉及条件拦截和改写查询，不单独归因于检测器精度。新的 seed 若没有触发错误，不据此宣称测得错误恢复能力。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_matrix.py --config configs/input_precheck.json --model models/ShowUI-2B --out runs/input-precheck-001
.venv/bin/gui-recover compare-precheck --baseline runs/input-precheck-001/dark-clean-off --checked runs/input-precheck-001/dark-clean-on --out runs/input-precheck-001/dark-clean-comparison.json
```

结果保留失败、unknown、未注入样本、实际调用和执行步数。离线审计重算事前/事后信号，DOM 只标注候选是否落在输入框及输入内容真值，不回流至策略。

## 首轮 v1 的负面结果

冻结实现为 `2365889`，新 seeds 70–74；两组都保留相同 v3 事后检查，只切换执行前检查：

| 条件 | precheck 关闭 | precheck v1 |
|---|---:|---:|
| 白底，正常输入 | 5/5 | 3/5 |
| 白底，整段漏输入 | 5/5 | 3/5 |
| 深色，正常输入 | 5/5 | 3/5 |
| 深色，整段漏输入 | 4/5 | 2/5 |

已见 seed 60 的两个独立回归对照（正常/漏输入）均为关闭 0/1、开启 0/1；开启后每次两次模型调用、零次环境动作，阻止了误点击，但未完成任务。
改写查询仍返回 Submit 区域，不能把停止包装成恢复成功。

44 个 episode 的 123 张轨迹截图哈希全部核验，22 对初始截图与首次预测匹配。
离线重放 22 个开启检查的 episode，初始截图、最终成败和 39 次事前信号均复现；其中 16 次候选实际在空输入框内却被拒绝，未观察到接受输入框外候选。
这些是检查次数，不是独立 episode 数。出现文字的 17 次事后检查也全部复现。

退化来自 v1 将文字检测的内侧裁边误用于点击位置判断。v2 仅移除点击位置的两像素内缩要求，不扩大到识别区域外，不修改文字检测、底色阈值、重新定位查询或预算。
seeds 70–74 已用于发现该问题，后续只作开发案例，不能继续作为修复后的独立验收成绩。

## v2 的独立验收

冻结实现为 `01a28d7`，新的 seeds 80–84；没有根据这轮结果继续放宽规则。

| 条件 | precheck 关闭 | precheck v2 | 模型调用：关闭/开启 |
|---|---:|---:|---:|
| 白底，正常输入 | 5/5 | 4/5 | 10/10 |
| 白底，整段漏输入 | 5/5 | 4/5 | 15/14 |
| 深色，正常输入 | 5/5 | 3/5 | 10/10 |
| 深色，整段漏输入 | 3/5 | 3/5 | 13/13 |

另外 12 个已知回归 episode 使用 seeds 60、70、73。正常输入为关闭 2/3、开启 1/3；漏输入为关闭 2/3、开启 0/3，不混入独立验收。
已知 seed 60 两个条件下仍为两次模型调用、零次环境动作的停止，不能计为成功恢复。

v2 虽去掉了纯色区域内部的两像素内缩要求，仍未覆盖真实控件的边框。例如白底 seed 83 的落点 y 为 102.72，而纯色区域从 y=104 开始；该点在实际可输入控件内，却在纯色区域外，改写查询也未纠正。
聚焦后边框还会改变纯色区域，使漏输入后的再尝试更容易被拒绝。不能以扩大到邻近区域的方式直接宣称识别了真实控件边界。

本轮保留默认关闭和草稿状态，不将减少的执行步数或提前停止造成的少量调用下降解释为效率提升。
后续应研究边框/控件身份识别和截图裁剪后的定位复核，再使用新的任务与 seed 验收；不继续针对本轮五个 seed 调阈值。

首轮和次轮使用不同 seed，不能把两轮百分比的变化当作同条件算法收益。次轮各自的关闭/开启配对中，独立样本未出现新增挽救：白底正常、白底漏输入、深色正常分别退化 1、1、2 对；深色漏输入无新增挽救或退化。

### 证据和工程验收

- 首轮 44 个 episode/123 张轨迹 PNG，次轮 52 个 episode/150 张 PNG，全部通过 SHA256 核验；两轮分别 22/26 对初始截图和首次预测完全匹配。
- 离线重放全部开启检查的 22/26 个 episode，初始截图和最终结果均一致；85 次事前信号、40 次事后信号均可由原始 PNG 复算。
- 次轮 46 次事前检查中，15 次候选实际在空输入框内却被拒绝，未观察到接受输入框外候选；与首轮的 16 次误拦来自不同样本，不能直接按计数比较准确率。
- 所有被拦候选的决策步均没有执行环境 transition；实际模型调用不超过四次预算，原始预测、检查决策和执行动作独立记录。
- 两轮均无 setup/runtime 错误或显存溢出；GPU 0 两秒采样峰值均为 5297 MiB，退出后显存归零。这不是速度或训练容量结论。
- v2 的 Linux 全部 132 项测试通过；Mac 106 项单元测试、26 项浏览器测试通过。[冻结代码 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44455030848) 的 Ubuntu 22.04、24.04 和 package 全部通过。

聚合记录：[v1](../examples/precheck-v1-evaluation.json)、[v2](../examples/precheck-v2-evaluation.json)。
示例：[拦截误提交但仍失败](../examples/precheck-blocked-submit/replay.html)、[正常输入的误拦反例](../examples/precheck-false-rejection/replay.html)。原始数据的失败和未注入样本均保留；DOM 标签仅作离线审计，不进入策略。
