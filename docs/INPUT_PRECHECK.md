# INPUT 执行前的视觉目标检查

## 目标和机制

v3 事后检查无法撤销已经点到 Submit 而触发的任务失败。可选 `--input-precheck visual` 在 INPUT 执行前，调用 `inspect_input_target(screenshot, proposed_position)`。
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
