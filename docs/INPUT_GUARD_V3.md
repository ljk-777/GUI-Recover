# v3：均匀底色输入框的视觉检查

## 目标与实现

v2 只寻找白底输入框，深色主题即使正常输入也会 unknown 并停止。v3 是显式选择的实验信号：

```bash
gui-recover run --policy showui-scripted --model models/ShowUI-2B --tasks enter-text --seeds 60 --max-steps 4 --parser mixed --visual-variant dark --fault drop-input --recovery input-guard --input-guard-version v3 --out runs/dark-v3-demo
```

从预测点附近的像素出发，寻找 RGB 各通道差值不超过 12 的连通区域。保留原有面积、长宽比、矩形填充率和封闭条件；在四侧各两像素邻接带内寻找边框，与底色的通道最大差至少 24，覆盖率至少 65%。
两像素邻接带覆盖 Linux Chromium 聚焦边框内侧的一行低对比抗锯齿像素；已加入该开发回归案例。
区域中位颜色作为底色；内部去掉两像素焦点边框，再以至少 45 的通道最大色差寻找笔画。沿用光标和孤立单像素的排除规则，至少 8 个有效像素才认为文字出现。
输入后中位底色改变超过 12、边框移位、笔画像素超过内部面积的 40% 或截图尺寸改变均返回 unknown。

输入 API 仍只有前后截图与预测坐标；不接收主题名、DOM、任务答案、奖励或故障标签。恢复控制不变：只在 empty 时重新定位并追加一次；unknown 或再次为空则停止。
默认仍为 v2，v1/v2 实现和旧信号名称保持不变。v3 的 manifest 信号为 `enclosed_uniform_input_contrast_v3`。

## 验收协议

开发使用合成颜色、焦点、光标、噪声、背景变化反例及真实 Chromium seed 7；在观察模型评测结果前冻结实现与阈值。
模型评测配置为 `configs/input_guard_v3.json`，使用新的 seeds 60–64、相同冻结 ShowUI、脚本规划和 4 次决策预算，共 50 个 episode：

- 默认与深色夹具各四组：无故障/无恢复、漏输入/无恢复、漏输入/v3、无故障/v3。
- 深色夹具额外两组：漏输入/v2、无故障/v2；用于同 seed 的旧信号对照。

两套四组对照分别使用 `gui-recover compare` 核验注入前截图、预测、动作、模型和运行环境。两种主题分开报告，不合成总成功率。
v2/v3 版本对照单独核对首步截图与预测，不放宽四组比较器的版本一致性规则。
保存完整轨迹、模型调用、步数、unknown、重试与故障标签。离线审计读取 DOM 仅用于标注真值，不回流策略。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/run_matrix.py --config configs/input_guard_v3.json --model models/ShowUI-2B --out runs/input-guard-v3-001
.venv/bin/gui-recover compare --clean runs/input-guard-v3-001/clean --faulted runs/input-guard-v3-001/faulted --recovered runs/input-guard-v3-001/recovered --clean-recovered runs/input-guard-v3-001/clean-recovered --out runs/input-guard-v3-001/default-comparison.json
.venv/bin/gui-recover compare --clean runs/input-guard-v3-001/dark-clean --faulted runs/input-guard-v3-001/dark-faulted --recovered runs/input-guard-v3-001/dark-recovered --clean-recovered runs/input-guard-v3-001/dark-clean-recovered --out runs/input-guard-v3-001/dark-comparison.json
```

## 边界

仅检查初始为空、底色均匀、边框可区分的单行矩形区域中是否出现明显笔画，不是 OCR，不校验文字正确性或完整性。
部分输入和错误文字仍可通过；低对比文字、细竖线字形、占位文字、渐变、透明背景、布局动画和其他缩放没有泛化保证。
固定两像素边距仍只按当前 MiniWoB 498×321 截图验证。彩色合成单元测试不等于真实彩色网页任务验证。
模型实验继续使用手写指令规划，不能作为自主规划能力提升的证据。

## 2026-11-19 验收结果

冻结模型实验提交为 `6637f65`，未根据 seeds 60–64 的结果修改阈值。完整聚合记录见 [v3-evaluation.json](../examples/v3-evaluation.json)。

| 条件 | 普通执行 | v2 输入检查 | v3 输入检查 |
|---|---:|---:|---:|
| 白底，正常输入 | 5/5 | 本轮未测 | 5/5 |
| 白底，整段漏输入 | 0/5 | 本轮未测 | 5/5 |
| 深色夹具，正常输入 | 4/5 | 0/5 | 4/5 |
| 深色夹具，整段漏输入 | 0/5 | 0/5 | 4/5 |

两种主题的故障对照各 5 对，注入前截图、原始预测、注入动作全部匹配；v2/v3 的 10 对初始截图、首步预测和注入标签也全部匹配。
深色 v2 的 10 次检查均为 unknown。v3 保住原本成功的 4 个深色正常任务，并挽救 4 个漏输入任务；白底正常任务没有观察到退化。
每组只有 5 个 seed，不能据此声称任意主题或网页上的恢复率。

深色 seed 60 的第一步 INPUT 预测为 `[0.21, 0.38]`，实际点击落在 Submit 而非输入框，立即触发任务失败。
离线 DOM 边界诊断确认落点，该数据没有传入策略。普通执行、v2、v3 均未挽救，全部保留在分母。
v3 在终止截图上返回 unknown，但此时环境已经失败；记录中的 block 决策不能解释为成功阻止了这次提交。
事后输入检查无法撤销错误点击造成的终止副作用，后续需要单独研究执行前检查。

### 成本与审计

- 白底普通/正常检查/漏输入恢复各组模型调用分别为 10/10/15 次；深色分别为 9/9/13 次。成功恢复的单个任务从 2 次调用、2 步增加到 3 次调用、3 步。
- 未将 v2 提前停止造成的较少调用解释为效率优势。
- 50 个 episode 无 setup/runtime 错误或显存溢出，145 张轨迹 PNG 全部通过 SHA256 核对。
- 独立重放 20 个 v3 episode，初始截图与最终成败全部复现；29 次信号均可由原始 PNG 复算。
- 按全部审计状态计算：真实空白 11，检出 9，unknown 2；非空 18，无误报为空。precision 为 9/9、recall 为 9/11。两次 unknown 都来自 seed 60 已终止的错误点击，没有事后删除。
- 检测函数服务器离线均值约 105.9 毫秒、最大约 210.8 毫秒，不含 PNG 读取、模型与浏览器执行，不作为端到端速度结论。
- GPU 0 的两秒采样峰值为 5197 MiB，模型退出后显存归零；仅对应本轮固定输入/输出预算，不是训练或并发容量结论。
- Linux 完整 108 项测试通过；Mac 87 项单元测试、21 项浏览器集成测试通过。[冻结代码的 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44451774413) 中 Ubuntu 22.04、24.04 和 package 全部通过。

[恢复示例](../examples/dark-recovered/replay.html) 为 seed 61；[定位失败反例](../examples/dark-grounding-failure/replay.html) 为 seed 60。两者均为修改后的合成页面，保留 MiniWoB 许可证和代码/模型版本，不代表完整评测集。
