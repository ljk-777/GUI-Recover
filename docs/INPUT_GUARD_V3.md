# v3：均匀底色输入框的视觉检查

## 目标与实现

v2 只寻找白底输入框，深色主题即使正常输入也会 unknown 并停止。v3 是显式选择的实验信号：

```bash
gui-recover run --policy showui-scripted --model models/ShowUI-2B --tasks enter-text --seeds 60 --max-steps 4 --parser mixed --visual-variant dark --fault drop-input --recovery input-guard --input-guard-version v3 --out runs/dark-v3-demo
```

从预测点附近的像素出发，寻找 RGB 各通道差值不超过 12 的连通区域。保留原有面积、长宽比、矩形填充率和封闭条件；四侧边框与底色的通道最大差至少 24，覆盖率至少 65%。
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
