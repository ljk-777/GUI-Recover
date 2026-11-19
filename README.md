# GUI-Recover

Fault injection, visual recovery experiments, and traceable evaluation for GUI agents.

**[V0.2.0a1 Alpha 已发布](https://github.com/ljk-777/GUI-Recover/releases/tag/v0.2.0a1)：实验工具，尚未发布到 PyPI。** 基于 ShowUI + BrowserGym/MiniWoB，保存截图、预测、执行动作和奖励，并核查恢复实验的配对与成本。

[下载源码与安装包](https://github.com/ljk-777/GUI-Recover/releases/tag/v0.2.0a1) · [发布版本测试](https://github.com/ljk-777/GUI-Recover/actions/runs/44449374272)

- **模型与环境接口**：视觉策略只接收任务文字、截图、动作历史和执行错误。
- **可复现故障**：漏点击、坐标偏移、漏输入、部分输入和错误文字。
- **有限恢复**：弱视觉反馈，以及脚本规划基线上的输入出现检查与最多一次补输入。
- **审计证据**：PNG/JSONL/HTML、截图哈希、固定版本、成功率和真实模型调用统计。

已发布 Alpha 的默认 v2 输入检查只验证白底、初始为空的输入框里出现了文字笔画；不核对内容正确性。脚本规划只覆盖三种 MiniWoB 指令语法，必须与自主导航成绩分开。
详见 [发布验证与已知限制](docs/RELEASE_VALIDATION.md)。历史实验及负面结果也保留在 docs/ 中。

```text
MiniWoB reset → 任务文本 + 截图 → Policy.predict → 严格动作解析
      ↑                                             ↓
      └──── 新截图 + 环境奖励 ← BrowserGym step ← 坐标动作
                             ↓
                 JSONL 轨迹 + PNG + 成功率 + HTML 查看
```

## 模块与验收

| 模块 | 原理 | 验收 |
|---|---|---|
| 环境 | BrowserGym MiniWoB Gymnasium reset/step | 固定 seed 能重置，按钮点击后获得终止奖励 |
| 视觉策略 | ShowUI-2B，任务 + 当前截图 + 最近 8 步历史 | 服务器生成合法单步动作，成功执行至少一条轨迹 |
| 动作 | 归一化坐标 → 截图像素 → BrowserGym 内部缩放 | 1.5 倍截图下仍能点击；拒绝越界、NaN 和代码 |
| 记录 | 每次重置和执行后保存 PNG；逐行写事件 | 原始输出、执行动作、奖励、终止标志、截图哈希可追溯 |
| 评测 | 按 task/seed 汇总环境终止结果 | 失败和异常留在分母，ANSWER 不等于成功 |

`src/gui_recover/` 分为 `environment.py`、`policies.py`、`actions.py`、`runner.py`、`cli.py`。
模型接口不包含 DOM、元素 ID、环境奖励或成功标记。`oracle-smoke` 是独立的 DOM 特权诊断策略，且只允许 `click-test`。

## 安装与本地验收

在项目目录执行，使用 Python 3.11。系统 Python 3.14 不属于本版本支持范围。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
bash scripts/setup_miniwob.sh
python -m playwright install chromium
bash scripts/demo_cpu.sh
```

Ubuntu 24.04 若缺系统库，使用 `bash scripts/setup_browser_ubuntu24.sh`；Ubuntu 22.04 使用 `python -m playwright install --with-deps chromium`。
国内下载可设置 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 和 `PLAYWRIGHT_DOWNLOAD_HOST=https://registry.npmmirror.com/-/binary/playwright`。
每次使用新的输出目录，防止覆盖实验。

CPU 演示使用 DOM 特权 oracle，只验证执行与记录链路，预期 3/3 成功；它不是 ShowUI 模型成绩。
输出目录的 `click-test-0/replay.html` 可离线打开。这是轨迹查看器，不是恢复浏览器完整状态的确定性重放器。
不安装浏览器也可查看 [保存的恢复示例](examples/recovered/replay.html) 和 [内容错误反例](examples/wrong-input/replay.html)。

## GPU 最小恢复演示

先选择已分配的一张 CUDA GPU，再执行：

```bash
python -m pip install -e '.[showui]'
python scripts/download_showui.py --out models/ShowUI-2B
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export CUDA_VISIBLE_DEVICES=0
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gui-recover run --policy showui-scripted --model models/ShowUI-2B --tasks enter-text --seeds 40 --max-steps 4 --parser mixed --fault drop-input --recovery input-guard --input-guard-version v2 --out runs/recovery-demo
```

模型不包含在软件包内，下载脚本会核对固定权重 SHA256。完整实验定义为 `configs/release_v0_2.json`，可用 `scripts/run_matrix.py` 执行。
该演示使用显式脚本规划和视觉输入检查，输出中的 `scripted_planner` 标记必须保留。

## 单张 RTX 4090

详见 [服务器操作单](docs/SERVER.md)。模型约 4.42 GB 权重；本次 BF16 单样本、三类简单任务的设备采样显存峰值为 6695 MiB，低于 12032 MiB 配额。
这是固定图像/生成预算下的推理实测，不能外推到高分辨率长上下文、训练或并发任务。
模型固定为 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`。

已实现 CLICK、INPUT、TYPE、HOVER、ENTER、SCROLL、ANSWER。INPUT 为点击后追加输入；TYPE 向已聚焦元素追加输入，两者不隐式清空。
SELECT、SELECT_TEXT、COPY 暂不支持，出现时计入 invalid_action。只执行一个预测动作，然后重新观察。INPUT 的点击和输入作为同一个逻辑动作。

默认保留初始 prompt 和 strict 解析；`--parser mixed` 可兼容单引号字典中的裸 JSON 常量，仍拒绝代码和多动作。
`--prompt-profile official-web` / `official-miniwob` 用于自主导航模板诊断，历史结果未改善输入任务，详见 [失败诊断](docs/DIAGNOSIS.md)。
`--input-guard-version v1` 保留原始信号；v2 忽略独立单像素噪声。布局反例通过 `--visual-variant shifted|dark` 启用，它们是修改后的测试夹具，不是原版 MiniWoB 成绩。

开发版增加可选 `--input-guard-version v3`，按区域自身底色检测笔画，面向均匀深色输入框；默认仍为 v2。新的 5 个 seed 中，深色漏输入恢复从 v2 的 0/5 到 v3 的 4/5，正常组也从 0/5 到 4/5；定位失败保留在分母。实现、完整 50 个 episode 对照和边界见 [v3 主题适应实验](docs/INPUT_GUARD_V3.md)。

后续开发增加 `--input-precheck visual`：INPUT 执行前检查空白区域，无法确认时最多重新定位一次，仍失败则停止；该功能默认关闭，不能证明控件可编辑。预算、记录和独立对照见 [执行前检查](docs/INPUT_PRECHECK.md)。

## 评测口径

成功要求同时满足 `terminated`、非 `truncated`、BrowserGym reward 为 1、`DONE_GLOBAL` 为真且 `RAW_REWARD_GLOBAL > 0`，与固定版本的 MiniWoB 二值奖励规则对应。
保留原始奖励，避免将此口径推广到其他 benchmark；时间截止和步数上限均不算成功。
汇总是 episode 加权成功率，并提供每个任务的独立分数；运行异常也计入总分母并分类报告。
manifest 记录模型版本、提示词、依赖、seed、步数预算、Git 版本、环境版本；每步记录推理耗时及 CUDA 峰值。
冷启动加载时间不计入每步推理耗时。PyTorch allocated/reserved 不等于整张显卡占用，显卡总占用另用服务器工具观察。

## 开源与后续

- [仓库、许可证和数据核查](docs/UPSTREAM_AUDIT.md)
- [本地实测验收记录](docs/LOCAL_VALIDATION.md)
- [单卡服务器实测](docs/SERVER_VALIDATION.md)
- [失败诊断与接口对照](docs/DIAGNOSIS.md)
- [V0.2 实验设计](docs/V0.2.md)
- [V0.2 首轮配对结果](docs/RECOVERY_RESULTS.md)
- [Grounding 与脚本规划诊断](docs/GROUNDING.md)
- [漏输入恢复与固定重试对照](docs/INPUT_GUARD.md)
- [发布验证、安装验收与已知限制](docs/RELEASE_VALIDATION.md)
- [贡献约定与 Git 操作](CONTRIBUTING.md)

本项目采用 Apache-2.0；第三方代码、权重和数据仍适用各自条款，见 [第三方说明](THIRD_PARTY.md)。没有把上游源码、模型权重或大数据集放进本仓库。
独立包通过接口组合三个项目，便于后续单独发布；有可复现的上游问题时，再在对应 fork 中准备小范围 PR。
