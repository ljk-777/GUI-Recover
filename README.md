# GUI-Recover

Reusable visual GUI agent trajectories, evaluation, and (planned) error recovery.

**状态：V0.1 开发预览。本地环境与记录链路可验收；ShowUI 单卡 CUDA 推理待服务器实测。**
诊断策略的成功率不代表模型成绩。V0.2 恢复模块尚未实现，不宣称性能提升。

目标是回答：视觉 GUI Agent 在哪里出错、能否恢复、恢复增加了多少成本。
V0.1 先构建可靠的测量基础，不进行训练。

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
python -m playwright install chromium
bash scripts/setup_miniwob.sh
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
gui-recover doctor
python -m pytest -q
gui-recover run --policy oracle-smoke --seeds 0 1 2 --out runs/smoke-001
```

Linux 若缺系统库，可由有安装权限的用户执行 `python -m playwright install --with-deps chromium`。
每次使用新的输出目录，防止覆盖实验。

预期 `runs/smoke-001/summary.json` 显示 3 个诊断 episode 全部成功；每条轨迹包含 reset、prediction、transition、episode_end 和两张截图。
`runs/smoke-001/click-test-0/replay.html` 可离线打开。这是轨迹查看器，不是恢复浏览器完整状态的确定性重放器。

## 单张 RTX 4090

详见 [服务器操作单](docs/SERVER.md)。模型约 4.42 GB 权重，24 GB 显存下做 BF16 单样本推理具备可行性；激活、图像 token、KV cache 和驱动占用仍需实测。
模型固定为 `showlab/ShowUI-2B@cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`。

已实现 CLICK、INPUT、HOVER、ENTER、SCROLL、ANSWER。INPUT 为点击后追加输入，不隐式清空；SELECT、SELECT_TEXT、COPY 暂不支持，出现时计入 invalid_action。只执行一个预测动作，然后重新观察。INPUT 的点击和输入作为同一个逻辑动作。

## 评测口径

成功要求同时满足 `terminated`、非 `truncated`、BrowserGym reward 为 1、`DONE_GLOBAL` 为真且 `RAW_REWARD_GLOBAL > 0`，与固定版本的 MiniWoB 二值奖励规则对应。
保留原始奖励，避免将此口径推广到其他 benchmark；时间截止和步数上限均不算成功。
汇总是 episode 加权成功率，并提供每个任务的独立分数；运行异常也计入总分母并分类报告。
manifest 记录模型版本、提示词、依赖、seed、步数预算、Git 版本、环境版本；每步记录推理耗时及 CUDA 峰值。
冷启动加载时间不计入每步推理耗时。PyTorch allocated/reserved 不等于整张显卡占用，显卡总占用另用服务器工具观察。

## 开源与后续

- [仓库、许可证和数据核查](docs/UPSTREAM_AUDIT.md)
- [V0.2 实验设计](docs/V0.2.md)
- [贡献约定与 Git 操作](CONTRIBUTING.md)

本项目采用 Apache-2.0；第三方代码、权重和数据仍适用各自条款，见 [第三方说明](THIRD_PARTY.md)。没有把上游源码、模型权重或大数据集放进本仓库。
独立包通过接口组合三个项目，便于后续单独发布；有可复现的上游问题时，再在对应 fork 中准备小范围 PR。
