# 单卡服务器验收：2026-11-19

## 结论

V0.1 的真实模型预测 → 动作执行 → 环境判定 → 截图/轨迹记录 → 基础评测已跑通。
12 GB 档位足够本次受控配置；策略在多步任务上仍明显不足，不能把可运行等同于可靠通用 GUI Agent。

实测代码提交：`3223a427a315da77798dcf8d8f9539c963a22a82`，运行时工作区干净。
本次没有进行训练、量化或恢复策略调优，也没有在评测过程中修改 prompt、解析器或 seed 清单。

## 环境与配置

| 项目 | 实际值 |
|---|---|
| 系统 | Ubuntu 24.04.4 LTS，Linux x86_64 |
| GPU | NVIDIA GeForce RTX 4090 D；实例可见显存 12032 MiB（11.75 GiB） |
| 系统内存 | 30 GiB |
| Python | 3.11.18，项目独立虚拟环境 |
| 模型运行时 | torch 2.6.0+cu124、torchvision 0.21.0、transformers 4.49.0 |
| GPU 驱动 | 595.84；本次 CUDA 12.4 PyTorch 实测正常 |
| 模型 | ShowUI-2B；BF16、SDPA、batch=1、greedy decoding |
| 图像/输出预算 | min_pixels=200704，max_pixels=614656；max_new_tokens=128；最近 8 步历史 |
| 环境 | BrowserGym core/miniwob 0.14.3；MiniWoB commit `7fd85d71a4b60325c6585396ec4f48377d049838` |
| 浏览器 | Playwright 1.44.0，Chromium 125.0.6422.26 |

权重从上游链接的 `AI-ModelScope/ShowUI-2B` 下载。全部文件按镜像 SHA256 校验；权重另外匹配 HF 固定 revision `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60` 的 SHA256：
`68080df785764e98976eb9cc93a07c6c69cf8a6933738496e02aef55b53d2aa3`。

下载路径：Python 依赖使用清华镜像，模型使用 ModelScope，浏览器使用 npmmirror，系统软件源主要为镜像原有的中科大源。
已处理 Ubuntu 24.04 的 t64 库名变化，未更改 BrowserGym/Playwright 版本。完整 Python 包快照见 `requirements/server-validated.txt`。

## 分层验收

1. 服务器测试：27 项全部通过（11.71 秒），含真实坐标点击、文本输入和提交。
2. 诊断策略：`click-test` seeds 0/1/2，3/3 成功，每个 1 步；仅说明执行链路正常。
3. 真实 ShowUI 首轮：`click-test` seeds 0/1/2，3/3 成功，每个 1 步；每次预测 2.403、0.748、0.759 秒。
4. 多任务基线：3 个任务各 seeds 0–9，每个最多 10 个逻辑动作，结果如下。

| 任务 | 成功数 / 总数 | 成功率 | 平均执行步数 |
|---|---:|---:|---:|
| click-test | 10 / 10 | 100% | 1.0 |
| click-button | 5 / 10 | 50% | 1.9 |
| enter-text | 0 / 10 | 0% | 1.0 |
| 合计 | 15 / 30 | 50% | 1.3 |

失败分类：环境判定失败 12、非法动作格式 2、步数用尽 1。没有 setup/runtime 错误，没有动作执行异常，没有显存溢出。
全部失败均留在分母。该结果仅是三个简单任务的小样本基线，不是完整 MiniWoB 成绩或恢复增益。

## 实测开销

- 共 41 次模型预测、39 个执行动作。
- 平均每次预测 0.797 秒，中位数 0.784 秒。
- PyTorch 最大 allocated：4.352 GiB；最大 reserved：6.189 GiB。reserved 包含 allocator 缓存。
- 每秒一次 `nvidia-smi` 采样得到设备显存峰值 6695 MiB，约 6.54 GiB；采样不能捕获所有瞬时峰值。
- 各 episode 耗时总和约 93.86 秒，不包含模型加载、下载与所有进程初始化开销。
- 运行结束后显存占用回到 0 MiB，没有遗留推理进程。

## 观察到的失败及下一步

**格式兼容问题：** click-button seeds 1/2 第二次输出形如 `{'action': 'CLICK', 'value': null, ...}`，混用 Python 单引号与 JSON null；当前严格解析器按非法格式拒绝。初始基线保留此结果。后续若增加安全的格式归一化或错误反馈，应建立新配置、独立报告，不能覆盖此基线。

**定位与停滞：** click-button seed 4 连续 10 次给出相同坐标 `[0.06, 0.5]`；环境未完成。此轨迹适合后续重复动作/停滞检测测试。

**多步指令遵循：** enter-text 全部在第一步点击 Submit，未输入目标文本便终止失败。输入执行器在集成测试中已验证可用。说明该 checkpoint 与当前 prompt 组合没有完成所需动作规划；尚未证明原因仅是 prompt、checkpoint 导航能力或任务分布中的某一项。

下一模块先做 checkpoint/官方导航 prompt 核对和格式兼容诊断，再用固定任务/seed 设计 V0.2 注入与恢复对照。不能用盲目重试或大规模训练替代原因分析。

## 证据与 Git

服务器 `runs/` 保留 `server-smoke-001`、`showui-first-001`、`showui-baseline-001`，含 manifest、summary、JSONL、PNG、HTML 查看页与 GPU CSV。
本地交付的 `server-validation/` 保存对应证据。36 个 episode 的 81 张截图已逐张核验 SHA256。

本模块关键 Git 操作：`git clone` 从 bundle 恢复仓库；`git fetch` 导入已测试提交；`git merge --ff-only` 同步；`git diff --check` 和 `git commit` 留存改动。
尚未创建 GitHub 远程仓库或提交上游 PR。
