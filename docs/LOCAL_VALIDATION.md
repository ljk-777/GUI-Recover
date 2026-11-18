# 本地验收记录

核查日期：2026-11-18，Asia/Shanghai。

已测试代码提交：`81ec05e6509f78065bf6e33dbb6b69685268f050`。
之后只补充本记录和服务器传输说明，不改变已测试代码。

| 检查 | 实测结果 |
|---|---|
| 主机 | Apple Silicon Mac，Darwin arm64；非用户 RTX 4090 服务器 |
| Python | 项目隔离环境 Python 3.11.17；系统 3.14.2 未用于运行 |
| 依赖 | BrowserGym core/miniwob 0.14.3，Playwright 1.44.0，Chromium 125.0.6422.26 |
| MiniWoB | `7fd85d71a4b60325c6585396ec4f48377d049838` |
| 单元与集成测试 | 27 passed，4.87 秒；包括真实点击与输入提交 |
| 诊断实验 | click-test，seeds 0/1/2：3/3 成功，各 1 步，执行错误为 0 |
| 坐标 | 332×214 视窗 → 498×321 截图；1.5 倍映射点击成功 |
| 轨迹 | 3 个 episode；各 2 张 PNG，JSONL 事件和 HTML 查看页 |
| 版本追溯 | 最终诊断 manifest 记录上述代码 commit，git_dirty=false，固定 MiniWoB commit |
| ShowUI | 适配代码已写；本机未安装 torch/transformers，未下载权重，未运行真实预测 |
| CUDA/4090 | 未连接、未测显存、未测速度，尚不能宣布 V0.1 模型闭环完成 |
| CI | 工作流已配置；尚未推送远程，因此没有 GitHub Actions 运行成绩 |

诊断成功率来自 DOM 特权定位器，仅验证环境、坐标、执行、记录和评测；不得写成 ShowUI 100% 成功率。
调用环境首次因本机进程启动限制失败，随后在允许启动专用 Chromium 的执行环境中通过；没有修改用户的浏览器或系统 Python。

完整依赖快照保存在 `requirements/local-validated.txt`。这是本次 macOS 环境记录，不是已验证的 Linux/CUDA lockfile。
若 Mac 上没有 `python3.11` 命令，但已安装 uv，可用 `uv venv --python 3.11 .venv` 创建环境后继续 README 的步骤。

交付目录旁的 `local-smoke-verified/` 包含原始验收证据。模型适配器、恢复效果、训练可行性需要各自后续实验，不能从诊断测试外推。
