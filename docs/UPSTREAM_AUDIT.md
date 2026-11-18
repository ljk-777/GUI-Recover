# 上游与数据核查

核查日期：2026-11-18（Asia/Shanghai）。依据公开仓库文件、模型卡、数据卡及本地安装包；历史会话中的 Star 数、Issue 状态不作为本次事实依据。

## 仓库与职责

| 项目 | 核查结果 | 在 GUI-Recover 中的职责 |
|---|---|---|
| [ShowUI](https://github.com/showlab/ShowUI) | 代码 Apache-2.0；提供 grounding、navigation、训练及评测代码；标准 Qwen2-VL 加载路径可用于公开 2B checkpoint | 视觉动作预测，不复制其训练仓库 |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | Apache-2.0；Gymnasium reset/step，Playwright 执行，截图及 DOM 观测，多 benchmark 接口 | 统一执行、观察、奖励 |
| [MiniWoB++](https://github.com/Farama-Foundation/MiniWoB-plusplus) | MIT；100 多种合成网页任务，当前 README 明示维护模式；原生 Python 接口使用 Selenium | 只取 HTML/JS 任务资产，通过 BrowserGym 运行 |

来源：[ShowUI LICENSE](https://github.com/showlab/ShowUI/blob/main/LICENSE)、[Quick Start](https://github.com/showlab/ShowUI/blob/main/QUICK_START.md)、[BrowserGym LICENSE](https://github.com/ServiceNow/BrowserGym/blob/main/LICENSE)、[MiniWoB LICENSE](https://github.com/Farama-Foundation/MiniWoB-plusplus/blob/main/LICENSE)、[MiniWoB README](https://github.com/Farama-Foundation/MiniWoB-plusplus/blob/main/README.md)。

源码调查时 GitHub 返回的最近提交：ShowUI `21ed7cb24be0cc877bb8352ee34d58a9aea2c876`；BrowserGym `62a678feefaff0f93e9bab03f51c5402a1e1db00`。
实际运行使用 PyPI `browsergym-core==0.14.3` / `browsergym-miniwob==0.14.3`，而非直接安装调查时 main。
MiniWoB 固定为 BrowserGym 安装说明推荐的 `7fd85d71a4b60325c6585396ec4f48377d049838`；不用上游 `reset --hard` 示例覆盖已有 checkout，使用独立目录和 detached checkout。

## 需要保留的边界

1. **代码、权重、数据许可证不同。** [ShowUI-2B 模型卡](https://huggingface.co/showlab/ShowUI-2B) 标注 MIT，不能把代码的 Apache-2.0 自动套用到权重；模型底座及再分发文件仍应保留对应通知。
2. **公开演示与真实在线任务不同。** ShowUI 已有导航示例，但不能据此声称其在我们选定的 MiniWoB seeds 上已成功。通用 `Qwen2VLForConditionalGeneration` 路径也不等于已启用其自定义 UI token selection。
3. **动作坐标有缩放。** 固定 BrowserGym 的 MiniWoB 视窗为 332×214，截图乘 1.5；coordinate action 内部会映射回 CSS 像素。本项目只把归一化坐标转换到截图像素，不重复除以 1.5。
4. **奖励按 benchmark 定义。** 固定 MiniWoB adapter 用 `RAW_REWARD_GLOBAL > 0` 得到二值奖励。其他任务若存在部分奖励，需要单独定义 validator。

接口来源：[MiniWoB setup](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md)、[task adapter](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/src/browsergym/miniwob/base.py)、[action functions](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py)。对应安装版也已在本地检查。

## 数据获取路线

| 资源 | 获取途径与实际检查 | 用途与限制 |
|---|---|---|
| MiniWoB 在线生成 | 固定源码，reset(seed) 生成指令/布局；交互产生截图、动作与环境奖励 | V0.1/V0.2 主数据，无须下载示范集；固定任务和 seed 清单 |
| ShowUI-2B | [HF 文件列表](https://huggingface.co/showlab/ShowUI-2B/tree/main)，revision `cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60`；权重文件 `pytorch_model.bin` 约 4.42 GB | 由服务器 transformers 下载；本机尚未下载或推理 |
| ShowUI-web | [数据卡](https://huggingface.co/datasets/showlab/ShowUI-web)、[文件列表](https://huggingface.co/datasets/showlab/ShowUI-web/tree/main)：21,988 条记录，parquet 的 image 为 null，image_url 为相对路径；图片另在 `images.tar.gz`，约 43.7 GB | 后续 grounding SFT；数据卡文字写 `image.tar.gz`，实际列表是复数 `images.tar.gz`；未解压验证图片和标注对齐，当前卡未见明确 license 字段，不承诺训练/再分发权利 |
| Multimodal-Mind2Web | [数据卡](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) 提供 parquet、screenshot 字段及多个 split；标注 OpenRAIL | 后续离线导航监督；先核对完整条款和拆分，不能把静态截图数据当可 reset 的在线环境 |

当前没有下载 43.7 GB 图片包，也没有开始训练。ShowUI-web 后续验收需要检查一条记录的 image_url 对应归档文件、图片尺寸、bbox/point 范围，以及许可来源。

## 单卡可行性结论

单卡 4090 的 V0.1 路径：一个浏览器环境 + 一个 ShowUI-2B BF16 实例，batch=1，SDPA，最大 784 个图像 token 对应的像素预算，最多生成 128 tokens，串行 episode。
约 4.42 GB 权重远小于 24 GB 显存，因此**推理最小闭环具备工程可行性**；这不是本项目实测显存数字，不能外推到全参数训练或 RL 吞吐。
模型还需要激活、KV cache、CUDA 上下文和 CPU 内存。起始建议准备至少 16 GB 可用 RAM、10 GB 以上可用磁盘用于模型及临时文件；安装 CUDA PyTorch 依赖还需额外磁盘。
完整验收以服务器实测的合法输出、成功轨迹、峰值显存与推理延迟为准。
