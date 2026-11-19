# 单卡服务器操作单

以下命令可由用户手动执行，也可在用户授权的 SSH 会话中执行。
将整个 GUI-Recover 项目目录复制到服务器，进入该目录。下面仅准备推理和评测，不启动训练。

也可以把交付的 `gui-recover.bundle` 复制到服务器，用 Git 恢复完整本地提交历史：

```bash
git clone /实际存放路径/gui-recover.bundle gui-recover
cd gui-recover
git remote remove origin
git log --oneline -2
```

移除的是指向离线 bundle 文件的 origin；以后创建独立 GitHub 仓库时再添加真实远程地址。

## 1. 环境与权限

确认 Python 3.11、空闲磁盘、可用 RAM、NVIDIA 驱动及允许使用的 GPU。
若是 `jiakun`，默认仅允许 GPU 0–3；0–3 忙时等待，不能改用 4–7。
先选择已获分配且空闲的一张卡；以下 `0` 只作示例，不代表已确认它空闲。

```bash
python3.11 --version
df -h .
free -h
# 仅查询已允许的 GPU（jiakun 默认允许 0–3）
nvidia-smi --id=0,1,2,3 --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0
```

单卡机器可将查询改为 `--id=0`。`CUDA_VISIBLE_DEVICES` 在启动 Python 前设置；程序内 `cuda:0` 指向选中的那张物理卡。

## 2. 安装与无模型验收

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,showui]'
python -m playwright install chromium
bash scripts/setup_miniwob.sh
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export HF_HOME="$PWD/.cache/huggingface"
gui-recover doctor --cuda > server-doctor.json
python -m pytest -q
gui-recover run --policy oracle-smoke --seeds 0 1 2 --out runs/server-smoke-001
```

`doctor` 应显示 CUDA 可用、所选显卡、BF16 支持和依赖版本。Linux 缺浏览器系统库时，由有权限的用户安装 Playwright 所需依赖。
标准 PyPI PyTorch 安装若与已有驱动不匹配，先解决驱动/运行时兼容，不改模型或评测口径。

Ubuntu 24.04 上，固定的 Playwright 1.44 安装器会请求旧名 `libasound2`，导致 `install --with-deps` 失败。
改用本项目的对应库名安装脚本：

```bash
bash scripts/setup_browser_ubuntu24.sh
```

## 国内下载路径

Python 包可使用清华镜像；只在当前命令指定，不改系统配置：

```bash
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -e '.[dev,showui]'
```

浏览器下载可使用本次验证通过的国内镜像，仍保留 Playwright 固定的 Chromium build：

```bash
PLAYWRIGHT_DOWNLOAD_HOST=https://registry.npmmirror.com/-/binary/playwright python -m playwright install chromium
```

无法访问 Hugging Face 时，可以使用 ShowUI 官方 README 链接的 ModelScope 镜像：

```bash
python scripts/download_showui.py --out models/ShowUI-2B
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 gui-recover run --policy showui --model models/ShowUI-2B --tasks click-test --seeds 0 1 2 --max-steps 5 --out runs/showui-first-001
```

脚本保存每个文件的镜像 revision 和 SHA256、支持断点续传；权重还必须匹配固定 HF checkpoint 的独立 SHA256：
`68080df785764e98976eb9cc93a07c6c69cf8a6933738496e02aef55b53d2aa3`。
小文件按镜像快照校验，不能仅凭权重相同声称整个目录逐字节等于 HF 仓库。manifest 会保存完整下载来源。

## 3. ShowUI 首个闭环

```bash
gui-recover run --policy showui --tasks click-test --seeds 0 1 2 --max-steps 5 --out runs/showui-first-001
```

首次运行会下载固定 revision 的模型（约 4.43 GB 总文件）。只使用 Transformers 标准实现，不启用远程自定义代码。
验收：无 startup/runtime 错误；至少一条 episode 的原始预测合法、执行成功、环境判定成功；PNG/JSONL 完整；显存未溢出。
若三条全失败，查看 raw output、截图和执行错误，先定位 prompt/动作/坐标/策略原因，不开始训练。
`ANSWER` 提前结束与非法动作均为失败。不要通过删除失败 episode 提高成功率。

## 4. 初始基线

首个闭环成功后，再跑 3 类任务 × 10 个 seeds，共 30 episodes：

```bash
gui-recover run --policy showui --tasks click-test click-button enter-text --seeds 0 1 2 3 4 5 6 7 8 9 --max-steps 10 --out runs/showui-baseline-001
```

这只是小样本基线，不是完整 MiniWoB benchmark 或泛化结论。保留任务级结果，记录错误类型、平均步数、逐步延迟和峰值显存；训练前另划分任务与 seeds，避免泄漏。

带回 `server-doctor.json`、`manifest.json`、`summary.json`，以及至少一条成功和一条失败轨迹目录即可继续下一模块。
