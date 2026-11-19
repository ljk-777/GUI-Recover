# 发布候选交接

目标版本：0.2.0a1，alpha。范围与未支持项见 RELEASE_VALIDATION.md。
当前提供完整 Git bundle、源码包和 wheel；未发布到 PyPI。GitHub 上的 CI 状态见下面的实际运行记录。

2026-11-19 已完成首次公开 Alpha：[仓库](https://github.com/ljk-777/GUI-Recover)、[预发布](https://github.com/ljk-777/GUI-Recover/releases/tag/v0.2.0a1)、[标签 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44449374272)。以下命令留作后续版本维护参考。

## 本地包验证

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
python3.11 -m venv /tmp/gui-recover-check
/tmp/gui-recover-check/bin/python -m pip install dist/gui_recover-0.2.0a1-py3-none-any.whl
/tmp/gui-recover-check/bin/gui-recover --version
```

从源码包运行演示：解压后安装上述 wheel，按 README 安装 Chromium 系统依赖，再运行 `GUI_RECOVER_PYTHON=/tmp/gui-recover-check/bin/python bash scripts/demo_cpu.sh`。
CPU 演示只验证诊断 oracle，GPU 模型恢复需要另外安装 showui extra 并下载固定模型。

## GitHub 发布

在拥有目标账号权限的终端先完成 `gh auth login --hostname github.com`，再用 `gh auth status` 确认登录。
完成仓库名称和公开性决策后，在项目目录创建远程仓库并推送：

```bash
# 将 OWNER 换为实际 GitHub 用户或组织；先检查目标名称是否已存在。
gh repo create OWNER/GUI-Recover --public --source . --remote origin --push
```

若已有目标仓库，添加真实 origin 后正常推送，不覆盖远程历史。
推送后检查 Actions 的单元、浏览器和构建任务；只有实际成功才能记录 CI 通过。
确认 GitHub 实际结果后可创建 alpha 预发布，并附源码包、wheel、SHA256SUMS 与 CHANGELOG。
交付 bundle 含本地 `v0.2.0a1` 标签；准备创建预发布时还需将该标签正常推送到 origin。
软件包内没有服务器密钥、模型权重或大型数据集；固定模型和环境各自保留许可证。
