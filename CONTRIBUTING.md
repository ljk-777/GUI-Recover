# Contributing

Keep environment, policy, action compiler, recording and evaluation separate. Never feed DOM or reward to a policy labelled visual-only. Label privileged diagnostic policies explicitly.

Use Python 3.11; run `python -m pytest -q`. Integration tests require MINIWOB_URL and Chromium. No model weights or private browser data in Git. CI runs unit and local browser tests; GPU inference is a separate manual acceptance gate.

## 按模块使用 Git

```bash
git status --short
git switch -c feat/recovery-injection
# 完成一个可验收模块后
git diff --check
python -m pytest -q
git add src tests docs
git diff --cached
git commit -m "feat: add deterministic action fault injection"
```

`status` 检查改动范围，分支隔离新功能，`diff --check` 检查补丁格式，提交保存一个可复现阶段。不要混入模型下载、运行截图或上游 checkout。
当前只有本地独立仓库。创建远程仓库后，可按实际 URL 添加 origin；不要将上游 ShowUI 的 remote 作为本项目 origin。

## 上游 PR 路径

先在固定版本复现真实问题，检查现有 issue/PR 是否已处理；将问题最小化为上游可运行的测试。再在对应项目的 fork 分支里修复，遵守其贡献指南。
GUI-Recover 的恢复框架留在本仓库；只有通用且属于上游的接口/文档/行为修复才提交上游。不预先承诺某条 Issue 仍开放或必然被接受。
