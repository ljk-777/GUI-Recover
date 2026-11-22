# Contributing

[Open issues](https://github.com/ljk-777/GUI-Recover/issues) · [简体中文](#提交问题与修改)

Include the affected command, task and seed, model revision if applicable, and a small trace with secrets and personal data removed. For a recovery change, include the paired baseline and candidate outcomes with the same task, seed and action budget.

Use Python 3.11 and run the unit suite before opening a pull request.

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not integration'
git diff --check
```

The browser integration suite requires Chromium and `MINIWOB_URL`. Follow the setup guide to run `python -m pytest -m integration`. For a benchmark change, attach the command, frozen task selection and CPU audit result. Visual policies receive screenshots, the goal and permitted action history. Mark diagnostic policies that use DOM or rewards. [Browser setup](docs/SERVER.md)

## 提交问题与修改

请附上复现命令、任务与种子、模型版本，以及去除密钥和个人信息的小段轨迹。修改恢复策略时，请提供相同任务、种子和预算下的基线与候选结果。提交前运行上面的单元测试和差异检查。

涉及浏览器的改动请补跑集成测试。属于 ShowUI、BrowserGym 或 MiniWoB++ 的通用修复，可以先在对应项目的固定版本上构造最小复现，再按该项目的贡献指南提交。 [浏览器环境](docs/SERVER.md)
