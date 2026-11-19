# V0.1 失败诊断：2026-11-19

## 问题与结果

初始 30 个 episode 中有 2 次动作格式错误，enter-text 10 次都在未输入文字时点击 Submit。
三组独立诊断均保留原始模型、任务、seeds 0–9、图像/生成预算和每 episode 最多 10 次预测/动作。没有训练或下载新 checkpoint。

| 配置 | click-test | click-button | enter-text | 合计 | 预测次数 / 执行动作 |
|---|---:|---:|---:|---:|---:|
| 初始 prompt + strict 解析 | 10/10 | 5/10 | 0/10 | 15/30 | 41 / 39 |
| 初始 prompt + mixed 解析 | 10/10 | 5/10 | 0/10 | 15/30 | 57 / 57 |
| 上游通用 web prompt + mixed | 10/10 | 4/10 | 0/10 | 14/30 | 66 / 66 |
| 上游 MiniWoB 模板 + mixed + TYPE 执行 | 10/10 | 5/10 | 0/10 | 15/30 | 38 / 35 |

原始基线、mixed/web、MiniWoB 诊断的代码版本分别为 `3223a42`、`dd0740f`、`c9c5801`，详见各运行 manifest。
最后一组新增 TYPE 支持、对应历史格式和无干预的诊断日志，不与旧基线合并。上述是小样本接口诊断，不是不同模型的性能排名。

## 格式兼容修复

模型输出曾混用单引号字典和 JSON `null`，严格 JSON 和 Python 字面量解析都会拒绝。
`--parser mixed` 只在 AST 中将裸 `null/true/false` 映射为字面量，字符串内容不变；仍使用 `ast.literal_eval`，不执行表达式。
重复键、额外键、多动作序列、非法坐标及调用表达式均被拒绝。修复默认不启用，strict 配置仍可重现。

mixed 运行共修正 3 个输出（2 个原始失败及继续执行后新增的混合格式）。原先两个格式失败最终都达到步数上限。
总成功数未变，动作从 39 增加到 57：仅让非法输出继续执行，不足以纠正原本错误的坐标。

## 输入任务与上游核查

已检查固定版本 ShowUI 的 [Quick Start](https://github.com/showlab/ShowUI/blob/21ed7cb24be0cc877bb8352ee34d58a9aea2c876/QUICK_START.md)、[训练说明](https://github.com/showlab/ShowUI/blob/21ed7cb24be0cc877bb8352ee34d58a9aea2c876/TRAIN.md)、[MiniWoB 模板](https://github.com/showlab/ShowUI/blob/21ed7cb24be0cc877bb8352ee34d58a9aea2c876/data/template/miniwob.py) 和 dataset loader。

通用 web 模板定义带坐标 INPUT；MiniWoB 训练模板只定义 CLICK 和向焦点输入的 TYPE。
新增 `official-miniwob` profile 后，历史动作按照上游文本顺序加入，每个动作仍重新观察；TYPE 使用 BrowserGym `keyboard_type`，不隐式点击或清空。
真实浏览器集成测试验证了 INPUT→TYPE 追加文字→Submit 的执行链路。

所有 3 组诊断的 enter-text 都是 0/10，专用模板没有生成 TYPE。失败原始轨迹显示先点击 Submit，导致环境立即终止。
因此，输入执行器缺失并不能解释当前全部输入失败。证据支持“当前公开 checkpoint + 测试 prompt 组合的多步指令遵循不足”；尚不能将原因单独归结为训练数据、权重导航能力或提示词中的某一项。

本项目使用标准 Qwen2-VL 推理路径；没有把它当作上游自定义 UI token selection 的复现，也未声明重现论文导航分数。

## 后续决策

1. 保留 mixed 作为明确配置的兼容选项，逐步记录修正次数；不把格式修复宣称为策略恢复增益。
2. V0.2 先用一次漏点击验证注入、检测、配对和成本统计。输入任务的错误 Submit 会立刻终止，动作后的反馈无法挽救，因此不放入这一首轮可恢复故障实验。
3. 若弱视觉反馈没有收益，下一步应测试 ShowUI 作为 grounding 模块、由独立 planner 提供明确子目标，并评估提交前的前置条件检查。任何手工任务规划都必须单独标注，不能当成模型自主规划成绩。
4. 这轮 seeds 0–9 用于开发诊断，恢复首轮改用 seeds 10–14；仍是相同任务的小样本，不能称为任务泛化测试。

## 验收与 Git

当前服务器 44 项测试通过（11.01 秒），包含安全解析、向聚焦输入框键入文字、真实漏点击和配对审计。
诊断交付目录保存原始 summary/manifest/JSONL/PNG/HTML；248 张诊断截图已核验 SHA256。

主要提交：`dd0740f` 解析器与上游 web profile；`13f2966` MiniWoB TYPE profile；`1460933` 故障/反馈/配对接口；`c9c5801` 真实 Chromium 验收。
操作方式沿用 `git diff --check`、测试、`git add`、`git commit`、bundle 传输及服务器 `git merge --ff-only`；原实验目录没有覆盖。
