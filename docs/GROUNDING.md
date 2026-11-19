# Grounding 与脚本规划诊断

## 本模块要验证什么

将多步指令直接交给模型时，文本输入任务经常提前提交。本模块检查同一 checkpoint 在明确元素描述下是否更擅长定位。
定位得分和完整任务成功率独立报告；输入框、提交按钮均由模型给出坐标。

## 实现边界

`ShowUIPolicy.ground(screenshot, query)` 使用上游 Quick Start 的 grounding 提示词及“文本→图片→元素描述”的顺序。
它与导航调用共用相同处理器、BF16 模型、图像 token 预算和生成预算，输出严格的归一化 `[x,y]`。

`ScriptedGroundingPolicy` 是有意设置的脚本规划基线，支持三种固定的 MiniWoB 指令语法：

| 指令类型 | 子目标与动作 |
|---|---|
| Click the button. | 定位 Click Me! 按钮 → CLICK |
| Click on the "…" button. | 定位同名按钮 → CLICK |
| Enter "…" into the text field and press Submit. | 定位 text field → INPUT 指令中的文字；重新观察后定位 Submit button → CLICK |

规划仅依赖任务文字，不读取 DOM 或目标坐标。它按执行步数推进，尚无视觉完成检查；漏点击或填错字段后可能仍推进到提交。
脚本不能处理任意网页任务，结果不能称为 ShowUI 自主规划或恢复成绩。

## 同图定位评分

`scripts/capture_grounding.py` 保存 PNG、文字输入和独立的 `evaluator-labels.json`。
标签通过 DOM 获得可见目标矩形，只供评分；模型调用只传入 PIL 截图和任务/元素描述。
click-button 的同名按钮可能有多个，按原任务定义接受其中任意一个。
坐标使用 BrowserGym 截图空间；将 CSS 元素框乘截图缩放系数并归一化，裁剪至可见视窗。

评分为“预测点位于至少一个可见目标框内”。无效输出保留在分母，目标不可见情况单列；矩形命中不保证没有遮挡或任务能完成。

对每张初始截图比较：

1. `navigation-full`：原始导航 prompt + 完整任务，提取下一动作坐标。
2. `grounding-full`：grounding prompt + 完整任务，输出一个点。
3. `grounding-subgoal`：grounding prompt + 明确元素描述，输出一个点。

前两种只按首个必要目标评分，三个任务共 30 个首步样本。输入任务首步目标为输入框。
另有 10 个 Submit 按钮子目标，只用于额外定位检查，不与首步 30 个样本混为同分母。
完整多步任务并不是单元素描述；grounding-full 是诊断压力条件，不是上游推荐的使用合同。

## 复现命令

在已配置环境的项目目录执行，使用新的输出目录：

```bash
source .env
.venv/bin/python scripts/capture_grounding.py --seeds 20 21 22 23 24 25 26 27 28 29 --out runs/grounding-cases-001
.venv/bin/python scripts/probe_grounding.py --model models/ShowUI-2B --cases runs/grounding-cases-001 --out runs/grounding-probe-001
.venv/bin/gui-recover run --policy showui-scripted --model models/ShowUI-2B --parser mixed --tasks click-test click-button enter-text --seeds 20 21 22 23 24 25 26 27 28 29 --max-steps 10 --out runs/showui-scripted-001
.venv/bin/gui-recover run --policy showui --model models/ShowUI-2B --parser mixed --tasks click-test click-button enter-text --seeds 20 21 22 23 24 25 26 27 28 29 --max-steps 10 --out runs/showui-navigation-control-001
```

两个在线 run 使用相同初始任务/seed，任务成功仍由环境验证，不能由定位框评分代替。
manifest 的 `scripted_planner` 和 `model_config.planner` 明确标记脚本策略。现有恢复 compare 命令不接受脚本策略，避免混用自主模型与手工规划实验。

源代码提交 `8e88798`。关键 Git 操作：`git diff --check` 和测试后提交，用 bundle 传输，再通过 `git fetch` / `git merge --ff-only` 同步；原诊断和恢复实验未覆盖。

## 2026-11-19 实测结果

### 同图定位

三种条件在同一初始截图上调用，以下只比较相同的 30 个首步必要目标：

| 条件 | click-test | click-button | enter-text 输入框 | 合计 |
|---|---:|---:|---:|---:|
| navigation-full | 10/10 | 2/10 | 0/10 | 12/30 |
| grounding-full | 10/10 | 2/10 | 10/10 | 22/30 |
| grounding-subgoal | 10/10 | 2/10 | 10/10 | 22/30 |

额外的 10 个 Submit 子目标，grounding-subgoal 命中 10/10。
所有 100 次模型输出均可解析，所有目标在截图中可见。
这轮没有观察到“明确子目标文字”相对于 grounding-full 的额外定位收益；明显差异来自 grounding 与 navigation 的输出合同/提示词模式。
不能由这些结果单独断言模型权重缺少某一种能力或某个提示词因素就是唯一原因。

### 在线任务

两组使用相同 seeds 20–29、相同初始截图、相同模型与最多 10 次决策/动作的预算：

| 策略 | click-test | click-button | enter-text | 总成功数 |
|---|---:|---:|---:|---:|
| 原导航策略 | 10/10 | 5/10 | 0/10 | 15/30（50%） |
| 脚本规划 + ShowUI grounding | 10/10 | 5/10 | 10/10 | 25/30（83.3%） |

新增成功来自 10 个输入任务，旧的成功没有退化。所有在线 reset 截图哈希均与同图诊断集一致。
输入任务使用 2 步：模型定位输入框后 INPUT，再重新观察、定位 Submit 并 CLICK。
没有将 DOM 坐标、字段内容读取结果或奖励给模型/脚本规划器。

这是手工规划基线的改善；不是自主规划增益、恢复增益、完整 MiniWoB 成绩或任务泛化结论。
本轮脚本策略的失败为 4 次计划耗尽、1 次点击错误。缺少动作完成验证的问题仍然存在。

### 执行与矩形评分的差异

按钮任务的静态严格矩形命中 2/10，但实际点击成功 5/10。
额外重置并重放保存的 10 个模型点，得到相同的 5 次成功；每个 reset 图片哈希与原始样本一致，没有重新调用模型。
严格框外但真实点击成功的样本：click-button seeds 20、22、29。
记录了连续 CSS 坐标、`elementFromPoint` 返回元素、矩形框和环境奖励，便于检查亚像素边缘命中差异；没有放宽原始矩形评分阈值。
因此矩形命中和真实执行应作为不同指标，不宜相互替代。

### 成本、统计修复与验收

- 原导航策略：66 次模型调用、66 个执行动作；平均每次预测约 0.781 秒，模型预测时间合计 51.55 秒。
- 脚本规划：40 次模型调用、40 个执行动作；平均每次预测约 0.442 秒，模型预测时间合计 17.67 秒。
- 脚本策略另外返回了 4 次本地“计划结束”，因此 `predictions`（policy 决策次数）为 44；不能将它们算作模型调用。
- 上述调用数按 `8e88798` 的原始轨迹和代码路径独立推导，原始文件未回写。单次模型预测计时不包含模型加载、完整浏览器步骤或网络下载。
- 提交 `93d27c0` / `3f7318f` 添加明确的 `model_invocations` / `model_calls` 字段并清理停止时的过期 token 指标。对策略不提供调用计数的情况，model_calls 记为 null，避免猜测。
- 服务器单样本验收验证了“2 次 policy 决策、1 次真实模型调用”，行为与原失败样本一致。
- 脚本/导航 PyTorch 峰值 reserved 分别约 4.73/6.20 GiB，无显存溢出。测试结束显存占用归零。
- 最终 54 项测试通过（13.06 秒）；取回的初始截图、在线轨迹截图、定位评分与边界重放均已核验。

本轮应保留 ShowUI grounding 作为独立适配器。下一模块优先在这个更稳定的脚本基线上加入可控的输入/点击故障和动作完成检查，验证恢复能否挽救真实失败；独立 learned planner 需要另设实验，不能继承脚本规划成绩。
