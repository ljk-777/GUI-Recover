# 漏输入恢复：视觉输入检查

## 目标与边界

在明确标记的 MiniWoB 脚本规划基线上，检查 INPUT 后输入框是否仍为空；只在确认空白时重新定位并补输入，最多一次。
输入仍未生效或区域无法识别时阻止提交。

这是对“整段键入丢失”的受控故障实验，不是通用文本识别、任意网页表单验证或模型自主规划。
只判断文字笔画出现，不核对内容是否正确、完整；部分输入、错误文字也可能通过该检查。

## 视觉信号

`input_guard.inspect_input(before_image, after_image, proposed_position)` 只接受两张截图和拟执行位置。
它在输入前截图中，从预测点附近寻找封闭的白色矩形区域，排除不符合面积、形状和边界条件的区域。
在相同截图位置检查输入后是否出现足够的深色笔画，并忽略窄竖线状光标。

真实浏览器测试发现 Chromium 聚焦边框会向内部加粗，初版将其误判为文字；当前在区域内侧再去掉两像素，测试覆盖了该焦点案例。
该边距按当前固定 MiniWoB 498×321 截图验证，不能直接承诺适用于不同缩放、主题、字体或动画。
大面积背景变化、布局移动、预先已有文字或无法推断空白矩形时返回 unknown，不进行盲目追加。

检测器、规划器和模型都不接收 DOM、输入值读取结果、环境奖励或注入标签。
单独的离线重放审计脚本可以读取 DOM 输入值，为原始检测结果标注真值；这些标签不会回流至策略。

## 故障与恢复

`--fault drop-input` 在首次 INPUT 时保留点击聚焦，将整段键入丢弃。执行动作从 INPUT 变为同坐标 CLICK，仍通过真实环境 step。
每个 episode 只注入一次；原动作、实际动作、注入 seed 和截图全部记录。

`--recovery input-guard` 使脚本计划等待视觉输入检查：

- text_visible：进入定位 Submit 的步骤。
- empty：重新观察、再次用 ShowUI 定位 text field，再 INPUT 原任务文字；最多一次。
- unknown 或连续两次 empty：本地停止并计为失败，不点击 Submit。

`--recovery input-retry` 是固定重试对照：每次第一次 INPUT 后都再 INPUT 一次，不看图像是否已有文字。它用于区分检测收益与多一次尝试的收益，默认关闭。
所有恢复策略保持显式启用，原先的默认脚本策略行为不变。

## 实验协议

开发校准使用此前保存的 seeds 20–29 成功轨迹，以及浏览器测试 seed 7；下面使用新的 seeds 30–39。
固定公开 ShowUI-2B 权重、脚本规划、mixed 解析、相同图像/输出预算；每个 episode 最多 4 次决策、4 个执行动作。

| 组 | 故障 | 恢复 |
|---|---|---|
| A | 无 | 无 |
| B | drop-input | 无 |
| C | drop-input | input-guard |
| D | 无 | input-guard |
| E | drop-input | input-retry |
| F | 无 | input-retry |

A/B/C/D 使用配对审计，模型与规划配置完全一致，唯一允许的配置差异为 input_guard.enabled。
B/C 注入前图片哈希、原始预测及注入动作必须完全匹配。E/F 提供相同尝试次数的机制对照，独立报告。
相同预算上限不代表实际调用一样多；恢复额外动作、真实模型调用和检测耗时必须报告。

## 复现

```bash
source .env
for arm in clean faulted recovered clean-recovered retry-faulted retry-clean; do
  fault=none
  recovery=none
  case "$arm" in
    faulted) fault=drop-input ;;
    recovered) fault=drop-input; recovery=input-guard ;;
    clean-recovered) recovery=input-guard ;;
    retry-faulted) fault=drop-input; recovery=input-retry ;;
    retry-clean) recovery=input-retry ;;
  esac
  .venv/bin/gui-recover run --policy showui-scripted --model models/ShowUI-2B --parser mixed --tasks enter-text --seeds 30 31 32 33 34 35 36 37 38 39 --max-steps 4 --fault "$fault" --fault-seed 20261119 --recovery "$recovery" --out "runs/input-guard-$arm-001"
done
.venv/bin/gui-recover compare --clean runs/input-guard-clean-001 --faulted runs/input-guard-faulted-001 --recovered runs/input-guard-recovered-001 --clean-recovered runs/input-guard-clean-recovered-001 --out runs/input-guard-comparison-001.json
```

输出目录不能已存在。compare 会核对任务/seed、预算、模型、依赖、固定环境版本与干净代码版本；脚本策略只允许 input-guard 模式进入相应对照，不与自主模型成绩混用。

## 2026-11-19 结果

模型实验代码提交 `ef4542a`，所有运行的 manifest 均记录同一干净版本。
后续只增加离线审计脚本（`e64547d` / `ef72fc9`），没有修改冻结模型实验的信号规则或阈值。

| 策略 | 正常输入 | 漏输入 |
|---|---:|---:|
| 普通脚本执行 | A：10/10 | B：0/10 |
| 视觉输入检查 | D：10/10 | C：6/10 |
| 固定追加重试 | F：0/10 | E：10/10 |

B/C 的 10 对样本均已注入，注入前截图、原始预测及注入动作全部匹配；未注入和前缀不匹配均为 0。
配对新增挽救 6/10，成功 seeds 为 30、31、32、33、36、37。正常任务加检查未观察到成功退化。
另外 4 个真实空白样本（34、35、38、39）在区域内残留一个深色像素，信号返回 unknown，任务被停止而没有完成。
这 4 个样本仍计为失败；本轮没有利用这些评测样本放宽阈值。

固定重试在故障组完成了全部任务，但在正常组将文字重复追加，导致全部失败。
说明本轮视觉检查的价值在于有条件重试并保留正常任务；其漏输入召回仍需提高。
不是所有成功恢复都能单独归因于检测算法：固定重试对照明确显示，多一次尝试也能修复已知输入丢失。

### 调用与步骤成本

| 组 | 实际模型调用 | 执行动作 | policy 决策 | 检查次数 | 有条件重试 |
|---|---:|---:|---:|---:|---:|
| A | 20 | 20 | 20 | 0 | 0 |
| B | 20 | 20 | 20 | 0 | 0 |
| C | 22 | 22 | 26 | 16 | 6 |
| D | 20 | 20 | 20 | 10 | 0 |
| E | 30 | 30 | 30 | 0 | 固定 10 |
| F | 30 | 30 | 30 | 0 | 固定 10 |

每个被成功挽救的任务使用 3 次模型调用/3 步，比原来的 2 次/2 步多一次。
C 中 4 次 unknown 停止只调用一次模型，再返回本地停止，因此全组总调用为 22，而不是 30。
不能将提前终止造成的较低开销解释为效率优势。D 无额外模型调用或输入追加。

### 独立审计

离线重放 C/D 共 20 个 episode，重置截图哈希和最终成败均与原实验一致。
审计脚本使用真实执行动作重放，在每次 INPUT 后读取 DOM 输入值，作为离线标签；未调用模型，标签不会影响原策略。
另外用原始 PNG 重新计算信号，26 次结果全部复现。

- 真实空白状态 10：检测为空 6，unknown 4；将 unknown 计为未检出时，空白召回率为 6/10。
- 真实非空状态 16：均识别出文字出现，没有误判为空。
- 本轮空白判断 precision 为 6/6，recall 为 6/10；仅限这批单一布局、单一故障的小样本。
- 检测函数服务器离线复测平均 48.63 毫秒、最大 57.83 毫秒；不含 PNG 读取、模型推理或完整浏览器流程，不能当成端到端耗时。

六组 60 个 episode 的 202 张截图、20 张重放 reset 图片均按 SHA256 核验；六组首步状态与预测一致。
固定重试两组的模型/规划配置与主对照相同，只有显式的重试模式开关不同。

### 测试、交付与后续

64 项测试通过（19.82 秒），包括真实漏输入、正常输入不重复、焦点边框和光标干扰、持续空白停止、固定重试和无注入标签泄漏。
模型实验无 runtime/setup 错误或显存溢出。所有成功/失败轨迹和结果均已取回，原始实验目录未覆盖；结束后 GPU 显存归零。

关键提交：`d36e498` 输入故障/视觉检查接口；`ef4542a` 焦点边框修复与固定重试对照；`ef72fc9` 独立审计。
操作遵循测试 → `git diff --check` → `git add` / `git commit` → bundle 传输 → 服务器 `git merge --ff-only`。

下一阶段先处理单像素噪声，并使用尚未参与开发的 seeds 验证召回和正常任务成本；同时增加“部分输入/错误文字”反例。
当前 ink 信号会接受已有的可见文字，不具备内容正确性验证能力，也没有证明适用于深色输入框、占位文字、任意布局或不同浏览器缩放。
