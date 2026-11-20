# Fixed crop controls and normal intermediate clicks

The contextual repair study recovered more natural failures with action-centered zoom and feedback. That experiment could not isolate the benefit of crop placement from the benefit of making the MiniWoB content larger. Its live healthy controls were all INPUT actions, so the detector could also rely on action type. This follow-up tests both explanations with the existing frozen actor and detector thresholds.

## Crop placement

The formal configuration reserves 80 seeds for each of click-button, click-dialog, and click-tab. The reset-only preflight accepts the first 20 unique layouts per task, excluding previously registered source layouts. A frozen ShowUI prediction supplies the first click in each scene. Every natural failure is retained in the conditional denominator, including terminal failures. The diagnostic uses the known first-click error to trigger each repair attempt and allows one additional click with zero restarts.

| Method | Text | Screenshot region |
|---|---|---|
| Plain | Original target | Full screenshot |
| Feedback | Same feedback in all four feedback conditions | Full screenshot |
| Action-centered zoom | Feedback | 60% of width and height centered on the first proposal, clamped at the edges |
| Fixed zoom | Feedback | Same width and height, fixed to the upper left |
| Fixed content crop | Feedback | MiniWoB's standard 160×210 CSS content panel, from the 332×214 CSS viewport |

All crop inputs are enlarged to the original canvas dimensions with bicubic interpolation. The output coordinates are mapped back to the original screenshot before they are executed. Native PNGs, derived inputs and crop coordinates are available separately. The fixed content crop includes more vertical content than the equal-size comparison and therefore changes the relative resize factors. Report actual task success, paired differences and post-evaluation target-center coverage separately. Target geometry participates only in evaluation. Keep the prior repair selection and report all methods.

## Normal clicks while the task is still live

The formal profile reserves 120 seeds per task and accepts 24 layouts per task for click-button-sequence and click-checkboxes. Before inference, filter exact screenshot duplicates, goals with zero required checkbox clicks, duplicate visible control labels and setups without an actionable point on any required control. The test uses the repository's existing deterministic rendering and capture fixture on stock MiniWoB task logic.

Each accepted setup provides a correct first click, a click in the blank right margin, an incorrect control click where available, and a first click from frozen ShowUI. Replays start from the same initial screenshot. The sequence log and checkbox checked state give the first-step labels. The selected checkbox must match the specified first target even if a different requested checkbox would eventually satisfy the full task. The correct control must still be live, and the remaining required actions are executed using fixture coordinates to prove the initial click allows full success. This continuation is an execution check with a privileged controller.

Detector inputs use the screenshot pair, the observed click coordinates, the goal, and the current target. A shared hover captures the hypothetical observation after waiting. Each detector's reported trigger uses the original threshold, with a second decision only when its immediate verdict is unknown. DOM state and fixture coordinates do not provide verifier inputs beyond the executed action itself. Detector comparisons are first-step decisions. They do not execute a recovery or report end-to-end policy success.

Report false triggers on healthy live clicks, detection on blank clicks and wrong controls, natural live first-click results, and unresolved verdicts. A button sequence can accept a click with no observable change to the screenshot. A checkbox can visibly change even when the wrong box is checked. These cases test separate limits of appearance-based detection. Natural and controlled transitions share layouts and should keep separate denominators.

## Run

From the repository root, set the same MiniWoB and browser environment as the contextual repair study and use the cached ShowUI model and detector artifacts.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_crop_control_multistep.py \
  --config configs/crop_control_multistep.json --profile formal \
  --model models/ShowUI-2B --inherited-detectors runs/visual-verification-003-models \
  --prior-development runs/contextual-repair-001-development \
  --reference runs/visual-verification-003-data \
  --reference runs/visual-verification-001-data \
  --reference runs/contextual-repair-001-test/data \
  --out runs/crop-control-multistep-001
```

The smoke profile has separate seeds and counts and is for checking the pipeline. The formal report uses only the committed formal config and its reserved layouts.

## 正式裁剪对照结果

冻结实现为 `2ef07eb`。60 个初始布局全部唯一且通过预检，24 个首击自然失败，其中 21 个仍可操作，3 个已终止。每组共享 24 个来源和同一次修复上限。120 次分支重放全部通过截图前缀核验，运行异常为零。原始成功由真正的浏览器动作和 MiniWoB 任务奖励判定。

| 修复输入 | 按钮 /10 | 对话框 /8 | 标签页 /6 | 实际恢复 /24 |
|---|---:|---:|---:|---:|
| 原图、无反馈 | 0 | 0 | 0 | 0 |
| 原图、同一反馈 | 0 | 1 | 0 | 1 |
| 60% 自适应裁剪 | 6 | 8 | 5 | 19 |
| 60% 固定裁剪 | 7 | 7 | 5 | 19 |
| 固定内容区裁剪 | 1 | 0 | 0 | 1 |

两个 60% 方案共同成功 17 个，各自独有成功 2 个，3 个共同失败。21 个能实际修复的案例中，6 个因边界夹紧得到了完全相同的裁剪，两种方案都成功。只看裁剪位置不同的另 15 个案例，两种方案仍各成功 13 个。

所有 21 次可执行修复均有可见目标，且每种裁剪都覆盖至少一个正确目标的中心。固定内容区裁剪的收益仍接近原图。这个对照同时改变了裁剪宽高和放大比例，结果无法分辨具体是哪一项导致差异。三个任务的受控布局支持把 60% 固定裁剪当作同预算基线，现有数据未测到按错误坐标调整位置带来的额外收益。

## 多步检测结果

按钮顺序和复选框各接受 24 个布局。另有两个零选项目标在重置预检被排除。189 条首击轨迹均保持可操作，48 个正确的受控首击随后都能按预定步骤完成任务。三个复选框布局中的所有选项均属于正确目标，因此无法构造点错控件的分支。

| 检测器 | 正常受控误触发 /48 | 空点触发 /48 | 错点控件触发 /45 | 自然正常误触发 /23 | 自然错误触发 /25 |
|---|---:|---:|---:|---:|---:|
| 冻结视觉 | 39 | 47 | 41 | 23 | 14 |
| 像素特征 | 41 | 39 | 40 | 23 | 18 |
| 仅动作与阶段 | 48 | 48 | 45 | 23 | 25 |
| 全图变化规则 | 48 | 48 | 45 | 23 | 25 |

动作与阶段模型会在所有受控正常首击上误触发。上一轮活跃状态中的正常动作和失败动作类型不同，本轮两类状态均来自点击。冻结视觉头仍误触发 39/48，超过原来的 5% 门槛。在自然首击组，23 个正确案例均来自按钮顺序；25 个错误中，1 个点错顺序按钮，7 个点错复选框，17 个没有改变勾选状态。分组大小和任务分布不同，受控组应作为检验正常点击误报的主要对照。

按钮顺序组 24 个正确首击和 24 个错误控件首击的前后截图均完全一致，内部状态却不同。[第一个接受布局的原图和三个分支证据](../examples/multistep-no-visible-change/evidence.json)展示了这个现象。复选框正确和错误控件首击的全图差异范围都约为 0.000993–0.001008，低于继承的 0.002 阈值。阈值下调可以识别变动，但两组差异范围相同，仍需判断哪个目标发生了什么变化。检测触发是在共享截图对上重算的路由决策，正确分支的任务成功来自受控续步。

## SFT 数据预备与验收

单独重放上一轮训练分区的 33 个自然错误，使用此前冻结组合给出的坐标。22 个案例实际恢复，筛去 4 个未通过目标框检查的动作后保留 18 个监督样本。输入保存原生错误后截图和反馈提示，答案是原图上的归一化坐标。每条样本关联其成功重放，包含截图、轨迹和训练清单的校验值。18 条均为按钮任务，完整训练还需要正常动作对照和更多任务来源。本次验收运行冻结模型，训练样本来自原有训练分区。

独立审计复核了 120 次裁剪比较、105 个派生输入和 189 次多步首击的轨迹、图像哈希和路由。两套 Ubuntu 测试和打包检查通过。正式运行的单卡显存采样峰值为 5427 MiB，分配上限 12032 MiB。[冻结实现 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44531903778) · [结构化汇总](../examples/crop-control-multistep-results.json)
