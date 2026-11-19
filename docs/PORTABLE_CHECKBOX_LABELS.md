# Portable checkbox label matching

The previous screenshot verifier abstained on two checkbox layouts where the top row led by a wide margin but the absolute glyph score was below its threshold. Pillow also selected different font layout engines on Linux and macOS. The new profile pins BASIC layout, renders a native 15 px reference and a 4× reference downsampled to the screenshot resolution, and smooths pixel coverage in the same way for the template and screenshot. The two scales must choose the same row. The stronger match must score at least 0.87 and lead by at least 0.18; both margins must reach 0.12. Armed recovery actions still require the previous checked-state, attribution and reinspection checks.

The profile is opt-in. It expects the known native MiniWoB checkbox rendering and a goal that refers to a label present on the form. A high similarity score alone cannot establish that an absent near-match string is present. For this reason the study explicitly probes absent lookalike labels offline, on the same untouched frame, without letting the probe trigger an action. Labels are collected by the benchmark fixture exclusively for ground-truth absence checks. The verifier sees the altered task text and the image, without access to that list.

## Study

Development used 38 previously recorded layouts. The fresh comparison reserves 100 seeds in the 44000 series and takes 20 reachable unique checkbox layouts while excluding previously registered layouts. The same before/after frames feed both the Linux legacy and the portable matcher. A correct first click, blank click and wrong checkbox click where available are real browser actions. Only the portable path executes repair. A scripted teacher continuation verifies full task success after a correct or repaired first step. Separate engineering seeds are excluded from the formal result.

Two previously ambiguous natural errors are replayed separately with cached prior actions and exact screenshot checks. These are regression cases from development. Fresh browser traces do not use actor calls, and the visual matcher requires no GPU. The baseline retains its existing RAQM font layout and thresholds. The portable renderer is audited again using the same font and saved screenshots on macOS.

```bash
python scripts/run_portable_checkbox_labels.py \
  --config configs/portable_checkbox_labels.json --profile test \
  --font /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \
  --previous-run runs/target-state-001 \
  --earlier-plan runs/crop-control-multistep-001/multistep-plan.json \
  --out runs/portable-labels-001
```

## 正式结果

实现冻结在 `42aa879`。20 个新布局通过截图去重，另有一个无需勾选的目标在预检排除。四个布局的所有选项都属于目标，无法构造误点控件分支。

| 新布局评测 | 旧版触发 | 双尺度触发 | 双尺度执行结果 |
|---|---:|---:|---:|
| 正常首击 /20 | 0 | 0 | 20 个保留正确状态 |
| 点空白 /20 | 20 | 19 | 19 个恢复 |
| 点错控件 /16 | 16 | 15 | 15 个恢复 |

同一个布局贡献了三条暂缓判断。两个尺度都定位到正确行，其中最高匹配分数为 0.867，低于预先登记的 0.87。该布局的正常点击仍得到保留，两个错误分支没有恢复。旧版触发是决策对照，新的恢复结果来自实际浏览器点击。34 次新布局修复均得到准确首步状态，后续在固定教师续步下完成任务。

历史种子 43009 和 43013 的自然错误在核验原始截图和模型点位后重放。旧版均暂缓。双尺度批准各一次补点，两个都恢复到正确勾选状态。历史案例用于回归检查，正式新布局样本与之独立。[历史重放证据](../examples/portable-checkbox-historical-cases.json)

18 个相似但不存在的标签探针中，旧版错误建议修复 10 次，新方案建议 7 次。三例从错误建议变为暂缓，未出现反向变化。这些探针只修改请求文字，使用原截图在离线条件下评测，未执行点击。缺失标签仍可能被误认，因此双尺度作为可选的研究配置。

同一组保存的截图、字体文件和 FreeType 2.14.3，在 Linux 和 macOS 上复算新方案的判断及坐标一致，最大分数或间隔差为 7.8 × 10⁻¹⁶。这项核验覆盖跨平台评分，浏览器动作统一在冻结的 Linux Chromium 夹具执行。

独立审计通过 56 条新轨迹、2 条历史回放、352 张截图与 51 次修复点击。整轮无 GPU 使用和新增模型推理。两套 Ubuntu 与打包检查通过。[机器可读结果](../examples/portable-checkbox-label-results.json) · [冻结实现 CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44543315857)
