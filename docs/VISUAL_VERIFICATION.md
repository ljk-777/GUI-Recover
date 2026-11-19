# 视觉动作结果验证与选择性恢复：冻结 V1 协议

## 目标与边界

前两轮给出了条件纠错成绩，错误由评测器提前选择。本轮撤掉该触发条件，在正常、注入和模型真实错误的混合场景上，运行“观察结果 → 判断 → 至多一次等待/恢复 → 继续任务”。
研究问题是：只用正常与可控故障训练的视觉验证器，能否在少误报下发现模型真实错误，并改善成功率与成本？
这是局部动作效果检查和固定控制器实验，不是自主规划。actor 是同一冻结 ShowUI-2B 底座；本轮不训练动作模型，不做 RL。
相关工作已有动作验证与恢复（[VeriGUI](https://arxiv.org/abs/2604.05477)）和真实错误/恢复轨迹合成（[RoTS](https://arxiv.org/abs/2605.29447)）。本试验聚焦小规模可复现、注入到真实错误迁移、误报和等待预算，不声称首次提出验证器。

## 数据：300 个新场景，630 个计划执行变体

`configs/visual_verification.json` 在收集前冻结：

| split | seeds | 场景 | 任务 |
|---|---|---:|---|
| train | 9000 系列 | 150 | click-button/enter-text 各60，click-test/focus-text 各15 |
| validation | 10000 系列 | 60 | 上述四任务各15 |
| test | 11000 系列 | 90 | 四个已见任务与 click-dialog/click-tab 各15 |

每个场景先由冻结 actor 根据截图预测一个真实动作，再同 seed 重置运行 none 和一个固定轮转故障。
CLICK 在 drop-click/offset-click 间轮转；INPUT 在 drop-input/partial-input/wrong-input 间轮转。wrong-input 仍明确限定为原文字追加 `!`，不代表任意文字错误。
所有 drop-input 场景另有 deferred-input 变体。因此计划 train 320、validation 125、test 185 个执行变体；实际异常与无效输出保留并报告。
同场景所有故障、截图和观察阶段都留在同一 split。初始图像跨 split 重合会使审计失败。已见旧场景仅用于工程 smoke，不计新验收。

none 指没有人为注入，不能直接当作动作成功。必须实际执行后标注效果；模型自然失败与注入故障分别报告。
主分类器只拟合原始参考动作成功的场景及其变体，排除自然失败及对已失败动作再注入的混杂情况；阈值选择也只用此类 validation。
自然失败保留作迁移诊断；test 的 click-dialog、click-tab 未参与拟合或阈值选择。

## 真实效果、不可观察性与等待

离线标签：INPUT 要求输入值与用户目标完全相等，focus 要求实际聚焦目标输入框，其余使用原 MiniWoB 成功规则。保留原始标签证据与 task_info。
模型只收到 TransitionView：前图、当前图、发出的原始动作、用户目标、当前目标、自己已观察的阶段。无 DOM、目标框、奖励、故障名、实际被注入动作、scene ID 或正确标签。

deferred-input 是明确的排队送达夹具，不伪装成真实网络延迟：首次只聚焦；等待150ms时送达原输入，或在控制器修复后、正常继续前送达。
过早重试可能与随后送达的输入叠加。这用于检验“先等还是重试”；不能外推到任意时长或真实网络故障。
原生截图完全保留，不通过改图伪造延迟。每个执行变体可记录 immediate 和 after_wait 两个观察，后者包含一次 hover no-op 或待送达输入的实际环境步。
Linux工程测试发现闪烁光标造成46像素的非确定性差异，Mac复核另发现141个原生按钮边缘像素的轻微变化。
本轮在新数据收集前统一隐藏input/textarea的caret，并为按钮指定固定实线边框、关闭原生appearance；不改写PNG、不改变文字、焦点或奖励。
这是明确标注的确定性控件渲染夹具，所有策略相同，不声称为原版MiniWoB渲染成绩。既有Alpha与普通环境reset不受此实验夹具影响。
阶段字段仅表示控制器自己的观察次数，不是环境提供的执行成功确认。

原始效果标签与视觉标签分开。同样的模型输入若出现不同隐藏结果，离线审计将这些观察标为 unknown 并保留原始真值；运行时不查此表。
pending 的首次观察标 unknown。每次验证只看原始前图和当前图；unknown 最多等待一次，再验证，仍 unknown 就不强制修复。

## 特征与轻量训练

- appearance：原有输入笔画检查与整图像素变化，明确不核对文字内容。
- pixels：固定全图与点击附近像素、差分、动作和观察阶段特征。
- frozen（预先指定主方法）：共享 actor 的冻结视觉编码器池化特征、冻结文本 embedding 和上述像素特征，不再加载第二个大模型。

小分类器为训练集标准化、PCA16加显式动作/阶段特征、三分类线性 softmax。场景等权，避免一个场景多种故障或重复观察被当作独立新样本。
L2固定候选0.01/0.1/1；600次CPU全批更新。PCA、标准化和权重只用 train，不读取 test；数据缓存虽可提前计算，三套特征分别落盘。
只在 validation 选择 L2 与置信阈值。目标为正常存活状态误触发不超过5%、pending过早触发为0，在合格候选里最大化注入失败召回；固定规则打破平局。
可选择1.01阈值作为完全拒判回退，不为追求召回而突破预算。小样本达到操作点不构成真实误报率的统计保证。
所有投影、特征、选择文件与数据记录均有哈希；test 前核验训练 ledger。概率只是分类头输出，不宣称已统计校准。

## 实际执行验收

test 每任务预定11000–11003四个场景，所有 none、注入与延迟变体都进入，共50个计划case；不按哪组成功选案例。
比较六种策略：none、不等待固定重试、先等待再固定重试、appearance触发、frozen验证器触发、特权oracle，共300次计划执行。
oracle 可以读取真实动作效果与待送达标志，明确是特权参考，不视为视觉模型成绩。其他路径不调用教师；集成测试将教师替换为报错函数验证隔离。
环境终止自然结束episode，不在已终止失败后重启；检测表另报告全部状态和仍可干预状态，避免将不可恢复终止错误当作恢复收益。

每组上限相同：一个来源动作、至多一次等待、一次修复、一次INPUT后的Submit，最多3个逻辑actor调用，0次重启。
修复位置由同一actor重新定位；INPUT修复采用点击、全选、替换用户指定文字的固定操作，CLICK采用重新定位后点击。目标切换和文字来自固定任务语法，不读取DOM。
记录逻辑调用、真正新增生成、视觉编码调用、等待、实际环境步与耗时。相同确定性预测可缓存，缓存命中与真实调用分开，不据此宣称通用吞吐提升。
正常误触发、pending过早重试、注入/真实失败召回、拒判与覆盖率、任务成功、正常退化、运行异常、前缀不匹配均保留。真实GUI结果由原始奖励/终止/task_info独立判定。
在线沿用离线test场景，是实际执行验收，不是额外独立样本。全部变体按场景统计，不将300次执行当作300个独立场景。

## 运行与验收命令

```bash
git switch feat/visual-outcome-verifier
git diff --check
python -m pytest -q
export CUDA_VISIBLE_DEVICES=0
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-sft/bin/python scripts/collect_verification_data.py --config configs/visual_verification.json --model models/ShowUI-2B --out runs/visual-verification-data-001
.venv-sft/bin/python scripts/train_visual_verifier.py --config configs/visual_verification.json --data runs/visual-verification-data-001 --out runs/visual-verifiers-001
.venv-sft/bin/python scripts/eval_visual_verifier.py --config configs/visual_verification.json --data runs/visual-verification-data-001 --classifiers runs/visual-verifiers-001 --model models/ShowUI-2B --out runs/visual-verification-eval-001
```

每个输出目录必须全新。先以 `configs/visual_verification_smoke.json` 在旧seed上验收工程链路，其成绩不进入新场景结果。
GPU0是当前已分配设备，忙时等待，不切其他GPU。实现与协议先提交，再运行，之后只补结果文档/脱敏示例。
主方法未达到验收时保留负面结果，不在test上换阈值、改解析或挑更好的模型。默认Alpha与原始模型保持原版本，分类器权重只作本地实验产物。
