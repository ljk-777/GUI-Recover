# 视觉动作结果验证与选择性恢复：修订后冻结协议

## 目标与边界

前两轮给出了条件纠错成绩，错误由评测器提前选择。本轮撤掉该触发条件，在正常、注入和模型真实错误的混合场景上，运行“观察结果 → 判断 → 至多一次等待/恢复 → 继续任务”。
研究问题是：只用正常与可控故障训练的视觉验证器，能否在少误报下发现模型真实错误，并改善成功率与成本？
这是局部动作效果检查和固定控制器实验，不是自主规划。actor 是同一冻结 ShowUI-2B 底座；本轮不训练动作模型，不做 RL。
相关工作已有动作验证与恢复（[VeriGUI](https://arxiv.org/abs/2604.05477)）和真实错误/恢复轨迹合成（[RoTS](https://arxiv.org/abs/2605.29447)）。本试验聚焦小规模可复现、注入到真实错误迁移、误报和等待预算，不声称首次提出验证器。

## 数据：300 个新场景，630 个计划执行变体

以下初始划分在收集前冻结；正式拟合前的一次去重调整见表后说明：

| split | seeds | 场景 | 任务 |
|---|---|---:|---|
| train | 9000 系列 | 150 | click-button/enter-text 各60，click-test/focus-text 各15 |
| validation | 10000 系列 | 60 | 上述四任务各15 |
| test | 12000 系列（标签审计后全新） | 90 | 四个已见任务与 click-dialog/click-tab 各15 |

每个场景先由冻结 actor 根据截图预测一个真实动作，再同 seed 重置运行 none 和一个固定轮转故障。
CLICK 在 drop-click/offset-click 间轮转；INPUT 在 drop-input/partial-input/wrong-input 间轮转。wrong-input 仍明确限定为原文字追加 `!`，不代表任意文字错误。
所有 drop-input 场景另有 deferred-input 变体。因此计划 train 320、validation 125、test 185 个执行变体；实际异常与无效输出保留并报告。
同场景所有故障、截图和观察阶段都留在同一 split。初始图像跨 split 重合会使审计失败。已见旧场景仅用于工程 smoke，不计新验收。

首次300场景采集在划分审计处停止，尚未拟合分类器或查看test效果：focus-text-9004（train）与focus-text-10001（validation）的初始图像、目标和提案相同。
按只看图像/任务/故障的规则，把10001移入train，与最低seed、同任务同offset-click且不重合的train场景9001交换。最终train的focus-text含10001而不含9001，validation反之；其余不变。
该去重阶段全部300场景和630变体保留，原11000系列test字节完全不变；随后因下述标签问题被整体弃作正式test。训练集中仍有两个seed共享初始图像，不能将150个train seed当作150个独立布局。
`scripts/regroup_verification_data.py`只重排开发集记录与已有特征，不重新推理、不依据效果挑样本；重新计算开发集内矛盾观察标签，并再次严格检查跨split重合为0。
原始采集目录保留；新manifest记录原代码、原配置、所有来源文件哈希与两条移动。历史trace.record.split保留采集时划分，当前划分以新JSONL为准。
这是拟合前的协议修订，不声称最终划分完全未经修改。后续可直接用已修订配置重新收集，或从原始完整失败采集构建：

```bash
git show 68087237da0b2bb3ebf27911d0ec0c6b89dd8571:configs/visual_verification.json > runs/visual-verification-original-config.json
python scripts/regroup_verification_data.py --source runs/visual-verification-001-data --out runs/visual-verification-002-data --original-config runs/visual-verification-original-config.json --config configs/visual_verification.json --collection-commit 68087237da0b2bb3ebf27911d0ec0c6b89dd8571
```

上面的重组命令对应去重修订提交`56b1785`时的配置；最终配置已更换test种子，不能拿它重组旧test。
第二次审计发现旧教师把focus-text成功后的失焦判为动作失败：上游`focus-text.html`在focus回调中明确先`this.blur()`再`endEpisode(1)`。
这会把原生任务成功误记为自然错误、错误排除健康训练参考。旧轮次在115次在线执行后停止，检测和恢复成绩均不作为正式结果；已生成的数据、权重与部分轨迹原样保留。
修正教师并增加真实浏览器回归测试后，只对train/validation的focus记录按原奖励/终止/task_info重新标注，保留旧教师证据、原图、原始trace及特征字节。
全部开发集参考资格和矛盾视觉标签重新计算。旧test的90个场景整体退出正式分母，改用12000–12014共90个新场景；阈值/L2候选、主方法、故障与预算不变。
新test先只reset预检布局重合，不执行动作或查看结果；随后按冻结实现采集。`assemble_revised_verification_data.py`验证开发集不变、新test不复用旧seed及特征ID对齐，记录完整谱系。
最终300个场景中210个开发场景复用，90个test场景全新；含被弃旧test的来源生成累计390次，不把它们计入新的独立研究分母。

none 指没有人为注入，不能直接当作动作成功。必须实际执行后标注效果；模型自然失败与注入故障分别报告。
主分类器只拟合原始参考动作成功的场景及其变体，排除自然失败及对已失败动作再注入的混杂情况；阈值选择也只用此类 validation。
自然失败保留作迁移诊断；test 的 click-dialog、click-tab 未参与拟合或阈值选择。

## 真实效果、不可观察性与等待

离线标签：INPUT 要求输入值与用户目标完全相等；focus使用原MiniWoB对聚焦事件的成功规则，结束后的activeElement只作诊断；其余使用原MiniWoB成功规则。保留原始标签证据与task_info。
模型只收到 TransitionView：前图、当前图、发出的原始动作、用户目标、当前目标、自己已观察的阶段。无 DOM、目标框、奖励、故障名、实际被注入动作、scene ID 或正确标签。

deferred-input 是明确的排队送达夹具，不伪装成真实网络延迟：首次只聚焦；等待150ms时送达原输入，或在下一条实际修复/Submit操作前按FIFO送达。
修复计划基于先前画面；若当时输入框为空，计划补输入，但先送达的原输入会与补输入叠加。这用于检验“先等还是重试”；不能外推到任意时长或真实网络故障。
原生截图完全保留，不通过改图伪造延迟。每个执行变体可记录 immediate 和 after_wait 两个观察，后者包含一次 hover no-op 或待送达输入的实际环境步。
Linux工程测试发现闪烁光标造成46像素的非确定性差异，Mac复核另发现141个原生按钮边缘像素的轻微变化。
本轮在新数据收集前统一隐藏input/textarea的caret，并为按钮指定固定实线边框、关闭原生appearance；不改写PNG、不改变文字、焦点或奖励。
实验截图在CDP切换到1.5倍设备缩放后等待两个requestAnimationFrame再原生捕获，避免立即采样的绘制抖动；普通BrowserGym截图库不打补丁。
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

test 每任务预定12000–12003四个场景，所有 none、注入与延迟变体都进入，共50个计划case；不按哪组成功选案例。
比较六种策略：none、不等待固定重试、先等待再固定重试、appearance触发、frozen验证器触发、特权oracle，共300次计划执行。
oracle 可以读取真实动作效果与待送达标志，明确是特权参考，不视为视觉模型成绩。其他路径不调用教师；集成测试将教师替换为报错函数验证隔离。
环境终止自然结束episode，不在已终止失败后重启；检测表另报告全部状态和仍可干预状态，避免将不可恢复终止错误当作恢复收益。

每组上限相同：一个来源动作、至多一次等待、一次修复、一次INPUT后的Submit，最多3个逻辑actor调用，0次重启。
修复位置由同一actor重新定位；INPUT修复由同一笔画检查器选择：画面为空时补输入，已有文字或无法确认时全选并替换用户指定文字，CLICK采用重新定位后点击。目标切换和文字来自固定任务语法，不读取DOM。
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

## V1 实测结果

原开发集采集冻结于 `68087237da0b2bb3ebf27911d0ec0c6b89dd8571`，教师修正、新test采集、拟合、阈值选择和实际执行冻结于 `b88e567fede19086c4ae83756333ca77a7c9437b`；之后只补充文档与脱敏案例。
完整机器可读记录：[visual-verification-results.json](../examples/visual-verification-results.json)。
旧seed工程smoke使用12个场景、28个变体和18次执行，只作工程验收，全部排除在下列研究分母外。

### 数据与可观察性

正式场景300个（train150/validation60/test90），执行变体630个（320/125/185），最终纳入300次来源生成；含被弃旧test的采集累计390次。旧错误标签轮次的115次在线执行不计下述分母。
同场景的正常、注入、排队输入和不同观察阶段整体划分；按前述拟合前去重修订后，跨split场景和初始截图重合为0。train含150个seed、147种初始画面。
状态统计为 `{'ok': 630}`。无效来源、异常和匹配失败保留在对应记录，不伪造可执行的成功轨迹。

| split | 观察数 | achieved | not_achieved | unknown | 矛盾输入组 |
|---|---:|---:|---:|---:|---:|
| train | 578 | 147 | 292 | 139 | 69 |
| validation | 213 | 43 | 102 | 68 | 34 |
| test | 312 | 63 | 179 | 70 | 35 |

unknown包含排队未送达的首次观察，以及同一可视输入对应不同隐藏结果的情形。原始动作效果真值保留，未将拒判包装成发现了错误。
观察数和多个故障变体不等于独立新场景。只有未终止的状态可按本轮0重启规则干预；终止失败仍在任务分母中。

### 训练与冻结阈值

两组都只用原始参考成功的场景及其正常/注入/排队变体，不拟合自然错误及对已失败动作再注入的样本。
PCA、标准化只用合格train观察；监督损失按场景等权，PCA/标准化统计按这些训练观察计算。
主方法提前指定为frozen，pixels只作次级对照，没有根据test切换主方法。

| 特征 | 拟合观察 / 场景 | L2 | 阈值 | validation健康误触发 | validation注入失败召回 | validation排队过早触发 |
|---|---|---:|---:|---|---|---|
| pixels | 450 / 117 | 0.1 | 0.6 | 0/15 | 37/52 | 0/5 |
| frozen | 450 / 117 | 0.1 | 0.6 | 0/15 | 48/52 | 0/5 |

经验操作点要求健康存活状态误触发不超过5%、排队首次观察过早触发为0。这是小样本选择规则，不能当作真实误报率的统计保证。
1.01阈值为完全拒判回退；低误报必须结合召回和拒判一起看。概率输出未经过另行统计校准。

### 独立test：触发与迁移

主方法在全新test的健康存活状态中误触发1/15（6.7%），高于5%的预设经验门槛，因此本轮不采用该分类器为默认。小分母不足以推断真实误报率已显著高于5%，但不能把validation达标当作test达标，也不在看到test后换阈值或主方法。

下面都是尚可干预的存活状态；健康误触发、注入失败、自然失败、排队过早触发的分母分别保留。
自然失败没有参与拟合或阈值选择。拒判后至多一次等待，再拒判就不强制恢复。

| 方法 | 健康误触发 | 注入失败触发 | 模型自然失败触发 | 排队过早触发 | 等待次数 |
|---|---|---|---|---|---:|
| pixels | 0/15 | 54/72 | 17/18 | 0/5 | 126 |
| frozen | 1/15 | 60/72 | 15/18 | 0/5 | 110 |
| appearance | 0/15 | 62/72 | 17/18 | 5/5 | 0 |

| test任务 | 原始动作达成 /15 | 自然失败可干预 | 自然失败已终止 | 主验证器可干预自然失败触发 |
|---|---:|---:|---:|---:|
| click-button | 7 | 8 | 0 | 8/8 |
| click-dialog | 8 | 7 | 0 | 7/7 |
| click-tab | 12 | 3 | 0 | 0/3 |
| click-test | 15 | 0 | 0 | 0/0（无样本） |
| enter-text | 15 | 0 | 0 | 0/0（无样本） |
| focus-text | 15 | 0 | 0 | 0/0（无样本） |

健康存活状态的分母只覆盖输入尚待Submit的状态；其他任务常在动作成功时立即终止，不能把该误触发率推广到所有正常GUI操作。

| 三分类输出（含终止态和等待后观察） | 接受 / 总观察 | 拒判unknown |
|---|---:|---:|
| pixels | 91/312 | 221 |
| frozen | 120/312 | 192 |
| appearance | 312/312 | 0 |

完整三分类混淆矩阵和每个case的触发/等待决定保留在JSON。三分类的unknown标签与“真实失败是否被恢复”不是同一个指标。
本轮含固定实线按钮/透明caret/同步原生截图夹具，以及FIFO排队输入模拟器，不能外推为原版MiniWoB或真实网络延迟成绩。

### 实际混合场景闭环

在提前固定的50个case上比较六种策略，共300次实际执行，所有策略共享完全相同的来源提案和故障机制。
50个case来自24个场景，不是300个独立场景。无故障组也包含actor自己的错误，并非全是健康状态。

| 策略 | 全部成功 /50 | 无注入 /24 | 人为注入 /24 | 排队 /2 | 修复 | 等待 | 逻辑actor调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 21 | 18 | 1 | 2 | 0 | 0 | 60 |
| retry | 36 | 18 | 18 | 0 | 35 | 0 | 95 |
| wait_retry | 38 | 18 | 18 | 2 | 35 | 35 | 95 |
| appearance | 34 | 18 | 16 | 0 | 28 | 0 | 88 |
| verifier | 36 | 18 | 16 | 2 | 24 | 30 | 84 |
| oracle | 38 | 18 | 18 | 2 | 29 | 2 | 89 |

oracle明确读取教师动作效果和排队状态，是特权参考；其修复位置仍由同一actor预测，不是正确坐标oracle。
wait_retry控制了“先等待再固定重试”，不能把单纯等待带来的收益全归因于学习。
相同输入预测/特征可缓存，逻辑调用和真正新增调用分别记录；共享缓存改变实际耗时，不据此宣称普适吞吐优势。

| 相对none的逐例变化 | 新增成功 | 丢失成功 | 健康原始参考上的丢失 |
|---|---:|---:|---:|
| none | 0 | 0 | 0 |
| retry | 17 | 2 | 0 |
| wait_retry | 17 | 0 | 0 |
| appearance | 15 | 2 | 0 |
| verifier | 15 | 0 | 0 |
| oracle | 17 | 0 | 0 |

运行异常 0 条，前缀不匹配 0 条，全部保留在分母。
每例预算为来源动作加至多一次修复、一次等待和INPUT后的Submit，最多3次逻辑actor调用，0次重启。
细分真假错误、终止状态、异常与成本的完整记录见JSON，不能将局部动作验证等同于自主规划。

### 结果解释与未解决问题

主方法相对不恢复新增15个成功、未丢失已有成功，达到36/50；但低于wait_retry的38/50。它使用24次修复、84次逻辑actor调用，对照为35次修复、95次调用；这只是本夹具和缓存口径下的调用节省，不能据此认定整体更优。
在线预定子集中的6个自然错误，六种策略全部0/6，连特权触发也未能恢复。同一个actor重新定位的修复能力仍不足，不能用注入故障的成功率上涨宣称解决了真实错误恢复。
跨任务检测也不均衡：主方法在按钮/对话框自然错误中分别触发8/8、7/7，在标签页中为0/3；全部观察中192/312被拒判。平均召回不能掩盖漏检和低覆盖率。
在线18个健康参考均保持成功，与离线15个健康存活状态中误触发1个并不矛盾：在线只取每任务前4个seed，未覆盖全部离线健康状态。

因此本轮完成的是可复现的视觉触发与审计工具，学习头尚未通过采用门槛，也没有证明优于简单等待重试。后续应在新的独立test上增加健康/自然错误覆盖、做仅动作/阶段特征消融，并研究能改变错误动作的修复策略，再考虑扩大训练。

### 验收证据与采用范围

本地独立审计验证了 2033 张数据截图、312 个test观察预测、884 张在线截图，
并重新核对教师标签证据、训练子集均值/标准差、冻结阈值选择、权重/ledger/特征/轨迹哈希、真实动作序列和原始task_info成功规则。
分类器与所有投影均为本地实验产物，未上传模型权重。ShowUI底座未改动，没有对动作模型做SFT或RL。
服务器109项测试通过；[冻结实现CI](https://github.com/ljk-777/GUI-Recover/actions/runs/44500189590) 的Ubuntu22.04、24.04和打包检查通过。
GPU0配额12032MiB，约每2秒采样2040次，设备采样峰值5201MiB；采样可能漏掉瞬时峰值。

可视检查器确实在执行时自行决定等待/恢复，未用正确标签选择触发。其效果与适用范围必须按上述全部对照、拒判、误报和迁移成绩评估。
默认Alpha与原始actor保持不变；原始负面结果、低覆盖率、无法观察的状态与恢复失败均保留。后续实验需要新的独立test划分。

- [视觉触发恢复](../examples/visual-verifier-recovery/provenance.json)：结果后的说明案例，原始PNG和日志不改写，不参与选模或计分。
- [未恢复反例](../examples/visual-verifier-counterexample/provenance.json)：结果后的说明案例，原始PNG和日志不改写，不参与选模或计分。
- [固定排队输入对照](../examples/visual-verifier-pending/provenance.json)：结果后的说明案例，原始PNG和日志不改写，不参与选模或计分。
