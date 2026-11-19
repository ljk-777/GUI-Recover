# 失败反馈与局部放大的冻结修复实验

## 目标和依据

上一轮视觉触发取得36/50，低于先等待重试的38/50；在线6个自然错误连特权触发也全部未恢复。
核对原始预测后，其中5个修复重复原坐标，另1个换坐标仍失败。这提示需区分修复输入不足、视觉分辨率和动作模型能力，不能仅靠更多触发改善任务成功。
本轮固定同一ShowUI-2B底座，先隔离修复动作，再接回视觉触发，不训练动作模型。
既有历史提示/SFT未稳定通过正常性能门槛，本轮分别检验反馈和视觉裁剪的影响。

## 输入对照与信息边界

| mode | 当前截图 | 查询 |
|---|---|---|
| plain | 完整原图 | 原始当前目标，字节保持原格式 |
| feedback | 完整原图 | 上一步动作类型/值、温和的重新检查请求、当前目标置于最后 |
| zoom | 固定局部放大 | 原始当前目标 |
| feedback_zoom | 固定局部放大 | 同feedback |

反馈不包含上一坐标，避免直接复制坐标；不声称与完整历史提示等信息。坐标仍可用于确定裁剪区域。
裁剪以模型上一提案位置为中心，宽高各取原图60%，靠边时整体平移以保持大小；再放大到与原图相同画布尺寸。
无目标框、DOM、成功标记或教师错误类别参与区域选择。预测先按局部坐标解析，再映射回原图归一化坐标执行；不强制换坐标，漏执行时仍允许同点重试。
原始PNG保持不变；派生模型输入另存为`repair-input.png`或按像素哈希保存，明确与原生环境截图区分。
每次修复一个模型请求，画布大小相同；文本token、真实生成、缓存命中、等待和环境步分别记账，不能把缓存后的耗时当作通用吞吐。

## 开发集、正常探针和真实执行

复用上一轮修正后数据的train150、validation60个无注入来源，保留原始PNG/trace字节、教师修正和来源哈希。
train有33个自然错误、117个健康来源；validation有8个自然错误、52个健康来源。此处自然错误都来自click-button，后续对话框/标签页检验为跨任务迁移。
只重放前缀以离线标注目标框并核对精确截图/动作效果。目标框只用于评分，不送入模型。

- 开发探针：全部33个自然错误、按任务轮转固定100个健康重查、20个正常Submit。
- 验证探针：全部8个自然错误、52个健康重查、15个正常Submit。
- 健康重查是成功后的截图加同一动作上下文，模拟一次错误警报。包含已终止状态，只有离线定位分数，**不等于执行了100个健康episode**。
- Submit探针使用真实输入后的截图、原INPUT提案及当前Submit目标，检查目标切换是否退化。
- 验证自然错误实际执行五种条件对照：四组特权触发修复，另加先等150ms再plain重试。全部错误保留，终止失败不重启。

条件错误集合由教师离线选出，是明确的特权诊断；其修复模型仍只接收截图、原提案及目标。正式混合闭环不按真实错误选择案例。

## 验证集选择与全新测试

候选顺序固定为feedback、zoom、feedback_zoom。候选须同时满足：

1. 验证健康重查命中数和Submit命中数分别不低于plain。
2. 全部验证探针无效输出数不多于plain。
3. 在8个验证自然错误上的**实际任务恢复数严格超过plain**。

达标者按实际恢复、健康命中、固定顺序选择；无候选达标就回退plain。验证数小，此规则不是统计非劣检验。
选择文件绑定代码、配置、开发数据、验证预测/执行和动作阶段分类器哈希；新test reset或推理前先验证ledger。
主方法不因test成绩切换，即使验证回退仍保留四种修复的条件test结果。

新test预定200个来源：click-button、click-dialog、click-tab、enter-text各40，click-test、focus-text各20。
候选seed从20000开始，按冻结配置顺序使用双倍预留池。先只reset，依次取与登记旧数据及本轮已接受初始截图都不重复的布局；不执行动作、不读取结果。
预留池不足就停止并保留记录，不根据模型成败补样本。模型随后实际执行每个原始提案，不假设无注入就成功。
全部自然错误进入条件执行的五组对照；健康/Submit探针只比较plain与验证选定方法，避免在test上扩大正常候选搜索。

混合闭环提前取每任务最前5个已接受seed，共30场景：不恢复、等待plain重试、原视觉检查器加plain、特权触发加选定修复；选定修复非plain时再加视觉检查器配选定修复。
因此计划120或150次实际混合执行，全部正常与自然错误保留。条件执行与混合执行可共享来源，不当作独立新场景。
每例最多1次等待、1次修复、INPUT后的1次Submit、3个逻辑actor调用，0次重启；Submit继续使用原始plain定位。

## 检查器消融

继承上轮冻结的完整视觉头与pixels头作为固定研究对照，未通过旧误报门槛的头不被视为已批准默认采用。
额外在原train/validation上拟合仅8个动作/阶段字段的小分类器：动作类型、提案x/y、值长度和自身观察阶段，去掉全部图片与目标文字特征。
训练子集、场景权重、L2/阈值候选、置信拒判与验证约束沿用原协议；只训练CPU分类头，选择在新test前冻结。
在新自然错误/健康来源上比较完整视觉、pixels、动作阶段和appearance规则。此轮无注入/排队test，不能报告不存在的此类分母。
成功CLICK常立刻终止，而健康存活状态主要来自INPUT，动作类型/终止门控可能解释高检测率；必须结合消融与任务分布解释，不能据此宣称语义理解。

## 命令与验收

```bash
git switch feat/contextual-repair
git diff --check
python -m pytest -q
export CUDA_VISIBLE_DEVICES=0
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers"
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_contextual_repair_study.py --stage development --config configs/contextual_repair.json --reference runs/visual-verification-003-data --inherited-detectors runs/visual-verification-003-models --model models/ShowUI-2B --out runs/contextual-repair-001-development
python scripts/run_contextual_repair_study.py --stage test --config configs/contextual_repair.json --reference runs/visual-verification-003-data --inherited-detectors runs/visual-verification-003-models --model models/ShowUI-2B --development runs/contextual-repair-001-development --exclude runs/visual-verification-001-data --out runs/contextual-repair-001-test
```

输出目录必须全新；先以`contextual_repair_smoke.json`验收旧开发片段和40000系列工程场景，全部排除研究分母，不据其效果改候选配方。
单用已授权GPU0，加载模型前再次检查空闲；不改底座、默认Alpha或旧实验，不向Git提交权重。源码先提交再运行，结果只补文档与原始案例。
验收包括正常/Submit定位变化、实际自然错误恢复、混合任务成功、误报与拒判、原生/派生图像哈希和坐标回映射、缓存/调用成本。
如无候选过门槛，保留回退和负面结果；本轮不以未验证收益启动SFT或RL。
