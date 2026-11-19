# 首轮 grounding SFT：训练与重新加载验收

本分支从已发布 Alpha 的 main 独立建立，不依赖事前检查的实验分支。目标是训练并重新加载第一个 adapter，保留独立的定位对照；暂不训练自主规划或多步恢复策略。

## 数据与划分

`configs/sft_smoke.json` 在模型训练前固定：

| 划分 | 任务与 seed | 计划监督对数 |
|---|---|---:|
| train | click-test、enter-text，1000–1015 | 64 |
| validation | 同两任务，2000–2003 | 16 |
| test | 同两任务，3000–3003；未训练的 click-button 3000–3007、focus-text 3000–3003 | 最多 28 |

enter-text 每张截图包含 text field、Submit button、输入框查询改写三种监督。64 对来自 32 个场景，不是 64 个独立 episode。
查询改写只是同一初始截图上的另一种定位描述，不包含实际失败前缀，不能据此称为已学会恢复决策。
DOM 只提供离线目标边框和中心坐标；模型输入严格由截图、目标描述组成，assistant 答案是归一化坐标。
同一场景的全部查询属于同一 split，并拒绝跨 split 的相同截图哈希。
过滤没有至少 4×4 像素可见正确目标的样本，写入 manifest；过滤发生在模型评测前，不删除模型预测失败。
生成数据保留 MiniWoB MIT 通知。数据目录不提交到 Git，公开示例与统计另行整理。

## 训练边界

- 固定 ShowUI-2B revision 和原始权重 SHA256；底座以 NF4/double quant 加载，BF16 计算。
- 只训练文本 decoder attention 的 q_proj/v_proj LoRA，rank 8、alpha 16、dropout 0；冻结视觉编码器、embedding、lm_head 和其余底座。
- batch 1、累积 2、固定 32 个 optimizer updates、学习率 1e-4，不按 test 成绩改配置。
- user、图像 token 和提示词全部 loss mask；只监督 assistant 坐标及结束 token。必须核对完整序列的 prompt 前缀，不能靠固定 token 数裁剪。
- 检查有限 loss/梯度、实际 adapter 参数变化、可训练参数范围；保存后先释放训练模型，再独立加载量化底座和 adapter，核对同一 probe 的 loss。
- 训练器只读取 train/validation，test 在训练成功后由独立程序读取。

## 独立对照

`eval_grounding_sft.py` 重新加载 BF16 底座与 adapter。baseline 禁用 adapter，SFT 启用 adapter，两组使用相同底座精度、图像预算、查询和 32-token 贪心生成。
这与训练时的量化重载检查是两个不同验收；避免用 4-bit baseline 对比 BF16 adapter 而混入精度差异。
记录原始输出、非法坐标、每个任务的目标框命中率、新增命中与退化。这里的命中率按定位查询计算，不是完整任务成功率；小样本不作为泛化结论。

## 可复现命令

GPU 0 只代表本次用户分配的卡。先确认空闲，再运行：

```bash
bash scripts/setup_sft_env.sh
export CUDA_VISIBLE_DEVICES=0
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers"
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-sft/bin/python scripts/collect_sft_grounding.py --config configs/sft_smoke.json --out runs/sft-data-001
.venv-sft/bin/python scripts/train_grounding_sft.py --config configs/sft_smoke.json --data runs/sft-data-001 --model models/ShowUI-2B --out runs/sft-adapter-001
.venv-sft/bin/python scripts/eval_grounding_sft.py --data runs/sft-data-001 --model models/ShowUI-2B --adapter runs/sft-adapter-001 --out runs/sft-eval-001
.venv-sft/bin/python -m gui_recover run --policy showui-scripted --model models/ShowUI-2B --adapter runs/sft-adapter-001 --tasks click-test enter-text --seeds 3000 3001 --max-steps 4 --parser mixed --out runs/sft-online-001
```

`.venv-sft` 是独立叠加环境，只读复用已有 `.venv` 的固定包，新增 peft 0.14.0、accelerate 1.4.0、bitsandbytes 0.45.3、psutil 6.1.1。它依赖原 `.venv` 保留，不是可搬移的独立环境；不升级原推理环境。
新机器可按 `.[sft]` 安装独立完整环境，但仍需匹配已验证的 CUDA PyTorch 安装和浏览器。训练命令要求干净 Git 版本和新输出目录。

`run --adapter` 仅加载本工具已通过训练和重载验收的本地 adapter，核对 adapter SHA256、base revision 和原始底座文件哈希。manifest 记录训练提交和更新次数；默认不加载 adapter。
端到端链路验证沿用测试 seed，仅确认训练结果可进入 BrowserGym 执行和记录，不作为额外独立泛化样本。baseline 与 adapter 的模型不同，不能使用要求同一模型的故障 `compare` 来宣称配对恢复收益。

参考：[PEFT 0.14 量化训练](https://huggingface.co/docs/peft/v0.14.0/en/developer_guides/quantization)、[Transformers 4.49 Qwen2-VL](https://huggingface.co/docs/transformers/v4.49.0/model_doc/qwen2_vl)。本轮不发布模型权重或启动 RL。
