# 主动核验末尾空格

正常输入和末尾额外空格可以产生逐字节相同的截图。上一轮在线 Q 在测试中对两种输入都选择了重输。本轮在视觉字形匹配且刚完成输入时，执行一次 `Shift+ArrowLeft` 并保存选区截图，再用 `ArrowRight` 取消选区。两个动作都通过 BrowserGym 记录。检测器从输入框内部的像素差异定位选区，以探测前截图在该区域的深色像素量区分字符和空白。判断只使用视觉证据，实验标签和 DOM 不进入策略。

Mac 的蓝色阈值原型在 Linux 上失败，因为浏览器配色不同。改用像素差异后，Linux 原型识别了工程集中的 6 对样本。它容许相邻字形泄入的一个深色像素。两到三个深色像素、非紧凑区域、无可见变化或复原后的图像差异都会使核验弃权。主动规则仅在确认末尾空白时重输。视觉不匹配和漏输入继续使用原字形规则。

启用 `RecoveryDecisionEnv(..., enable_tail_probe=True)` 时，Enter Text 的动作空间增加一次选区探测，默认决策预算为四步。探测后观察包含原始选区截图、调用标记和复原标记。轨迹使用版本 2 并带 PNG 哈希，旧的四动作环境仍记录版本 1。

## 正式协议

四个新布局各测七种条件和三种策略，共 84 条轨迹。对照为字形规则、立即重输和主动核验。三组共享冻结 ShowUI、同一重输执行器、同一浏览器版本和精确提交奖励。工程布局、此前登记的输入布局和可用参考语料在推理前排除。预设门槛要求保留字形规则的所有成功、正常组零重输、末尾空格组新增成功。报告同时保留键盘操作、等待、重输、探测耗时和逻辑推理请求。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_active_text_verification.py \
  --config configs/active_text_verification.json --profile formal --model models/ShowUI-2B \
  --prior-plan runs/pre-rl-recovery-001/splits.json \
  --prior-plan runs/recovery-offline-rl-001/splits.json \
  --prior-plan runs/idempotent-input-repair-001/preflight.json \
  --prior-plan runs/visual-text-content-002/preflight.json \
  --prior-plan runs/online-visual-recovery-rl-001/preflight.json \
  --prior-plan runs/online-visual-recovery-rl-smoke-001/preflight.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --exclude-run runs/active-text-verification-smoke-001 \
  --out runs/active-text-verification-001
```

先在独立目录运行 `--profile smoke`，省略 `--exclude-run`。当前使用单行短 ASCII 输入和固定 Chromium 视口。光标由之前的输入动作留在行尾。
