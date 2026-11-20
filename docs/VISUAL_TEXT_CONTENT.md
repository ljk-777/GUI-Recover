# Visual text comparison for automatic input recovery

The previous executor could replace a field with the original target text after recovery was selected. Its trigger detected text presence and missed nonempty mistakes. This follow-up adds a rendered-text comparator for the native MiniWoB Enter Text profile.

An independent local Chromium process renders each requested string into four fixed reference controls. Those positions cover the native pixel phases at a 1.5× scale. Reference generation reads the visible goals recorded during reset-only preflight and runs before any action or outcome. The reference process contains only its own controls. It never opens the task. The bank records the browser version and is checksummed before and after evaluation.

The verifier receives the before/after screenshots, requested input and prior proposed coordinates. It finds the same initially blank white field as the original visual guard, then compares the cropped visible ink with the requested glyph mask. The crop excludes the focused outline with a three-pixel inset. The comparison allows up to one pixel of translation on each axis. A best intersection-over-union score of at least 0.995 is a visual match. A score at or below 0.965 indicates different visible text. The gap, a clipped crop, an unsupported geometry and a missing reference cause abstention. The profile covers short ASCII letter strings on the tested Chromium viewport.

Visible mismatch triggers one grounded replacement of the original goal text. An empty field can wait once to let deferred text arrive; the second empty observation triggers replacement. An uncertain observation can wait once and then proceeds without a forced edit. A screenshot cannot in general establish exact string identity. The formal experiment explicitly includes an invisible trailing space.

## Frozen protocol

Six fresh layouts are selected by reset and screenshot comparison, excluding registered prior layouts. Each runs in seven conditions: normal, drop, deferred, partial, appended exclamation mark, first-letter case change, and appended trailing space. Both policies execute the same frozen ShowUI actor, idempotent replacement, waiting budget and exact reward. The presence-only baseline waits once when it sees no text and leaves nonempty fields alone. The new policy compares the screenshot with its prepared templates. Native outcomes are saved for evaluation, and the actor gets the same screenshot and target query as the baseline.

The effect gate requires preserving paired normal, missing and delayed successes, zero replacements on normal inputs, and a positive paired gain in each of the three visible-error groups. The trailing-space group is a registered negative visibility control. Both paths operate on the same task repeated per layout. The experiments use exact terminal MiniWoB rewards, including failure when a trailing space changes the required value.

The first Linux smoke run exposed a focused outline pixel inside the original two-pixel crop. It caused the comparator to abstain. The three-pixel inset was verified on the same engineering layout and on the browser integration fixture before the held-out rollouts began. The test layout selection, thresholds and effect gate were frozen before that evaluation.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_visual_text_content.py \
  --config configs/visual_text_content.json --profile test --model models/ShowUI-2B \
  --prior-pre-rl runs/pre-rl-recovery-001/splits.json \
  --prior-rl runs/recovery-offline-rl-001/splits.json \
  --prior-idempotent runs/idempotent-input-repair-001/preflight.json \
  --reference runs/visual-verification-003-data \
  --reference runs/contextual-repair-001-test/data \
  --reference runs/crop-control-multistep-001/crop-data \
  --out runs/visual-text-content-002
```

Use a fresh output directory for each run. The earlier interrupted run is preserved under `runs/visual-text-content-001`.

## CPU audit

Run this with the project installed. The source checkout also provides `scripts/audit_visual_text_content.py` as an alias.

```bash
gui-recover audit-text-recovery runs/visual-text-content-002
```

The audit reads the frozen preflight, reference bank and recorded rollout files. It checks file hashes, the layout and policy order, the screenshot sequence, the policy decision reconstructed from each screenshot and the success reconstructed from terminal browser evidence. It runs without model weights, a GPU or a browser. A complete run also has its paired condition totals and effect gate checked against the verified traces.

An interrupted run reports `status: incomplete` and the count of verified episodes and pairs. The command reads completed index rows and reports any bytes left in a partial final row. It issues effect totals only when the full registered index and summary are present and consistent.

## 正式结果

冻结源码为 `8cfb665`，Chromium 为 `125.0.6422.26`。6 个留出布局各执行 7 种条件和两套策略，共 84 条原生浏览器轨迹。重跑使用独立输出目录。两次运行的布局清单与参考字形文件逐字节一致，前 40 条轨迹的截图、决策和奖励也一致。

| 条件 | 只检查有无文字 | 核对字形 | 净新增成功 | 核对字形的等待 / 重输 |
|---|---:|---:|---:|---:|
| 正常 | 6/6 | 6/6 | 0 | 0 / 0 |
| 延迟输入 | 6/6 | 6/6 | 0 | 6 / 0 |
| 漏输入 | 6/6 | 6/6 | 0 | 6 / 6 |
| 少输 | 0/6 | 6/6 | 6 | 0 / 6 |
| 多输一个 `!` | 0/6 | 6/6 | 6 | 0 / 6 |
| 首字母大小写错误 | 0/6 | 6/6 | 6 | 0 / 6 |
| 末尾多一个空格 | 0/6 | 0/6 | 0 | 0 / 0 |

核对字形通过预设效果门槛，共新增 18 次成功，配对回退为 0。正常、延迟与漏输入的等待和重输次数与基线一致。每次新增修复使用一次额外的 ShowUI 定位，逻辑模型请求从 90 增至 108。整个配对实验共享推理缓存，共执行 54 次实际生成。RTX 4090 D 的设备采样峰值为 5309 MiB，实测配额为 12032 MiB。

六个布局的末尾空格截图均与各自的正常输入截图逐字节一致，两种策略在精确提交时均失败。漏输入和延迟输入的首次动作后截图也逐对相同，等待提供了区分它们的下一帧。结果适用于这批 MiniWoB Enter Text、短 ASCII 字符串和固定渲染条件。六个布局在多个条件下重复使用。跨字体、长文本、不同输入控件与网站的表现仍需验证。

两套 CPU 复核共核对 192 个决策。公开复核命令验证了 276 张分支内截图，研究审计另核对了 222 次浏览器操作。用相同布局清单独立重渲染字形，文件哈希一致。[机器可读结果](../examples/visual-text-content-results.json) · [首字母错误的配对轨迹](../examples/visual-text-content-pair/provenance.json) · [实现检查](https://github.com/ljk-777/GUI-Recover/actions/runs/44562726672) · [复核工具检查](https://github.com/ljk-777/GUI-Recover/actions/runs/44668688098)
