# Multistep recovery

The benchmark measures recovery at the first live checkpoint of native MiniWoB button-sequence, checkbox and short form tasks. All controllers use the same scripted suffix after the checkpoint. Natural first actions come from a frozen vision model. Controlled healthy, missed and wrong actions are paired by layout, and the report retains wrong order, invalid predictions and terminal failures. The audit reports binary task reward and full credit separately. The checkbox task can award a positive binary result for selecting only part of the requested set. Full credit requires the recorded raw reward to equal one.

## Compare three controllers

Install the package and pinned MiniWoB assets. The Linux command uses the local Arial-compatible font reported by fontconfig. A BF16 GPU and local ShowUI checkpoint are required for the ShowUI profile.

```bash
python -m pip install -e '.[showui]'
bash scripts/setup_miniwob.sh
python -m playwright install chromium
python scripts/download_showui.py --out models/ShowUI-2B
export MINIWOB_URL="file://$PWD/vendor/miniwob-plusplus/miniwob/html/miniwob/"
export CUDA_VISIBLE_DEVICES=0

python scripts/run_multistep_recovery.py \
  --config configs/multistep_recovery.json --profile smoke \
  --model models/ShowUI-2B --font "$(fc-match -f '%{file}' Arial)" \
  --out runs/multistep-smoke
python scripts/audit_multistep_recovery.py runs/multistep-smoke --report
```

Open `report.html` inside the audited run. It links every native frame and shows full credit, binary reward, false repairs, natural failure recovery and operation cost. Use `--profile formal` for the frozen larger population. `--prior-evidence` accepts old rollout directories to exclude duplicate native observations. Use a fresh output directory for each run.

## Formal comparison

The ShowUI evaluation ran 540 trajectories across 36 held-out layouts, including 12 layouts from the reserved checkbox family. All methods use the same extra-operation cap.

| Controller | Full credit | Native binary success | Natural misses recovered with full credit | Healthy checkpoints repaired | Extra operations |
|---|---:|---:|---:|---:|---:|
| Continue | 71/180 | 96/180 | 0/13 | 0/59 | 0 |
| Fixed retry | 104/180 | 138/180 | 1/13 | 59/59 | 204 |
| Visual recovery | 167/180 | 167/180 | 12/13 | 0/59 | 255 |

Full-credit outcomes were recomputed from the original terminal rewards after identifying partial credit in the checkbox task. The registered binary comparison also passed both gates. The visual controller spent 51 more operations than fixed retry. It maintained the same zero-repair rate as continuation on healthy checkpoints.

Each arm has a shared cap of three additional operations. The no-recovery arm proceeds to the suffix. The fixed baseline performs one zoomed re-grounding and retry. The visual rule can wait once for text delivery, inspect a reversible tail selection or button focus marker, and take a bounded repair. The checkbox verifier reads pixels and may undo a specific wrong toggle before setting the intended box. The sequence probe briefly exposes button focus with a trusted browser style, checks that the unmarked frame was restored, and can retry when neither button acquired focus. An already recorded click on the wrong sequence button remains a failure under a zero-reset budget.

On the first registered button layout, the correct click, omitted click and wrong click produced the same original PNG. The focus probe distinguished the omitted click and let the controller retry once. The correct click continued with no retry. The wrong click still failed. All three probes restored the original frame. [Original screenshot](../examples/multistep-focus-probe/without-probe.png) · [Correct focus](../examples/multistep-focus-probe/healthy-focus.png) · [Omitted click](../examples/multistep-focus-probe/miss-focus.png) · [Wrong focus](../examples/multistep-focus-probe/wrong-focus.png) · [Provenance](../examples/multistep-focus-probe/provenance.json).

The protocol freezes screenshot hashes and layout splits before acting. Checkbox layouts form the entire transfer task family. DOM geometry builds the controlled actions and common suffix, and privileged state labels are collected for reporting. The recovery controller takes the visible goal and proposal, screenshots and the outputs of its bounded operations. The benchmark reports extra checks of healthy states separately from repairs of healthy states.

## Learn only from the development split

The train sequence explores probe and skip on native form outcomes. Rewards include the same wait, probe and repair costs in all arms. The runner freezes Q estimates before using test scenes and reports zero regression on deferred trailing spaces and a reduction in probe operations as independent promotion criteria.

Export successful repair coordinates and clean anchors from the development evidence. The export verifies the source trajectory and frame checksums and checks that train, validation and test share neither layouts nor exact screenshots. The SFT loop uses assistant-only LoRA on the fixed ShowUI checkpoint, with no test input to the trainer.

```bash
python -m pip install -e '.[sft]'
python scripts/export_multistep_sft.py runs/multistep-smoke runs/multistep-sft-data
python scripts/train_grounding_sft.py --config configs/multistep_sft.json \
  --data runs/multistep-sft-data --model models/ShowUI-2B --out runs/multistep-adapter
python scripts/eval_multistep_sft.py --run runs/multistep-smoke \
  --model models/ShowUI-2B --adapter runs/multistep-adapter --out runs/multistep-adapted
```

The held-out adapter check reuses frozen original proposals and replaces the re-grounding model only. It requires a stable committed worktree for recording training provenance.

The online probe learner completed 64 training episodes. Independent CPU replay reproduced every seeded selection and both Q-value tables. Its frozen policy reached 48/48 on held-out form episodes, including 12/12 on delayed trailing spaces. The visual rule also reached 48/48. Both executed 48 probes, so the learned policy missed the cost-reduction gate.

The ShowUI SFT pilot used 56 training and 28 validation examples, with 40 and 20 verified corrections in those splits. It completed 32 decoder LoRA updates. Validation loss went from 4.150 to 1.830, and the adapter reload matched the saved test loss exactly. On 18 matched browser rollouts the frozen retry model achieved 11 full-credit outcomes and the adapted retry model achieved 10. Both achieved 14 native binary successes. These results keep the visual rule and frozen weights as the evaluation reference.

## Transfer to a second model

Download a separate local Qwen2.5-VL-3B-Instruct snapshot at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. See [third-party terms](../THIRD_PARTY.md). This backend returns integer pixels on its prepared image. The adapter validates the output and maps it into the normalized action interface used by the same evaluator and audit.

```bash
python scripts/run_multistep_recovery.py \
  --config configs/multistep_recovery.json --actor qwen --device cpu \
  --model models/Qwen2.5-VL-3B-Instruct --font "$(fc-match -f '%{file}' Arial)" \
  --transfer-from runs/multistep-smoke --layouts-per-task 1 --out runs/multistep-qwen
python scripts/audit_multistep_recovery.py runs/multistep-qwen --report
```

In the three-layout Qwen compatibility study, the three controllers ran 45 native trajectories and the CPU audit checked 199 screenshots. Full credit was 14/15 with visual recovery, 10/15 with fixed retry and 7/15 with continuation. The respective binary scores were 14/15, 13/15 and 9/15. The visual method repaired zero of six healthy intermediate states and used 20 extra operations. Fixed retry repaired four of those states and used 13 extra operations. All three natural Qwen first actions in this sample were successful, so the comparison measures controlled faults.

## Inspect VisualWebArena recordings

The offline diagnostic checks six native viewport screenshots from VisualWebArena Classifieds tasks 118, 120 and 149. It compares a last-frame answer with a bounded second look when Qwen explicitly reports uncertainty. It reports exact short-answer matches separately from the historical environment reward. The upstream assets stay local. See [AgentRewardBench dataset terms](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench#terms-of-use).

```bash
python scripts/replay_visualwebarena.py --download configs/visualwebarena_replay.json \
  --recordings runs/vwa-recordings --out runs/vwa-cpu-check.json
python scripts/replay_visualwebarena.py --recordings runs/vwa-recordings \
  --qwen models/Qwen2.5-VL-3B-Instruct --device cpu --out runs/vwa-qwen-check.json
```

A native run requires a Classifieds instance you control. The opt-in adapter sends answers to the BrowserGym task evaluator and uses its terminal reward. The registered tasks use the native string judge, so the runner skips the unused BLIP captioner and external judging clients. The first command reports the setup status without loading a model or starting a browser. Configure `VWA_*` and a private NLTK data directory with `punkt_tab` before the live command. [BrowserGym setup](https://github.com/ServiceNow/BrowserGym/tree/main/browsergym/visualwebarena)

```bash
python -m pip install 'browsergym-visualwebarena==0.14.3'
python scripts/run_visualwebarena_live.py --preflight
python scripts/run_visualwebarena_live.py --model models/ShowUI-2B --out runs/visualwebarena-live
```

For CPU Qwen navigation, pass `--actor qwen --device cpu --model models/Qwen2.5-VL-3B-Instruct`.

The offline answer check matched one of three reference answers. The model returned an answer in every case, so an uncertainty gate based on an explicit `UNKNOWN` requested zero additional checks. The recording comparison is independent of the live benchmark.

The server ran the optimized Classifieds site with fixed web and database image digests. The optimized site uses re-encoded AVIF photos. A known-answer wiring check on the live site reached the native evaluator and earned full reward for all three tasks. [Classifieds image source](https://github.com/bgrins/vwa_classifieds_optimized)

Qwen then ran the same three task IDs in each arm on the single GPU with a two-step budget. Each arm executed three actions in total and finished zero tasks. One task hit the step limit and two produced invalid model actions. The feedback arm issued zero recovery messages. This run validates the live connection and action loop and leaves the recovery effect on VisualWebArena unmeasured.

### Evaluate invalid-action recovery

The Qwen adapter recognizes the observed labelled scroll form as one fixed-size scroll. Clicks and text input still require a valid position. The format study compares stopping, repeating the same request, and asking again with guidance about the action schema. A retry uses the same screenshot and action history. Each arm shares the same prediction budget, and a retry consumes one prediction from it. The study allows one retry per episode.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --model models/Qwen2.5-VL-3B-Instruct \
  --study format --task-set transfer --max-steps 6 --out runs/visualwebarena-format-transfer
python scripts/audit_visualwebarena_live.py runs/visualwebarena-format-transfer \
  --out runs/visualwebarena-format-transfer-audit.json
```

Use `--task-set diagnostic` to reproduce the three tasks that exposed the original parsing failures. The transfer set is a separate group of 10 registered Classifieds tasks with local string grading. The CPU audit checks the paired reset screenshots, prediction budgets, executed actions, and terminal reward. Its report tracks how often a retry produced a valid action and how often the original task earned full native reward.

The Qwen CUDA run used six predictions per episode as its cap. The audit verified 39 trajectories and 117 screenshots. All 13 task trios shared identical reset pixels and identical pixels and provider output at the first rejection.

| Transfer strategy | Model calls | Executed actions | Valid retries | Native successes |
| --- | ---: | ---: | ---: | ---: |
| Stop after rejection | 29 | 19 | – | 0/10 |
| Repeat the request | 38 | 19 | 0/9 | 0/10 |
| Retry with format guidance | 43 | 24 | 3/9 | 0/10 |

On the three diagnostic tasks, zero of three plain retries and one of three guided retries yielded a valid action. No diagnostic arm completed a task. The guided transfer arm executed one click and two text inputs immediately after its successful retries. It submitted zero answers across the 10 tasks. This establishes a benefit for generating executable actions under the tested prompt and budget. Task completion on Classifieds remains unresolved.

### Evaluate unchanged-scroll feedback

Pass `--study scroll --task-set diagnostic` to compare the existing click and execution-error feedback with the same controller plus scroll feedback. Both arms use the one-retry format guidance and the same prediction budget. The scroll detector fires when a scroll leaves the screenshot pixel-identical. Its message describes the observed lack of movement. Scroll messages share the controller's two-message limit with click and execution-error feedback.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --model models/Qwen2.5-VL-3B-Instruct \
  --study scroll --task-set diagnostic --max-steps 6 --out runs/visualwebarena-scroll-diagnostic
python scripts/audit_visualwebarena_live.py runs/visualwebarena-scroll-diagnostic \
  --out runs/visualwebarena-scroll-diagnostic-audit.json
```

Two of the three paired tasks reached an identical unchanged viewport after a scroll. The next action was a scroll that moved the viewport on both eligible tasks with scroll feedback, and on one eligible task with the ordinary feedback controller. The third task never reached a scroll. Both arms earned full native reward on zero of three tasks. The audit checked all six trajectories and 20 screenshots in this diagnostic comparison.

[Live format and scroll results with run provenance](../examples/vwa-live-recovery-results.json)

### Check whether the current viewport supports an answer

The answer study keeps the same Qwen checkpoint, navigation prompt, invalid-action retry, and scroll feedback in both arms. The added check asks a separate question about the current screenshot before the first navigation prediction on that image. A short answer is submitted to the native judge. An explicit abstention or malformed response leaves the browser untouched and gives the remaining budget to navigation. A repeated screenshot is checked once per episode. Each check, retry, and navigation prediction counts against the same eight-call cap.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --model models/Qwen2.5-VL-3B-Instruct \
  --study answer --task-set answer-transfer --max-steps 8 --out runs/visualwebarena-answer-transfer
python scripts/audit_visualwebarena_live.py runs/visualwebarena-answer-transfer \
  --out runs/visualwebarena-answer-transfer-audit.json
```

The diagnostic set reuses tasks 118, 128 and 221 from earlier runs. The transfer set contains eight different tasks fixed before running the check. It includes category navigation, gallery inspection, reading details in listing photos, and a quantity question on a listing page. The CPU audit reconstructs which screenshot each check read, whether an answer reached the benchmark, the source of that answer, and the native reward.

On the three diagnostic tasks, the navigation arm submitted zero answers. The added check made four judgments across distinct screenshots and submitted two answers. Both submissions occurred on the initial screenshot and received zero reward. Both arms finished with zero native successes.

| Transfer strategy | Model calls | Answer checks | Answers submitted | Native successes |
| --- | ---: | ---: | ---: | ---: |
| Navigation | 35 | 0 | 0 | 0/8 |
| Check each new viewport | 25 | 13 | 4 | 0/8 |

All four transfer submissions came from the initial check and received zero reward. The other checks returned eight explicit abstentions and one malformed reply. The lower model-call count reflects the four early terminal failures. The check makes answer submission observable and exposes premature decisions on these tasks. The CPU audit verified 22 trajectories, 66 screenshots, and identical reset pixels in all 11 task pairs.

A separate native judge check used answers built from the official references on four of these tasks. All four received full reward. That check made zero model calls and has its own result record.

[Live answer-check results and run provenance](../examples/vwa-answer-check-results.json)

### Locate and read answer evidence

The evidence study runs the existing answer-check controller and a controller that first locates a relevant region, then reads that region from the original screenshot pixels. Localization and crop reading each use one prediction. Navigation can seek missing evidence. Only a crop read can submit an answer in the evidence arm. For price ranges, the answer must contain two endpoints present in the reported visual facts. A row region clipped at the bottom of the viewport sends control back to navigation.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --model models/Qwen2.5-VL-3B-Instruct \
  --study evidence --task-set evidence-transfer --max-steps 8 --out runs/visualwebarena-evidence-transfer
python scripts/audit_visualwebarena_live.py runs/visualwebarena-evidence-transfer \
  --out runs/visualwebarena-evidence-transfer-audit.json
```

The evidence transfer set was frozen before observing model outcomes. It contains tasks 23, 40, 43, 110, 161, 173, 174 and 199. The paired comparison uses the same Qwen checkpoint and a cap of eight calls per episode. The acceptance check requires at least two native successes, more successes than the existing answer check, and fewer wrong submissions. The CPU audit recomputes every evidence crop from its recorded screenshot and traces the final answer to the crop-reading response.

Development used the six screenshots that previously led to wrong first-frame submissions. The final development replay withheld three answers through the clipping and range checks. One other reply reached the model's token limit and could not be parsed. Two wrong candidates remained. Both the crop localization and the reported facts come from the model, so these checks leave visual recognition errors possible.

| Transfer strategy | Model calls | Wrong submissions | Native successes |
| --- | ---: | ---: | ---: |
| Answer check | 18 | 4 | 0/8 |
| Evidence localization and crop read | 22 | 5 | 0/8 |

All eight pairs started from pixel-identical resets. The crop-read arm submitted five incorrect answers from the initial viewport and executed zero navigation actions. The other three episodes ended after invalid navigation replies. The audit verified 16 live trajectories and 26 screenshots. The study missed the acceptance gate.

[Frozen plan and per-task evidence study results](../examples/vwa-evidence-results.json)

### Confirm the answer using the viewport and crop

The confirmation study lets the evidence search navigate directly if the current viewport lacks the requested result. It reads the native-pixel crop when a region is available. Before submission, the same checkpoint independently reads a panel containing the full viewport with the region outlined and the cropped detail enlarged. The verifier receives the question and images. Its answer must agree with the crop reader's candidate after formatting normalization. The CPU audit reconstructs both images from saved screenshots and rejects an answer that bypasses agreement.

Both arms use the selected checkpoint and one shared call cap per episode. The confirmation comparison uses eight tasks that were outside the previous cohorts. The optional 7B profile loads locally using 4-bit NF4 with BF16 compute. Its downloader verifies the size and SHA-256 of each pinned ModelScope artifact. [Upstream 7B model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

```bash
python scripts/download_qwen_7b.py
python scripts/run_visualwebarena_live.py --actor qwen --qwen-7b-4bit \
  --model models/Qwen2.5-VL-7B-Instruct --study evidence-confirm \
  --task-set evidence-confirm-transfer --max-steps 8 --out runs/visualwebarena-confirm-transfer
python scripts/audit_visualwebarena_live.py runs/visualwebarena-confirm-transfer \
  --out runs/visualwebarena-confirm-transfer-audit.json
```

The CUDA profile needs the pinned Torch and Transformers versions plus `bitsandbytes==0.45.3` and `accelerate==1.4.0`.

| Transfer strategy | Model calls | Navigation actions | Wrong submissions | Native successes |
| --- | ---: | ---: | ---: | ---: |
| Answer check | 36 | 7 | 1 | 0/8 |
| Evidence and independent confirmation | 45 | 22 | 0 | 0/8 |

The audit verified 16 trajectories, 46 screenshots, and identical initial pixels for all eight pairs. The candidate produced no evidence crop on these homepage-start tasks. It either kept navigating or received an invalid navigation reply, so the transfer run did not reach independent confirmation. The measured peak allocation was 7.04 GiB on an RTX 4090 D. This comparison missed the success gate.

[Frozen plan and per-task confirmation results](../examples/vwa-confirmation-results.json)

### Carry the preceding click into item confirmation

The context study keeps the preceding viewport and location of the most recent click that changed the rendered page. The independent reading sees that click alongside the current viewport and the selected crop. This lets it check item identity using a listing view and read details using the item view. The audit traces the orange click marker back to the executed action and reconstructs all three panels. The same profile accepts finite JSON numbers from the independent reader so a count such as `6` can agree with the string `"6"`.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --qwen-7b-4bit \
  --model models/Qwen2.5-VL-7B-Instruct --study evidence-context \
  --task-set evidence-context-transfer --max-steps 8 --out runs/visualwebarena-context-transfer
python scripts/audit_visualwebarena_live.py runs/visualwebarena-context-transfer \
  --out runs/visualwebarena-context-transfer-audit.json
```

The context transfer set contains seven tasks that start on search results or category pages and one that starts on the homepage. All eight were fixed before model outcomes. The comparison keeps the model, call cap, and native score definition constant within each task pair.

The numeric fix was also checked using the recorded replies on diagnostic task 221. The crop reader returned the string `"6"` and the independent reader returned the number `6`. The replay reconstructed the same viewport, crop, and confirmation image and received full reward from the native judge. The replay used three recorded model replies and made zero new model calls.

| Transfer strategy | Model calls | Navigation actions | Wrong submissions | Native successes |
| --- | ---: | ---: | ---: | ---: |
| Answer check | 49 | 21 | 3 | 0/8 |
| Evidence with click context | 55 | 25 | 0 | 0/8 |

The audit verified 16 trajectories, 65 screenshots, and identical reset pixels in all eight task pairs. The candidate opened two crops for reading. One read abstained and the other produced a candidate on the eighth call. No transfer episode reached the independent confirmation stage within the cap. The comparison missed the success gate. The task 221 diagnostic remains separate from this transfer result.

The navigation adapter was then tested on three previously observed tasks. The existing strategy scored 0/3 and the candidate scored 1/3 at the same eight-call cap. On task 221, the frozen Qwen 7B checkpoint produced the verified numeric answer on call three and earned full reward in a fresh browser. On the two search-results tasks, the candidate executed six and five navigation actions. The earlier frozen candidate executed two and one actions on those tasks. All three pairs had identical reset pixels. One environment startup timed out before inference and was repeated once with its original attempt retained.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --qwen-7b-4bit \
  --model models/Qwen2.5-VL-7B-Instruct --study evidence-context \
  --task-set evidence-navigation-diagnostic --max-steps 8 --out runs/visualwebarena-nav-diagnostic
```

[Frozen click-context results and native replay receipt](../examples/vwa-context-evidence-results.json)

### Navigate from rendered text

The optional OCR profile supplies indexed text boxes from the current screenshot during navigation. Clicking a text box uses its validated center in the prepared canvas. A task that begins with an exact quoted search query can submit the OCR-identified empty keyword box without calling Qwen. The action clicks the field, enters the query, and presses Enter. It consumes one decision from the eight-decision limit, so every trajectory still makes at most eight model calls.

The profile preserves image-based crop reading on listing detail pages. A fast response from a navigation page can propose a crop and a candidate together when the candidate is literally present in OCR text within the proposed region. The candidate still requires a separate matching visual read. The logs record OCR coordinates and the decision type. The audit reconstructs indexed actions from those records and distinguishes the deterministic search from a model invocation. [RapidOCR usage](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/usage/)

```bash
python -m pip install -e '.[ocr]'
python scripts/run_visualwebarena_live.py --actor qwen --qwen-7b-4bit \
  --model models/Qwen2.5-VL-7B-Instruct --study evidence-ocr \
  --task-set evidence-direct-search-diagnostic --max-steps 8 --out runs/visualwebarena-ocr
python scripts/audit_visualwebarena_live.py runs/visualwebarena-ocr \
  --out runs/visualwebarena-ocr-audit.json
```

The Classifieds adapter also exposes a bounded browser link preview at specific viewport positions. A live test read two same-origin listing destinations without changing the page. A subsequent click reached the first previewed destination. This probe used a known gallery and zero model calls. It verifies the browser observation needed for link-returning tasks.

| Diagnostic profile | Existing answer check | OCR candidate | Wrong submissions, existing and candidate |
| --- | ---: | ---: | ---: |
| Indexed OCR with fast candidates | 0/6 | 0/6 | 1 and 0 |
| Literal text guard and submitted search action | 0/4 | 0/4 | 1 and 0 |
| Exact query routing and detail-page reading | 0/3 | 1/3 | 1 and 0 |

All three comparisons reused previously observed tasks and the same 7B checkpoint within each pair. The final variant submitted the exact dishwasher search without a Qwen call, reached genuine results, and ran out of decisions before finding a verified answer. It earned full native reward on task 221 with three model calls. The diagnostics cover 26 trajectories and 103 screenshots. The held-out success gate remains open.

[OCR diagnostic metrics and browser action receipts](../examples/vwa-ocr-navigation-results.json)

### Verify links from visible listing titles

For a link question, the evidence policy selects up to three visible listing titles by OCR index or by a unique literal match to rendered text. A visible price directly below a single aligned title can resolve to that title. The browser reads the same-origin link under each selected title and records the title position, exact destination, and source pixel digest. A separate model call identifies the requested listing set from the same screenshot and indexed text. The second call receives the task and screenshot evidence without the first selection or the destinations. The policy submits the captured destinations only if both selections cover the same distinct listings. The auditor checks pointer coordinates, frame provenance, both selections, the submitted URLs, and the native reward.

The locked comparison uses eight previously unrun official Classifieds tasks with `answer-check` as the control and the same pinned Qwen 7B NF4 model in both arms. It uses an eight-decision cap with at most eight model calls per episode. Acceptance needs at least two native full-credit tasks, more successes than the control, and fewer submitted answers without full credit. Screenshots and raw trajectories stay in the run folder. The audit reports each paired start, model call count, submission outcome, and native success.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --qwen-7b-4bit \
  --model models/Qwen2.5-VL-7B-Instruct --study evidence-links \
  --task-set evidence-links-holdout --max-steps 8 --out runs/visualwebarena-links-holdout
python scripts/audit_visualwebarena_live.py runs/visualwebarena-links-holdout \
  --out runs/visualwebarena-links-holdout-audit.json
```

[Locked task IDs, upstream provenance, and scoring criteria](../examples/vwa-classifieds-heldout-protocol.json)

Two synthetic prompts on the previously observed task 14 gallery requested three listings by title and two listings by visible price. Each took two model calls. All five returned destinations matched the actual clicks. The eight-task official comparison then ran from frozen commit `827e42c515fca9ab82d16cc1df6f2ca523ca07a4`.

| Strategy | Native full credit | Wrong submissions | Model calls across eight tasks |
| --- | ---: | ---: | ---: |
| Existing answer check | 1/8 | 0 | 50 |
| Listing-link evidence | 0/8 | 1 | 61 |

The audit reconstructed 16 trajectories and 76 screenshots, verified identical starting pixels for all eight pairs, and verified the eight-call cap. The comparison missed all three acceptance thresholds. On task 34, both model calls chose the same three paintings, but only one satisfied the native reference. On task 106, the control opened a matching card directly from the homepage and received full credit. The evidence policy opened a category menu and spent its remaining calls without reaching the target. These diagnosed tasks now belong to development. Subsequent acceptance needs a new isolated cohort.

[Paired results, native rewards, and evidence archive digest](../examples/vwa-classifieds-link-results.json)

### Inspect ranked visual results across pages

The `evidence-cards` study handles explicit keyword searches for recently listed items with a pictured subject requirement. After submitting the visible search field, it verifies the rendered query, sort label, and result range. A browser action selects the visible grid and expands the viewport to 1,800 pixels so the complete result page can be inspected. The adapter pairs visible title and thumbnail rectangles with their actual listing destinations. Qwen reads numbered screenshot crops in groups of six. Only complete consecutive result pages contribute to the ordered prefix, and each selected photo receives a separate confirmation before the policy submits the links.

The study also supports direct navigation to an evidently relevant listing on the homepage for seller photo questions. A single bounds-checked model box can supply the click. On the opened listing, a literal email in the reader's evidence can become a candidate for a separate read. The confirmation checks the opened item and the displayed field using the crop and the previous click context.

The auditor reconstructs page ranges, crop panels, visible navigation, the reviewed result prefix, submitted actions, and native rewards from the screenshots and saved replies. The development comparison reused tasks 34 and 106. The card policy earned native full credit on both tasks in six and four Qwen calls respectively. The concurrent control scored 0/2. Both arms made zero wrong submissions, and both pairs had identical reset pixels. The previous locked comparison recorded a control success on task 106. This development result records progress on the two diagnosed failures.

```bash
python scripts/run_visualwebarena_live.py --actor qwen --qwen-7b-4bit \
  --model models/Qwen2.5-VL-7B-Instruct --study evidence-cards \
  --task-set evidence-card-diagnostic --max-steps 8 --out runs/visualwebarena-card-diagnostic
python scripts/audit_visualwebarena_live.py runs/visualwebarena-card-diagnostic \
  --out runs/visualwebarena-card-diagnostic-audit.json
```

[Development scores, calls, and archive digest](../examples/vwa-card-evidence-diagnostic-results.json)

### Run the registered image-and-tab holdout

The second eight-task registration adds official task reference images and multiple browser start pages. The runner verifies the installed task entries and reference file hashes before opening an episode. It resolves BrowserGym's homepage image URLs to the registered local assets. Both strategies receive the same primary viewport, ordered reference images, and one-based tab list. Mouse coordinates still refer to the primary viewport. The auditor checks the reference PNGs, active tabs, executed tab transitions, native rewards, and the eight-call cap.

For a request for a listing title, the evidence policy can use a strong and distinct visual match among rendered card thumbnails to localize the title and thumbnail crop. Qwen reads that crop and independently checks the title before submitting. Weak or ambiguous image matches continue through the standard evidence policy.

Prepare the reference archive from the upstream checkout at revision 89f5af29305c3d1e9f97ce4421462060a70c9a03. Configure the usual VisualWebArena variables and genuine Shopping and Reddit instances at distinct origins. The runner derives required sites from the official start URLs. BrowserGym's login metadata can omit public pages that a task needs. For example, task 233 logs in only to Classifieds and also opens a Reddit post. Both arms use the pinned Qwen 7B NF4 profile and the original local native scoring rules.

On a host with Docker, the WebArena-Verified Shopping image can provide the storefront and seeded catalog. The following pins the Linux amd64 publisher image and puts the storefront on loopback. Use this origin for `VWA_SHOPPING`. Wait until BrowserGym's original Shopping login, a catalog search, and a product image load succeed before running the pair. [Upstream Shopping environment](https://servicenow.github.io/webarena-verified/v1.2.3/environments/shopping/)

~~~bash
docker run --detach --name gui-recover-shopping --platform linux/amd64 \
  --publish 127.0.0.1:19981:80 \
  --env WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:19981 \
  am1n3e/webarena-verified-shopping@sha256:3e8cb9b945ea9b1c94ab26dba53e8d12dd0406abbf4bf686fd3bb2b6a5908feb
export VWA_SHOPPING=http://127.0.0.1:19981
~~~

Set `VWA_REDDIT` to the origin that serves the real forum and its post images. The audit verifies the origin of each initial browser tab. [Upstream Reddit environment](https://servicenow.github.io/webarena-verified/v1.2.3/environments/reddit/)

~~~bash
python scripts/prepare_vwa_input_holdout.py --upstream /path/to/visualwebarena \
  --protocol examples/vwa-input-heldout-protocol.json --out runs/vwa-input-cohort
python scripts/run_visualwebarena_live.py --actor qwen --qwen-7b-4bit \
  --model models/Qwen2.5-VL-7B-Instruct --study evidence-cards \
  --task-set evidence-cards-input-holdout --cohort-dir runs/vwa-input-cohort \
  --max-steps 8 --out runs/visualwebarena-input-holdout
python scripts/audit_visualwebarena_live.py runs/visualwebarena-input-holdout \
  --out runs/visualwebarena-input-holdout-audit.json
~~~

If Shopping or Reddit is temporarily unavailable, add `--input-phase classifieds` to the run command to record the six single-site pairs. BrowserGym expects all site variables to contain valid URLs. During this phase, `VWA_SHOPPING` and `VWA_REDDIT` can use the Classifieds URL as a placeholder. Once both backends are available, set their actual URLs and run the same command with `--input-phase finish` and the same output directory. The saved plan requires the original rollout code, model configuration, package versions, shared backend URLs, and registered inputs. Each completed trajectory is kept so an interrupted phase can be retried. The full summary and the acceptance report require all eight pairs.

[Second registered holdout protocol](../examples/vwa-input-heldout-protocol.json)

The first attempt at this registration recorded 16 trajectories and 90 screenshots on an RTX 4090 D. The fixture for task 233 substituted the Classifieds origin for the required Reddit post. Its pair is invalid. Both strategies scored 0/7 on the seven valid pairs, with zero answer submissions. The two Shopping reset screenshots differed by one pixel. Across the archived attempt the evidence policy performed 41 scrolls and reached zero evidence reads or confirmations. Even a full score on the remaining pair would leave this attempt below the two-success threshold. This cohort has been inspected for development, so subsequent acceptance requires fresh tasks.

[Audited per-task results](../examples/vwa-input-heldout-results.json)

On a synthetic development fixture derived from an already used Classifieds task, both strategies started from the same screenshot, tab state, and reference image. Each switched browser tabs once, made three model calls, and earned native reward `1.0`. The evidence policy also recorded the thumbnail match, title crop read, and independent confirmation. The registered holdout does not include this fixture.

[Audited development smoke results](../examples/vwa-input-context-smoke-results.json)

[Machine-readable results, method costs and model provenance](../examples/multistep-recovery-results.json)
