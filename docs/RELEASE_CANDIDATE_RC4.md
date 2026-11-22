# GUI-Recover 0.2.0rc4

GUI-Recover provides visual error detection, bounded recovery and replay for browser agents. Start with the installed CPU demo, or use the source distribution to run the pinned MiniWoB++ and VisualWebArena studies.

## Installed CPU demo

```bash
python -m pip install dist/gui_recover-0.2.0rc4-py3-none-any.whl
python -m playwright install chromium
gui-recover --version
gui-recover demo-tail-probe --out runs/tail-demo
```

Open `runs/tail-demo/report.html` to inspect the matched screenshots and submission outcomes. The run writes a JSON summary, the original screenshots and the individual case records. Use a fresh output directory for each run. [Demo method and prior browser validation](TAIL_PROBE_PORTABILITY.md)

## Recovery evidence

| Evaluation | Outcome |
| --- | --- |
| Installed tail-selection demo | 12/12 clean and 12/12 faulted submissions succeeded. The demo rewrote zero clean fields and abstained on all eight ambiguous controls. |
| Classifieds clean price-and-city tasks | Upright retry and whole-line recovery each passed 4/4. |
| Classifieds controlled detector dropout | Upright retry passed 0/4 and abstained on all four. Whole-line recognition passed 4/4, with zero wrong submissions for either policy. |

The Classifieds comparison used 16 real browser episodes on four fresh listings. Each episode made one Qwen call. Recovery spent one extra CPU OCR read on each injected fault. The 4/4 gain is for controlled deletion of the two detector outputs on this fixture. [Protocol, replay commands and scope](MULTISTEP_RECOVERY.md#test-recovery-when-both-price-detector-reads-are-empty) · [Per-task scores and archive checksums](../examples/vwa-catalog-price-transfer-results-010.json)

The 1.25× capture scan exhausted its 192-listing budget before enrolling a double detector failure. The 166 recorded enlarged crops all contained parseable prices. That leaves the natural-failure recovery rate on this capture profile unmeasured. [Higher-resolution screening record](../examples/vwa-price-retry-holdout-screening-011.json)

## Packaging and validation

- The wheel and source distribution passed Twine metadata checks. A clean Apple Silicon install ran the packaged CLI and the full source-distribution suite with Chromium. All 303 tests passed.
- Fresh wheel environments on macOS and Linux produced the same tail-selection outcomes above. The HTML report was checked at desktop and mobile widths with its screenshot links and expandable cases.
- The source distribution includes the bilingual README, demo animation, benchmark charts, run scripts, protocols, tests and third-party notices. The local CPU audit reproduced the saved Classifieds dropout report byte for byte.

See the [changelog](../CHANGELOG.md), [contribution guide](../CONTRIBUTING.md) and [third-party notices](../THIRD_PARTY.md) for the release changes and experimental setup.
