"""Render the two README charts from the audited Classifieds dropout results."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/vwa-catalog-price-transfer-results-010.json"
OUTPUT = ROOT / "examples/readme-demo"
TEXT = {
    "en": {
        "subtitle": "Classifieds · 4 new price-and-city tasks · cohort 010",
        "clean": "Normal OCR",
        "fault": "Both local OCR reads erased",
        "prior": "Upright retry",
        "current": "Whole-line recovery",
        "footer": "{wrong} wrong submissions · {calls} model calls across {episodes} episodes",
    },
    "zh-CN": {
        "subtitle": "Classifieds · 4 道新价格与城市题 · 第 010 批",
        "clean": "正常 OCR",
        "fault": "受控清空两次局部 OCR",
        "prior": "横排重读",
        "current": "整行恢复",
        "footer": "{episodes} 轮实跑，{wrong} 次错误提交，{calls} 次模型调用",
    },
}
INK, MUTED = "#182e28", "#465f56"
PAPER, BACKGROUND, RIM = "#f9fcf9", "#eef4ee", "#cedfd4"
PRIOR, CURRENT = "#4e6d61", "#146b55"


def render(report, locale, fonts):
    language = TEXT[locale]
    canvas = Image.new("RGB", (1254, 620), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    latin, bold, cjk = fonts
    tasks = report["registered_tasks"]

    def label(x, y, message, size, *, weight=False, color=INK, anchor="lm"):
        face = cjk if locale == "zh-CN" and not message.isascii() else bold if weight else latin
        draw.text((x, y), message, fill=color, font=ImageFont.truetype(str(face), size), anchor=anchor)

    label(65, 53, "VisualWebArena", 57, weight=True)
    label(70, 105, language["subtitle"], 30, color=MUTED)
    wrong = calls = episodes = 0
    for condition, top in (("clean", 140), ("fault", 352)):
        draw.rounded_rectangle((32, top, 1222, top + 190), radius=18, fill=PAPER, outline=RIM, width=2)
        label(70, top + 38, language[condition], 35, weight=True)
        for arm, row_y, color in (("prior", top + 99, PRIOR), ("current", top + 150, CURRENT)):
            rows = report["by_task"][condition][arm]
            successes = [rows[task]["success"] for task in tasks]
            label(70, row_y, language[arm], 31)
            for index, success in enumerate(successes):
                x = 590 + index * 72
                draw.ellipse((x - 13, row_y - 13, x + 13, row_y + 13),
                             fill=color if success else PAPER, outline=color, width=3)
            label(1135, row_y, f"{sum(successes)}/{len(tasks)}", 37, weight=True,
                  color=color if any(successes) else INK, anchor="rm")
            wrong += report[condition][arm]["wrong_submissions"]
            calls += report[condition][arm]["model_calls"]
            episodes += len(tasks)
    label(70, 580, language["footer"].format(wrong=wrong, calls=calls, episodes=episodes), 27, color=MUTED)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, required=True, help="Latin regular TTF or TTC")
    parser.add_argument("--bold-font", type=Path, required=True, help="Latin bold TTF or TTC")
    parser.add_argument("--cjk-font", type=Path, required=True, help="Simplified Chinese TTF or TTC")
    args = parser.parse_args()
    report = json.loads(RESULTS.read_text())
    for locale in TEXT:
        render(report, locale, (args.font, args.bold_font, args.cjk_font)).save(
            OUTPUT / f"benchmark-visualwebarena.{locale}.png"
        )


if __name__ == "__main__":
    main()
