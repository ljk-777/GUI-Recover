"""Render bilingual README video replays from saved browser evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
INPUT_RUN = ROOT / "examples/visual-text-content-pair"
OUTPUT = ROOT / "examples/readme-demo"
SIZE = (1280, 720)
FPS = 24
INK, MUTED = "#182e28", "#435e54"
GREEN, RED = "#12634e", "#9b3d32"
PAPER, BACKGROUND, RIM = "#ffffff", "#edf4ee", "#d2e3d7"
MINT, ROSE = "#e2f1e5", "#f9e9e6"
MARK_PROMPT = (
    "Use case: logo-brand. Make one clean, original graphic mark for the open-source GUI-Recover toolkit. "
    "Output is an opaque 1:1 square app tile, with a perfectly uniform evergreen color (#155b50), edge to "
    "edge across the entire canvas. Draw one bold ivory inspection symbol in the center: three thick rounded "
    "viewfinder corners with the lower-right corner becoming a single checkmark. Place one small bright mint "
    "square inside the upper-right quadrant as the inspected pixel. The symbol should fill the middle 62% of "
    "the square, with clear negative space, generous padding and crisp geometry readable at 32px. Ensure "
    "flat solid ink throughout, with all areas fully painted and opaque. No text, no letters, no texture, no "
    "shadows, no black spot, no gradient, no outside border, no transparent holes. Contemporary, restrained "
    "and professional software identity."
)

COPY = {
    "en": {
        "recorded": "Recorded browser replay",
        "input_title": "Fix the case before Submit",
        "input_target": "MiniWoB++ · Target input",
        "input_stage": ("Case error injected", "Compare the decisions", "Native result"),
        "baseline": "Text-presence baseline",
        "input_left": ("Detected text in the field", "Kept the lowercase input", "Wrong text submitted"),
        "input_right": ("Detected the case mismatch", "Restored the expected text", "Correct text submitted"),
        "input_end_left": "FAIL · Reward 0",
        "input_end_right": "PASS · Reward 1",
        "input_foot": "Both policies started with the same corrupted input.",
        "price_title": "Read the whole price line",
        "price_sub": "VisualWebArena · Classifieds · task 11002",
        "price_stage": ("Controlled OCR dropout", "Whole-line retry", "Native result"),
        "fields": "Listing screenshot",
        "reads": "Price OCR evidence",
        "reads_labels": ("Enlarged crop", "Upright crop", "Whole line"),
        "empty": "Empty",
        "waiting": "Pending",
        "model_check": "OCR agrees with the model reply",
        "abstained": "Upright retry abstained",
        "paid": "PASS · Reward 1",
        "price_foot": "The fault removed OCR output. The listing pixels stayed the same.",
    },
    "zh-CN": {
        "recorded": "真实浏览器轨迹回放",
        "input_title": "提交前修复大小写",
        "input_target": "MiniWoB++ · 目标输入",
        "input_stage": ("注入大小写错误", "对照策略决策", "原生评分"),
        "baseline": "只检查有无文字",
        "input_left": ("检测到字段中有字", "保留了小写输入", "提交了错误文本"),
        "input_right": ("发现大小写不符", "恢复了目标文本", "提交了正确文本"),
        "input_end_left": "失败 · 奖励 0",
        "input_end_right": "通过 · 奖励 1",
        "input_foot": "两套策略从同一个错误输入开始。",
        "price_title": "局部 OCR 失败后重读整行",
        "price_sub": "VisualWebArena · Classifieds · 任务 11002",
        "price_stage": ("受控清空 OCR 结果", "整行重读", "原生评分"),
        "fields": "商品原始截图",
        "reads": "价格读取证据",
        "reads_labels": ("放大裁剪", "横排重读", "整行识别"),
        "empty": "空结果",
        "waiting": "待重读",
        "model_check": "整行读数与模型回复一致",
        "abstained": "横排重读版选择弃答",
        "paid": "通过 · 奖励 1",
        "price_foot": "故障清空了 OCR 的输出，商品截图保持一致。",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def input_evidence():
    provenance = json.loads((INPUT_RUN / "provenance.json").read_text())
    original = INPUT_RUN / "glyph_check/001.png"
    ends, rewards = [], []
    for arm in ("presence_wait", "glyph_check"):
        trace = INPUT_RUN / arm / "transitions.jsonl"
        assert sha(trace) == provenance["policies"][arm]["trace_sha256"]
        injected, final = read_jsonl(trace)
        frame = INPUT_RUN / arm / final["next_observation"]["current"]["image"]
        assert sha(frame) == final["next_observation"]["current"]["sha256"]
        assert sha(original) == injected["next_observation"]["current"]["sha256"]
        ends.append(frame)
        rewards.append(final["reward"])
    assert rewards == [0.0, 1.0]
    return original, ends


def price_evidence(runs):
    folders = [runs / f"visualwebarena-catalog-price-transfer-fault-{arm}-010" / "visualwebarena.11002"
               for arm in ("prior", "current")]
    traces = [folder / "trajectory.jsonl" for folder in folders]
    prior, current = (read_jsonl(trace) for trace in traces)
    before = folders[1] / current[0]["screenshot"]
    assert sha(before) == current[0]["sha256"] == prior[0]["sha256"]
    confirmations = [next(event for event in trace if event["event"] == "prediction"
                          and event["metrics"].get("evidence_stage") == "confirm") for trace in (prior, current)]
    assert confirmations[0]["raw"] is None and prior[-1]["reason"] == "evidence_abstained"
    recovery = confirmations[1]["metrics"]
    assert recovery["provider_output"] == confirmations[0]["metrics"]["provider_output"]
    for event in confirmations:
        assert event["metrics"]["listing_price_check"]["rows"] == []
        assert event["metrics"]["listing_price_upright_retry"]["rows"] == []
    value = json.loads(confirmations[1]["raw"])["value"]
    reading, = recovery["listing_price_line_retry"]["rows"]
    assert value == "Lanham, 30 dollars" and reading["text"] == "30.00 $"
    assert current[-1]["success"] is True and current[-1]["reward"] == 1.0
    locate = next(event["metrics"] for event in current if event["event"] == "prediction"
                  and event["metrics"].get("evidence_stage") == "locate")
    city, = [row for row in locate["screen_text"] if row["text"].startswith("Location: Lanham,")]
    with Image.open(before) as screen:
        width, height = locate["provider_image_size"]
        x1, y1, x2, y2 = city["bounds"]
        city_box = (round(x1 * screen.width / width) - 8, round(y1 * screen.height / height) - 5,
                    round(x2 * screen.width / width) + 8, round(y2 * screen.height / height) + 5)
        return screen.crop(recovery["listing_price_line_retry"]["bounds"]).convert("RGB"), \
            screen.crop(city_box).convert("RGB"), value, reading["text"], before, traces


def canvas(lang, font_files, text, kind, phase):
    scene = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(scene)

    def label(x, y, message, size, color=INK, *, bold=False, anchor="lm"):
        cjk = any("\u4e00" <= glyph <= "\u9fff" for glyph in message)
        face = font_files[2] if cjk else font_files[1 if bold else 0]
        draw.text((x, y), message, font=ImageFont.truetype(str(face), size), fill=color, anchor=anchor,
                  stroke_width=int(cjk and bold), stroke_fill=color)

    draw.rectangle((0, 0, 1280, 232), fill="#144c42")
    with Image.open(OUTPUT / "gui-recover-mark.png") as logo:
        mark = logo.convert("RGBA").resize((49, 49), Image.Resampling.LANCZOS)
        scene.paste(mark, (42, 29), mark)
    label(103, 54, "GUI-Recover", 28, "#dcefeb", bold=True)
    label(48, 123, text[f"{kind}_title"], 56 if lang == "en" else 48, "#ffffff", bold=True)
    subtitle = text["input_target"] + "  Cristin" if kind == "input" else text["price_sub"]
    label(48, 186, subtitle, 26, "#e2f2e8")
    label(1232, 54, text["recorded"], 22, "#dcefeb", anchor="rm")
    label(1232, 186, text[f"{kind}_stage"][phase], 25, "#bbefc8", bold=True, anchor="rm")
    return scene, draw, label


def photo(scene, image, box):
    x1, y1, x2, y2 = box
    scaled = ImageOps.contain(image, (x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    scene.paste(scaled, (x1, y1 + (y2 - y1 - scaled.height) // 2))


def input_scenes(lang, fonts, before, ends):
    text = COPY[lang]
    with Image.open(before) as frame:
        injected = frame.crop((0, 76, 225, 124)).convert("RGB")
    final = []
    for endpoint in ends:
        with Image.open(endpoint) as frame:
            final.append(frame.crop((0, 76, 225, 124)).convert("RGB"))
    for phase in range(3):
        scene, draw, label = canvas(lang, fonts, text, "input", phase)
        recovery = phase > 0
        for index, left in enumerate((48, 650)):
            draw.rounded_rectangle((left, 270, left + 582, 610), radius=24, fill=PAPER,
                                   outline=GREEN if index == 1 and recovery else RIM,
                                   width=3 if index == 1 and recovery else 2)
            label(left + 33, 311, text["baseline"] if index == 0 else "GUI-Recover", 30, bold=True)
            photo(scene, final[index] if recovery else injected, (left + 31, 350, left + 551, 486))
            label(left + 33, 525, text[f"input_{'left' if index == 0 else 'right'}"][phase], 27, MUTED)
            if phase == 2:
                label(left + 33, 570, text[f"input_end_{'left' if index == 0 else 'right'}"], 27,
                      RED if index == 0 else GREEN, bold=True)
        label(48, 657, text["input_foot"], 24, MUTED)
        yield scene


def price_scenes(lang, fonts, price, city, answer, reading):
    text = COPY[lang]
    for phase in range(3):
        scene, draw, label = canvas(lang, fonts, text, "price", phase)
        draw.rounded_rectangle((48, 270, 614, 610), radius=24, fill=PAPER, outline=RIM, width=2)
        draw.rounded_rectangle((634, 270, 1232, 610), radius=24, fill=PAPER, outline=RIM, width=2)
        label(80, 312, text["fields"], 30, bold=True)
        label(667, 312, text["reads"], 30, bold=True)
        photo(scene, price, (79, 355, 581, 489))
        photo(scene, city, (80, 508, 581, 567))
        for index, name in enumerate(text["reads_labels"]):
            y = 379 + index * 59
            accepted = index == 2 and phase > 0
            draw.rounded_rectangle((666, y - 21, 1200, y + 22), radius=9, fill=MINT if accepted else BACKGROUND)
            label(684, y, name, 25, MUTED)
            label(1183, y, reading if accepted else text["waiting" if index == 2 else "empty"], 25,
                  GREEN if accepted else MUTED, bold=accepted, anchor="rm")
        if phase == 1:
            label(669, 568, text["model_check"], 24, GREEN)
        if phase == 2:
            label(669, 568, answer, 26, GREEN, bold=True)
            label(48, 657, text["abstained"], 24, MUTED)
            label(1232, 657, text["paid"], 27, GREEN, bold=True, anchor="rm")
        else:
            label(48, 657, text["price_foot"], 24, MUTED)
        yield scene


def export(ffmpeg, name, scenes):
    images = tuple(scenes)
    durations = (60, 55, 75)
    fade = 7
    video = OUTPUT / f"{name}.mp4"
    web_video = OUTPUT / f"{name}.webm"
    preview = OUTPUT / f"{name}.gif"
    options = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y"]
    with subprocess.Popen([*options, "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
                           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
                          stdin=subprocess.PIPE) as encoder:
        for index, (scene, duration) in enumerate(zip(images, durations, strict=True)):
            for frame in range(duration):
                if index and frame < fade:
                    encoder.stdin.write(Image.blend(images[index - 1], scene, (frame + 1) / fade).tobytes())
                else:
                    encoder.stdin.write(scene.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError(f"Encoding failed for {video}")
    subprocess.run([*options, "-i", str(video), "-an", "-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0",
                    "-deadline", "good", "-cpu-used", "4", str(web_video)], check=True)
    subprocess.run([*options, "-i", str(video), "-filter_complex",
                    "fps=6,scale=720:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse",
                    "-loop", "0", str(preview)], check=True)
    return {file.name: sha(file) for file in (video, web_video, preview)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", type=Path, required=True, help="FFmpeg executable with libx264")
    parser.add_argument("--vwa-runs", type=Path, required=True, help="Extracted cohort 010 run directories")
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--bold-font", type=Path, required=True)
    parser.add_argument("--cjk-font", type=Path, required=True)
    args = parser.parse_args()
    fonts = (args.font, args.bold_font, args.cjk_font)
    before, ends = input_evidence()
    price, city, answer, reading, listing, traces = price_evidence(args.vwa_runs)
    assets = {}
    for lang in COPY:
        assets.update(export(args.ffmpeg, f"miniwob-case-replay.{lang}", input_scenes(lang, fonts, before, ends)))
        assets.update(export(args.ffmpeg, f"vwa-ocr-replay.{lang}", price_scenes(lang, fonts, price, city, answer, reading)))
    record = {"scope": "Edited replays of recorded native browser frames and actions",
              "mark": {"file": "gui-recover-mark.png", "sha256": sha(OUTPUT / "gui-recover-mark.png"),
                       "method": "Built-in image generation", "prompt": MARK_PROMPT},
              "assets_sha256": assets, "miniwob": {"task": "enter-text", "seed": 69000,
                                                    "provenance": "../visual-text-content-pair/provenance.json"},
              "visualwebarena": {"cohort": "010", "task": "11002", "original_viewport_sha256": sha(listing),
                                 "trajectory_sha256": {arm: sha(trace) for arm, trace in zip(("prior", "current"), traces)},
                                 "results": "../vwa-catalog-price-transfer-results-010.json"}}
    (OUTPUT / "replay-media-provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
