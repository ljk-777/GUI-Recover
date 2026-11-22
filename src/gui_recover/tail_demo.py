"""A CPU-only browser demonstration across editable controls and visual styles."""

from dataclasses import dataclass
import sys

from .portable_tail import inspect_selection
from .runner import write_json
from .sft import file_sha256
from .tail_demo_report import write_tail_demo_report
from .verification_capture import capture_context_frame, capture_selected_tail


@dataclass(frozen=True)
class Profile:
    name: str
    control: str
    font: str
    size: int
    background: str
    foreground: str
    selection: str
    scale: float
    viewport: tuple[int, int]


PROFILES = (
    Profile("light-input", "input", "Arial", 15, "#fff", "#161616", "#a8cff9", 1, (490, 320)),
    Profile("dark-input", "input", "Courier New", 21, "#272933", "#f2f3f6", "#25654a", 1.5, (630, 430)),
    Profile("light-textarea", "textarea", "Georgia", 18, "#fff9ed", "#3a261d", "#ffd480", 2, (540, 360)),
    Profile("dark-textarea", "textarea", "Arial", 24, "#1e2835", "#fff", "#72549c", 1.25, (710, 410)),
)
SAMPLES = ("Ivor", "Hazel", "Mavis")


def _load_case(browser, profile, layout, expected, value):
    width, height = profile.viewport
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=profile.scale)
    control = ('<input id="entry" autocomplete="off">' if profile.control == "input"
               else '<textarea id="entry" rows="3"></textarea>')
    # Four CSS pixels also align at the 1.25 and 1.5 device scales. Avoid
    # platform-painted outlines and scrollbars in this exact-frame fixture.
    line_height = 4 * round(profile.size * 1.35 / 4)
    page.set_content(f"""
        <style>
        body {{ margin:0; background:#eceef3; font:14px Arial; }}
        main {{ position:absolute; left:{36 + layout * 32}px; top:{40 + layout * 20}px; }}
        #entry {{ display:block; width:{232 + layout * 16}px; box-sizing:border-box;
            appearance:none; outline:none; resize:none; overflow:hidden;
            padding:8px 12px; border:4px solid #64748b; border-radius:0;
            background:{profile.background}; color:{profile.foreground};
            font:{profile.size}px/{line_height}px '{profile.font}'; caret-color:transparent; }}
        #entry::selection {{ background:{profile.selection}; color:{profile.foreground}; }}
        button {{ appearance:none; outline:none; box-shadow:none; margin-top:16px;
            padding:4px 12px; border:4px solid #64748b; border-radius:0;
            background:#fafbfc; color:#161616; font:14px/20px Arial; }}
        button:focus-visible {{ outline:2px solid #145287; outline-offset:2px; }}
        </style>
        <main><form>{control}<button id="submit">Submit</button></form><output id="result"></output></main>
    """)
    page.evaluate("""expected => document.querySelector('form').addEventListener('submit', event => {
        event.preventDefault();
        document.querySelector('#result').value =
            document.querySelector('#entry').value === expected ? 'exact' : 'incorrect';
    })""", expected)
    field = page.locator("#entry")
    field.click()
    page.keyboard.insert_text(value)
    box = field.bounding_box()
    bounds = tuple(round(value * profile.scale) for value in
                   (box["x"] + 4, box["y"] + 4, box["x"] + box["width"] - 4, box["y"] + box["height"] - 4))
    return page, field, bounds


def _save_frames(folder, **frames):
    folder.mkdir(exist_ok=False)
    records = {}
    for label, frame in frames.items():
        path = folder / f"{label}.png"
        frame.save(path)
        records[label] = {"file": path.name, "sha256": file_sha256(path)}
    return records


def _run_case(browser, out, profile, layout, expected, fault):
    value = expected + (" " if fault else "")
    page, field, bounds = _load_case(browser, profile, layout, expected, value)
    try:
        before = capture_context_frame(page)
        selected = capture_selected_tail(page, capture_context_frame)
        restored = capture_context_frame(page)
        unchanged = field.input_value() == value
        same_pixels = before.tobytes() == restored.tobytes()
        evidence = (inspect_selection(restored, selected, bounds) if same_pixels
                    else {"status": "unknown", "reason": "visual_state_changed"})
        repaired = evidence["status"] == "blank_tail"
        if repaired:
            page.keyboard.press("Meta+A" if sys.platform == "darwin" else "Control+A")
            page.keyboard.insert_text(expected)
        page.locator("#submit").click()
        success = page.locator("#result").text_content() == "exact"
        name = f"{profile.name}-{layout}-{'trailing-space' if fault else 'clean'}"
        frames = _save_frames(out / name, before=before, selected=selected, restored=restored,
                              submitted=capture_context_frame(page))
        row = {"id": name, "profile": profile.name, "layout": layout, "trailing_space": fault,
               "field_interior": bounds, "evidence": evidence, "value_unchanged_by_probe": unchanged,
               "probe_restored_screen": same_pixels, "repaired": repaired, "success": success, "frames": frames}
        write_json(out / name / "result.json", row)
        return row
    finally:
        page.close()


def _run_controls(browser, out, profile):
    expected = "A long visibly selected value"
    if profile.control == "textarea":
        expected += "\non a second line"
    rows = []
    for control in ("blurred", "many_characters"):
        page, field, bounds = _load_case(browser, profile, 0, expected, expected)
        try:
            if control == "blurred":
                page.locator("#submit").focus()
            before = capture_context_frame(page)
            if control == "blurred":
                selected = capture_selected_tail(page, capture_context_frame)
            else:
                page.keyboard.press("Meta+A" if sys.platform == "darwin" else "Control+A")
                selected = capture_context_frame(page)
            evidence = inspect_selection(before, selected, bounds)
            name = f"{profile.name}-{control}"
            frames = _save_frames(out / name, before=before, selected=selected)
            row = {"id": name, "profile": profile.name, "negative_control": control, "field_interior": bounds,
                   "evidence": evidence, "value_unchanged": field.input_value() == expected, "frames": frames}
            write_json(out / name / "result.json", row)
            rows.append(row)
        finally:
            page.close()
    return rows


def run_tail_demo(out, samples=SAMPLES):
    """Run paired native selections and negative controls without model weights."""
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=False)
    rows, controls = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            version = browser.version
            for profile in PROFILES:
                for index, sample in enumerate(samples):
                    expected = ("Note\n" + sample) if profile.control == "textarea" else sample
                    rows.extend(_run_case(browser, out, profile, index, expected, fault) for fault in (False, True))
                controls.extend(_run_controls(browser, out, profile))
        finally:
            browser.close()
    clean, faulted = rows[::2], rows[1::2]
    result = {
        "browser_version": version, "profiles": [p.name for p in PROFILES],
        "layout_pairs": len(clean), "browser_cases": len(rows),
        "identical_unselected_pairs": sum(a["frames"]["before"]["sha256"] == b["frames"]["before"]["sha256"]
                                          for a, b in zip(clean, faulted, strict=True)),
        "clean_successes": sum(r["success"] for r in clean), "clean_rewrites": sum(r["repaired"] for r in clean),
        "trailing_space_successes": sum(r["success"] for r in faulted),
        "negative_control_abstentions": sum(r["evidence"]["status"] == "unknown" for r in controls),
        "negative_controls": len(controls),
        "selection_preserved_input": sum(r["value_unchanged_by_probe"] and r["probe_restored_screen"] for r in rows),
        "cases": [r["id"] for r in rows], "controls": [r["id"] for r in controls],
    }
    write_json(out / "summary.json", result)
    write_tail_demo_report(out, result, zip(clean, faulted, strict=True))
    return result
