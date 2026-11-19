import json

import numpy as np
import pytest
from PIL import Image

from gui_recover.actions import Action, parse_action
from gui_recover.environment import VisualObservation
from gui_recover.grounding import ScriptedGroundingPolicy
from gui_recover.input_guard import inspect_input
from gui_recover.recovery import ActionFault
from gui_recover.runner import run_episode


def field(kind="empty"):
    pixels = np.full((80, 200, 3), 85, dtype=np.uint8)
    pixels[28:54, 18:182] = 120
    pixels[30:52, 20:180] = 255
    if kind in ("caret", "text"):
        pixels[32:50, 25:27] = 0
    if kind == "focus":
        pixels[30:32, 20:180] = 0
        pixels[50:52, 20:180] = 0
        pixels[30:52, 20:22] = 0
        pixels[30:52, 178:180] = 0
    if kind == "text":
        pixels[35:46, 40:55] = 0
    if kind == "speckle":
        pixels[34, 40] = 0
    return Image.fromarray(pixels)


def test_visual_signal_ignores_caret_and_detects_ink():
    assert inspect_input(field(), field("caret"), (0.5, 0.5))["status"] == "empty"
    assert inspect_input(field(), field("focus"), (0.5, 0.5))["status"] == "empty"
    assert inspect_input(field(), field("text"), (0.5, 0.5))["status"] == "text_visible"
    assert inspect_input(field("text"), field("text"), (0.5, 0.5))["status"] == "unknown"
    assert inspect_input(field(), Image.new("RGB", (200, 80)), (0.5, 0.5))["status"] == "unknown"
    assert inspect_input(Image.new("RGB", (200, 80), "white"), field(), (0.5, 0.5))["status"] == "unknown"


def test_guard_is_appearance_only_and_does_not_validate_content():
    # Any sufficiently visible text can pass; no claim of exact value verification.
    result = inspect_input(field(), field("text"), (0.5, 0.5))
    assert result["status"] == "text_visible"
    assert result["reason"] == "appearance_only_not_text_correctness"


def test_speckle_fix_is_versioned_and_preserves_v1_result():
    assert inspect_input(field(), field("speckle"), (0.5, 0.5), version="v1")["status"] == "unknown"
    fixed = inspect_input(field(), field("speckle"), (0.5, 0.5), version="v2")
    assert fixed["status"] == "empty" and fixed["ignored_speckle_pixels"] == 1


def themed_field(kind, background, foreground):
    pixels = np.asarray(field(kind)).copy()
    interior = pixels[30:52, 20:180]
    is_background = (interior == 255).all(axis=2)
    interior[is_background] = background
    interior[~is_background] = foreground
    return Image.fromarray(pixels)


@pytest.mark.parametrize("background,foreground", [
    ((255, 255, 255), (0, 0, 0)),
    ((51, 51, 51), (255, 255, 255)),
    ((25, 50, 90), (230, 240, 255)),
    ((230, 220, 205), (30, 40, 50)),
])
@pytest.mark.parametrize("kind,expected", [
    ("empty", "empty"), ("caret", "empty"), ("focus", "empty"),
    ("speckle", "empty"), ("text", "text_visible"),
])
def test_v3_uses_field_color_and_ignores_focus_caret_and_speckle(background, foreground, kind, expected):
    result = inspect_input(themed_field("empty", background, foreground),
                           themed_field(kind, background, foreground), (0.5, 0.5), version="v3")
    assert result["status"] == expected
    assert result["background_rgb"] == list(background)


def test_v3_rejects_changed_fill_moved_outline_and_preexisting_text():
    before = themed_field("empty", (51, 51, 51), (255, 255, 255))
    changed = themed_field("empty", (75, 75, 75), (255, 255, 255))
    assert inspect_input(before, changed, (0.5, 0.5), version="v3")["reason"] == "region_background_changed"
    assert inspect_input(before, before.resize((400, 160)), (0.5, 0.5), version="v3")["status"] == "unknown"
    assert inspect_input(before, Image.new("RGB", before.size, (51, 51, 51)),
                         (0.5, 0.5), version="v3")["reason"] == "input_outline_moved_or_changed"
    text = themed_field("text", (51, 51, 51), (255, 255, 255))
    assert inspect_input(text, text, (0.5, 0.5), version="v3")["status"] == "unknown"


def test_v3_requires_a_contrasting_enclosed_region_and_nontrivial_ink():
    blank = Image.new("RGB", (200, 80), (51, 51, 51))
    assert inspect_input(blank, blank, (0.5, 0.5), version="v3")["status"] == "unknown"
    pixels = np.asarray(field()).copy()
    pixels[36:38, 40:42] = 0
    assert inspect_input(field(), Image.fromarray(pixels), (0.5, 0.5), version="v3")["status"] == "unknown"


def test_v3_does_not_change_legacy_dark_theme_results():
    before = themed_field("empty", (51, 51, 51), (255, 255, 255))
    after = themed_field("text", (51, 51, 51), (255, 255, 255))
    for version in ("v1", "v2"):
        assert inspect_input(before, after, (0.5, 0.5), version=version)["status"] == "unknown"
    assert inspect_input(before, after, (0.5, 0.5), version="v3")["status"] == "text_visible"


def test_v3_reidentifies_focused_field_with_antialiased_inner_border():
    # Linux Chromium's focused dark field has one low-contrast row just inside
    # the stronger outline. A retry must recognize this already-focused field.
    blank = np.asarray(themed_field("empty", (51, 51, 51), (255, 255, 255))).copy()
    blank[51, 20:180] = 33
    typed = blank.copy()
    typed[35:46, 40:55] = 255
    assert inspect_input(Image.fromarray(blank), Image.fromarray(blank),
                         (0.5, 0.5), version="v3")["status"] == "empty"
    assert inspect_input(Image.fromarray(blank), Image.fromarray(typed),
                         (0.5, 0.5), version="v3")["status"] == "text_visible"


def test_input_fault_focuses_but_does_not_type_and_is_once():
    injector = ActionFault("drop-input", 7)
    assert injector.apply(Action("CLICK", position=(0.3, 0.3)))[1] is None
    proposed = Action("INPUT", "abc", (0.5, 0.5))
    executed, label = injector.apply(proposed)
    assert executed == Action("CLICK", position=(0.5, 0.5))
    assert label["kind"] == "drop-input"
    assert injector.apply(proposed) == (proposed, None)


def test_partial_and_wrong_input_faults_are_deterministic_counterexamples():
    proposed = Action("INPUT", "Myron", (0.5, 0.5))
    assert ActionFault("partial-input").apply(proposed)[0].value == "My"
    assert ActionFault("wrong-input").apply(proposed)[0].value == "Myron!"


class Backend:
    config = {"model": "test"}
    last_metrics = {}

    def __init__(self):
        self.queries = []

    def reset(self, seed):
        self.queries = []

    def ground(self, screenshot, query):
        self.queries.append(query)
        return "[0.5,0.5]"


class TextEnv:
    def __init__(self, persistent_loss=False):
        self.persistent_loss = persistent_loss
        self.submissions = 0

    def reset(self, seed):
        self.text = ""
        self.picture = field()
        return self.obs(), {}

    def obs(self):
        return VisualObservation('Enter "abc" into the text field and press Submit.', self.picture)

    def metadata(self):
        return {}

    def step(self, action):
        if action.action == "INPUT":
            if not self.persistent_loss:
                self.text += action.value
            self.picture = field("text" if self.text else "caret")
        elif action.action == "CLICK" and not self.text:
            self.picture = field("caret")
        done = action.action == "CLICK" and self.text
        done = bool(done)
        reward = int(done and self.text == "abc")
        self.submissions += int(done)
        return self.obs(), reward, done, False, {"task_info": {"DONE_GLOBAL": done, "RAW_REWARD_GLOBAL": reward}}, "compiled"


def test_missing_input_regrounds_and_recovers_without_label_leakage(tmp_path):
    backend = Backend()
    result = run_episode(TextEnv(), ScriptedGroundingPolicy(backend, input_guard=True),
                         "enter-text", 0, 4, tmp_path / "episode",
                         fault_kind="drop-input", recovery="input-guard")
    assert result["success"] and result["injected"]
    assert result["model_calls"] == result["steps"] == 3
    assert result["input_guard"]["retries"] == 1
    assert backend.queries == ["text field", "text field", "Submit button"]
    events = [json.loads(line) for line in (tmp_path / "episode/trajectory.jsonl").read_text().splitlines()]
    assert [e["decision"] for e in events if e["event"] == "policy_observation"] == ["reground_and_retry", "advance"]


def test_persistent_empty_input_blocks_submission(tmp_path):
    backend = Backend()
    env = TextEnv(persistent_loss=True)
    result = run_episode(env, ScriptedGroundingPolicy(backend, input_guard=True),
                         "enter-text", 0, 4, tmp_path / "episode", recovery="input-guard")
    assert not result["success"]
    assert result["input_guard"]["blocked"] == 1
    assert result["model_calls"] == result["steps"] == 2
    assert result["predictions"] == 3
    assert backend.queries == ["text field", "text field"]
    assert env.submissions == 0


def test_clean_input_does_not_duplicate_text(tmp_path):
    backend = Backend()
    result = run_episode(TextEnv(), ScriptedGroundingPolicy(backend, input_guard=True),
                         "enter-text", 0, 4, tmp_path / "episode", recovery="input-guard")
    assert result["success"]
    assert result["input_guard"]["retries"] == 0
    assert result["model_calls"] == result["steps"] == 2
    assert backend.queries == ["text field", "Submit button"]


def test_blind_retry_succeeds_only_when_first_input_is_lost(tmp_path):
    for fault in ("none", "drop-input"):
        backend = Backend()
        result = run_episode(TextEnv(), ScriptedGroundingPolicy(backend, blind_input_retry=True),
                             "enter-text", 0, 4, tmp_path / fault,
                             fault_kind=fault, recovery="input-retry")
        assert result["success"] == (fault == "drop-input")
        assert result["model_calls"] == result["steps"] == 3
        assert result["input_guard"]["checks"] == 0
