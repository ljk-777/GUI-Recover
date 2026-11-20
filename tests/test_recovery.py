from PIL import Image

from gui_recover.actions import Action
from gui_recover.environment import VisualObservation
from gui_recover.recovery import ClickFault, FeedbackRecovery, detect


def obs(color="white", error=""):
    return VisualObservation("a task", Image.new("RGB", (30, 20), color), error)


def test_fault_is_once_at_first_eligible_click_and_deterministic():
    click = Action("CLICK", position=(0.3, 0.7))
    fault = ClickFault("drop-click", 12)
    assert fault.apply(Action("TYPE", "test"))[1] is None
    executed, label = fault.apply(click)
    assert executed == Action("HOVER", position=click.position)
    assert label["proposed"] == click.to_dict()
    assert fault.apply(click) == (click, None)
    assert ClickFault("offset-click", 42).apply(click) == ClickFault("offset-click", 42).apply(click)


def test_detector_is_pixel_only_and_feedback_is_bounded():
    click = Action("CLICK", position=(0.2, 0.3))
    unchanged = detect(obs(), obs(), click)
    changed = detect(obs(), obs("black"), click)
    assert unchanged.unchanged_after_click
    assert not changed.unchanged_after_click
    assert not detect(obs(), obs(), Action("TYPE", "test")).unchanged_after_click
    controller = FeedbackRecovery(enabled=True, limit=1)
    assert controller.feedback(changed) is None
    assert controller.feedback(unchanged)
    assert controller.feedback(unchanged) is None
    assert FeedbackRecovery().feedback(unchanged) is None


def test_scroll_feedback_requires_an_unchanged_viewport_and_explicit_opt_in():
    scroll = Action("SCROLL", "up")
    before = obs()
    nearly_identical = obs()
    nearly_identical.screenshot.putpixel((0, 0), (254, 254, 254))
    unchanged = detect(before, obs(), scroll)
    assert unchanged.unchanged_after_scroll
    assert not detect(before, nearly_identical, scroll).unchanged_after_scroll
    assert FeedbackRecovery(enabled=True).feedback(unchanged) is None

    controller = FeedbackRecovery(enabled=True, scroll=True, limit=1)
    assert "viewport unchanged" in controller.feedback(unchanged)
    assert controller.scroll_used == controller.used == 1
    assert controller.feedback(unchanged) is None


def test_injection_metadata_never_reaches_policy(tmp_path):
    from gui_recover.runner import run_episode

    class Policy:
        messages = []
        histories = []

        def predict(self, observation, history):
            self.histories.append(list(history))
            return "{'action':'CLICK','value':None,'position':[0.3,0.7]}"

        def set_feedback(self, message):
            self.messages.append(message)

    class Env:
        def reset(self, seed):
            return obs(), {}

        def metadata(self):
            return {}

        def step(self, action):
            if action.action == "HOVER":
                return obs(), 0, False, False, {}, "mouse_move(9,14)"
            return obs(), 1, True, False, {"task_info": {"DONE_GLOBAL": True, "RAW_REWARD_GLOBAL": 1}}, "mouse_click(9,14)"

    policy = Policy()
    result = run_episode(Env(), policy, "fake", 0, 2, tmp_path / "episode",
                         fault_kind="drop-click", recovery="feedback")
    assert result["success"] and result["injected"]
    assert result["steps"] == result["predictions"] == 2
    assert result["recovery_feedbacks"] == 1
    assert policy.histories[1] == [{"action": Action("CLICK", position=(0.3, 0.7)).to_dict(), "execution_error": ""}]
    assert all("drop-click" not in str(message) for message in policy.messages)
