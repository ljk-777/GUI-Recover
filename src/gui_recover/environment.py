from dataclasses import dataclass
import os

from PIL import Image

from .actions import Action, compile_action


@dataclass(frozen=True)
class VisualObservation:
    goal: str
    screenshot: Image.Image
    last_action_error: str = ""


class MiniwobEnvironment:
    """Only visual observations cross the model boundary; no DOM or reward."""

    def __init__(self, task: str):
        import browsergym.miniwob  # noqa: F401 -- registers environments
        import gymnasium as gym
        from browsergym.core.action.highlevel import HighLevelActionSet

        if not os.environ.get("MINIWOB_URL", "").endswith("/"):
            raise ValueError("Set MINIWOB_URL to the MiniWoB HTML directory URL ending in /")
        self.task = task
        actions = HighLevelActionSet(subsets=["coord"], strict=True, multiaction=True)
        self.env = gym.make(f"browsergym/miniwob.{task}", headless=True,
                            action_mapping=actions.to_python_code)
        self.obs = None

    @staticmethod
    def _visual(obs):
        return VisualObservation(obs["goal"], Image.fromarray(obs["screenshot"]).convert("RGB"),
                                 obs.get("last_action_error", ""))

    def reset(self, seed):
        obs, info = self.env.reset(seed=seed)
        self.obs = self._visual(obs)
        return self.obs, info

    def step(self, action: Action):
        code = compile_action(action, *self.obs.screenshot.size)
        obs, reward, terminated, truncated, info = self.env.step(code)
        self.obs = self._visual(obs)
        return self.obs, float(reward), bool(terminated), bool(truncated), info, code

    def diagnostic_button_position(self):
        """DOM oracle for plumbing tests ONLY. Never used by ShowUI."""
        if self.task != "click-test":
            raise ValueError("oracle-smoke is restricted to click-test")
        page = self.env.unwrapped.page
        box = page.locator("#subbtn").bounding_box()
        scale = getattr(page, "_bgym_scale_factor", 1.0)
        width, height = self.obs.screenshot.size
        return ((box["x"] + box["width"] / 2) * scale / width,
                (box["y"] + box["height"] / 2) * scale / height)

    def metadata(self):
        base = self.env.unwrapped
        return {"browser_version": base.browser.version,
                "viewport": base.page.viewport_size,
                "screenshot_size": self.obs.screenshot.size,
                "screenshot_scale": getattr(base.page, "_bgym_scale_factor", 1.0)}

    def close(self):
        self.env.close()
