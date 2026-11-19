import json

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from PIL import Image
import pytest

from gui_recover.environment import VisualObservation
from gui_recover.recovery_env import Decision, RecoveryDecisionEnv, exact_success
from gui_recover.recovery_rollouts import iter_learning_transitions, record_rollout
from gui_recover.repair_study_data import png_sha
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256


class Fixture:
    def __init__(self, task, seed, proposed, regime):
        self.obs = self.before = VisualObservation("Click on the \"yes\" button.",
                                                   Image.new("RGB", (498, 321), (seed % 255, 0, 0)))
        self.proposed = proposed
        self.events = []
        self.info = {"task_info": {}}
        self.reward, self.done, self.truncated = 0., False, False

    def first(self):
        self.events.append({"event": "first"})

    def wait_once(self):
        self.events.append({"event": "wait_hover"})

    def close(self):
        pass


class Grounder:
    def ground(self, picture, target):
        return {"point": [.1,.2]}

    def repair(self, view, mode, **options):
        return {"point": None}


def test_exact_reward_rejects_partial_browsergym_positive_reward():
    assert exact_success(True, False, 1., {"DONE_GLOBAL":True,"RAW_REWARD_GLOBAL":1.})
    assert not exact_success(True, False, 1., {"DONE_GLOBAL":True,"RAW_REWARD_GLOBAL":.5})
    assert not exact_success(False, False, 1., {"DONE_GLOBAL":True,"RAW_REWARD_GLOBAL":1.})
    assert not exact_success(True, True, 1., {"DONE_GLOBAL":True,"RAW_REWARD_GLOBAL":1.})


def test_gym_masks_and_budget_do_not_generate_positive_reward(monkeypatch):
    monkeypatch.setattr("gui_recover.recovery_env.TransitionExecution", Fixture)
    env = RecoveryDecisionEnv("click-button", Grounder(), max_decisions=2)
    check_env(env, skip_render_check=True, skip_close_check=True)
    obs, info = env.reset(seed=42)
    assert info == {} and obs["action_mask"].tolist() == [1,0,0,0]
    _, reward, terminated, truncated, info = env.step(Decision.RETRY_PLAIN)
    assert (reward, terminated, truncated, info["outcome"], info["actor_requests"]) == (0.,True,False,"masked_action",0)
    env.reset(seed=42)
    obs, *_ = env.step(Decision.CONTINUE)
    assert obs["action_mask"].tolist() == [1,1,1,1] and obs["remaining_decisions"] == 1
    _, reward, terminated, truncated, info = env.step(Decision.WAIT)
    assert (reward,terminated,truncated,info["outcome"]) == (0.,False,True,"decision_budget")
    with pytest.raises(gym.error.ResetNeeded):
        env.step(Decision.CONTINUE)
    env.close()


def test_training_reader_validates_images_and_keeps_test_and_evidence_out(monkeypatch, tmp_path):
    monkeypatch.setattr("gui_recover.recovery_env.TransitionExecution", Fixture)
    env=RecoveryDecisionEnv("click-button",Grounder())
    example = Fixture("click-button",17,None,"none")
    scene={"task":"click-button","seed":17,"split":"train","initial_sha256":png_sha(example.obs.screenshot)}
    episode = record_rollout(env,scene,"continue",tmp_path/"rollouts/train/sample")
    trace="rollouts/train/sample/transitions.jsonl"
    index=tmp_path/"train.jsonl"
    index.write_text(json.dumps({**episode,"trace":trace})+"\n")
    write_json(tmp_path/"summary.json",{"index_sha256":{"train":file_sha256(index)}})
    bundle = list(iter_learning_transitions(tmp_path))
    assert len(bundle)==2 and len(bundle[0])==6 and bundle[1]["terminated"]
    assert env.observation_space.contains(bundle[0]["observation"])
    with pytest.raises(ValueError,match="train and validation"):
        list(iter_learning_transitions(tmp_path,"test"))
    (tmp_path/"rollouts/train/sample/001.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="checksum"):
        list(iter_learning_transitions(tmp_path))
    env.close()
