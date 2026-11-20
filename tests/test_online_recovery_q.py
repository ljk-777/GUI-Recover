from copy import deepcopy

import pytest

from gui_recover import online_recovery_q
from gui_recover.recovery_env import Decision


def test_live_episode_updates_wait_before_freezing(monkeypatch):
    monkeypatch.setattr(online_recovery_q, "recovery_state", lambda observation, verifier: observation["state"])
    first = {"state": "immediate:empty", "action_mask": [1, 1, 1, 1]}
    later = {"state": "after_wait:empty", "action_mask": [1, 0, 1, 1]}
    closed = {"state": None, "action_mask": [0, 0, 0, 0]}
    learner = online_recovery_q.OnlineRecoveryQ(None, seed=13, epsilon=0)

    learner.observe(first, Decision.WAIT, 0., later, False, False)
    with pytest.raises(ValueError, match="unfinished episode"):
        learner.checkpoint()
    learner.observe(later, Decision.RETRY_FIXED_ZOOM, 1., closed, True, False)
    checkpoint = learner.checkpoint()
    assert checkpoint["episodes"] == 1
    assert checkpoint["q"]["after_wait:empty"]["RETRY_FIXED_ZOOM"] == pytest.approx(0.95)
    assert checkpoint["q"]["immediate:empty"]["WAIT"] == pytest.approx(0.93)
    frozen = online_recovery_q.FrozenOnlinePolicy(None, checkpoint)
    original = deepcopy(checkpoint)
    assert frozen(first) == Decision.WAIT
    assert frozen(later) == Decision.RETRY_FIXED_ZOOM
    assert checkpoint == original


def test_truncated_episode_updates_without_bootstrapping(monkeypatch):
    monkeypatch.setattr(online_recovery_q, "recovery_state", lambda observation, verifier: observation["state"])
    learner = online_recovery_q.OnlineRecoveryQ(None, seed=13, epsilon=0)
    first = {"state": "immediate:different", "action_mask": [1, 1, 1, 1]}
    after = {"state": "after_wait:different", "action_mask": [0, 0, 0, 0]}
    learner.observe(first, Decision.WAIT, 0., after, False, True)
    assert learner.checkpoint()["q"] == {"immediate:different": {"WAIT": -0.02}}
