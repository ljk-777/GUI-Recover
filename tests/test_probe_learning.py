from copy import deepcopy

from gui_recover.probe_learning import FrozenProbeChoice, ProbeLearner
from gui_recover.recovery_env import ProbeDecision


def test_online_updates_learn_when_probe_cost_is_worthwhile():
    learner = ProbeLearner(seed=7, epsilon=0., min_visits=2)
    immediate, waited = {"waited": 0}, {"waited": 1}
    first = FrozenProbeChoice(learner.checkpoint())
    assert first(waited) == ProbeDecision.PROBE_LAST_CHAR
    for _ in range(4):
        action = learner.choose(immediate)
        learner.finish_episode(.93 if action == ProbeDecision.PROBE_LAST_CHAR else 0.)
        action = learner.choose(waited)
        learner.finish_episode(.96 if action == ProbeDecision.PROBE_LAST_CHAR else .98)
    checkpoint = learner.checkpoint()
    assert checkpoint["visits"] == {
        "immediate": {"CONTINUE": 2, "PROBE_LAST_CHAR": 2},
        "after_wait": {"CONTINUE": 2, "PROBE_LAST_CHAR": 2},
    }
    original = deepcopy(checkpoint)
    policy = FrozenProbeChoice(checkpoint)
    assert policy(immediate) == ProbeDecision.PROBE_LAST_CHAR
    assert policy(waited) == ProbeDecision.CONTINUE
    assert checkpoint == original
