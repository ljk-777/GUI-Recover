"""Online contextual reinforcement learning for one bounded probe decision."""

from collections import Counter, defaultdict
from math import isfinite
from random import Random

from .recovery_env import Decision, ProbeDecision


ALGORITHM = "online_probe_contextual_q_v1"
ACTIONS = (ProbeDecision.PROBE_LAST_CHAR, ProbeDecision.CONTINUE)
CONTEXTS = ("immediate", "after_wait")
COSTS = {"wait": .02, "probe": .02, "repair": .05}


def context(observation):
    return "after_wait" if observation["waited"] else "immediate"


def adjusted_return(row):
    return row["return"] - (COSTS["wait"] * row["waits"] + COSTS["probe"] * row["probes"]
                            + COSTS["repair"] * row["repairs"])


def estimate_action(values):
    # Prefer probing when evidence does not distinguish the values.
    return max(ACTIONS, key=lambda action: values.get(action.name, float("-inf")))


class ProbeLearner:
    """Explore against live browser reward and update after each episode."""

    def __init__(self, *, seed, epsilon, min_visits):
        if not 0 <= epsilon <= 1 or type(min_visits) is not int or min_visits < 1:
            raise ValueError("Invalid probe exploration parameters")
        self.rng = Random(seed)
        self.epsilon = epsilon
        self.min_visits = min_visits
        self.q = defaultdict(dict)
        self.visits = defaultdict(Counter)
        self.pending = None
        self.episodes = 0

    def choose(self, observation):
        if self.pending is not None:
            raise ValueError("The learner can make only one matched-state decision per episode")
        state = context(observation)
        visits = self.visits[state]
        under_sampled = [a for a in ACTIONS if visits[a.name] < self.min_visits]
        if under_sampled:
            action = self.rng.choice(under_sampled)
        elif self.rng.random() < self.epsilon:
            action = self.rng.choice(ACTIONS)
        else:
            action = estimate_action(self.q[state])
        self.pending = state, action.name
        return action

    def finish_episode(self, episode_return):
        self.episodes += 1
        if self.pending is None:
            return
        if not isfinite(episode_return):
            raise ValueError("Expected a finite browser return")
        state, key = self.pending
        previous = self.q[state].get(key, 0.)
        self.visits[state][key] += 1
        self.q[state][key] = previous + (episode_return - previous) / self.visits[state][key]
        self.pending = None

    def checkpoint(self):
        if self.pending is not None:
            raise ValueError("Cannot freeze a learner during an episode")
        return {"algorithm": ALGORITHM, "costs": COSTS, "episodes": self.episodes,
                "q": {s: dict(values) for s, values in self.q.items()},
                "visits": {s: dict(visits) for s, visits in self.visits.items()}}


class FrozenProbeChoice:
    """Route matched-state observations through a frozen, fully sampled table."""

    def __init__(self, checkpoint):
        if checkpoint["algorithm"] != ALGORITHM or checkpoint["costs"] != COSTS:
            raise ValueError("Unsupported learned probe checkpoint")
        self.q = {}
        expected = {action.name for action in ACTIONS}
        for state, values in checkpoint["q"].items():
            if state not in CONTEXTS or not values.keys() <= expected or not all(map(isfinite, values.values())):
                raise ValueError("Invalid learned probe values")
            if values.keys() == expected:
                self.q[state] = dict(values)
        self.counts = Counter()

    def __call__(self, observation):
        state = context(observation)
        if state not in self.q:
            self.counts["fallback_probe"] += 1
            return ProbeDecision.PROBE_LAST_CHAR
        self.counts["learned_choice"] += 1
        return estimate_action(self.q[state])


def skip_after_wait(observation):
    return Decision.CONTINUE if observation["waited"] else ProbeDecision.PROBE_LAST_CHAR
