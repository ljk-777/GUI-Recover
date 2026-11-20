"""Learn bounded recovery choices from newly executed browser episodes."""

from collections import Counter, defaultdict
from math import isfinite
from random import Random

from .recovery_env import Decision, Phase, transition_view
from .recovery_q import DEFAULT_COSTS
from .visual_text import ContentRecoveryPolicy


ALGORITHM = "online_episodic_q_v1"
ACTIONS = (Decision.CONTINUE, Decision.WAIT, Decision.RETRY_FIXED_ZOOM)


def recovery_state(observation, verifier):
    if observation["phase"] != Phase.REVIEW:
        return None
    stage = "after_wait" if observation["waited"] else "immediate"
    glyph = verifier.verify(transition_view(observation))["status"]
    return f"{stage}:{glyph}"


def legal_actions(observation):
    return tuple(action for action in ACTIONS if observation["action_mask"][action])


class OnlineRecoveryQ:
    """Epsilon-greedy collection with a backward Q update after each episode."""

    def __init__(self, verifier, *, seed, epsilon):
        if not 0 <= epsilon <= 1:
            raise ValueError("Exploration probability must be in [0, 1]")
        self.verifier = verifier
        self.rng = Random(seed)
        self.epsilon = epsilon
        self.q = defaultdict(dict)
        self.support = defaultdict(Counter)
        self.pending = []
        self.episodes = 0

    def choose(self, observation):
        state = recovery_state(observation, self.verifier)
        if state is None:
            return Decision.CONTINUE
        legal = legal_actions(observation)
        values = self.q[state]
        unseen = [action for action in legal if action.name not in values]
        if unseen:
            return self.rng.choice(unseen)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        return max(legal, key=lambda action: values[action.name])

    def observe(self, before, action, reward, after, terminated, truncated):
        state = recovery_state(before, self.verifier)
        closed = terminated or truncated
        if state is not None:
            following = None if closed else recovery_state(after, self.verifier)
            next_legal = () if closed else legal_actions(after)
            self.pending.append((state, action, reward, following, next_legal))
        if closed:
            for state, action, reward, following, next_legal in reversed(self.pending):
                future = self.q.get(following, {})
                continuation = max((future[a.name] for a in next_legal if a.name in future), default=0.)
                target = reward - DEFAULT_COSTS[action] + continuation
                key = action.name
                self.support[state][key] += 1
                previous = self.q[state].get(key, 0.)
                self.q[state][key] = previous + (target - previous) / self.support[state][key]
            self.pending.clear()
            self.episodes += 1

    def checkpoint(self):
        if self.pending:
            raise ValueError("Cannot freeze a learner with an unfinished episode")
        return {
            "algorithm": ALGORITHM, "discount": 1., "costs": list(DEFAULT_COSTS),
            "epsilon": self.epsilon, "episodes": self.episodes,
            "q": {state: dict(values) for state, values in self.q.items() if values},
            "support": {state: dict(counts) for state, counts in self.support.items()},
        }


class FrozenOnlinePolicy:
    def __init__(self, verifier, checkpoint):
        if checkpoint["algorithm"] != ALGORITHM or checkpoint["costs"] != list(DEFAULT_COSTS):
            raise ValueError("Unsupported online recovery checkpoint")
        self.verifier = verifier
        self.q = checkpoint["q"]
        if any(key not in {action.name for action in ACTIONS} or not isfinite(value)
               for values in self.q.values() for key, value in values.items()):
            raise ValueError("Invalid action value in checkpoint")
        self.counts = Counter()
        self.fallback = ContentRecoveryPolicy(verifier)

    def __call__(self, observation):
        state = recovery_state(observation, self.verifier)
        if state is None:
            return Decision.CONTINUE
        values = self.q.get(state, {})
        known = [action for action in legal_actions(observation) if action.name in values]
        if known:
            self.counts["learned_review"] += 1
            return max(known, key=lambda action: values[action.name])
        self.counts["fallback_review"] += 1
        action = self.fallback(observation)
        # Track usage in counts; the trial already has the original observations.
        self.fallback.evidence.clear()
        return action
