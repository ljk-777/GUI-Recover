"""Finite-horizon fitted Q-learning over real visual recovery transitions."""

from collections import Counter, defaultdict
from math import isfinite

from .recovery_env import Decision, Phase, transition_view
from .recovery_rollouts import baseline_decision
from .verification import appearance_verdict


ALGORITHM = "finite_horizon_fitted_q_v1"
DEFAULT_COSTS = (0., .02, .05, .05)


def feature_state(observation):
    """The compact policy state excludes seed, intervention, DOM and outcome."""
    view = transition_view(observation)
    if view.proposed.action != "INPUT":
        return None
    stage = "after_wait" if observation["waited"] else "immediate"
    return stage + ":" + appearance_verdict(view)["status"]


def _costs(costs):
    result = tuple(map(float, costs))
    if len(result) != len(Decision) or not all(isfinite(x) and 0 <= x < 1 for x in result):
        raise ValueError("Invalid decision costs")
    return result


def fit_recovery_q(episodes, costs=DEFAULT_COSTS):
    """Average real one-step Bellman targets, fitting the post-wait stage first."""
    costs = _costs(costs)
    unique = {}
    seen = 0
    for group_id, episode in episodes:
        for row in episode:
            obs, after = row["observation"], row["next_observation"]
            if obs["phase"] != Phase.REVIEW:
                continue
            s = feature_state(obs)
            if s is None:
                raise ValueError("Learning corpus contains a non-input recovery state")
            action = Decision(row["action"])
            if not obs["action_mask"][action] or row["reward"] not in (0., 1.):
                raise ValueError("Invalid learning transition")
            closed = row["terminated"] or row["truncated"]
            next_s = feature_state(after) if not closed else None
            if not closed and (action != Decision.WAIT or obs["waited"] or
                               not after["waited"] or next_s is None or row["reward"] != 0.):
                raise ValueError("The only nonterminal review decision must be a first wait")
            following = tuple(a for a in Decision if after["action_mask"][a]) if next_s else ()
            key = (group_id, s, action)
            sample = (row["reward"], next_s, following)
            if unique.setdefault(key, sample) != sample:
                raise ValueError("Repeated exploratory branch produced a different outcome")
            seen += 1
    if not unique:
        raise ValueError("No review transitions available for Q-learning")

    q = {}
    support = {}
    for stage in ("after_wait", "immediate"):
        values = defaultdict(list)
        for (_, state, action), (reward, next_s, following) in unique.items():
            if not state.startswith(stage+":"):
                continue
            future = [q[next_s][a.name] for a in following if a.name in q.get(next_s, {})]
            if next_s is not None and not future:
                raise ValueError("A nonterminal training branch has no measured continuation")
            values[state, action].append(reward-costs[action]+(max(future) if future else 0.))
        for (state, action), targets in sorted(values.items()):
            q.setdefault(state, {})[action.name] = sum(targets)/len(targets)
            support.setdefault(state, {})[action.name] = len(targets)
    return {"algorithm": ALGORITHM, "discount": 1., "costs": list(costs), "q": q, "support": support,
            "fit": {"review_rows": seen, "independent_state_action_outcomes": len(unique),
                    "duplicate_branch_rows_removed": seen-len(unique),
                    "fitted_entries": sum(map(len, q.values()))}}


class FittedRecoveryPolicy:
    """Greedy over learned, legal actions. Unsupported visual states use the frozen rule."""

    def __init__(self, checkpoint):
        if checkpoint["algorithm"] != ALGORITHM or checkpoint["discount"] != 1. or not checkpoint["q"]:
            raise ValueError("Unsupported recovery policy checkpoint")
        _costs(checkpoint["costs"])
        self.q = checkpoint["q"]
        for state, options in self.q.items():
            if state not in {phase+":"+status for phase in ("immediate", "after_wait")
                             for status in ("achieved", "not_achieved", "unknown")} or not options:
                raise ValueError("Unrecognized Q state")
            if any(a not in Decision.__members__ or not isfinite(v) for a, v in options.items()):
                raise ValueError("Invalid learned action value")
            if state.startswith("after_wait:") and Decision.WAIT.name in options:
                raise ValueError("Wait is illegal after waiting")
        self.counts = Counter()

    def __call__(self, observation):
        if observation["phase"] != Phase.REVIEW:
            return Decision.CONTINUE
        state = feature_state(observation)
        scores = self.q.get(state, {})
        legal = [a for a in Decision if a.name in scores and observation["action_mask"][a]]
        if not legal:
            self.counts["fallback_review"] += 1
            return baseline_decision(observation, "visual_fixed")
        self.counts["learned_review"] += 1
        return max(legal, key=lambda a: scores[a.name])
