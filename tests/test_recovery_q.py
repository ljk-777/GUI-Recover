import pytest

from gui_recover.recovery_env import Decision, Phase
from gui_recover.recovery_q import FittedRecoveryPolicy, fit_recovery_q


def test_wait_value_bootstraps_after_wait_and_repeated_prefix_is_counted_once(monkeypatch):
    monkeypatch.setattr("gui_recover.recovery_q.feature_state", lambda obs: obs["state"])
    empty={"phase":Phase.REVIEW,"state":"immediate:not_achieved","waited":0,"action_mask":[1,1,1,1]}
    empty_after={"phase":Phase.REVIEW,"state":"after_wait:not_achieved","waited":1,"action_mask":[1,0,1,1]}
    visible_after={"phase":Phase.REVIEW,"state":"after_wait:achieved","waited":1,"action_mask":[1,0,1,1]}
    closed={"phase":Phase.CLOSED,"action_mask":[0,0,0,0]}

    def sample(start,act,reward,end=closed):
        return {"observation":start,"action":act,"reward":reward,"next_observation":end,
                "terminated":end is closed,"truncated":False}

    drop_wait=sample(empty,Decision.WAIT,0.,empty_after)
    late_wait=sample(empty,Decision.WAIT,0.,visible_after)
    checkpoint=fit_recovery_q([
      ("drop",[sample(empty,Decision.RETRY_PLAIN,1.)]),
      ("late",[sample(empty,Decision.RETRY_PLAIN,0.)]),
      ("drop",[drop_wait,sample(empty_after,Decision.RETRY_PLAIN,1.)]),
      ("drop",[drop_wait,sample(empty_after,Decision.CONTINUE,0.)]),
      ("late",[late_wait,sample(visible_after,Decision.CONTINUE,1.)]),
      ("late",[late_wait,sample(visible_after,Decision.RETRY_PLAIN,0.)]),
    ])
    assert checkpoint["fit"]["duplicate_branch_rows_removed"]==2
    assert checkpoint["q"]["immediate:not_achieved"]["WAIT"]==pytest.approx(.955)
    assert checkpoint["q"]["immediate:not_achieved"]["RETRY_PLAIN"]==pytest.approx(.45)
    policy=FittedRecoveryPolicy(checkpoint)
    assert policy(empty)==Decision.WAIT
    assert policy(empty_after)==Decision.RETRY_PLAIN
    assert policy(visible_after)==Decision.CONTINUE
