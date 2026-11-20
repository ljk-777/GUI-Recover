import pytest

from gui_recover.sft_balanced import balanced_heads, balanced_training_records, validation_gate


def test_mixture_pairs_contexts_without_replacing_real_action_or_creating_new_scenes():
    previous = {"action": "INPUT", "value": "sample", "position": [0.2, 0.3]}
    rows = [{"id": kind, "split": "train", "kind": kind, "previous_action": action,
             "image_sha256": kind, "answer": "[0.4,0.5]"}
            for kind, action in (("normal_entry", None), ("normal_submit", previous), ("correction", previous))]
    mixture = balanced_training_records(rows)
    assert len(mixture) == 11 and len({r["id"] for r in mixture}) == 11
    for kind in ("normal_submit", "correction"):
        chosen = [r for r in mixture if r["kind"] == kind]
        assert {r["history_input"] for r in chosen} == {True, False}
        assert all(r["previous_action"] == previous and r["answer"] == "[0.4,0.5]" for r in chosen)
    assert {r["image_sha256"] for r in mixture} == {r["image_sha256"] for r in rows}
    with pytest.raises(ValueError, match="train only"):
        balanced_training_records([{**rows[0], "split": "test"}])


def test_all_seed_gate_does_not_choose_only_the_best_replicate():
    seeds = [1, 2, 3]
    heads = balanced_heads(seeds)
    summary = {g: {"heads": {h: {"hits": n} for h in heads}} for g, n in
               (("normal", 17), ("normal_submit", 4), ("correction", 2))}
    assert validation_gate(summary, seeds)["passed_all_replicates"]
    summary["normal"]["heads"]["balanced_3_plain"]["hits"] = 16
    gate = validation_gate(summary, seeds)
    assert not gate["passed_all_replicates"] and gate["primary_seed"] == 1
    assert not gate["per_seed"]["3"]["normal_pass"]
