"""Fixed recipe and seed replicates for normal-transition preservation."""

import json
from pathlib import Path

from .policies import MODEL_REVISION, sft_adapter_metadata
from .sft import file_sha256, grounding_query
from .sft_history import HistoryStudyGrounder, study_query


def balanced_training_records(rows, submit_repeats=4):
    result = []
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Mixture recipe accepts train only")
        contexts = (False, True) if row.get("previous_action") is not None else (False,)
        repeats = submit_repeats if row["kind"] == "normal_submit" else 1
        for repeat in range(repeats):
            for context in contexts:
                result.append({**row, "id": f"{row['id']}-context-{int(context)}-repeat-{repeat}",
                               "origin_id": row["id"], "history_input": context})
    return result


def balanced_heads(seeds):
    return ["base_plain", "history_plain", "history_legacy", "history_target_last"] + [
        f"balanced_{seed}_{context}" for seed in seeds for context in ("plain", "target_last")]


def balanced_query(query, previous, head):
    if not head.startswith("balanced_"):
        return study_query(query, previous, head)
    return grounding_query(query, previous if head.endswith("target_last") else None, history_profile="target-last")


def validation_gate(summary, seeds):
    floor = max(summary["normal"]["heads"][h]["hits"] for h in ("base_plain", "history_plain"))
    submit_floor = summary["normal_submit"]["heads"]["base_plain"]["hits"]
    correction_floor = max(summary["correction"]["heads"][h]["hits"] for h in ("history_legacy", "history_target_last"))
    per_seed = {}
    for seed in seeds:
        plain, history = (f"balanced_{seed}_{c}" for c in ("plain", "target_last"))
        per_seed[str(seed)] = {
            "normal_pass": all(summary["normal"]["heads"][h]["hits"] >= floor for h in (plain, history)),
            "submit_pass": all(summary["normal_submit"]["heads"][h]["hits"] >= submit_floor for h in (plain, history)),
            "correction_pass": summary["correction"]["heads"][history]["hits"] >= correction_floor,
        }
    return {"primary_seed": seeds[0], "per_seed": per_seed,
            "passed_all_replicates": all(all(values.values()) for values in per_seed.values()),
            "normal_floor": floor, "submit_floor": submit_floor, "correction_floor": correction_floor,
            "selection": "Primary seed fixed before training; no best-seed selection. All replicates retained even on gate failure."}


class BalancedStudyGrounder(HistoryStudyGrounder):
    def __init__(self, model_path, reference, visual, history, balanced_data, adapters, config):
        super().__init__(model_path, reference, visual, history)
        self.replicates = []
        seeds = config["balanced_training"]["seeds"]
        if len(adapters) != len(seeds):
            raise ValueError("Expected all predeclared seed replicates")
        for seed, path in zip(seeds, adapters):
            m = json.loads((Path(path) / "training.json").read_text())
            sft_adapter_metadata(path, MODEL_REVISION)
            expected_config = {**config, "seed": seed}
            if (m["config"] != expected_config or m["context_mode"] != "record-action"
                    or m["history_profile"] != "target-last" or m["completed_updates"] != config["training"]["steps"]
                    or m["dataset_manifest_sha256"] != file_sha256(Path(balanced_data) / "manifest.json")
                    or m["base_weights_sha256"] != self.training[0]["base_weights_sha256"]
                    or m["reload_probe_loss_abs_difference"] > 0.002):
                raise ValueError("Balanced training provenance differs")
            if m["runtime"]["git_dirty"] is not False:
                raise ValueError("Replicate was trained from dirty code")
            if file_sha256(Path(path) / "tokenizer.json") != file_sha256(Path(visual) / "tokenizer.json"):
                raise ValueError("Replicate tokenizer differs")
            self.model.load_adapter(path, adapter_name=f"balanced_{seed}", is_trainable=False)
            self.replicates.append(m)
        if (len({m["adapter_sha256"] for m in self.replicates}) != len(seeds)
                or len({m["runtime"]["git_commit"] for m in self.replicates}) != 1):
            raise ValueError("Expected distinct replicate weights from one frozen code revision")
        self.model.eval()

    def predict(self, picture, query, previous, head):
        if not head.startswith("balanced_"):
            return super().predict(picture, query, previous, head)
        _, seed, _ = head.split("_", 2)
        action = previous if head.endswith("target_last") else None
        return self.predict_weights(picture, query, action, f"balanced_{seed}", history_profile="target-last")
