"""Run the portable label matcher with paired decisions and real browser repairs."""

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageFont, features

from gui_recover.actions import Action
from gui_recover.cli import fingerprint
from gui_recover.multistep_study import first_click, register_multistep
from gui_recover.portable_labels import PortableCheckboxStateVerifier
from gui_recover.runner import write_json
from gui_recover.sft import file_sha256
from gui_recover.target_state import CheckboxStateVerifier
from gui_recover.verification import TransitionView, pixel_hash


_TASK = "click-checkboxes"
_CONFUSIONS = ("0oOQ", "1Ilij", "5Ss", "2Zz", "6Gg", "8B", "ccae", "rnmn", "FE", "kh")


class ArchivedGrounder:
    """Replay an earlier agent decision only if both visual input and query match."""
    def __init__(self, prediction):
        self.prediction = prediction

    def ground(self, screenshot, target):
        p = self.prediction
        if p["query"] != target or p["input_pixel_hash"] != pixel_hash(screenshot):
            raise ValueError("Archived actor input does not match")
        return p


def absent_label(label, visible):
    for i, char in enumerate(label):
        for group in _CONFUSIONS:
            if char in group:
                for alternative in group:
                    candidate = label[:i]+alternative+label[i+1:]
                    if candidate not in visible:
                        return candidate
    return None


def visual_stress(picture, actual_target, visible, models):
    candidate = absent_label(actual_target, visible)
    if candidate is None:
        return None
    goal = f"Select {candidate} and click Submit."
    target = f'The checkbox labeled "{candidate}"'
    unchanged = TransitionView(picture, picture, Action("CLICK", position=(.9,.9)), goal, target)
    return {"matched_real_label": actual_target, "absent_query": candidate,
            "goal": goal, "current_target": target,
            "decisions": {name: verify(unchanged) for name, verify in models.items()}}


def aggregate(rows, stress):
    role_summary = {}
    for role in sorted({r["role"] for r in rows}):
        group = [r for r in rows if r["role"] == role]
        counts = {"traces": len(group), "initial_correct": sum(r["achieved"] for r in group),
                  "final_correct": sum(r["achieved_after_recovery"] for r in group),
                  "regressions": sum(r["achieved"] and not r["achieved_after_recovery"] for r in group),
                  "repair_clicks": sum(len(r["recovery"]["actions"]) for r in group),
                  "completed_with_oracle_continuation": sum(
                      r.get("completed_with_oracle_continuation", False) for r in group)}
        for name in ("legacy", "target_state"):
            d = [r["decisions"][name] for r in group]
            counts[name] = {"triggers": sum(p["trigger"] for p in d),
                "unknown_after_routing": sum(p["after_wait" if p["wait"] else "immediate"]["status"]=="unknown"
                                             for p in d)}
        role_summary[role] = counts
    return {"roles": role_summary, "absent_label_probes": {"count": len(stress),
        **{name: Counter(r["decisions"][name]["status"] for r in stress) for name in ("legacy", "target_state")}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "font", "previous-run", "earlier-plan", "out"):
        parser.add_argument("--"+option, type=Path, required=True)
    parser.add_argument("--profile", choices=("test", "smoke"), required=True)
    args = parser.parse_args()
    runtime = fingerprint()
    config = json.loads(args.config.read_text())
    if runtime["git_dirty"] is not False or runtime["miniwob_commit"] != config["miniwob_commit"]:
        raise ValueError("Use the recorded clean source and MiniWoB revision")
    if not features.check("raqm"):
        raise ValueError("The frozen Linux legacy baseline requires RAQM")
    previous_plan = args.previous_run/"plan.json"
    for path, key in ((args.font, "font_sha256"), (previous_plan, "previous_plan_sha256"),
                      (args.earlier_plan, "earlier_plan_sha256")):
        if file_sha256(path) != config[key]:
            raise ValueError(f"Protocol input changed: {key}")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"runtime": runtime, "config": config, "profile": args.profile,
        "config_sha256": file_sha256(args.config), "legacy_layout": "RAQM", "new_layout": "BASIC",
        "freetype": features.version("freetype2"), "raqm": features.version("raqm")})
    old = json.loads(previous_plan.read_text())
    earlier = json.loads(args.earlier_plan.read_text())
    profile = config[args.profile]
    candidates = range(profile["first_seed"], profile["first_seed"]+profile["reserve"])
    spec = {"multistep_tasks":[{"task": _TASK, "count": profile["count"], "candidate_seeds": list(candidates)}]}
    plan = register_multistep(spec, known_layouts=[r["initial_sha256"] for p in (old,earlier) for r in p["accepted"]])
    write_json(out/"plan.json", plan)
    print("registered", len(plan["accepted"]), "rejected", len(plan["rejected"]), flush=True)

    legacy = CheckboxStateVerifier(args.font)
    assert legacy.font.layout_engine == ImageFont.Layout.RAQM
    candidate = PortableCheckboxStateVerifier(args.font,
        **{name: config[name] for name in ("minimum_score", "minimum_margin", "minimum_agreement_margin")})
    verifiers = {"legacy": legacy.verify, "target_state": candidate.verify}
    (out/"fresh").mkdir()
    (out/"historical").mkdir()
    rows, stress = [], []
    for scene in plan["accepted"]:
        scene_id = f"{scene['task']}-{scene['seed']}"
        for role in ("correct", "blank", "wrong_control"):
            fixture = scene["fixture"]
            if role=="wrong_control" and fixture[role] is None:
                continue
            branch = out/"fresh"/f"{scene_id}-{role}"
            record = first_click(scene, role, None, verifiers, branch, recovery=candidate)
            rows.append(record)
            if role == "correct":
                picture = Image.open(branch/record["before"]["image"]).convert("RGB")
                probe = visual_stress(picture, fixture["targets"][0], fixture["visible_labels"], verifiers)
                if probe is not None:
                    stress.append({"seed":scene["seed"], **probe})
            print("fresh", scene_id, role, record["achieved_after_recovery"], flush=True)
    historic = []
    for scene in old["accepted"]:
        seed = scene["seed"]
        if scene["task"] != _TASK or seed not in config["known_natural_regressions"]:
            continue
        previous_record = args.previous_run/"episodes"/f"{_TASK}-{seed}-natural"/"record.json"
        original = json.loads(previous_record.read_text())
        branch = out/"historical"/f"{_TASK}-{seed}-natural"
        record = first_click(scene, "natural", ArchivedGrounder(original["source_prediction"]), verifiers,
                             branch, recovery=candidate)
        if record["immediate"]["sha256"] != original["immediate"]["sha256"]:
            raise ValueError("Prior natural-error screenshot did not replay")
        historic.append(record)
        print("historical", seed, record["achieved_after_recovery"], flush=True)
    write_json(out/"absent-labels.json", stress)
    write_json(out/"summary.json", {"profile":args.profile, "plan_sha256":file_sha256(out/"plan.json"),
        "fresh":aggregate(rows, stress), "historical_natural":aggregate(historic, []),
        "historical_seeds":[r["seed"] for r in historic], "new_model_generations":0,
        "near_absent_scope":"Offline malformed-goal challenge on unaltered screenshots; no actions executed"})
    print("completed", out.name, flush=True)


if __name__ == "__main__":
    main()
