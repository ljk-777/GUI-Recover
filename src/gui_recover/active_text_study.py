"""Collect paired active-tail verification artifacts for a study or an installed demo."""

from collections import Counter, defaultdict
import json
from statistics import median

from .active_text import ActiveTextPolicy
from .contextual_repair import RepairGrounder
from .policies import ShowUIPolicy
from .recovery_env import RecoveryDecisionEnv, SUPPORTED_REGIMES
from .recovery_rollouts import record_rollout
from .repair_study_data import preflight
from .runner import write_json
from .sft import file_sha256
from .text_goals import NATIVE_TEXT_TASKS
from .visual_text import ContentRecoveryPolicy, VisualTextVerifier
from .visual_text_study import prepare_references


TASK = "enter-text"
POLICIES = ("glyph_check", "fixed_zoom", "active_tail")


def summarize(rows, regimes):
    conditions = {}
    for regime in regimes:
        by_seed = defaultdict(dict)
        for row in rows:
            if row["regime"] == regime:
                pair = by_seed[row["task"], row["seed"]]
                if row["policy"] in pair:
                    raise ValueError("Duplicate paired rollout")
                pair[row["policy"]] = row
        if not by_seed or any(set(pair) != set(POLICIES) for pair in by_seed.values()):
            raise ValueError("Missing paired rollout")
        metrics = {}
        for policy in POLICIES:
            group = [pair[policy] for pair in by_seed.values()]
            metrics[policy] = {
                "episodes": len(group), "successes": sum(row["success"] for row in group),
                **{key: sum(row[key] for row in group) for key in
                   ("decisions", "waits", "repairs", "probes", "actor_requests", "browser_operations")},
            }
        for baseline in POLICIES[:2]:
            metrics[f"active_vs_{baseline}"] = {
                "gains": sum(pair["active_tail"]["success"] and not pair[baseline]["success"] for pair in by_seed.values()),
                "losses": sum(pair[baseline]["success"] and not pair["active_tail"]["success"] for pair in by_seed.values()),
            }
        conditions[regime] = metrics
    return conditions


def collect(out, plan, regimes, grounder, verifier, browser_version, *,
            policies=POLICIES, batched_probe=False, alternate_order=False):
    rows = []
    with (out / "episodes.jsonl").open("x", encoding="utf-8") as index:
        for layout_index, layout in enumerate(plan["selected"]):
            scene = {**layout, "split": "test"}
            task = scene["task"]
            for regime_index, regime in enumerate(regimes):
                ordered = policies[::-1] if alternate_order and (layout_index + regime_index) % 2 else policies
                for policy in ordered:
                    choose = None
                    if policy == "glyph_check":
                        choose = ContentRecoveryPolicy(verifier)
                    elif policy in ("active_tail", "batched_tail"):
                        choose = ActiveTextPolicy(verifier)
                    branch = out / "rollouts" / f"{task}-{layout['seed']}-{regime}-{policy}"
                    env = RecoveryDecisionEnv(task, grounder, regime=regime, input_repair="replace", enable_tail_probe=True,
                        batched_probe=policy == "batched_tail" or (policy == "active_tail" and batched_probe))
                    try:
                        row = record_rollout(env, scene, policy, branch, choose=choose)
                        if env.execution.env.metadata()["browser_version"] != browser_version:
                            raise ValueError("Task and reference browsers differ")
                        row["browser_operations"] = len(env.execution.events)
                    finally:
                        env.close()
                    evidence = choose.evidence if choose else []
                    evidence_path = branch / "visual-verification.json"
                    write_json(evidence_path, evidence)
                    row.update(trace=(branch / "transitions.jsonl").relative_to(out).as_posix(),
                               visual_verification_sha256=file_sha256(evidence_path),
                               selection_statuses=[item["selection"]["status"] for item in evidence if item.get("selection")])
                    rows.append(row)
                    index.write(json.dumps(row) + "\n")
                    index.flush()
                    print(branch.name, row["outcome"], flush=True)
    return rows


def evaluate_registered(out, plan, profile_name, regimes, actor, bank_sha, *, batched_probe=False):
    """Execute a sealed layout plan with a shared frozen grounding cache."""
    bank_path = out / "text-references.json"
    if file_sha256(bank_path) != bank_sha:
        raise ValueError("Frozen text references changed before evaluation")
    bank = json.loads(bank_path.read_text())
    verifier = VisualTextVerifier(bank)
    grounder = RepairGrounder(actor, out / "inference")
    rows = collect(out, plan, regimes, grounder, verifier, bank["browser_version"], batched_probe=batched_probe)
    if file_sha256(bank_path) != bank_sha:
        raise RuntimeError("Frozen text references changed")
    conditions = summarize(rows, regimes)
    probe_rows = [row for row in rows if row["probes"]]
    write_json(out / "summary.json", {
        "profile": profile_name, "layouts": len(plan["selected"]), "episodes": len(rows), "conditions": conditions,
        "plan_sha256": file_sha256(out / "preflight.json"), "bank_sha256": bank_sha,
        "index_sha256": file_sha256(out / "episodes.jsonl"),
        "selection_statuses": dict(Counter(status for row in rows for status in row["selection_statuses"])),
        "probe_time_seconds": {"total": sum(row["probe_seconds"] for row in probe_rows),
                               "median": median(row["probe_seconds"] for row in probe_rows) if probe_rows else None},
        "gate": {
            "preserved_glyph_successes": not any(m["active_vs_glyph_check"]["losses"] for m in conditions.values()),
            "no_clean_repairs": conditions["none"]["active_tail"]["repairs"] == 0,
            "added_trailing_space_success": conditions["trailing-space-input"]["active_vs_glyph_check"]["gains"] > 0,
        },
        "inference": grounder.counters(), "peak_cuda_reserved_bytes": actor.torch.cuda.max_memory_reserved(),
    })
    print("completed", out.name, flush=True)


def run_active_quickstart(out, seeds, model, revision, runtime, *, tasks=(TASK,)):
    """Run the paired active-tail demo without historical experiment directories."""
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(NATIVE_TEXT_TASKS):
        raise ValueError("Expected distinct supported native text tasks")
    out.mkdir(parents=True, exist_ok=False)
    regimes = tuple(regime for regime in SUPPORTED_REGIMES
                    if regime != "case-input" or "enter-text-dynamic" not in tasks)
    profile = {"layouts": len(seeds) * len(tasks), "seeds": seeds, "tasks": tasks, "regimes": regimes}
    write_json(out / "protocol.json", {"profile": "quickstart", "config": {"quickstart": profile},
                                      "model": {"identifier": model, "revision": revision}, "runtime": runtime,
                                      "probe_execution": "single_step"})
    write_json(out / "development-layouts.json", [])
    plan = preflight((), out, [{"task": task, "count": len(seeds), "candidate_seeds": seeds} for task in tasks],
                     include_goal=True)
    bank_sha = prepare_references(out)
    actor = ShowUIPolicy(model, revision=revision, max_new_tokens=32)
    evaluate_registered(out, plan, "quickstart", regimes, actor, bank_sha, batched_probe=True)
