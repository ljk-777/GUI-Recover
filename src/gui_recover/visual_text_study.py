"""Run paired screenshot-based text recovery in the supported MiniWoB task."""

from collections import Counter
import json
import subprocess
import sys

from .contextual_repair import RepairGrounder
from .policies import ShowUIPolicy
from .recovery_env import Decision, Phase, RecoveryDecisionEnv, SUPPORTED_REGIMES, transition_view
from .recovery_rollouts import baseline_decision, record_rollout
from .repair_study_data import preflight
from .runner import write_json
from .sft import file_sha256
from .verification import appearance_verdict
from .visual_text import ContentRecoveryPolicy, VisualTextVerifier


POLICIES = ("presence_wait", "glyph_check")
VISIBLE_ERRORS = ("partial-input", "wrong-input", "case-input")


def presence_wait(observation):
    if (observation["phase"] == Phase.REVIEW and observation["action_mask"][Decision.WAIT]
            and appearance_verdict(transition_view(observation))["status"] != "achieved"):
        return Decision.WAIT
    return baseline_decision(observation,"visual_fixed")


def summarize(rows):
    scores = {}
    for condition in SUPPORTED_REGIMES:
        paired = {}
        for r in rows:
            if r["regime"] == condition:
                group = paired.setdefault(r["seed"],{})
                if r["policy"] in group:
                    raise ValueError("Duplicate policy branch")
                group[r["policy"]] = r
        if not paired or any(set(p)!=set(POLICIES) for p in paired.values()):
            raise ValueError("Missing paired branch")
        totals = {}
        for policy in POLICIES:
            branch = [p[policy] for p in paired.values()]
            totals[policy] = {"episodes":len(branch),"successes":sum(r["success"] for r in branch),
                **{k:sum(r[k] for r in branch) for k in ("waits","repairs","actor_requests")},
                "outcomes":dict(Counter(r["outcome"] for r in branch))}
        totals["paired_glyph_vs_presence"] = {
            "gains":sum(p["glyph_check"]["success"] and not p["presence_wait"]["success"] for p in paired.values()),
            "losses":sum(not p["glyph_check"]["success"] and p["presence_wait"]["success"] for p in paired.values())}
        totals["glyph_statuses"] = dict(Counter(s for p in paired.values() for s in p["glyph_check"]["visual_statuses"]))
        scores[condition] = totals
    preserved = ("none","drop-input","deferred-input")
    gate = {"preserved_normal_and_missing":all(scores[r]["paired_glyph_vs_presence"]["losses"]==0 for r in preserved),
            "no_normal_repair":scores["none"]["glyph_check"]["repairs"]==0,
            "gained_visible_errors":all(scores[r]["paired_glyph_vs_presence"]["gains"]>0 for r in VISIBLE_ERRORS)}
    return {"conditions":scores,"gate":gate}


def prepare_references(out):
    """Build the reference bank and seal it before the first policy action."""
    bank_path=out/"text-references.json"
    # Rendering runs in an isolated browser process with visible goal text
    # as its only input. It never opens the task or consumes rollout outcomes.
    subprocess.run([sys.executable,"-m","gui_recover.visual_text","--plan",str(out/"preflight.json"),
                    "--out",str(bank_path)],check=True)
    bank_sha=file_sha256(bank_path)
    write_json(out/"before-rollouts.json",{"plan_sha256":file_sha256(out/"preflight.json"),
        "bank_sha256":bank_sha,"actor_calls":0,"outcomes_observed":False})
    return bank_sha


def record_text_study(out, plan, profile, actor, bank_sha, *, split="test"):
    """Run and record both policies on every registered input fault."""
    bank_path = out / "text-references.json"
    if file_sha256(bank_path) != bank_sha:
        raise ValueError("Reference templates changed before evaluation")
    bank = json.loads(bank_path.read_text())
    verifier = VisualTextVerifier(bank)
    grounder=RepairGrounder(actor,out/"inference")
    rows=[]
    with (out/"episodes.jsonl").open("x") as index:
        for source in plan["selected"]:
            scene={**source,"split":split}
            for condition in SUPPORTED_REGIMES:
                for policy in POLICIES:
                    content=ContentRecoveryPolicy(verifier) if policy=="glyph_check" else None
                    name=f"{scene['task']}-{scene['seed']}-{condition}-{policy}"
                    branch=out/"rollouts"/name
                    env=RecoveryDecisionEnv(scene["task"],grounder,regime=condition,input_repair="replace")
                    try:
                        result=record_rollout(env,scene,policy,branch,choose=content or presence_wait)
                        if env.execution.env.metadata()["browser_version"]!=bank["browser_version"]:
                            raise ValueError("Task and reference Chromium builds differ")
                    finally:
                        env.close()
                    diagnostics=content.evidence if content else []
                    write_json(branch/"visual-verification.json",diagnostics)
                    trace=branch/"transitions.jsonl"
                    result.update(trace=str(trace.relative_to(out)),
                        visual_verification_sha256=file_sha256(branch/"visual-verification.json"),
                        visual_statuses=[d["status"] for d in diagnostics])
                    index.write(json.dumps(result)+"\n")
                    index.flush()
                    rows.append(result)
                    print(name,result["outcome"],flush=True)
    if file_sha256(bank_path)!=bank_sha:
        raise ValueError("Reference templates changed during evaluation")
    write_json(out/"summary.json",{**summarize(rows),"profile":profile,"episodes":len(rows),
        "index_sha256":file_sha256(out/"episodes.jsonl"),"plan_sha256":file_sha256(out/"preflight.json"),
        "bank_sha256":bank_sha,"reference_texts":len(bank["templates"]),
        "inference":grounder.counters(),"peak_cuda_reserved_bytes":actor.torch.cuda.max_memory_reserved()})
    print("completed",out.name,flush=True)


def run_text_quickstart(out, seeds, model, revision, runtime):
    """Run a stand-alone paired demo using only the installed environment and model."""
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", {
        "profile": "quickstart",
        "config": {"quickstart": {"layouts": len(seeds), "seeds": seeds}},
        "model": {"identifier": model, "revision": revision},
        "runtime": runtime,
    })
    plan = preflight((), out, [{"task": "enter-text", "count": len(seeds), "candidate_seeds": seeds}], include_goal=True)
    bank_sha = prepare_references(out)
    actor = ShowUIPolicy(model, revision=revision, max_new_tokens=32)
    record_text_study(out, plan, "quickstart", actor, bank_sha, split="demo")
