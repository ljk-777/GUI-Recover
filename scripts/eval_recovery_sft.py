"""Six-way prompt/training ablation plus bounded real GUI correction replay."""

import argparse
import json
from pathlib import Path

from PIL import Image

from gui_recover.cli import fingerprint
from gui_recover.grounding import point_hits_boxes
from gui_recover.runner import write_json
from gui_recover.sft import audit_dataset, file_sha256, grounding_query, load_records
from gui_recover.sft_ablation import AblationGrounder, DEFAULT_HEADS, HEADS
from gui_recover.sft_execution import execute_case
from gui_recover.sft_recovery import audit_corrections


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--visual-adapter",required=True)
    parser.add_argument("--history-adapter",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    data,out=Path(args.data),Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    audit=audit_dataset(data)
    rows=load_records(data,"test")
    corrections=audit_corrections(data,rows)
    source=json.loads((data/"manifest.json").read_text())
    runtime=fingerprint()
    if runtime["git_dirty"] is not False: raise ValueError("Freeze clean evaluation code")
    model=AblationGrounder(args.model,data,args.visual_adapter,args.history_adapter)
    write_json(out/"manifest.json",{"runtime":runtime,"data_audit":audit,"corrections":corrections,
        "dataset_manifest_sha256":file_sha256(data/"manifest.json"),"training_code":model.training[0]["runtime"]["git_commit"],
        "design":"same BF16 base and budgets; six weight/context heads; reused identical normal prompts",
        "heads":HEADS,"online_defaults":DEFAULT_HEADS,"source_model":source["model_config"],
        "limitations":"oracle-selected real failures; conditional repair, at most one same-seed restart, not autonomous detection"})
    predictions=[]
    index={}
    for row in rows:
        picture=Image.open(data/row["image"]).convert("RGB")
        result={k:row[k] for k in ("id","task","seed","kind","image_sha256")}
        result["recovery_mode"]=row.get("recovery_mode")
        result["previous_action"]=row.get("previous_action")
        result["target_boxes"]=row["target_boxes"]
        result["copy_previous_point_hit"]=point_hits_boxes((row.get("previous_action") or {}).get("position"),row["target_boxes"])
        result["heads"]={}
        cache={}
        for head,(weights,history) in HEADS.items():
            previous=row.get("previous_action") if history else None
            key=(weights,grounding_query(row["query"],previous))
            if key not in cache:
                cache[key]=model.predict(picture,row["query"],row.get("previous_action"),head)
            result["heads"][head]={**cache[key],"hit":point_hits_boxes(cache[key]["point"],row["target_boxes"])}
        predictions.append(result)
        index[row["id"]]=result
        with (out/"predictions.jsonl").open("a") as f: f.write(json.dumps(result)+"\n")
        print("grounded",row["id"],flush=True)
    def summary(items):
        return {"queries":len(items),"heads":{head:{"hits":sum(r["heads"][head]["hit"] for r in items),
            "invalid":sum(r["heads"][head]["point"] is None for r in items),
            "newly_hit_vs_base_plain":sum(not r["heads"]["base_plain"]["hit"] and r["heads"][head]["hit"] for r in items),
            "degraded_vs_base_plain":sum(r["heads"]["base_plain"]["hit"] and not r["heads"][head]["hit"] for r in items)} for head in HEADS},
            "paired_effects":[{"from":a,"to":b,
                "newly_hit":sum(not r["heads"][a]["hit"] and r["heads"][b]["hit"] for r in items),
                "degraded":sum(r["heads"][a]["hit"] and not r["heads"][b]["hit"] for r in items)}
                for a,b in (("base_plain","base_history"),("base_plain","visual_plain"),
                            ("visual_history","history_history"),("history_plain","history_history"))],
            "copy_previous_point_hits":sum(r["copy_previous_point_hit"] for r in items)}
    offline={"all":summary(predictions),"normal":summary([r for r in predictions if r["kind"]!="correction"]),
        "correction":summary([r for r in predictions if r["kind"]=="correction"]),
        "by_task":{task:summary([r for r in predictions if r["task"]==task]) for task in sorted({r["task"] for r in predictions})}}
    write_json(out/"offline-summary.json",offline)
    selected=set(source["online_case_ids"])
    normal_seeds=set(source["config"]["online_normal_seeds"])
    online=[]
    for row in rows:
        conditional=row["id"] in selected
        normal=row["kind"]=="normal_entry" and row["seed"] in normal_seeds
        if not conditional and not normal: continue
        for head in DEFAULT_HEADS:
            result=execute_case(model,head,row,index[row["id"]]["heads"][head],data,out,conditional)
            online.append(result)
            write_json(out/"online-results.json",online)
            print("executed",row["id"],head,result["reason"],flush=True)
    def online_summary(items):
        return {head:{"episodes":sum(r["head"]==head for r in items),
            "successes":sum(r["head"]==head and r["success"] for r in items),
            "prefix_mismatches":sum(r["head"]==head and not r["prefix_matched"] for r in items),
            "resets":sum(r["resets_after_failure"] for r in items if r["head"]==head)} for head in DEFAULT_HEADS}
    totals={"offline":offline,"online_normal":online_summary([r for r in online if not r["conditional"]]),
        "online_conditional":online_summary([r for r in online if r["conditional"]]),
        "actual_generation_calls_including_submit":model.calls,
        "peak_cuda_allocated_bytes":model.torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes":model.torch.cuda.max_memory_reserved()}
    write_json(out/"summary.json",totals)
    print(json.dumps(totals,indent=2))
    if any(r["reason"]=="runtime_error" for r in online): raise SystemExit(1)


if __name__=="__main__": main()
