"""Repair a pre-fit development collision without changing the held-out test."""
import argparse
import json
from pathlib import Path

from gui_recover.cli import fingerprint
from gui_recover.verification_regroup import regroup_development_data

parser = argparse.ArgumentParser()
for name in ("source", "out", "original-config", "config", "collection-commit"):
    parser.add_argument("--"+name, required=True)
args = parser.parse_args()
runtime = fingerprint()
if runtime["git_dirty"] is not False:
    raise ValueError("Freeze clean regrouping implementation")
manifest = regroup_development_data(args.source, args.out,
    json.loads(Path(args.original_config).read_text()), json.loads(Path(args.config).read_text()),
    args.collection_commit, runtime)
print(json.dumps({"audit": manifest["audit"], "moves": manifest["regroup"]["moves"]}, indent=2))
