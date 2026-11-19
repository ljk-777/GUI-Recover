"""Assemble corrected development labels and an unexposed test collection."""
import argparse
import json
from pathlib import Path

from gui_recover.cli import fingerprint
from gui_recover.verification_revision import assemble_revised_data

parser = argparse.ArgumentParser()
for name in ("development", "fresh-test", "out", "config"):
    parser.add_argument("--"+name, required=True)
args = parser.parse_args()
runtime = fingerprint()
if runtime["git_dirty"] is not False:
    raise ValueError("Freeze clean revision implementation")
m = assemble_revised_data(args.development, args.fresh_test, args.out,
                          json.loads(Path(args.config).read_text()), runtime)
print(json.dumps({"audit": m["audit"], "corrected_records": len(m["label_revision"]["changed_development_records"])}, indent=2))
