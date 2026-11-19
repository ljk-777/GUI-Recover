"""Run an explicit JSON experiment matrix with one shared frozen VLM instance."""

import argparse
import json
from pathlib import Path
import re

from gui_recover.cli import fingerprint, main as evaluate
from gui_recover.policies import MODEL_REVISION, ShowUIPolicy
from gui_recover.runner import write_json


def flags(settings):
    result = []
    for key, value in settings.items():
        if not re.fullmatch(r"[a-z][a-z-]*", key) or key in ("out", "model"):
            raise ValueError(f"Invalid matrix setting: {key}")
        values = value if isinstance(value, list) else [value]
        if not values or any(type(v) not in (str, int) for v in values):
            raise ValueError(f"Invalid value for {key}")
        result.extend(["--" + key, *map(str, values)])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    names = [arm["name"] for arm in config["arms"]]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[a-z0-9-]+", name) for name in names):
        raise ValueError("Matrix arm names must be unique and safe directory names")
    for arm in config["arms"]:
        flags({**config["common"], **arm["settings"]})
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "matrix.json", {"config": config, "runtime": fingerprint(),
        "cache_policy": "reuse frozen weights; clear unused CUDA cache before each arm; reset policy/seed per episode"})
    backend = ShowUIPolicy(model=args.model, revision=config["common"].get("revision", MODEL_REVISION),
                           prompt_profile=config["common"].get("prompt-profile", "baseline"))
    summaries = {}
    for arm in config["arms"]:
        settings = {**config["common"], **arm["settings"]}
        backend.torch.cuda.empty_cache()
        output = directory / arm["name"]
        evaluate(["run", *flags(settings), "--model", args.model, "--out", str(output)], backend=backend)
        summaries[arm["name"]] = json.loads((output / "summary.json").read_text())
        write_json(directory / "summary.json", summaries)
        print(f"Completed matrix arm: {arm['name']}", flush=True)


if __name__ == "__main__":
    main()
