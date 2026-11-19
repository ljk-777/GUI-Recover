"""Build a tiny diagnostic set; privileged boxes are evaluator-only labels."""

import argparse
import hashlib
from pathlib import Path
import re

from gui_recover.cli import fingerprint
from gui_recover.environment import MiniwobEnvironment
from gui_recover.grounding import plan_from_goal
from gui_recover.runner import write_json


def visible_boxes(env, goal, step_index):
    page = env.env.unwrapped.page
    if env.task == "click-test":
        elements = [page.locator("#subbtn")]
    elif env.task == "enter-text":
        elements = [page.locator("#tt" if step_index == 0 else "#subbtn")]
    else:
        target = re.fullmatch(r'Click on the "([^"\n]+)" button\.', goal)[1]
        elements = [element for element in page.locator("button").all()
                    if element.inner_text().strip() == target]
    if not elements:
        raise RuntimeError("Missing target elements")
    width, height = env.obs.screenshot.size
    scale = getattr(page, "_bgym_scale_factor", 1.0)
    boxes = []
    for element in elements:
        box = element.bounding_box()
        if box is None:
            continue
        normalized = [max(0, box["x"] * scale / width), max(0, box["y"] * scale / height),
                      min(1, (box["x"] + box["width"]) * scale / width),
                      min(1, (box["y"] + box["height"]) * scale / height)]
        if normalized[0] < normalized[2] and normalized[1] < normalized[3]:
            boxes.append(normalized)
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20, 30)))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be unique and nonnegative")
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)
    inputs, labels = [], {}
    tasks = ["click-test", "click-button", "enter-text"]
    write_json(directory / "manifest.json", {"runtime": fingerprint(), "tasks": tasks, "seeds": args.seeds,
               "label_source": "privileged DOM bounding boxes; never model inputs",
               "secondary_phase": "submit localization on initial screenshot; not a next-action score"})
    for task in tasks:
        for seed in args.seeds:
            env = MiniwobEnvironment(task)
            try:
                obs, _ = env.reset(seed)
                path = directory / f"{task}-{seed}.png"
                obs.screenshot.save(path)
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                for phase, step in enumerate(plan_from_goal(obs.goal)):
                    identifier = f"{task}-{seed}-{phase}"
                    inputs.append({"id": identifier, "task": task, "seed": seed, "phase": phase,
                                   "goal": obs.goal, "query": step.query,
                                   "screenshot": path.name, "sha256": sha})
                    labels[identifier] = {"boxes": visible_boxes(env, obs.goal, phase)}
            finally:
                env.close()
            write_json(directory / "inputs.json", inputs)
            write_json(directory / "evaluator-labels.json", labels)
    print(f"Captured {len(inputs)} target cases from {len(tasks) * len(args.seeds)} screenshots")


if __name__ == "__main__":
    main()
