"""Download the upstream-linked ModelScope mirror with per-file SHA256 checks.

No custom model code is downloaded. The weight hash is independently checked
against the pinned Hugging Face checkpoint's published SHA256.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

REPOSITORY = "AI-ModelScope/ShowUI-2B"
HF_REVISION = "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"
WEIGHT_SHA256 = "68080df785764e98976eb9cc93a07c6c69cf8a6933738496e02aef55b53d2aa3"
FILENAMES = {"README.md", "config.json", "generation_config.json", "preprocessor_config.json",
             "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
             "added_tokens.json", "special_tokens_map.json", "chat_template.json", "pytorch_model.bin"}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/ShowUI-2B")
    args = parser.parse_args()
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "download_manifest.json"
    if metadata_path.exists():
        manifest = json.loads(metadata_path.read_text())
    else:
        url = f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo/files?Revision=master&Recursive=true"
        with urllib.request.urlopen(url, timeout=30) as stream:
            files = json.load(stream)["Data"]["Files"]
        files = [f for f in files if f.get("Path") in FILENAMES]
        if {f["Path"] for f in files} != FILENAMES:
            raise RuntimeError("Incomplete model file list")
        manifest = {"source": "ModelScope", "repository": REPOSITORY,
                    "huggingface_weight_revision": HF_REVISION, "files": files}
        metadata_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if {f["Path"] for f in manifest["files"]} != FILENAMES:
        raise RuntimeError("Incomplete saved model manifest")
    for file in manifest["files"]:
        name = file["Path"]
        if name not in FILENAMES:
            raise ValueError("Unexpected model file")
        if name == "pytorch_model.bin" and file["Sha256"] != WEIGHT_SHA256:
            raise RuntimeError("Mirror weight differs from the pinned upstream checkpoint")
        path = directory / name
        if path.exists() and path.stat().st_size == file["Size"] and digest(path) == file["Sha256"]:
            print(f"Verified cached {name}", flush=True)
            continue
        params = urllib.parse.urlencode({"Revision": file["Revision"], "FilePath": name})
        url = f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo?{params}"
        temporary = path.with_name(path.name + ".part")
        if temporary.exists() and temporary.stat().st_size == file["Size"] and digest(temporary) == file["Sha256"]:
            temporary.replace(path)
            print(f"Verified completed partial {name}", flush=True)
            continue
        start = temporary.stat().st_size if temporary.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(request, timeout=90) as stream:
            if start and stream.status != 206:
                start = 0
            total = start
            milestone = start // (128 * 1024 * 1024)
            with temporary.open("ab" if start else "wb") as output:
                while block := stream.read(8 * 1024 * 1024):
                    output.write(block)
                    total += len(block)
                    if total // (128 * 1024 * 1024) > milestone:
                        milestone = total // (128 * 1024 * 1024)
                        print(f"{name}: {total / file['Size']:.0%}", flush=True)
        if temporary.stat().st_size != file["Size"] or digest(temporary) != file["Sha256"]:
            raise RuntimeError(f"Checksum/size mismatch: {name}; inspect {temporary}")
        temporary.replace(path)
        print(f"Verified {name}", flush=True)
    print(f"Model ready: {directory}", flush=True)


if __name__ == "__main__":
    main()
