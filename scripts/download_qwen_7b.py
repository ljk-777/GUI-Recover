"""Stage a pinned Qwen 7B snapshot on the project server."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
URL = f"https://modelscope.cn/api/v1/models/{MODEL}/repo"
SHA256 = {
    "model-00001-of-00005.safetensors": "e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e",
    "model-00002-of-00005.safetensors": "a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024",
    "model-00003-of-00005.safetensors": "111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507",
    "model-00004-of-00005.safetensors": "ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed",
    "model-00005-of-00005.safetensors": "0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67",
}
WEIGHT_REVISION = "c03ef5ea2cb88c1aaf8c1231f58dcd8b4a80374f"
FILENAMES = set(SHA256) | {"config.json", "generation_config.json", "preprocessor_config.json",
                           "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
                           "chat_template.json", "model.safetensors.index.json"}


def verify(path, receipt):
    if not path.is_file() or path.stat().st_size != receipt["Size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == receipt["Sha256"]


def download(root, receipt):
    name = receipt["Path"]
    path = root / name
    partial = path.with_name(path.name + ".part")
    if verify(path, receipt):
        print("verified cached", name, flush=True)
        return
    params = urlencode({"Revision": receipt["Revision"], "FilePath": name})
    for attempt in range(1, 5):
        if verify(partial, receipt):
            partial.replace(path)
            print("verified", name, flush=True)
            return
        start = partial.stat().st_size if partial.exists() else 0
        if start >= receipt["Size"]:
            raise RuntimeError(f"Complete file does not match checksum: {name}")
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with urlopen(Request(f"{URL}?{params}", headers=headers), timeout=45) as response:
                with partial.open("ab" if start and response.status == 206 else "wb") as output:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
        except (HTTPError, URLError, TimeoutError) as error:
            print("retrying", name, attempt, type(error).__name__, flush=True)
            sleep(2 * attempt)
    if not verify(partial, receipt):
        raise RuntimeError(f"Unable to verify {name}")
    partial.replace(path)
    print("verified", name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models/Qwen2.5-VL-7B-Instruct"))
    root = parser.parse_args().out
    with urlopen(f"{URL}/files?Revision=master&Recursive=true", timeout=20) as response:
        entries = json.load(response)["Data"]["Files"]
    files = [entry for entry in entries if entry.get("Path") in FILENAMES]
    if {file["Path"] for file in files} != FILENAMES:
        raise RuntimeError("Model mirror is missing a required asset")
    for file in files:
        name = file["Path"]
        if name in SHA256 and (file["Sha256"] != SHA256[name] or file["Revision"] != WEIGHT_REVISION):
            raise RuntimeError(f"Model mirror differs from pinned weights: {name}")
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda receipt: download(root, receipt), files))
    receipt = {"repository": MODEL, "source": "ModelScope", "weight_revision": WEIGHT_REVISION, "files": files}
    staged = root / "download_manifest.json.part"
    staged.write_text(json.dumps(receipt, indent=2) + "\n")
    staged.replace(root / "download_manifest.json")


if __name__ == "__main__":
    main()
