#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Create the validated inference .venv first.' >&2
  exit 1
fi
if [[ ! -x .venv-sft/bin/python ]]; then
  .venv/bin/python -m venv .venv-sft
fi
.venv-sft/bin/python - <<'PY'
from pathlib import Path
import sysconfig
root = Path.cwd()
site = Path(sysconfig.get_paths()['purelib'])
(site / 'gui_recover_sft_base.pth').write_text(
    str(root / '.venv/lib/python3.11/site-packages') + '\n' + str(root / 'src') + '\n')
PY
# Only add the four pinned packages in the overlay. Never upgrade base torch,
# transformers, browsergym or their CUDA dependencies during this smoke test.
.venv-sft/bin/python -m pip install --no-deps --timeout 25 --retries 1 \
  --index-url "${GUI_RECOVER_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  peft==0.14.0 accelerate==1.4.0 bitsandbytes==0.45.3 psutil==6.1.1
.venv-sft/bin/python -m pip check
