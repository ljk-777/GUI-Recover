#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_root"
if [[ -n "${GUI_RECOVER_PYTHON:-}" ]]; then
  python_bin="$GUI_RECOVER_PYTHON"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  python_bin="$project_root/.venv/bin/python"
else
  python_bin=python3
fi
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$project_root/.cache/browsers}"
bash scripts/setup_miniwob.sh
"$python_bin" -m playwright install chromium
export MINIWOB_URL="file://$project_root/vendor/miniwob-plusplus/miniwob/html/miniwob/"
output="${1:-runs/cpu-demo-$(date -u +%Y%m%dT%H%M%SZ)}"
echo 'Running privileged diagnostic oracle; this is NOT a ShowUI model score.'
"$python_bin" -m gui_recover run --policy oracle-smoke --seeds 0 1 2 --max-steps 2 --out "$output"
echo "Inspect $output/summary.json and $output/click-test-0/replay.html"
