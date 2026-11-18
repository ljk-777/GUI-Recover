#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
target="$repo_root/vendor/miniwob-plusplus"
revision=7fd85d71a4b60325c6585396ec4f48377d049838
if [[ -e "$target" ]]; then
  actual="$(git -C "$target" rev-parse HEAD)"
  if [[ "$actual" != "$revision" ]] || [[ -n "$(git -C "$target" status --porcelain)" ]]; then
    echo "Existing MiniWoB checkout differs or has edits; leaving it untouched." >&2
    exit 1
  fi
else
  mkdir -p "$repo_root/vendor"
  git clone --no-checkout --depth 1 https://github.com/Farama-Foundation/miniwob-plusplus.git "$target"
  git -C "$target" fetch --depth 1 origin "$revision"
  git -C "$target" checkout --detach "$revision"
fi
echo "MiniWoB ready at $target"
echo 'Set MINIWOB_URL using the command in README.md.'
