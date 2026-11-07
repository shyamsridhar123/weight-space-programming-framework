#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  printf 'Usage: bash scripts/check_proofs.sh .proof-runs/<new-run-name>\n' >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  printf 'Python 3.10 or later is required as python3; see PROOFS.md.\n' >&2
  exit 127
fi
script_directory="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$script_directory/check_proofs.py" --output "$1"
