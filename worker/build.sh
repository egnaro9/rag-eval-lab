#!/usr/bin/env bash
# Rebuild the wheel and re-lock against it. BOTH steps, always: see the note in
# worker/pyproject.toml for what happens if you do only the first.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pick a python that can ACTUALLY build, not merely one that exists. The repo
# venv may be present without the `build` module, and `python3 -m build` then
# fails with "'build' is a package and cannot be directly executed", which reads
# like a broken package rather than a missing dependency.
PY=""
for cand in ./.venv/bin/python python3; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
        if "$cand" -c "import build.__main__" 2>/dev/null; then PY="$cand"; break; fi
    fi
done
if [ -z "$PY" ]; then
    echo "No python here has the 'build' module. Install it:" >&2
    echo "    ./.venv/bin/pip install build      # or: python3 -m pip install build" >&2
    exit 1
fi

"$PY" -m build --wheel --outdir worker/dist
cd worker
rm -f uv.lock pylock.toml
uv sync
echo
shasum -a 256 dist/*.whl
