#!/usr/bin/env bash
# Linux/WSL reproducibility wrapper for reproduce_all.py.
# Usage: bash reproduce_all.sh [--check-only]

set -Eeuo pipefail
trap 'printf "ERROR: line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR

[[ $# -le 1 && ${1:-} =~ ^(--check-only)?$ ]] || {
    echo "Usage: bash reproduce_all.sh [--check-only]" >&2
    exit 2
}

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$ROOT"

PYTHON="$ROOT/.venv/bin/python"

[[ -x "$PYTHON" ]] || {
    echo "Missing virtual environment: $ROOT/.venv" >&2
    exit 1
}

[[ -f pipeline.py && -f reproduce_all.py && -f requirements.txt ]] || {
    echo "Missing required repository files." >&2
    exit 1
}

[[ -x /usr/bin/time ]] || {
    echo "GNU /usr/bin/time is required for runtime measurement." >&2
    exit 1
}

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE


echo "Preflight: checking environment and source-only state..."

"$PYTHON" - <<'PY'
from pathlib import Path
import sys

root = Path.cwd()

if sys.prefix == sys.base_prefix:
    raise SystemExit("Python is not running inside a virtual environment.")

if Path(sys.prefix).resolve() != (root / ".venv").resolve():
    raise SystemExit("Python must use this repository's .venv.")

for target in ("data", "models", "results"):
    output = root / target
    if output.exists() or output.is_symlink():
        raise SystemExit(
            f"Refusing to overwrite {output}; "
            "a from-scratch run requires it to be absent."
        )

import pipeline as P

if Path(P.__file__).resolve() != root / "pipeline.py":
    raise SystemExit("Imported the wrong pipeline.py.")

if str(P.DEVICE) != "cpu":
    raise SystemExit("Expected the reproducibility pipeline to use CPU.")

print(f"Python: {sys.version.split()[0]}")
print(f"Executable: {sys.executable}")
print(f"Pipeline: {P.__file__}")
print(f"Device: {P.DEVICE}")
PY

"$PYTHON" -m pip check
/usr/bin/time --version >/dev/null

if [[ ${1:-} == "--check-only" ]]; then
    echo "Preflight passed. No production computation was executed."
    exit 0
fi


AUDIT=$(mktemp -d "$ROOT/reproduction-report-XXXXXXXX")
REPORT="$AUDIT/runtime-environment.txt"

{
    printf 'Repository: %s\n' "$ROOT"
    printf 'UTC start: %s\n' "$(date -u +%FT%TZ)"
    printf '%s\n' \
        'Timing scope: reproduce_all.py from process start through final result writes.'
    printf '%s\n' \
        'Environment checks and report collection are excluded from production runtime.'

    echo
    echo "=== Operating system ==="
    uname -a
    if [[ -r /etc/os-release ]]; then
        cat /etc/os-release
    fi

    echo
    echo "=== CPU ==="
    if command -v lscpu >/dev/null 2>&1; then
        lscpu | sed -n \
            '/^Architecture:/p;
             /^CPU(s):/p;
             /^On-line CPU(s) list:/p;
             /^Model name:/p;
             /^Thread(s) per core:/p;
             /^Core(s) per socket:/p;
             /^Socket(s):/p'
    fi

    echo
    echo "=== Memory ==="
    if [[ -r /proc/meminfo ]]; then
        sed -n \
            '/^MemTotal:/p;
             /^SwapTotal:/p' \
            /proc/meminfo
    fi

    echo
    echo "=== GPU ==="
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=name,memory.total,driver_version \
            --format=csv,noheader
        echo "Pipeline computation device: CPU"
    else
        echo "nvidia-smi unavailable"
        echo "Pipeline computation device: CPU"
    fi

    echo
    echo "=== Git ==="
    if command -v git >/dev/null 2>&1; then
        printf 'Commit: '
        git rev-parse HEAD
        echo "Working tree:"
        git status --short
    else
        echo "git unavailable"
    fi

    echo
    echo "=== Source hashes ==="
    sha256sum \
        pipeline.py \
        reproduce_all.py \
        reproduce_all.sh \
        requirements.txt

    echo
    echo "=== Python environment ==="
    "$PYTHON" - <<'PY'
import importlib.metadata
import platform
import sys

import pipeline as P

print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("Platform:", platform.platform())

for package in (
    "numpy",
    "torch",
    "scikit-learn",
    "scipy",
    "threadpoolctl",
):
    print(f"{package}: {importlib.metadata.version(package)}")

for name in (
    "DEVICE",
    "SEED",
    "STAGE3_SEED",
    "TORCH_THREADS",
    "NATIVE_THREAD_LIMIT",
):
    print(f"{name}: {getattr(P, name)}")
PY

} > "$REPORT"

echo "Report directory: $AUDIT"
echo
echo "Starting complete from-scratch production run..."


set +e

/usr/bin/time \
    -f 'Wall seconds: %e
User CPU seconds: %U
System CPU seconds: %S
Peak RSS KiB: %M
Process exit status: %x' \
    -o "$AUDIT/timing.txt" \
    "$PYTHON" -u reproduce_all.py \
    2>&1 | tee "$AUDIT/production.log"

statuses=("${PIPESTATUS[@]}")
set -e

status=${statuses[0]}

if [[ $status -eq 0 && ${statuses[1]} -ne 0 ]]; then
    status=${statuses[1]}
fi


{
    echo
    echo "=== Runtime ==="
    cat "$AUDIT/timing.txt"

    printf 'UTC finish: %s\n' "$(date -u +%FT%TZ)"
    printf 'Overall exit status: %s\n' "$status"

    if [[ $status -eq 0 ]]; then
        echo "Outcome: COMPLETE"
    else
        echo "Outcome: FAILED; partial output retained for inspection."
    fi
} >> "$REPORT"


echo
cat "$REPORT"

exit "$status"
