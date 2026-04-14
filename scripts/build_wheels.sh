#!/usr/bin/env bash
# build_wheels.sh — Build Python wheels locally using maturin.
#
# Usage:
#   ./scripts/build_wheels.sh [--out DIR] [--python VERSION ...]
#
# Options:
#   --out DIR         Output directory for built wheels (default: dist/)
#   --python VERSION  One or more CPython versions to target (default: 3.10 3.11 3.12)
#
# Requirements:
#   - Rust stable toolchain (rustup)
#   - maturin >= 1.4 (pip install maturin)
#
# Exit codes:
#   0  success
#   1  invalid argument
#   2  build failure

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/rust/vectro_py/Cargo.toml"
OUT="$ROOT/dist"
PYTHONS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT="$2"; shift 2 ;;
        --python)
            PYTHONS+=("$2"); shift 2 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//' | head -20
            exit 0 ;;
        *)
            echo "error: unknown option '$1'" >&2
            exit 1 ;;
    esac
done

# Default Python versions if none supplied
[[ ${#PYTHONS[@]} -eq 0 ]] && PYTHONS=(3.10 3.11 3.12)

# ── Preconditions ─────────────────────────────────────────────────────────────
if ! command -v maturin &>/dev/null; then
    echo "error: maturin not found. Install with: pip install maturin" >&2
    exit 2
fi
if ! command -v cargo &>/dev/null; then
    echo "error: cargo not found. Install Rust: https://rustup.rs" >&2
    exit 2
fi

mkdir -p "$OUT"

# ── Build ─────────────────────────────────────────────────────────────────────
echo "Building vectro Python wheels → $OUT"
echo "Targets: ${PYTHONS[*]}"
echo ""

BUILT=0
SKIPPED=0

for pyver in "${PYTHONS[@]}"; do
    INTERP="python${pyver}"
    if ! command -v "$INTERP" &>/dev/null; then
        echo "  skipping py${pyver}: $INTERP not found"
        (( SKIPPED++ )) || true
        continue
    fi
    echo "  Building py${pyver}..."
    if maturin build \
        --release \
        --out "$OUT" \
        --manifest-path "$MANIFEST" \
        --interpreter "$INTERP" \
        --quiet; then
        echo "  ✓ py${pyver} ok"
        (( BUILT++ )) || true
    else
        echo "error: maturin build failed for py${pyver}" >&2
        exit 2
    fi
done

echo ""
echo "Results: $BUILT built, $SKIPPED skipped"
echo ""
echo "Wheels:"
find "$OUT" -name "*.whl" -newer "$MANIFEST" -print 2>/dev/null | sort || true
echo ""
echo "Done. To install: pip install \$(ls $OUT/*.whl | tail -1)"
