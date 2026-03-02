#!/usr/bin/env bash
# build.sh — Orchestrates the make4ht + MathJax pipeline.
#
# Usage: bash website/latex_to_html/build.sh [tex_file] [output_dir]
#
# Arguments:
#   tex_file    TeX source file (default: book-main.tex)
#   output_dir  Final HTML output directory (default: website/html)
#
# Stages:
#   1. Run make4ht → _build_{base}/
#   2. Generate MathJax macros from .sty files → macros.json
#   3. Inject macros into MathJax config in each HTML file
#   4. Post-process: rename, inject CSS/JS, mini-TOC, search index → output_dir/

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPELINE_DIR="$SCRIPT_DIR"

TEX_FILE="${1:-book-main.tex}"
TEX_PATH="$REPO_ROOT/$TEX_FILE"
TEX_BASE="$(basename "$TEX_FILE" .tex)"

OUTPUT_DIR="${2:-$REPO_ROOT/website/html}"
# Make output_dir absolute if relative
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"

BUILD_DIR="$REPO_ROOT/website/_build_${TEX_BASE}"

CFG_FILE="$PIPELINE_DIR/book.cfg"
MK4_FILE="$PIPELINE_DIR/book.mk4"
MACROS_JSON="$BUILD_DIR/macros.json"

echo "=========================================="
echo "make4ht + MathJax Pipeline"
echo "=========================================="
echo "  Repo root:   $REPO_ROOT"
echo "  TeX file:    $TEX_FILE"
echo "  Build dir:   $BUILD_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Pre-build: clean stale tex4ht artifacts from repo root
# ---------------------------------------------------------------------------
# make4ht writes .xref, .4ct, .4tc, .idv, .lg files to the repo root.
# Stale versions from a previous (possibly failed) run can poison the
# next build — e.g. malformed .xref entries cause immediate parse errors.
echo "[Pre-build] Cleaning stale tex4ht artifacts..."
rm -f "$REPO_ROOT/${TEX_BASE}".{xref,4ct,4tc,idv,lg}
# ---------------------------------------------------------------------------
# Pre-build: ensure XeTeX format has enough main_memory
# ---------------------------------------------------------------------------
# The XeTeX format bakes main_memory at build time; the repo texmf.cnf
# sets a larger value.  Rebuild a user-level xelatex.fmt if needed so
# make4ht (which uses xelatex) has enough memory for CJK + tex4ht.
# Uses a temp dir for the probe so parallel builds don't collide.
export TEXMFCNF="$REPO_ROOT:"

NEEDED_MEM=$(kpsewhich --var-value main_memory 2>/dev/null || echo 0)
PROBE_DIR=$(mktemp -d)
ACTUAL_MEM=$(cd "$PROBE_DIR" \
    && xelatex -interaction=batchmode '\tracingstats=1 \stop' >/dev/null 2>&1 \
    && grep -o 'out of [0-9]*' texput.log 2>/dev/null | grep -o '[0-9]*' || echo 0)
rm -rf "$PROBE_DIR"

if [ "$ACTUAL_MEM" -lt "$NEEDED_MEM" ] 2>/dev/null; then
    echo "[Pre-build] Rebuilding xelatex format (main_memory $ACTUAL_MEM -> $NEEDED_MEM)..."
    fmtutil-user --byfmt xelatex >/dev/null 2>&1
fi

# ---------------------------------------------------------------------------
# Stage 1: make4ht
# ---------------------------------------------------------------------------
echo "[Stage 1] Running make4ht..."
mkdir -p "$BUILD_DIR"
cd "$REPO_ROOT"

# Run make4ht with XeTeX engine, mathjax passthrough, chapter splitting
make4ht -x -u -s \
    -c "$CFG_FILE" \
    -e "$MK4_FILE" \
    -d "$BUILD_DIR/" \
    "$TEX_PATH" \
    "html,mathjax,2,fn-in" \
    "" \
    "" \
    "-shell-escape"

echo "  make4ht output in $BUILD_DIR/"
echo "  Files: $(ls "$BUILD_DIR"/*.html 2>/dev/null | wc -l) HTML files"
echo ""

# ---------------------------------------------------------------------------
# Stage 2: Generate MathJax macros
# ---------------------------------------------------------------------------
echo "[Stage 2] Generating MathJax macros..."
cd "$REPO_ROOT"

uv run python3 "$PIPELINE_DIR/generate_macros.py" \
    "$MACROS_JSON" \
    "$REPO_ROOT/math-macros.sty" \
    "$REPO_ROOT/book-macros.sty" \
    "$REPO_ROOT/chapters"

echo "  Macros written to $MACROS_JSON"
echo ""

# ---------------------------------------------------------------------------
# Stage 3: Inject macros into MathJax config
# ---------------------------------------------------------------------------
echo "[Stage 3] Injecting MathJax macros..."

uv run python3 "$PIPELINE_DIR/inject_mathjax_macros.py" "$BUILD_DIR" "$MACROS_JSON"

echo ""

# ---------------------------------------------------------------------------
# Stage 4: Post-processing
# ---------------------------------------------------------------------------
echo "[Stage 4] Post-processing..."

uv run python3 "$PIPELINE_DIR/postprocess.py" \
    --input "$BUILD_DIR" \
    --output "$OUTPUT_DIR"

echo ""

# ---------------------------------------------------------------------------
# Cleanup: remove stale make4ht outputs from repo root (generated despite -d flag)
# ---------------------------------------------------------------------------
echo "[Cleanup] Removing stale make4ht artifacts from repo root..."
rm -f "$REPO_ROOT/${TEX_BASE}"*.html \
      "$REPO_ROOT/${TEX_BASE}"*.svg \
      "$REPO_ROOT/${TEX_BASE}".css \
      "$REPO_ROOT/${TEX_BASE}".tmp

# Remove make4ht's dash-suffixed PNGs from source chapter directories
echo "[Cleanup] Removing *-.png from source chapter directories..."
find "$REPO_ROOT/chapters" -name '*-.png' -delete 2>/dev/null || true

echo "=========================================="
echo "Pipeline complete!"
echo "Output: $OUTPUT_DIR/"
echo "=========================================="
