#!/usr/bin/env bash
# build.sh — Build PDF + HTML from a TeX file in an isolated temp directory.
#
# Usage: bash website/latex_to_html/build.sh [tex_file] [output_dir]
#
# Arguments:
#   tex_file    TeX source file (default: book-main.tex)
#   output_dir  Final output directory for HTML + PDF (default: website/html)
#
# The build runs in a temp directory named after the TeX file, allowing
# parallel builds of different files without collisions. The PDF is placed
# at $output_dir/$tex_base.pdf alongside the HTML output.
#
# Set KEEP_BUILD_DIR=1 to preserve the temp directory for debugging.
# Set BOOK_BUILD_STRICT=1 to fail on compilation errors and unresolved PDF refs.
# Set BOOK_BUILD_DIAGNOSTICS to retain stage logs even after a failed build.
# Set BOOK_PYTHON to a Python executable to bypass the full uv project environment.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIPELINE_DIR="$SCRIPT_DIR"

TEX_FILE="${1:-book-main.tex}"
if [[ "$TEX_FILE" = /* ]]; then
    case "$TEX_FILE" in
    "$REPO_ROOT"/*) TEX_REL="${TEX_FILE#$REPO_ROOT/}" ;;
    *)
        echo "Error: tex_file must be inside the repository: $TEX_FILE" >&2
        exit 1
        ;;
    esac
else
    TEX_REL="$TEX_FILE"
fi
TEX_BASE="$(basename "$TEX_REL" .tex)"

OUTPUT_DIR="${2:-$REPO_ROOT/website/html}"
[[ "$OUTPUT_DIR" != /* ]] && OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"

STRICT_BUILD="${BOOK_BUILD_STRICT:-0}"
if [[ "$STRICT_BUILD" != "0" && "$STRICT_BUILD" != "1" ]]; then
    echo "Error: BOOK_BUILD_STRICT must be 0 or 1" >&2
    exit 1
fi
DIAGNOSTICS_DIR="${BOOK_BUILD_DIAGNOSTICS:-}"
if [[ -n "$DIAGNOSTICS_DIR" && "$DIAGNOSTICS_DIR" != /* ]]; then
    DIAGNOSTICS_DIR="$REPO_ROOT/$DIAGNOSTICS_DIR"
fi
if [[ -n "${BOOK_PYTHON:-}" ]]; then
    PYTHON_CMD=("$BOOK_PYTHON")
else
    PYTHON_CMD=(uv run python3)
fi

# Create temp build directory named after the tex file (parallel-safe)
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/build-${TEX_BASE}-XXXXXX")"
BUILD_PHASE="setup"
collect_stage_logs() {
    [[ -n "$DIAGNOSTICS_DIR" && -d "${SOURCE_DIR:-}" ]] || return 0
    mkdir -p "$DIAGNOSTICS_DIR"
    local ext
    for ext in log aux blg lg; do
        if [[ -f "$SOURCE_DIR/$TEX_BASE.$ext" ]]; then
            cp "$SOURCE_DIR/$TEX_BASE.$ext" "$DIAGNOSTICS_DIR/$BUILD_PHASE.$ext"
        fi
    done
}
cleanup() {
    local result=$?
    collect_stage_logs || echo "Warning: could not preserve build logs" >&2
    if [ "${KEEP_BUILD_DIR:-}" = "1" ]; then
        echo "Build dir preserved: $BUILD_DIR"
    else
        rm -rf "$BUILD_DIR"
    fi
    return "$result"
}
trap cleanup EXIT

SOURCE_DIR="$BUILD_DIR/source"
MAKE4HT_DIR="$BUILD_DIR/make4ht_raw"
MACROS_DIR="$BUILD_DIR/macros"
MATHJAX_DIR="$BUILD_DIR/mathjax_injected"
POST_INPUT_DIR="$BUILD_DIR/postprocess_input"
POST_OUTPUT_DIR="$BUILD_DIR/postprocess_output"

echo "=========================================="
echo "Build Pipeline (PDF + HTML)"
echo "=========================================="
echo "  Repo root:   $REPO_ROOT"
echo "  TeX file:    $TEX_REL"
echo "  Build dir:   $BUILD_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# Stage 0: Create source snapshot
# ---------------------------------------------------------------------------
echo "[Stage 0] Creating source snapshot..."
mkdir -p "$SOURCE_DIR" "$MAKE4HT_DIR" "$MACROS_DIR" \
    "$MATHJAX_DIR" "$POST_INPUT_DIR" "$POST_OUTPUT_DIR"

rsync -a \
    --exclude=".git/" \
    --exclude="/_build_*/" \
    --exclude="website/_build_*/" \
    --exclude="website/html/" \
    "$REPO_ROOT/" "$SOURCE_DIR/"

SNAPSHOT_TEX="$SOURCE_DIR/$TEX_REL"
if [ ! -f "$SNAPSHOT_TEX" ]; then
    echo "Error: TeX file not found: $SNAPSHOT_TEX" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-build: ensure XeTeX format has enough main_memory
# ---------------------------------------------------------------------------
export TEXMFCNF="$SOURCE_DIR:"

NEEDED_MEM=$(kpsewhich --var-value main_memory 2>/dev/null || echo 0)
PROBE_DIR=$(mktemp -d)
ACTUAL_MEM=$(cd "$PROBE_DIR" &&
    xelatex -interaction=batchmode '\tracingstats=1 \stop' >/dev/null 2>&1 &&
    grep -o 'out of [0-9]*' texput.log 2>/dev/null | grep -o '[0-9]*' || echo 0)
rm -rf "$PROBE_DIR"

echo "[Pre-build] main_memory: needed=$NEEDED_MEM actual=$ACTUAL_MEM"

if [ "$ACTUAL_MEM" -lt "$NEEDED_MEM" ] 2>/dev/null; then
    echo "[Pre-build] Rebuilding xelatex format (main_memory $ACTUAL_MEM -> $NEEDED_MEM)..."
    fmtutil-user --byfmt xelatex
fi

# ---------------------------------------------------------------------------
# Stage 1: Build PDF
# ---------------------------------------------------------------------------
echo "[Stage 1] Building PDF..."
BUILD_PHASE="pdf"
PDF_ARGS=(-pdf -interaction=nonstopmode -shell-escape)
if [[ "$STRICT_BUILD" == "1" ]]; then
    PDF_ARGS+=(-halt-on-error)
else
    PDF_ARGS+=(-f)
fi
PDF_STATUS=0
(
    cd "$SOURCE_DIR"
    latexmk "${PDF_ARGS[@]}" "$TEX_REL"
) || PDF_STATUS=$?
collect_stage_logs
if [[ "$STRICT_BUILD" == "1" && "$PDF_STATUS" != "0" ]]; then
    echo "Error: PDF compilation failed (exit $PDF_STATUS)" >&2
    exit "$PDF_STATUS"
fi

PDF_FILE="$SOURCE_DIR/${TEX_BASE}.pdf"
if [ -f "$PDF_FILE" ]; then
    echo "  PDF built successfully"
else
    echo "  Warning: PDF was not produced"
    [[ "$STRICT_BUILD" == "0" ]] || exit 1
fi

if [[ "$STRICT_BUILD" == "1" ]]; then
    LOG_ARGS=(tex-log "$SOURCE_DIR/$TEX_BASE.log" --references)
    if [[ -f "$SOURCE_DIR/$TEX_BASE.blg" ]]; then
        LOG_ARGS+=(--biber-log "$SOURCE_DIR/$TEX_BASE.blg")
    fi
    "${PYTHON_CMD[@]}" "$PIPELINE_DIR/validate_build.py" "${LOG_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 2: Capture reference AUX (from PDF build) for eqref fallback
# ---------------------------------------------------------------------------
echo "[Stage 2] Capturing reference AUX..."
SNAPSHOT_AUX="$(dirname "$SNAPSHOT_TEX")/${TEX_BASE}.aux"
REF_AUX="$BUILD_DIR/reference.aux"

if [ -f "$SNAPSHOT_AUX" ] && ! grep -q '\\ifx\\rEfLiNK\\UnDef' "$SNAPSHOT_AUX"; then
    cp "$SNAPSHOT_AUX" "$REF_AUX"
    echo "  Using AUX from PDF build"
else
    echo "  Warning: no usable AUX; eqref fallback may be limited"
    [[ "$STRICT_BUILD" == "0" ]] || exit 1
fi

# ---------------------------------------------------------------------------
# Stage 3: make4ht
# ---------------------------------------------------------------------------
echo "[Stage 3] Running make4ht..."
BUILD_PHASE="html"
CFG_FILE="$PIPELINE_DIR/book.cfg"
MK4_FILE="$PIPELINE_DIR/book.mk4"
(
    cd "$SOURCE_DIR"
    make4ht -x -u -s \
        -c "$CFG_FILE" \
        -e "$MK4_FILE" \
        -d "$MAKE4HT_DIR/" \
        "$SNAPSHOT_TEX" \
        "html,mathjax,2,fn-in" \
        "" \
        "" \
        "-shell-escape"
)
collect_stage_logs
if [[ "$STRICT_BUILD" == "1" ]]; then
    # TeX4ht refs may still need postprocess.py's PDF-AUX recovery at this stage.
    "${PYTHON_CMD[@]}" "$PIPELINE_DIR/validate_build.py" tex-log "$SOURCE_DIR/$TEX_BASE.log"
    if ! compgen -G "$MAKE4HT_DIR/*.html" >/dev/null; then
        echo "Error: make4ht produced no HTML files" >&2
        exit 1
    fi
fi

echo "  make4ht output: $(ls "$MAKE4HT_DIR"/*.html 2>/dev/null | wc -l) HTML files"
echo ""

# ---------------------------------------------------------------------------
# Stage 4: Generate MathJax macros
# ---------------------------------------------------------------------------
echo "[Stage 4] Generating MathJax macros..."
MACROS_JSON="$MACROS_DIR/macros.json"

(cd "$REPO_ROOT" && "${PYTHON_CMD[@]}" "$PIPELINE_DIR/generate_macros.py" \
    "$MACROS_JSON" \
    "$SOURCE_DIR/math-macros.sty" \
    "$SOURCE_DIR/book-macros.sty" \
    "$SOURCE_DIR/chapters")

echo "  Macros written to $MACROS_JSON"
echo ""

# ---------------------------------------------------------------------------
# Stage 5: Inject macros into copied HTML
# ---------------------------------------------------------------------------
echo "[Stage 5] Injecting MathJax macros..."

rsync -a --delete "$MAKE4HT_DIR/" "$MATHJAX_DIR/"

(cd "$REPO_ROOT" && "${PYTHON_CMD[@]}" "$PIPELINE_DIR/inject_mathjax_macros.py" \
    "$MATHJAX_DIR" "$MACROS_JSON")

echo ""

# ---------------------------------------------------------------------------
# Stage 6: Post-processing
# ---------------------------------------------------------------------------
echo "[Stage 6] Post-processing..."

rsync -a --delete "$MATHJAX_DIR/" "$POST_INPUT_DIR/"

(cd "$REPO_ROOT" && "${PYTHON_CMD[@]}" "$PIPELINE_DIR/postprocess.py" \
    --input "$POST_INPUT_DIR" \
    --output "$POST_OUTPUT_DIR" \
    --aux "$REF_AUX" \
    --shared-asset-prefix "")

echo ""

# ---------------------------------------------------------------------------
# Stage 7: Publish to output directory
# ---------------------------------------------------------------------------
echo "[Stage 7] Publishing to $OUTPUT_DIR..."
mkdir -p "$OUTPUT_DIR"
rsync -a "$POST_OUTPUT_DIR/" "$OUTPUT_DIR/"

# Copy PDF to output directory
if [ -f "$PDF_FILE" ]; then
    cp "$PDF_FILE" "$OUTPUT_DIR/${TEX_BASE}.pdf"
    echo "  PDF: $OUTPUT_DIR/${TEX_BASE}.pdf"
fi

# Keep chapter assets resolvable when publishing to non-default output paths.
SHARED_ASSETS_DIR="$REPO_ROOT/website/html"
for file in common.css chapter.css common.js chapter.js; do
    if [ -f "$SHARED_ASSETS_DIR/$file" ] && [ "$SHARED_ASSETS_DIR/$file" != "$OUTPUT_DIR/$file" ]; then
        cp "$SHARED_ASSETS_DIR/$file" "$OUTPUT_DIR/$file"
    fi
done
if [ -d "$SHARED_ASSETS_DIR/assets" ] && [ "$SHARED_ASSETS_DIR/assets" != "$OUTPUT_DIR/assets" ]; then
    mkdir -p "$OUTPUT_DIR/assets"
    rsync -a "$SHARED_ASSETS_DIR/assets/" "$OUTPUT_DIR/assets/"
fi

echo "=========================================="
echo "Build complete!"
echo "  HTML: $OUTPUT_DIR/"
if [ -f "$PDF_FILE" ]; then
    echo "  PDF:  $OUTPUT_DIR/${TEX_BASE}.pdf"
fi
echo "=========================================="
