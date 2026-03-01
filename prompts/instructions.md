# Proofreading Instructions

The contained Markdown files each contain a prompt that can be used for proofreading paragraphs or snippets using AI models and environments like Cursor. This prompt-sharing is a way to consolidate the desired format and notation. It is \_highly recommended_that you copy-edit new text that you output, if only to check for grammar errors and fix the English.

## Prompt Modules

Individual instruction modules live in `modules/` and are passed directly via `--prompts`. Each module is focused on a single concern so models follow instructions reliably.

**Math:**

| Module | Description |
| --- | --- |
| `modules/math_macros.md` | Replace raw commands with book macros (`\v`, `\c`, `\bb` prefixes, `\KL`, `\ip`, `\norm`, `\mat`, etc.) + notation table |
| `modules/math_delimiters.md` | `$`->`\(\)`, `equation*`->`equation`, `align*`->`align`, `\[...\]` conversion, numbering |
| `modules/math_style.md` | Subscript braces, bracket sizing (`\bp`/`\bs`/`\bc`), `\colon`, display equation punctuation |

**Text:**

| Module | Description |
| --- | --- |
| `modules/text_editing.md` | Grammar, emphasis (`\textit`), dashes, tone, paragraph/section titles |
| `modules/references.md` | `\Cref`, `\eqref`, citation style (`\citep`/`\citet`) |
| `modules/punctuation.md` | Punctuation in and around math |

**Chapter 8 specific:**

| Module | Description |
| --- | --- |
| `modules/ch8_structural.md` | Tables, figure paths, floats, captions, image dimension ordering |
| `modules/ch8_notation.md` | Notation conflicts (theta, encoder/decoder, dataset var) |
| `modules/ch8_cleanup.md` | Typos, review artifacts, section-by-section issues |

## Automated Proofreading Script

`proofread.py` automates the proofreading workflow for entire `.tex` files. It splits the file into chunks, sends each to an LLM with given instruction files, and presents an interactive diff for review. When multiple prompts are provided, they are chained sequentially on each chunk (each prompt transforms the output of the previous one).

### Setup

```bash
export OPENROUTER_API_KEY="your-key-here"
```

### Usage

```bash
uv run prompts/proofread.py --file <file.tex> --prompts <prompt.md> [<prompt.md> ...] [options]
```

### Options

| Flag | Description |
| --- | --- |
| `--file FILE` | Path to the `.tex` file (required) |
| `--prompts P [P ...]` | One or more markdown instruction files (required) |
| `--model MODEL` | OpenRouter model ID (default: `moonshotai/kimi-k2`) |
| `--dry-run` | Show chunks without calling the LLM |
| `--skip-to N` | Resume from chunk N (1-based) |
| `--no-thinking` | Disable reasoning/thinking for models that support it |
| `--verbose` | Print chunk details |

### Interactive Review

For each chunk with changes, a colored diff is shown and you can respond with a single keypress:

- **a** — Accept the change
- **r** — Reject the change
- **e** — Edit the LLM suggestion in `$EDITOR`
- **A** — Accept all remaining changes without prompting
- **q** — Quit (all previously accepted changes are already saved)

A backup of the original file is created as `file.tex.bak` before any writes.

### Examples

```bash
# Dry run to inspect how the file is chunked
uv run prompts/proofread.py --file chapters/chapter3/denoising.tex \
    --prompts prompts/modules/text_editing.md --dry-run --verbose

# Proofread English prose
uv run prompts/proofread.py --file chapters/chapter4/lossy-compression.tex \
    --prompts prompts/modules/text_editing.md

# Chain math macro substitution then text editing
uv run prompts/proofread.py --file chapters/chapter3/denoising.tex \
    --prompts prompts/modules/math_macros.md prompts/modules/text_editing.md

# Full math pass (macros, delimiters, style)
uv run prompts/proofread.py --file chapters/chapter3/denoising.tex \
    --prompts prompts/modules/math_macros.md \
             prompts/modules/math_delimiters.md \
             prompts/modules/math_style.md

# Resume a previous session from chunk 15
uv run prompts/proofread.py --file chapters/chapter3/denoising.tex \
    --prompts prompts/modules/text_editing.md --skip-to 15
```

---

**NOTE:** The above prompts are used solely for _proofreading and copy-editing_. Please do NOT use AI models for long-form generation of content (except for translation). Write all the ideas yourself.
