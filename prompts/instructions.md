# Proofreading Instructions

The contained Markdown files each contain a prompt that can be used for proofreading paragraphs or snippets using AI models and environments like Cursor. This prompt-sharing is a way to consolidate the desired format and notation. It is \_highly recommended_that you copy-edit new text that you output, if only to check for grammar errors and fix the English.

- `text_proofreading.md` --- use for improving the writing of one or a handful of paragraphs of English prose. This prompt should be used with the Kimi K2 model (in Cursor).
- `math_proofreading.md` --- use for improving the math formatting of a snippet of math code. This prompt should be used with a model which has good instruction-following capabilities, e.g., OpenAI GPT 5.2, Gemini 3 Flash/Pro, or public models such as DeepSeek 3.2.

## Automated Proofreading Script

`proofread.py` automates the proofreading workflow for entire `.tex` files. It splits the file into chunks, sends each to an LLM with a given instructions file, and presents an interactive diff for review. The LLM decides which chunks are relevant to the instructions; unchanged chunks are auto-skipped.

### Setup

```bash
export OPENROUTER_API_KEY="your-key-here"
```

### Usage

```bash
uv run prompts/proofread.py <file.tex> <instructions.md> [options]
```

### Options

| Flag            | Description                                         |
| --------------- | --------------------------------------------------- |
| `--model MODEL` | OpenRouter model ID (default: `moonshotai/kimi-k2`) |
| `--dry-run`     | Show chunks without calling the LLM                 |
| `--skip-to N`   | Resume from chunk N (1-based)                       |
| `--verbose`     | Print chunk details                                 |

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
uv run prompts/proofread.py chapters/chapter3/denoising.tex prompts/text_proofreading.md --dry-run --verbose

# Proofread English prose
uv run prompts/proofread.py chapters/chapter4/lossy-compression.tex prompts/text_proofreading.md

# Proofread math formatting with a different model
uv run prompts/proofread.py chapters/chapter3/denoising.tex prompts/math_proofreading.md --model openai/gpt-5-nano

# Resume a previous session from chunk 15
uv run prompts/proofread.py chapters/chapter3/denoising.tex prompts/text_proofreading.md --skip-to 15
```

---

**NOTE:** The above prompts are used solely for _proofreading and copy-editing_. Please do NOT use AI models for long-form generation of content (except for translation). Write all the ideas yourself.
