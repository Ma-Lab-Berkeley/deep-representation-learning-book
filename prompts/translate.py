#!/usr/bin/env python3
"""Automated LaTeX translation script (English -> Chinese).

Splits a .tex file into fixed-size line chunks, sends each to an LLM
via litellm for translation, and writes the combined result to a _zh.tex file.

Usage:
    python translate.py chapters/chapter1/introduction.tex
    python translate.py chapters/chapter1/introduction.tex --model openrouter/moonshotai/kimi-k2
    python translate.py chapters/chapter1/introduction.tex --skip-to 3
    python translate.py chapters/chapter1/introduction.tex --dry-run
"""

import argparse
import re
import sys
import time
from pathlib import Path

import litellm

DEFAULT_MODEL = "openrouter/google/gemini-3.1-pro-preview"
CHUNK_SIZE = 500

SUFFIX_INSTRUCTION = (
    "\n\nTranslate the following LaTeX excerpt from English to Chinese "
    "according to the rules above. "
    "Return ONLY the translated LaTeX code. "
    "No explanations, no markdown formatting."
)

RED = "\033[31m"
GREEN = "\033[32m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:latex|tex)?\s*\n(.*?)```\s*$", text, re.DOTALL)
    return m.group(1).rstrip("\n") if m else text


def split_into_chunks(
    content: str, chunk_size: int = CHUNK_SIZE
) -> list[tuple[int, int, str]]:
    """Split content into chunks of ``chunk_size`` lines.

    Returns a list of (start_line, end_line, text) where lines are 1-based.
    """
    lines = content.split("\n")
    chunks: list[tuple[int, int, str]] = []
    for i in range(0, len(lines), chunk_size):
        chunk_lines = lines[i : i + chunk_size]
        start = i + 1
        end = i + len(chunk_lines)
        chunks.append((start, end, "\n".join(chunk_lines)))
    return chunks


def call_llm(
    model: str,
    system_prompt: str,
    user_text: str,
    max_retries: int = 3,
) -> str | None:
    """Call the LLM via litellm with retries."""
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                timeout=1200
            )
            result = response.choices[0].message.content
            return strip_code_fences(result) if result else None

        except litellm.RateLimitError:
            wait = 2**attempt * 5
            print(f"  Rate limited, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except litellm.AuthenticationError:
            print(
                "Error: Authentication failed. Check your API key.",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"  Error on attempt {attempt + 1}: {e}",
                file=sys.stderr,
            )
            if attempt < max_retries - 1:
                wait = 2**attempt * 2
                print(f"  Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                return None

    print("  Max retries exceeded.", file=sys.stderr)
    return None


def get_output_path(input_path: Path) -> Path:
    return input_path.with_stem(input_path.stem + "_zh")


def main():
    parser = argparse.ArgumentParser(
        description="Translate an English LaTeX file to Chinese using an LLM.",
    )
    parser.add_argument("file", type=Path, help="Path to the .tex file")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path(__file__).parent / "translate.md",
        help="Path to the translation prompt (default: prompts/translate.md)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"litellm model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: <input>_zh.tex)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Lines per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--skip-to",
        type=int,
        default=1,
        help="Resume from chunk N (1-based), keeping prior chunks from output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show chunk layout without calling the LLM",
    )
    args = parser.parse_args()

    tex_path = args.file.resolve()
    if not tex_path.exists():
        print(f"Error: file not found: {tex_path}", file=sys.stderr)
        sys.exit(1)

    prompt_path = args.prompt.resolve()
    if not prompt_path.exists():
        print(f"Error: prompt not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)

    output_path = (args.output or get_output_path(tex_path)).resolve()

    base_prompt = prompt_path.read_text().strip()
    system_prompt = base_prompt + SUFFIX_INSTRUCTION

    content = tex_path.read_text()
    chunks = split_into_chunks(content, args.chunk_size)
    total = len(chunks)

    print(f"\n{BOLD}Translating: {tex_path.name}{RESET}")
    print(f"  Model:      {args.model}")
    print(f"  Output:     {output_path}")
    print(f"  Chunks:     {total}  ({args.chunk_size} lines each)")
    print(f"  Source:     {len(content.splitlines())} lines\n")

    # When resuming, load already-translated chunks from the output file.
    translated: list[str] = []
    if args.skip_to > 1 and output_path.exists():
        existing = output_path.read_text()
        existing_chunks = split_into_chunks(existing, args.chunk_size)
        for idx in range(min(args.skip_to - 1, len(existing_chunks))):
            translated.append(existing_chunks[idx][2])
        print(
            f"  Resuming from chunk {args.skip_to} "
            f"({len(translated)} chunks loaded from {output_path.name})\n"
        )

    # If the output file had fewer chunks than skip_to-1, pad with originals.
    while len(translated) < args.skip_to - 1 and len(translated) < total:
        translated.append(chunks[len(translated)][2])

    for i, (start, end, text) in enumerate(chunks):
        chunk_num = i + 1

        if chunk_num < args.skip_to:
            continue

        print(
            f"  {BOLD}{CYAN}[{chunk_num}/{total}]{RESET} "
            f"lines {start}–{end}  ",
            end="",
            flush=True,
        )

        if args.dry_run:
            print(f"{DIM}(dry run, {len(text.splitlines())} lines){RESET}")
            translated.append(text)
            continue

        result = call_llm(args.model, system_prompt, text)

        if result is None:
            print(f"{RED}FAILED{RESET} — keeping original")
            translated.append(text)
        else:
            print(f"{GREEN}OK{RESET}  ({len(result.splitlines())} lines)")
            translated.append(result)

        output_path.write_text("\n".join(translated))

    output_path.write_text("\n".join(translated))
    print(f"\n{BOLD}{GREEN}Done!{RESET}  Output written to {output_path}\n")


if __name__ == "__main__":
    main()
