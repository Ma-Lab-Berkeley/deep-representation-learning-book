#!/usr/bin/env python3
"""Automated LaTeX proofreading script.

Splits a .tex file into chunks, sends each to an LLM via OpenRouter with
a user-provided instructions file, and presents an interactive diff for
accept/reject/edit. The LLM decides which chunks are relevant to the
instructions; unchanged chunks are auto-skipped.
"""

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "moonshotai/kimi-k2"


def _clean_tex(content: str) -> str:
    """Strip comment-only lines from TeX content."""
    return "\n".join(
        line for line in content.splitlines() if not re.match(r"^\s*%", line)
    )


def resolve_includes(text: str, base_dir: Path) -> str:
    """Replace <!-- include: path --> directives with file contents."""
    repo_root = base_dir
    for parent in [base_dir, *base_dir.parents]:
        if (parent / ".git").exists():
            repo_root = parent
            break

    def _replace(m: re.Match) -> str:
        path = (repo_root / m.group(1).strip()).resolve()
        if not path.exists():
            print(f"  Warning: include file not found: {path}", file=sys.stderr)
            return m.group(0)
        content = path.read_text()
        if path.suffix == ".tex":
            content = _clean_tex(content)
        return content

    return re.sub(r"<!--\s*include:\s*(.+?)\s*-->", _replace, text)


SYSTEM_PROMPT = (
    "You are a LaTeX proofreading assistant for an academic textbook. "
    "You will receive a LaTeX excerpt along with specific editing instructions.\n\n"
    "Rules:\n"
    "- Apply the given instructions precisely to the LaTeX excerpt.\n"
    "- If the instructions are not relevant to this excerpt, return it UNCHANGED.\n"
    "- Return ONLY the corrected LaTeX. No explanations, no markdown code fences or any other delimiters.\n"
    "- Preserve all whitespace, indentation, and line structure exactly."
)


# ---------------------------------------------------------------------------
# Chunk Splitting
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    start_line: int  # 1-based
    end_line: int  # 1-based, inclusive
    is_blank: bool = False
    is_preamble: bool = False


def split_into_chunks(content: str) -> list[Chunk]:
    """Split LaTeX content into chunks on blank lines, respecting environments."""
    lines = content.split("\n")
    chunks: list[Chunk] = []

    # Find \begin{document} and \end{document}
    doc_start = None
    doc_end = None
    for i, line in enumerate(lines):
        if re.match(r"\s*\\begin\{document\}", line):
            doc_start = i
        if re.match(r"\s*\\end\{document\}", line):
            doc_end = i

    # Preamble chunk
    if doc_start is not None and doc_start > 0:
        chunks.append(
            Chunk(
                text="\n".join(lines[: doc_start + 1]),
                start_line=1,
                end_line=doc_start + 1,
                is_preamble=True,
            )
        )
        body_start = doc_start + 1
    else:
        body_start = 0

    body_end = doc_end if doc_end is not None else len(lines)

    # Split body on blank lines, tracking environment nesting
    env_stack: list[str] = []
    current_lines: list[str] = []
    chunk_start = body_start

    def flush(end_idx: int):
        if not current_lines:
            return
        chunks.append(
            Chunk(
                text="\n".join(current_lines),
                start_line=chunk_start + 1,
                end_line=end_idx + 1,
            )
        )

    for i in range(body_start, body_end):
        line = lines[i]

        for m in re.finditer(r"\\begin\{([^}]+)\}", line):
            env_stack.append(m.group(1))
        for m in re.finditer(r"\\end\{([^}]+)\}", line):
            if env_stack and env_stack[-1] == m.group(1):
                env_stack.pop()

        if line.strip() == "" and not env_stack:
            flush(i - 1 if current_lines else i)
            current_lines = []
            chunks.append(
                Chunk(
                    text="",
                    start_line=i + 1,
                    end_line=i + 1,
                    is_blank=True,
                )
            )
            chunk_start = i + 1
        else:
            if not current_lines:
                chunk_start = i
            current_lines.append(line)

    if current_lines:
        flush(body_end - 1)

    # Postamble chunk
    if doc_end is not None:
        chunks.append(
            Chunk(
                text="\n".join(lines[doc_end:]),
                start_line=doc_end + 1,
                end_line=len(lines),
                is_preamble=True,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# LLM Interface
# ---------------------------------------------------------------------------


def get_openai_client():
    """Create an OpenAI client pointing at OpenRouter."""
    try:
        from openai import OpenAI
    except ImportError:
        print(
            "Error: openai package not installed. Run: pip install openai",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr
        )
        sys.exit(1)

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def strip_llm_wrapping(text: str) -> str:
    """Remove markdown fences, horizontal rules, and other LLM wrapper artifacts."""
    text = text.strip()
    # Strip ```latex ... ``` or ```tex ... ``` or ``` ... ```
    m = re.match(r"^```(?:latex|tex)?\s*\n(.*?)```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # Strip leading/trailing horizontal rules that LLMs add as delimiters.
    # Must be on its own line and consist only of repeated separator tokens.
    # Matches: ---, - - -, -- -- --, ***, * * *, ___, etc.
    # Anchored to line boundaries to avoid matching em-dashes/en-dashes in content.
    hr = r"[-*_]+(?:\s+[-*_]+)+"  # e.g. "-- -- --", "- - -", "* * *"
    hr_solid = r"[-]{3,}|[*]{3,}|[_]{3,}"  # e.g. "---", "***", "___"
    hr_pat = rf"(?:{hr}|{hr_solid})"
    text = re.sub(rf"^{hr_pat}\s*\n", "", text)
    text = re.sub(rf"\n\s*{hr_pat}$", "", text)
    return text.strip()


def unwrap_tex(text: str) -> str:
    """Join hard-wrapped lines into single-line paragraphs for comparison."""
    paragraphs = re.split(r"\n[ \t]*\n", text)
    return "\n\n".join(
        " ".join(line.strip() for line in p.splitlines() if line.strip())
        for p in paragraphs
    )


def call_llm(
    client,
    model: str,
    chunk_text: str,
    instructions: str,
    max_retries: int = 3,
    no_thinking: bool = False,
) -> str | None:
    """Call the LLM with retries. Returns corrected text or None on error.

    Messages are structured for prefix caching: the static system prompt is
    identical across all calls, and the chunk text sits at the start of the
    user message so it is shared across multiple prompt passes on the same chunk.
    """
    from openai import APIStatusError, RateLimitError

    user_message = (
        "Here is the LaTeX excerpt to proofread:\n\n"
        f"{chunk_text}\n\n"
        "---\n\n"
        "Apply the following editing instructions to the excerpt above:\n\n"
        f"{instructions}"
    )

    extra_body = {}
    if no_thinking:
        extra_body["reasoning"] = {"effort": "none"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                **({"extra_body": extra_body} if extra_body else {}),
            )
            result = response.choices[0].message.content
            return strip_llm_wrapping(result) if result else None

        except RateLimitError:
            wait = 2**attempt * 5
            print(f"  Rate limited, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code == 401:
                print(
                    "Error: Authentication failed. Check OPENROUTER_API_KEY.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"  API error ({e.status_code}): {e.message}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  Unexpected error: {e}", file=sys.stderr)
            return None

    print("  Max retries exceeded.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Diff Display
# ---------------------------------------------------------------------------

RED = "\033[31m"
GREEN = "\033[32m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
STRIKETHROUGH = "\033[9m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
BG_RED = "\033[48;2;50;20;20m"
BG_GREEN = "\033[48;2;20;50;20m"
CLEAR_LINE = "\033[2K"


def print_progress(current: int, total: int, status: str = ""):
    """Overwrite the current line with a progress bar."""
    width = shutil.get_terminal_size().columns
    bar_width = min(30, width - 50)
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = current * 100 // total if total > 0 else 0
    line = f"  [{bar}]  {current}/{total}  ({pct}%)"
    if status:
        # Truncate status to fit
        max_status = width - len(line) - 4
        if max_status > 0:
            if len(status) > max_status:
                status = status[: max_status - 1] + "…"
            line += f"  {DIM}{status}{RESET}"
    print(f"\r{CLEAR_LINE}{line}", end="", flush=True)


def end_progress():
    """Move past the progress bar line so subsequent output doesn't overwrite it."""
    print()


def _bg(text: str, bg: str) -> str:
    """Wrap *text* with a background color, re-applying it after any resets."""
    return bg + text.replace(RESET, RESET + bg) + RESET


def compute_char_diff(old_line: str, new_line: str) -> tuple[str, str]:
    """Character-level diff within a single changed line pair."""
    sm = difflib.SequenceMatcher(None, old_line, new_line)
    old_parts: list[str] = []
    new_parts: list[str] = []

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            old_parts.append(old_line[i1:i2])
            new_parts.append(new_line[j1:j2])
        elif op == "delete":
            old_parts.append(f"{RED}{STRIKETHROUGH}{old_line[i1:i2]}{RESET}")
        elif op == "insert":
            new_parts.append(f"{GREEN}{new_line[j1:j2]}{RESET}")
        elif op == "replace":
            old_parts.append(f"{RED}{STRIKETHROUGH}{old_line[i1:i2]}{RESET}")
            new_parts.append(f"{GREEN}{new_line[j1:j2]}{RESET}")

    return "".join(old_parts), "".join(new_parts)


def render_diff(old_text: str, new_text: str, context: int = 3) -> str:
    """Render a colored inline diff with context lines."""
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    sm = difflib.SequenceMatcher(None, old_lines, new_lines)
    all_ops = sm.get_opcodes()

    if all(op == "equal" for op, *_ in all_ops):
        return ""

    # Which old-side lines are near a change?
    shown_old: set[int] = set()
    for op, i1, i2, j1, j2 in all_ops:
        if op != "equal":
            for k in range(max(0, i1 - context), min(len(old_lines), i2 + context)):
                shown_old.add(k)

    output: list[str] = []
    prev_end = -1

    for op, i1, i2, j1, j2 in all_ops:
        if op == "equal":
            for k in range(i1, i2):
                if k in shown_old:
                    if prev_end >= 0 and k > prev_end + 1:
                        output.append(f"{DIM}  ...{RESET}")
                    output.append(f"{DIM}  {old_lines[k]}{RESET}")
                    prev_end = k
        elif op == "delete":
            for k in range(i1, i2):
                output.append(_bg(f"{RED}- {old_lines[k]}", BG_RED))
                prev_end = k
        elif op == "insert":
            for k in range(j1, j2):
                output.append(_bg(f"{GREEN}+ {new_lines[k]}", BG_GREEN))
        elif op == "replace":
            for oi, ni in zip(range(i1, i2), range(j1, j2)):
                old_fmt, new_fmt = compute_char_diff(old_lines[oi], new_lines[ni])
                output.append(_bg(f"{RED}-{RESET} {old_fmt}", BG_RED))
                output.append(_bg(f"{GREEN}+{RESET} {new_fmt}", BG_GREEN))
                prev_end = oi
            # Leftover lines on either side
            if i2 - i1 > j2 - j1:
                for k in range(j2 - j1 + i1, i2):
                    output.append(_bg(f"{RED}- {old_lines[k]}", BG_RED))
                    prev_end = k
            elif j2 - j1 > i2 - i1:
                for k in range(i2 - i1 + j1, j2):
                    output.append(_bg(f"{GREEN}+ {new_lines[k]}", BG_GREEN))

    return "\n".join(output)


def _wrap_ansi(text: str, width: int = 0, indent: str = "  ") -> str:
    """Wrap text containing ANSI codes to fit terminal width."""
    if width <= 0:
        width = shutil.get_terminal_size().columns - 2
    ansi_pattern = re.compile(r"\033\[[0-9;]*m")
    words = text.split(" ")
    lines: list[str] = []
    line = indent
    col = len(indent)

    for word in words:
        if not word:
            continue
        vis_len = len(ansi_pattern.sub("", word))
        needed = (1 if col > len(indent) else 0) + vis_len
        if col + needed > width and col > len(indent):
            lines.append(line)
            line = indent + word
            col = len(indent) + vis_len
        else:
            if col > len(indent):
                line += " " + word
                col += 1 + vis_len
            else:
                line += word
                col += vis_len

    if line.strip():
        lines.append(line)

    return "\n".join(lines)


def render_word_diff(old_text: str, new_text: str) -> str:
    """Inline word-level diff that ignores line-break differences."""
    old_norm = unwrap_tex(old_text)
    new_norm = unwrap_tex(new_text)

    if old_norm == new_norm:
        return ""

    # Detect leading indent from original text
    indent = "  "
    for line in old_text.splitlines():
        if line.strip():
            indent = "  " + re.match(r"(\s*)", line).group(1)
            break

    old_words = old_norm.split()
    new_words = new_norm.split()

    sm = difflib.SequenceMatcher(None, old_words, new_words)
    parts: list[str] = []

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            parts.extend(old_words[i1:i2])
        elif op == "delete":
            for w in old_words[i1:i2]:
                parts.append(f"{RED}{STRIKETHROUGH}{w}{RESET}")
        elif op == "insert":
            for w in new_words[j1:j2]:
                parts.append(f"{GREEN}{w}{RESET}")
        elif op == "replace":
            for w in old_words[i1:i2]:
                parts.append(f"{RED}{STRIKETHROUGH}{w}{RESET}")
            for w in new_words[j1:j2]:
                parts.append(f"{GREEN}{w}{RESET}")

    return _wrap_ansi(" ".join(parts), indent=indent)


# ---------------------------------------------------------------------------
# Interactive Review
# ---------------------------------------------------------------------------


def getch() -> str:
    """Read a single character without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def edit_in_editor(text: str) -> str:
    """Open text in $EDITOR and return the edited result."""
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        subprocess.run([editor, tmp_path], check=True)
        with open(tmp_path) as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def review_change(
    chunk_idx: int,
    total: int,
    chunk: Chunk,
    corrected: str,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Display diff and prompt for a decision. Returns (action, final_text)."""
    current = corrected

    while True:
        print(
            f"\n{BOLD}{CYAN}--- Chunk {chunk_idx}/{total}  "
            f"lines {chunk.start_line}-{chunk.end_line} ---{RESET}\n"
        )

        diff_output = render_word_diff(chunk.text, current)
        if not diff_output:
            print(f"  {DIM}(no changes){RESET}")
            return "reject", chunk.text

        print(diff_output)

        if dry_run:
            print(f"\n  {DIM}(dry run){RESET}")
            return "reject", chunk.text

        print(
            f"\n  {BOLD}[a]{RESET} Accept   "
            f"{BOLD}[r]{RESET} Reject   "
            f"{BOLD}[e]{RESET} Edit in $EDITOR   "
            f"{BOLD}[A]{RESET} Accept all remaining   "
            f"{BOLD}[q]{RESET} Quit"
        )

        while True:
            ch = getch()
            if ch == "a":
                print("  -> Accepted")
                return "accept", current
            elif ch == "r":
                print("  -> Rejected")
                return "reject", chunk.text
            elif ch == "e":
                print("  -> Opening editor...")
                current = edit_in_editor(current)
                break  # re-render diff with edited text
            elif ch == "A":
                print("  -> Accepting all remaining")
                return "accept_all", current
            elif ch == "q" or ch == "\x03":
                print("  -> Quitting (progress saved)")
                return "quit", chunk.text


# ---------------------------------------------------------------------------
# File Writing
# ---------------------------------------------------------------------------


def reconstruct_file(chunks: list[Chunk]) -> str:
    """Reconstruct the full file from chunks."""
    return "\n".join("" if c.is_blank else c.text for c in chunks)


def write_file(path: Path, chunks: list[Chunk]) -> None:
    path.write_text(reconstruct_file(chunks))


def backup_file(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(path.read_text())
    print(f"  Backup: {bak}")
    return bak


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOKEN_WARN = 12000


def main():
    parser = argparse.ArgumentParser(
        description="Proofread a LaTeX file with an LLM and a set of instructions.",
    )
    parser.add_argument(
        "--file", type=Path, required=True, help="Path to the .tex file"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        nargs="+",
        required=True,
        help="One or more markdown instruction files",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show chunks without calling the LLM"
    )
    parser.add_argument(
        "--skip-to", type=int, default=1, help="Resume from chunk N (1-based)"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable reasoning/thinking for models that support it",
    )
    parser.add_argument("--verbose", action="store_true", help="Show chunk details")
    args = parser.parse_args()

    # Load inputs
    tex_path = args.file.resolve()
    if not tex_path.exists():
        print(f"Error: file not found: {tex_path}", file=sys.stderr)
        sys.exit(1)

    # Load and resolve each prompt file
    prompts: list[tuple[str, str]] = []
    for prompt_path in args.prompts:
        p = prompt_path.resolve()
        if not p.exists():
            print(f"Error: prompt not found: {p}", file=sys.stderr)
            sys.exit(1)
        raw = p.read_text().strip()
        resolved = resolve_includes(raw, p.parent)
        prompts.append((p.stem, resolved))

    content = tex_path.read_text()
    chunks = split_into_chunks(content)

    # Processable = not blank, not preamble/postamble
    processable = [
        (i, c) for i, c in enumerate(chunks) if not c.is_blank and not c.is_preamble
    ]
    total = len(processable)

    prompt_names = ", ".join(name for name, _ in prompts)
    print(f"\n{BOLD}Proofreading: {tex_path.name}{RESET}")
    print(f"  Prompts: {prompt_names}")
    print(f"  Model: {args.model}")
    print(f"  {len(chunks)} total chunks, {total} to process\n")

    if args.verbose:
        for i, chunk in enumerate(chunks):
            kind = "BLANK" if chunk.is_blank else "PRE" if chunk.is_preamble else "    "
            preview = chunk.text[:60].replace("\n", " ")
            print(
                f"  [{i + 1:3d}] {kind}  L{chunk.start_line}-{chunk.end_line}  {DIM}{preview}{RESET}"
            )
        print()

    if total == 0:
        print("  Nothing to process.")
        return

    if not args.dry_run:
        backup_file(tex_path)

    client = None if args.dry_run else get_openai_client()

    accept_all = False

    for list_idx, (chunk_idx, chunk) in enumerate(processable):
        display_idx = list_idx + 1

        if display_idx < args.skip_to:
            print_progress(display_idx, total, "skipped")
            continue

        loc = f"L{chunk.start_line}-{chunk.end_line}"

        # Token warning
        est_tokens = len(chunk.text) // 4
        if est_tokens > TOKEN_WARN:
            end_progress()
            print(
                f"{YELLOW}  Warning: Chunk {display_idx} is ~{est_tokens} tokens.{RESET}",
                file=sys.stderr,
            )

        if args.dry_run:
            print_progress(display_idx, total, f"{loc}  dry run")
            if args.verbose:
                end_progress()
                print(
                    f"  {BOLD}{CYAN}--- Chunk {display_idx}/{total}  {loc} ---{RESET}"
                )
                print(f"  {DIM}(dry run){RESET}")
            continue

        # Chain prompts: each transforms the previous output
        current_text = chunk.text
        for prompt_name, instructions in prompts:
            print_progress(display_idx, total, f"{loc}  {prompt_name}")
            result = call_llm(
                client,
                args.model,
                current_text,
                instructions,
                no_thinking=args.no_thinking,
            )
            if result is None:
                end_progress()
                print(
                    f"  Skipping prompt '{prompt_name}' for chunk {display_idx} (LLM error).",
                    file=sys.stderr,
                )
            else:
                current_text = result

        corrected = current_text

        # No changes — skip
        if unwrap_tex(corrected) == unwrap_tex(chunk.text):
            print_progress(display_idx, total, f"{loc}  no changes")
            continue

        if accept_all:
            diff_out = render_word_diff(chunk.text, corrected)
            if diff_out:
                end_progress()
                print(
                    f"\n{BOLD}{CYAN}--- Chunk {display_idx}/{total}  {loc} ---{RESET}"
                )
                print(diff_out)
                print("  -> Auto-accepted")
            chunk.text = corrected
            write_file(tex_path, chunks)
            print_progress(display_idx, total, f"{loc}  accepted")
            continue

        end_progress()
        decision, final_text = review_change(
            display_idx, total, chunk, corrected, args.dry_run
        )

        if decision in ("accept", "accept_all"):
            chunk.text = final_text
            write_file(tex_path, chunks)
            if decision == "accept_all":
                accept_all = True
        elif decision == "quit":
            print(f"\n  Progress saved. Resume with: --skip-to {display_idx}")
            return

    end_progress()
    print(f"\n{BOLD}{GREEN}Done!{RESET} Processed {total} chunks.")


if __name__ == "__main__":
    main()
