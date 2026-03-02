#!/usr/bin/env python3
"""Parse .sty files and generate mathjax-macros.json for MathJax rendering.

Handles:
  1. Simple macros: \\NewDocumentCommand{\\R}{}{\\mathbb{R}}
  2. Macros with args: \\NewDocumentCommand{\\bp}{m}{\\left(#1\\right)}
  3. \\DeclareMathOperator(*): \\DeclareMathOperator*{\\argmin}{arg\\ min}
  4. \\forcsvlist/\\csdef metaprogramming (font maker patterns)
  5. Star variants (\\IfBooleanTF): non-starred form as main macro
  6. \\newcommand / \\renewcommand forms
  7. Fixups: \\mleft→\\left, \\mright→\\right, strip \\ensuremath, \\given→\\mid

Output: mathjax-macros.json with {"macros": {...}, "starVariants": {...}}
"""

import json
import re
import sys
from pathlib import Path

# Letters for font maker expansion
LATIN_LOWER = list("abcdefghijklmnopqrstuvwxyz")
LATIN_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
GREEK_LOWER = [
    "alpha", "beta", "gamma", "delta", "epsilon", "eps", "phi", "zeta",
    "eta", "theta", "vartheta", "kappa", "lambda", "mu", "nu", "xi",
    "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon",
    "varphi", "chi", "psi", "omega",
]
GREEK_UPPER = [
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
]


def apply_fixups(expansion: str) -> str:
    """Apply KaTeX compatibility fixups to a macro expansion."""
    s = expansion
    s = s.replace("\\mleft", "\\left")
    s = s.replace("\\mright", "\\right")
    # Strip \ensuremath{...} → ...
    s = re.sub(r"\\ensuremath\{([^}]*)\}", r"\1", s)
    return s


def count_args(argspec: str) -> int:
    """Count number of mandatory args in xparse argspec (each 'm')."""
    return argspec.lower().count("m")


def parse_newdocumentcommand(line: str, macros: dict, star_variants: dict):
    """Parse \\NewDocumentCommand or \\RenewDocumentCommand lines."""
    # Match: \NewDocumentCommand{\name}{argspec}{body}
    # or \RenewDocumentCommand{\name}{argspec}{body}
    # Handle multi-line by requiring the body to have balanced braces
    pattern = r"\\(?:New|Renew|Provide)DocumentCommand\{(\\[a-zA-Z]+)\}\{([^}]*)\}"
    m = re.search(pattern, line)
    if not m:
        return

    cmd_name = m.group(1)  # e.g., \abs
    argspec = m.group(2)    # e.g., sm

    # Extract body: everything after the argspec closing brace, inside { }
    rest = line[m.end():]
    body = extract_braced(rest)
    if body is None:
        return

    body = apply_fixups(body)

    # Check for star variant (argspec starts with 's')
    if argspec.startswith("s"):
        remaining_args = argspec[1:]
        n_args = count_args(remaining_args)
        # Parse IfBooleanTF to get starred and unstarred forms
        starred, unstarred = parse_if_boolean(body, n_args)
        if starred is not None and unstarred is not None:
            # Register unstarred form as the main macro
            if n_args > 0:
                macros[cmd_name] = unstarred
            else:
                macros[cmd_name] = unstarred
            # Register starred expansion
            star_variants[cmd_name] = {"expansion": starred, "nargs": n_args}
        else:
            # Fallback: just use the body with adjusted arg numbers
            # Skip the star arg (shift #2→#1, #3→#2, etc.)
            adjusted = shift_args(body)
            if n_args > 0:
                macros[cmd_name] = adjusted
            else:
                macros[cmd_name] = adjusted
    else:
        n_args = count_args(argspec)
        if n_args > 0:
            macros[cmd_name] = body
        else:
            macros[cmd_name] = body


def parse_if_boolean(body: str, n_args: int):
    """Parse \\IfBooleanTF{#1}{starred_body}{unstarred_body}.
    Returns (starred_expansion, unstarred_expansion) with args shifted.
    """
    m = re.search(r"\\IfBooleanTF\{#1\}", body)
    if not m:
        return None, None

    rest = body[m.end():]
    starred = extract_braced(rest)
    if starred is None:
        return None, None
    rest = rest[len(starred) + 2:]  # +2 for the braces
    unstarred = extract_braced(rest)
    if unstarred is None:
        return None, None

    # Shift arg numbers: #2→#1, #3→#2, etc.
    starred = shift_args(starred)
    unstarred = shift_args(unstarred)
    starred = apply_fixups(starred)
    unstarred = apply_fixups(unstarred)
    return starred, unstarred


def shift_args(s: str) -> str:
    """Shift #2→#1, #3→#2, etc. (for removing the star boolean arg)."""
    def replace_arg(m):
        n = int(m.group(1))
        if n <= 1:
            return m.group(0)
        return f"#{n - 1}"
    return re.sub(r"#(\d)", replace_arg, s)


def extract_braced(s: str) -> str | None:
    """Extract content of first { ... } group with balanced braces."""
    s = s.lstrip()
    if not s or s[0] != "{":
        return None
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{" and (i == 0 or s[i-1] != "\\"):
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}" and (i == 0 or s[i-1] != "\\"):
            depth -= 1
            if depth == 0:
                return s[start:i]
    return None


def parse_newcommand(line: str, macros: dict):
    """Parse \\newcommand{\\name}[nargs]{body} or \\newcommand{\\name}{body}."""
    # With optional arg count
    m = re.match(r"\\(?:new|renew|provide)command\{?(\\[a-zA-Z]+)\}?\s*(?:\[(\d+)\])?\s*\{", line)
    if not m:
        return
    cmd_name = m.group(1)
    n_args = int(m.group(2)) if m.group(2) else 0
    rest = line[m.start():]
    # Find the body after the arg spec
    after_pattern = re.search(r"\\(?:new|renew|provide)command\{?\\[a-zA-Z]+\}?\s*(?:\[\d+\])?\s*", rest)
    if not after_pattern:
        return
    body_str = rest[after_pattern.end():]
    body = extract_braced("{" + body_str) if not body_str.startswith("{") else extract_braced(body_str)
    if body is None:
        # Try to get from the original line
        brace_start = line.find("{", m.end() - 1)
        if brace_start >= 0:
            body = extract_braced(line[brace_start:])
    if body is None:
        return
    body = apply_fixups(body)
    macros[cmd_name] = body


def parse_declare_math_operator(line: str, macros: dict):
    """Parse \\DeclareMathOperator(*){\\name}{text}."""
    m = re.match(r"\\DeclareMathOperator(\*?)\{(\\[a-zA-Z]+)\}\{([^}]*)\}", line)
    if not m:
        return
    is_star = m.group(1) == "*"
    cmd_name = m.group(2)
    text = m.group(3)
    # Normalize spacing: "arg\ min" → "arg\\,min"
    text = re.sub(r"\\\s+", r"\\,", text)
    op = "\\operatorname*" if is_star else "\\operatorname"
    macros[cmd_name] = f"{op}{{{text}}}"


def expand_font_makers(macros: dict):
    """Expand the \\forcsvlist/\\csdef font maker patterns.

    Uses the bm fallback branch (non-unicode-math), which is what
    the book uses with pdflatex/xelatex without unicode-math.
    """
    # Latin lowercase: \fr_, \sf_, \rm_, \bf_, \v_
    for letter in LATIN_LOWER:
        macros[f"\\fr{letter}"] = f"\\mathfrak{{{letter}}}"
        macros[f"\\sf{letter}"] = f"\\mathsf{{{letter}}}"
        macros[f"\\rm{letter}"] = f"\\mathrm{{{letter}}}"
        macros[f"\\bf{letter}"] = f"\\mathbf{{{letter}}}"
        macros[f"\\v{letter}"] = f"\\bm{{{letter}}}"

    # Latin uppercase: \fr_, \sf_, \rm_, \bf_, \v_, \b_, \c_, \sc_
    for letter in LATIN_UPPER:
        macros[f"\\fr{letter}"] = f"\\mathfrak{{{letter}}}"
        macros[f"\\sf{letter}"] = f"\\mathsf{{{letter}}}"
        macros[f"\\rm{letter}"] = f"\\mathrm{{{letter}}}"
        macros[f"\\bf{letter}"] = f"\\mathbf{{{letter}}}"
        macros[f"\\v{letter}"] = f"\\bm{{{letter}}}"
        macros[f"\\b{letter}"] = f"\\mathbb{{{letter}}}"
        macros[f"\\c{letter}"] = f"\\mathcal{{{letter}}}"
        macros[f"\\sc{letter}"] = f"\\mathscr{{{letter}}}"

    # Greek lowercase: \v{name}
    for name in GREEK_LOWER:
        macros[f"\\v{name}"] = f"\\bm{{\\{name}}}"

    # Greek uppercase: \v{Name}
    for name in GREEK_UPPER:
        macros[f"\\v{name}"] = f"\\bm{{\\{name}}}"

    # Special cases
    macros["\\vzero"] = "\\bm{0}"
    macros["\\vone"] = "\\bm{1}"


def parse_sty_file(filepath: Path, macros: dict, star_variants: dict):
    """Parse a single .sty file for macro definitions."""
    text = filepath.read_text(encoding="utf-8")

    # Join continuation lines: lines ending with %, and brace-unbalanced lines
    lines = text.split("\n")
    joined_lines = []
    buf = ""
    brace_depth = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith("%") and not stripped.endswith("\\%"):
            buf += stripped[:-1]
            # Update brace depth
            for ch in stripped[:-1]:
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
        else:
            buf += stripped
            # Update brace depth
            for ch in stripped:
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
            # Only emit the line when braces are balanced
            if brace_depth <= 0:
                joined_lines.append(buf)
                buf = ""
                brace_depth = 0
            # else: keep accumulating
    if buf:
        joined_lines.append(buf)

    for line in joined_lines:
        # Skip comments
        line_stripped = line.lstrip()
        if line_stripped.startswith("%"):
            continue

        # Remove inline comments (not preceded by \)
        line_clean = re.sub(r"(?<!\\)%.*$", "", line_stripped)

        if "\\NewDocumentCommand" in line_clean or "\\RenewDocumentCommand" in line_clean or "\\ProvideDocumentCommand" in line_clean:
            parse_newdocumentcommand(line_clean, macros, star_variants)
        elif "\\DeclareMathOperator" in line_clean:
            parse_declare_math_operator(line_clean, macros)
        elif re.search(r"\\(?:new|renew|provide)command", line_clean):
            parse_newcommand(line_clean, macros)


def add_extra_fixups(macros: dict):
    """Add standalone fixup macros that MathJax needs.

    These are macros that use LaTeX internals (etoolbox, counters, etc.)
    that MathJax can't process natively.
    """
    # \given → \mid (MathJax doesn't have etoolbox's \currentgrouptype)
    macros["\\given"] = "\\mid"
    # \wrt → \mid\!\mid
    macros["\\wrt"] = "\\mid\\!\\mid"
    # \numberthis / \labelthis — no-op (MathJax handles equation numbering via AMS)
    macros["\\numberthis"] = ""
    macros["\\labelthis"] = ""
    # \vocab — bold text
    macros["\\vocab"] = "\\textbf{#1}"
    # \ind — double perpendicular
    macros["\\ind"] = "\\perp\\!\\!\\!\\perp"
    # \simiid
    macros["\\iid"] = "\\mathrm{i.i.d.}"
    macros["\\simiid"] = "\\stackrel{\\mathrm{i.i.d.}}{\\sim}"
    macros["\\equid"] = "\\stackrel{\\mathrm{d}}{=}"
    # \indvar
    macros["\\indvar"] = "\\mathbf{1}"
    # \adj
    macros["\\adj"] = "^{\\ast}"
    # \mmid
    macros["\\mmid"] = "\\;\\|\\;"
    # Operator name helpers (these take 1 arg, expand to \operatorname{\mathXX{#1}})
    macros["\\operatornamerm"] = "\\operatorname{\\mathrm{#1}}"
    macros["\\operatornamebb"] = "\\operatorname{\\mathbb{#1}}"
    macros["\\operatornormbf"] = "\\operatorname{\\mathbf{#1}}"
    macros["\\operatornamett"] = "\\operatorname{\\mathtt{#1}}"
    macros["\\operatornormcal"] = "\\operatorname{\\mathcal{#1}}"
    macros["\\operatornamesf"] = "\\operatorname{\\mathsf{#1}}"
    macros["\\operatornamescr"] = "\\operatorname{\\mathscr{#1}}"
    # \casework
    macros["\\casework"] = "\\begin{cases}#1\\end{cases}"
    # \mat
    macros["\\mat"] = "\\begin{bmatrix}#1\\end{bmatrix}"
    # \qq, \qm, \qc, etc.
    macros["\\qq"] = "\\quad\\text{#1}\\quad"
    macros["\\qm"] = "\\quad #1 \\quad"
    macros["\\qc"] = ",\\quad"
    macros["\\qqq"] = "\\qquad\\text{#1}\\qquad"
    macros["\\qqm"] = "\\qquad #1 \\qquad"
    macros["\\qqc"] = ",\\qquad"
    # \posteriorsample
    macros["\\posteriorsample"] = "\\bm{x}^{\\mathrm{c}}"
    macros["\\spcdot"] = "\\,\\cdot\\,"
    # \phi remapping
    macros["\\plainphi"] = "\\phi"
    # Annoying names (text mode)
    macros["\\Holder"] = "\\text{H\\\"older}"
    macros["\\Levy"] = "\\text{L\\'evy}"
    macros["\\Ito"] = "\\text{It\\^o}"
    macros["\\Caratheodory"] = "\\text{Carath\\'eodory}"
    macros["\\Erdos"] = "\\text{Erd\\\"os}"
    macros["\\Renyi"] = "\\text{R\\'enyi}"
    macros["\\Frechet"] = "\\text{Fr\\'echet}"
    # Book-macros extras that use scalerel (not in KaTeX)
    macros["\\odotsc"] = "\\odot"
    macros["\\odivsc"] = "\\oslash"
    macros["\\otimessc"] = "\\otimes"
    macros["\\hada"] = "\\odot"
    macros["\\kron"] = "\\otimes"
    macros["\\haddiv"] = "\\oslash"
    # \Eta
    macros["\\Eta"] = "\\mathrm{H}"
    # \circm
    macros["\\circm"] = "\\mathsf{circ}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output.json> [sty_or_tex_file_or_dir ...]", file=sys.stderr)
        sys.exit(1)

    output_path = Path(sys.argv[1])

    # Collect input files: accept .sty, .tex files, and directories (recursively glob .tex)
    if len(sys.argv) > 2:
        inputs = [Path(f) for f in sys.argv[2:]]
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        inputs = [
            repo_root / "math-macros.sty",
            repo_root / "book-macros.sty",
            repo_root / "chapters",
        ]

    sty_files: list[Path] = []
    tex_files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            tex_files.extend(sorted(p.rglob("*.tex")))
        elif p.suffix == ".sty":
            sty_files.append(p)
        elif p.suffix == ".tex":
            tex_files.append(p)
        else:
            sty_files.append(p)  # treat unknown as sty-like

    macros: dict[str, str] = {}
    star_variants: dict[str, dict] = {}

    # Parse .sty files first (authoritative macro definitions)
    for sty_file in sty_files:
        if sty_file.exists():
            print(f"Parsing {sty_file.name}...")
            parse_sty_file(sty_file, macros, star_variants)
        else:
            print(f"Warning: {sty_file} not found, skipping", file=sys.stderr)

    # Parse .tex files for local \newcommand definitions
    if tex_files:
        print(f"Scanning {len(tex_files)} .tex files for local macros...")
        tex_count_before = len(macros)
        for tex_file in tex_files:
            if tex_file.exists():
                parse_sty_file(tex_file, macros, star_variants)
        print(f"  Found {len(macros) - tex_count_before} additional macros from .tex files")

    # Expand font maker metaprogramming
    print("Expanding font maker metaprogramming...")
    expand_font_makers(macros)

    # Add extra fixups
    print("Adding MathJax fixups...")
    add_extra_fixups(macros)

    result = {
        "macros": macros,
        "starVariants": star_variants,
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(macros)} macros + {len(star_variants)} star variants → {output_path}")


if __name__ == "__main__":
    main()
