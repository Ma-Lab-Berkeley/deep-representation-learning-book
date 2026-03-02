#!/usr/bin/env python3
"""Inject MathJax macro definitions into make4ht HTML output.

make4ht with the `mathjax` option embeds a minimal MathJax config:
    <script>window.MathJax = { tex: { tags: "ams", }, }; </script>

This script replaces that config with one that includes all custom macros
from the book's .sty files (parsed by generate_macros.py).

MathJax v3 macro format:
    macros: {
        R: "\\\\mathbb{R}",              // no args
        bp: ["\\\\left(#1\\\\right)", 1], // 1 arg
    }

Usage: python inject_mathjax_macros.py <build_dir> <macros.json>
"""

import json
import re
import sys
from pathlib import Path


def build_mathjax_config(macros: dict, star_variants: dict) -> str:
    """Build a MathJax v3 config script from parsed macros.

    Args:
        macros: dict mapping "\\cmd" → "expansion" (with #1, #2, etc.)
        star_variants: dict mapping "\\cmd" → {"expansion": ..., "nargs": N}
    """
    mj_macros: dict[str, str | list] = {}

    for cmd, expansion in macros.items():
        # Strip leading backslash for MathJax key
        name = cmd.lstrip("\\")
        if not name:
            continue

        # Count args (#1, #2, ...) in expansion
        arg_nums = re.findall(r"#(\d)", expansion)
        nargs = max(int(n) for n in arg_nums) if arg_nums else 0

        # Escape backslashes for JS string literal
        js_expansion = expansion.replace("\\", "\\\\")

        if nargs > 0:
            mj_macros[name] = [js_expansion, nargs]
        else:
            mj_macros[name] = js_expansion

    # Star variants: register the starred expansion as a separate macro
    # e.g., \abs* → register as \absstar or handle via MathJax's star support
    # MathJax v3 supports star variants natively when macros are defined,
    # but for safety we also add the starred forms as explicit macros
    # with names like "abs*" — MathJax doesn't support this in macro keys,
    # so we skip star variants (MathJax will use the non-starred form).

    # Commands MathJax doesn't provide natively — define as macros.
    # \bm → \boldsymbol (from boldsymbol package)
    if "bm" not in mj_macros:
        mj_macros["bm"] = ["\\\\boldsymbol{#1}", 1]
    # \textsc — MathJax has no small-caps; approximate with \text
    if "textsc" not in mj_macros:
        mj_macros["textsc"] = ["\\\\text{#1}", 1]
    # \scalebox — no-op, just pass through the content (2nd arg)
    if "scalebox" not in mj_macros:
        mj_macros["scalebox"] = ["#2", 2]
    # \em, \bf — old-style font declarations; no-op inside MathJax \text/\mbox
    if "em" not in mj_macros:
        mj_macros["em"] = ""
    if "bf" not in mj_macros:
        mj_macros["bf"] = ""

    # Build the JS config
    macros_json = json.dumps(mj_macros, ensure_ascii=False)

    config = f"""window.MathJax = {{
  loader: {{load: ['[tex]/boldsymbol', '[tex]/mathtools', '[tex]/ams', '[tex]/centernot']}},
  tex: {{
    tags: "ams",
    packages: {{'[+]': ['boldsymbol', 'mathtools', 'ams', 'centernot']}},
    macros: {macros_json}
  }}
}};"""

    return config


def preprocess_star_variants(html: str, star_variants: dict) -> str:
    r"""Expand star-variant macros in-place in the HTML.

    MathJax doesn't support \cmd*{...} syntax in its macro system.
    We expand them before MathJax processes the page.
    e.g. \abs*{x} → \left\lvert x\right\rvert
    """
    if not star_variants:
        return html

    # Multiple passes to handle chained expansions (e.g. \ipip* → \ip* → expanded)
    for _pass in range(3):
        prev = html
        html = _expand_star_variants_once(html, star_variants)
        if html == prev:
            break

    return html


def _expand_star_variants_once(html: str, star_variants: dict) -> str:
    """Single pass of star variant expansion."""
    for cmd, info in star_variants.items():
        expansion = info["expansion"]
        nargs = info["nargs"]

        # make4ht may insert whitespace between \cmd and *, e.g. \abs *{...}
        # Use regex to match \cmd followed by optional whitespace then *
        pattern = re.compile(re.escape(cmd) + r"\s*\*")

        result = []
        i = 0
        while i < len(html):
            m = pattern.search(html, i)
            if m is None:
                result.append(html[i:])
                break

            result.append(html[i:m.start()])
            j = m.end()

            # Extract braced arguments
            args = []
            success = True
            for _ in range(nargs):
                # Skip whitespace
                while j < len(html) and html[j] in " \t\n\r":
                    j += 1
                if j >= len(html) or html[j] != "{":
                    success = False
                    break
                end = _find_matching_brace(html, j)
                if end == -1:
                    success = False
                    break
                args.append(html[j + 1 : end])
                j = end + 1

            if success:
                # Substitute #1, #2, etc. in the expansion.
                # First, insert a space between \command and #N placeholders
                # so that arguments starting with a letter don't concatenate
                # with the preceding command name. e.g. \Vert#1 with #1=h(...)
                # would produce \Verth (undefined); adding a space gives \Vert h(...)
                expanded = re.sub(r"(\\[a-zA-Z]+)(#\d)", r"\1 \2", expansion)
                for idx, arg in enumerate(args):
                    expanded = expanded.replace(f"#{idx + 1}", arg)
                result.append(expanded)
                i = j
            else:
                # Can't parse — keep original
                result.append(m.group())
                i = m.end()

        html = "".join(result)

    return html


def preprocess_scalebox(html: str) -> str:
    r"""Remove \scalebox{...}{...} from math, keeping only the content.

    Handles nested \( \) delimiters inside the content argument, e.g.:
        \scalebox{0.8}{\(\odot\)}  →  \odot
        \scalebox{0.8}{\odot}      →  \odot
    """
    result = []
    i = 0
    tag = "\\scalebox"
    while i < len(html):
        pos = html.find(tag, i)
        if pos == -1:
            result.append(html[i:])
            break

        result.append(html[i:pos])

        # Skip past \scalebox
        j = pos + len(tag)

        # Skip whitespace
        while j < len(html) and html[j] in " \t\n\r":
            j += 1

        # Skip first arg {scale factor}
        if j < len(html) and html[j] == "{":
            end = _find_matching_brace(html, j)
            if end == -1:
                result.append(html[pos:j])
                i = j
                continue
            j = end + 1
        else:
            # No brace — not a real \scalebox, keep it
            result.append(tag)
            i = pos + len(tag)
            continue

        # Skip whitespace
        while j < len(html) and html[j] in " \t\n\r":
            j += 1

        # Extract second arg {content}
        if j < len(html) and html[j] == "{":
            end = _find_matching_brace(html, j)
            if end == -1:
                result.append(html[pos:j])
                i = j
                continue
            content = html[j + 1 : end]
            j = end + 1

            # Strip nested \( \) or \[ \] delimiters from the content
            content = re.sub(r"^\\\(|\\\)$", "", content.strip())
            content = re.sub(r"^\\\[|\\\]$", "", content.strip())

            result.append(content)
            i = j
        else:
            result.append(tag)
            i = pos + len(tag)

    return "".join(result)


def _find_matching_brace(s: str, start: int) -> int:
    """Find matching } for s[start] == '{', respecting nesting."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{" and (i == 0 or s[i - 1] != "\\"):
            depth += 1
        elif s[i] == "}" and (i == 0 or s[i - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return i
    return -1


def inject_into_html(html: str, config_script: str, star_variants: dict) -> str:
    """Replace the existing MathJax config <script> with our enhanced one."""
    # Pre-process: expand star variants (\abs*{x} etc.) before MathJax sees them
    html = preprocess_star_variants(html, star_variants)
    # Pre-process: strip \scalebox before MathJax sees it
    html = preprocess_scalebox(html)

    # Pattern: <script>window.MathJax = { ... }; </script>
    pattern = r"<script[^>]*>\s*window\.MathJax\s*=[\s\S]*?</script>"
    replacement = f"<script>{config_script}</script>"

    new_html, count = re.subn(pattern, replacement, html, count=1)
    if count == 0:
        # No existing config found — inject before the MathJax loader script
        loader_pattern = r'(<script[^>]*src="[^"]*mathjax[^"]*"[^>]*></script>)'
        new_html, count = re.subn(
            loader_pattern,
            replacement + "\n" + r"\1",
            html,
            count=1,
        )
        if count == 0:
            # No MathJax at all — inject before </head>
            head_end = html.find("</head>")
            if head_end != -1:
                mathjax_loader = (
                    '<script async id="MathJax-script" '
                    'src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js">'
                    "</script>"
                )
                new_html = (
                    html[:head_end]
                    + f"<script>{config_script}</script>\n"
                    + mathjax_loader
                    + "\n"
                    + html[head_end:]
                )

    return new_html


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <build_dir> <macros.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    build_dir = Path(sys.argv[1])
    macros_path = Path(sys.argv[2])

    # Load macros
    data = json.loads(macros_path.read_text(encoding="utf-8"))
    macros = data.get("macros", {})
    star_variants = data.get("starVariants", {})

    print(f"Loaded {len(macros)} macros, {len(star_variants)} star variants")

    # Build MathJax config
    config_script = build_mathjax_config(macros, star_variants)

    # Find and process HTML files
    html_files = sorted(build_dir.glob("*.html"))
    print(f"Processing {len(html_files)} HTML files...")

    for html_file in html_files:
        html = html_file.read_text(encoding="utf-8")

        # Skip files with no MathJax or math content
        if "mathjax" not in html.lower() and "\\(" not in html and "\\[" not in html:
            print(f"  {html_file.name}: no math, skipping")
            continue

        new_html = inject_into_html(html, config_script, star_variants)
        html_file.write_text(new_html, encoding="utf-8")
        print(f"  {html_file.name}: OK")

    print("MathJax macro injection complete.")


if __name__ == "__main__":
    main()
