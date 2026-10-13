#!/usr/bin/env python3
"""Validate final TeX logs and generated book artifacts without network access."""

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment


REQUIRED_HTML = tuple(f"Ch{i}.html" for i in range(1, 10)) + (
    "Chx1.html", "Chx2.html", "A1.html", "A2.html",
)
FATAL_TEX = re.compile(
    r"^!|(?:LaTeX|Package\s+\S+|Class\s+\S+)\s+Error:|"
    r"Emergency stop|Fatal error|TeX capacity exceeded|"
    r"Undefined control sequence|Missing \$ inserted|No pages of output\.",
    re.IGNORECASE | re.MULTILINE,
)
REFERENCE_WARNINGS = (
    (r"(?:Reference|Citation)\s+.{0,600}?\bundefined\b", "undefined reference/citation"),
    (r"There were undefined (?:references|citations)", "undefined references/citations"),
    (r"(?:Label\s+.{0,600}?multiply defined|multiply[- ]defined labels)", "duplicate label"),
    (r"Label\(s\) may have changed", "labels need another TeX pass"),
    (r"Rerun to get (?:cross-references|citations|outlines) right", "rerun required"),
    (r"(?:Please\s+)?(?:\(re\)run|rerun)\s+(?:Biber|BibTeX|LaTeX)", "rerun required"),
)
UNCONVERTED_REF = re.compile(r"\\(?:ref|eqref|cref|Cref)\*?\s*\{")


def read_nonempty(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read required file ({exc})") from exc
    if not text.strip():
        raise ValueError(f"{path}: required file is empty")
    return text


def validate_tex_log(path: Path, references: bool = False,
                     biber_log: Path | None = None) -> list[str]:
    errors = []
    try:
        text = read_nonempty(path)
        for match in FATAL_TEX.finditer(text):
            errors.append(f"{path}: TeX error: {match.group(0)}")
        if references:
            # TeX wraps warnings and may prefix continuation lines with a package name.
            flat = re.sub(r"\n\s*\([\w.-]+\)\s*", " ", text)
            flat = re.sub(r"\s+", " ", flat)
            for pattern, description in REFERENCE_WARNINGS:
                if re.search(pattern, flat, re.IGNORECASE):
                    errors.append(f"{path}: {description}")
    except ValueError as exc:
        errors.append(str(exc))
    if biber_log is not None:
        try:
            text = re.sub(r"\s+", " ", read_nonempty(biber_log))
            if re.search(r"\bERROR\b|I didn't find (?:a )?database entry|"
                         r"not found in (?:the )?database", text, re.IGNORECASE):
                errors.append(f"{biber_log}: Biber error or missing bibliography entry")
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def validate_search_index(path: Path) -> list[str]:
    try:
        payload = json.loads(read_nonempty(path))
        if not isinstance(payload, dict):
            raise ValueError("expected an object")
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("entries must be a nonempty list")
        if type(payload.get("count")) is not int or payload["count"] != len(entries):
            raise ValueError("count must match the number of entries")
        if not isinstance(payload.get("generated"), str):
            raise ValueError("generated must be an ISO timestamp")
        datetime.fromisoformat(payload["generated"].replace("Z", "+00:00"))
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or any(
                not isinstance(entry.get(key), str) or not entry[key].strip()
                for key in ("page", "href", "title")
            ) or not isinstance(entry.get("snippet"), str):
                raise ValueError(f"entry {i} needs page, href, title and snippet strings")
            href = urlsplit(entry["href"])
            if href.scheme or href.netloc or not href.path.endswith(".html") or not href.fragment:
                raise ValueError(f"entry {i} needs a local HTML section href")
    except (ValueError, TypeError) as exc:
        return [f"{path}: invalid search index: {exc}"]
    return []


def validate_html(output_dir: Path, pdf: Path | None = None) -> list[str]:
    errors = []
    bibliography_found = False
    filenames = set(REQUIRED_HTML)
    filenames.update(path.name for path in output_dir.glob("*.html")
                     if re.fullmatch(r"(?:Chx?\d+|Ax?\d+|bib)\.html", path.name))
    for filename in sorted(filenames):
        path = output_dir / filename
        try:
            soup = BeautifulSoup(read_nonempty(path), "html.parser")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for node in soup.find_all(string=lambda value: isinstance(value, Comment)):
            node.extract()
        for node in soup.select("script, style, pre, code, [hidden], [aria-hidden='true']"):
            node.decompose()
        body = soup.body
        if soup.html is None or body is None or not body.find(re.compile(r"^h[1-6]$")):
            errors.append(f"{path}: missing HTML document, body or heading")
            continue
        # make4ht may put references in a front-matter page or the last appendix.
        if any(entry.get_text(strip=True) for entry in body.select("dl.thebibliography dd")):
            bibliography_found = True
        content = body.select("p, li, dd, td, .mathjax-env, .newtheorem")
        if not any(node.get_text(strip=True) for node in content):
            errors.append(f"{path}: no chapter content beyond headings")
        visible = body.get_text(" ", strip=True)
        if "??" in visible:
            errors.append(f"{path}: unresolved reference marker '??'")
        if UNCONVERTED_REF.search(visible):
            errors.append(f"{path}: unconverted LaTeX reference command")
    if not bibliography_found:
        errors.append(f"{output_dir}: missing or empty bibliography in generated pages")
    errors.extend(validate_search_index(output_dir / "search-index.json"))
    if pdf is not None:
        try:
            with pdf.open("rb") as stream:
                if stream.read(5) != b"%PDF-" or pdf.stat().st_size <= 5:
                    errors.append(f"{pdf}: missing PDF signature or empty PDF")
        except OSError as exc:
            errors.append(f"{pdf}: cannot read required PDF ({exc})")
    return errors


def write_link_manifest(output_dir: Path) -> None:
    """Expose validated search destinations to the CI link checker, not the website."""
    entries = json.loads(read_nonempty(output_dir / "search-index.json"))["entries"]
    links = "\n".join(
        f'<li><a href="{escape(entry["href"], quote=True)}">{escape(entry["title"])}</a></li>'
        for entry in entries
    )
    (output_dir / "_ci-search-links.html").write_text(
        '<!DOCTYPE html><html><head><title>CI search link targets</title></head>'
        f'<body><ul>\n{links}\n</ul></body></html>\n', encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tex = commands.add_parser("tex-log", help="check a final TeX compilation log")
    tex.add_argument("path", type=Path)
    tex.add_argument("--references", action="store_true", help="reject unresolved references and reruns")
    tex.add_argument("--biber-log", type=Path)
    html = commands.add_parser("html", help="check a generated full-book website")
    html.add_argument("output_dir", type=Path)
    html.add_argument("--pdf", type=Path, help="also require a generated PDF signature")
    html.add_argument("--write-link-manifest", action="store_true",
                      help="after validation, write a CI-only search link helper (not book output)")
    args = parser.parse_args(argv)
    if args.command == "tex-log":
        errors = validate_tex_log(args.path, args.references, args.biber_log)
    else:
        errors = validate_html(args.output_dir, args.pdf)
        if not errors and args.write_link_manifest:
            try:
                write_link_manifest(args.output_dir)
            except (OSError, ValueError) as exc:
                errors.append(f"{args.output_dir}: cannot write CI search link manifest ({exc})")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Build validation passed: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
