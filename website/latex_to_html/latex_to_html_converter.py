#!/usr/bin/env python3
"""
Portable LaTeX → HTML converter and post-processor built around LaTeXML.

Key capabilities:
- Drive LaTeXML to convert a main `.tex` file into one or more `.html` files
- Ensure shared assets are present (chapter.css, chapter.js)
- Post-process generated HTML to inject viewport/meta, CSS/JS, topbar, and navigation
- Build a compact search index over chapter/appendix content

CLI usage:
  python latex_to_html_converter.py <source_filepath> [output_dir]
  python latex_to_html_converter.py <source_filepath> --postprocess-only [output_dir]

Requirements:
- LaTeXML installed (latexmlc available) unless using --postprocess-only
- Python packages: beautifulsoup4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, cast

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag
except ImportError as import_error:
    print("Error: beautifulsoup4 is required. Install with: pip install beautifulsoup4")
    raise SystemExit(1) from import_error


# ------------------------------
# Configuration and constants
# ------------------------------

# Fixed directory structure paths
_CONVERTER_FILE = Path(__file__).resolve()
CONVERTER_DIR = _CONVERTER_FILE.parent  # website/latex_to_html/
WEBSITE_ROOT = CONVERTER_DIR.parent # website/latex_to_html/ -> website/
PROJECT_ROOT = WEBSITE_ROOT.parent # website -> repo_root/
HTML_OUTPUT_DIR = WEBSITE_ROOT / "html"  # website/html/

LATEXML_TIMEOUT_SEC: int = 0

# Example lines in LaTeXML log we want to capture as missing packages
MISSING_PACKAGE_RE = re.compile(
    r"^Warning:missing_file:(\S+)\s(?:Can't\sfind\s(package|binding for class))?",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class ResourceConfig:
    """Resource bundles injected into <head>."""

    css_urls: Tuple[str, ...] = (
        "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
        "https://cdn.jsdelivr.net/gh/arXiv/arxiv-browse@master/arxiv/browse/static/css/ar5iv.0.8.2.min.css",
        "https://cdn.jsdelivr.net/gh/arXiv/arxiv-browse@master/arxiv/browse/static/css/ar5iv-fonts.0.8.2.min.css",
        "https://cdn.jsdelivr.net/gh/arXiv/arxiv-browse@master/arxiv/browse/static/css/latexml_styles.0.8.2.css",
    )
    js_urls: Tuple[str, ...] = (
        "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.3.3/html2canvas.min.js",
    )


@dataclass(frozen=True)
class LaTeXMLConfig:
    """Conversion knobs for LaTeXML execution."""

    timeout_sec: int = LATEXML_TIMEOUT_SEC
    split_at: str = "chapter"
    format: str = "html5"
    navigation_toc: str = "context"
    whats_in: str = "directory"
    preload: str = (
        "[nobibtex,nobreakuntex,localrawstyles,mathlexemes,"
        "magnify=1.2,zoomout=1.2,tokenlimit=2499999999,iflimit=35999999,"
        "absorblimit=1299999,pushbacklimit=599999999]latexml.sty"
    )
    pmml: bool = True
    mathtex: bool = True
    noinvisibletimes: bool = True
    nodefaultresources: bool = True
    extra_paths: Tuple[Path, ...] = field(default_factory=tuple)


# ------------------------------
# Logging
# ------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ------------------------------
# Utilities
# ------------------------------

def ensure_tag(obj: Any, name: str | None = None) -> Optional[Tag]:
    """Return obj if it is a Tag (and optionally matching name), else None."""
    if isinstance(obj, Tag) and (name is None or obj.name == name):
        return obj
    return None


def strip_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def check_latexml_available() -> bool:
    """Quickly check whether `latexmlc` is invokable on the system."""
    try:
        # Keep noisy output off the console; we only care if it runs
        subprocess.run(["latexmlc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def format_missing_dependency(name: str, message_fragment: Optional[str]) -> Optional[str]:
    if name.endswith((".sty", ".cls")):
        return name
    if name.endswith((".css", ".js", ".tex", ".ltx", ".def")):
        return None
    ext = "cls" if message_fragment == "binding for class" else "sty"
    return f"{name}.{ext}"


def list_missing_packages(log_path: Path) -> List[str]:
    """Parse LaTeXML log for missing package hints.

    Returns a de-duplicated list of `.sty`/`.cls` names.
    """
    matches: List[re.Match[str]] = []
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    m = re.search(MISSING_PACKAGE_RE, line)
                    if m:
                        matches.append(m)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(f"Could not read log file {log_path}: {exc}")

    deps: List[str] = []
    for m in matches:
        dep = format_missing_dependency(m[1], m[2])
        if dep:
            deps.append(dep)
    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique = [d for d in deps if not (d in seen or seen.add(d))]
    return unique


# ------------------------------
# LaTeXML runner
# ------------------------------

class LaTeXMLRunner:
    def __init__(self, resource_cfg: ResourceConfig, latexml_cfg: LaTeXMLConfig) -> None:
        self.resource_cfg = resource_cfg
        self.latexml_cfg = latexml_cfg

    def _build_command(self, source: Path, dest_html: Path, log_path: Path, source_dir: Path) -> List[str]:
        args: List[str] = [
            "latexmlc",
            f"--preload={self.latexml_cfg.preload}",
            f"--timeout={self.latexml_cfg.timeout_sec}",
            f"--splitat={self.latexml_cfg.split_at}",
            "--format=" + self.latexml_cfg.format,
            f"--navigationtoc={self.latexml_cfg.navigation_toc}",
            f"--whatsin={self.latexml_cfg.whats_in}",
            f"--sourcedirectory={source_dir}",
            f"--source={source}",
            f"--log={log_path}",
            f"--dest={dest_html}",
        ]
        if self.latexml_cfg.pmml:
            args.append("--pmml")
        if self.latexml_cfg.mathtex:
            args.append("--mathtex")
        if self.latexml_cfg.noinvisibletimes:
            args.append("--noinvisibletimes")
        if self.latexml_cfg.nodefaultresources:
            args.append("--nodefaultresources")

        # Make local .ltxml bindings discoverable
        extra_paths: Sequence[Path] = (
            tuple(self.latexml_cfg.extra_paths)
            if self.latexml_cfg.extra_paths
            else (CONVERTER_DIR,)
        )
        for p in extra_paths:
            args.append(f"--path={str(p)}")

        # Inject external resources
        for css in self.resource_cfg.css_urls:
            args.append(f"--css={css}")
        for js in self.resource_cfg.js_urls:
            args.append(f"--javascript={js}")
        return args

    def run(self, source_filepath: Path, output_dir: Path, verbose: bool = False) -> bool:
        if not source_filepath.is_file():
            logging.error(f"Source file does not exist: {source_filepath}")
            return False
        output_dir.mkdir(parents=True, exist_ok=True)

        source_dir = source_filepath.parent
        output_html = output_dir / f"{source_filepath.stem}.html"
        log_file = output_dir / "latexml.log"

        cmd = self._build_command(source_filepath, output_html, log_file, source_dir)
        logging.info(f"Converting {source_filepath} → {output_html} with LaTeXML…")
        if verbose:
            logging.debug("LaTeXML command: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(self.latexml_cfg.timeout_sec + 5, 3600),
            )
            if proc.returncode != 0:
                logging.error("LaTeXML failed (code %s)", proc.returncode)
                if proc.stdout:
                    logging.error("Output: %s", proc.stdout)
                return False

            logging.info("LaTeXML conversion successful: %s", output_html)
            missing = list_missing_packages(log_file)
            if missing:
                logging.warning("Missing packages detected: %s", ", ".join(missing))
                logging.warning("Consider installing these LaTeX packages for better results")
            return True
        except subprocess.TimeoutExpired:
            logging.error("LaTeXML conversion timed out after %s seconds", self.latexml_cfg.timeout_sec + 5)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logging.error("LaTeXML conversion failed: %s", exc)
            return False


# ------------------------------
# HTML post-processing
# ------------------------------

class HTMLPostProcessor:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        # Relative prefix from the directory of processed HTML files to the asset root (website/html)
        self._asset_rel_prefix: str = ""

    # ---------- top-level driver ----------
    def post_process_all(self, output_dir: Path) -> None:
        # Compute relative path prefix from current output_dir to the shared asset root (HTML_OUTPUT_DIR)
        try:
            rel = os.path.relpath(HTML_OUTPUT_DIR.resolve(), output_dir.resolve())
            # Use empty prefix for '.'; otherwise ensure trailing '/'
            self._asset_rel_prefix = "" if rel == "." else (rel.replace("\\", "/") + "/")
        except Exception:
            self._asset_rel_prefix = ""
        self._remove_unwanted_pages(output_dir)
        html_files = sorted(output_dir.glob("*.html"))
        file_by_name = {f.name: f for f in html_files}

        preface_file, chapter_files, appendix_files = self._compute_collections(file_by_name)
        for html_file in html_files:
            try:
                soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
                self._ensure_document_skeleton(soup)
                self._inject_head_basics(soup, html_file.name)
                # Remove LaTeXML-emitted document relation links that we don't use
                # (e.g., rel="up", rel="start", rel="chapter", rel="part", rel="appendix", "prev", "next").
                self._remove_legacy_head_relations(soup)
                self._ensure_body_top_anchor(soup)
                # Attempt to resolve and fix missing images and unhelpful alt text early.
                self._fix_images_in_document(soup, html_file.name, output_dir)
                self._remove_center_header_navigation(soup)
                self._remove_preface_toc_if_needed(soup, html_file.name)
                self._remove_bibliography_backlinks(soup)
                self._clean_footer_prev_next(soup)
                self._convert_tcolorbox_svg_to_div(soup)
                self._build_static_mini_toc(soup, html_file.name)
                html_serialized = str(soup)
                html_serialized = self._collapse_excess_blank_lines(html_serialized)
                html_file.write_text(html_serialized, encoding="utf-8")
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("Post-processing failed for %s: %s", html_file, exc)

    # ---------- helpers: structure ----------
    def _remove_unwanted_pages(self, output_dir: Path) -> None:
        try:
            unwanted = []
            unwanted.extend(output_dir.glob("book-main*.html"))
            unwanted.extend(output_dir.glob("Ptx*.html"))
            # unwanted.extend([output_dir / "Chx2.html", output_dir / "Chx3.html"])
            for f in unwanted:
                if f.exists():
                    f.unlink()
                    logging.info("Removed unwanted HTML page: %s", f.name)
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to remove unwanted pages: %s", exc)

    def _ensure_document_skeleton(self, soup: BeautifulSoup) -> None:
        html_tag = ensure_tag(soup.find("html"), "html")
        if not html_tag:
            html_tag = soup.new_tag("html")
            soup.append(html_tag)
        _head_tag = ensure_tag(soup.find("head"), "head")
        if not _head_tag:
            _head_tag = soup.new_tag("head")
            html_tag.insert(0, _head_tag)

    def _should_include_chapter_styling(self, html_filename: str) -> bool:
        """Determine if a file should include chapter-specific CSS and JS.
        
        Returns True for:
        - A*.html, Ax*.html (appendices)
        - Ch*.html, Chx*.html (chapters)
        - bib.html (bibliography)
        """
        filename_lower = html_filename.lower()
        
        # Check for chapter files (Ch1.html, Ch2.html, Chx1.html, Chx2.html, etc.)
        if re.match(r"^chx?[0-9]+\.html$", filename_lower):
            return True
        
        # Check for appendix files (A1.html, A2.html, Ax1.html, Ax2.html, etc.)
        if re.match(r"^ax?[0-9]+\.html$", filename_lower):
            return True
        
        # Check for bibliography
        if filename_lower == "bib.html":
            return True
        
        return False

    def _inject_head_basics(self, soup: BeautifulSoup, html_filename: str) -> None:
        head_tag = ensure_tag(soup.find("head"), "head")
        if not head_tag:
            return
        if head_tag.find("meta", {"name": "viewport"}) is None:
            head_tag.insert(0, soup.new_tag("meta", attrs={"name": "viewport", "content": "width=device-width, initial-scale=1"}))
        
        # Always add common CSS/JS for all files
        common_css_href = f"{self._asset_rel_prefix}common.css" if self._asset_rel_prefix else "common.css"
        common_components_js_src = "common_components.js"
        common_js_src = f"{self._asset_rel_prefix}common.js" if self._asset_rel_prefix else "common.js"
        # Avoid duplicates: check both bare and prefixed variants
        if head_tag.find("link", {"href": common_css_href}) is None and head_tag.find("link", {"href": "common.css"}) is None:
            head_tag.append(soup.new_tag("link", rel="stylesheet", href=common_css_href, type="text/css"))
        if head_tag.find("script", {"src": common_components_js_src}) is None and head_tag.find("script", {"src": "common_components.js"}) is None:
            sui = soup.new_tag("script", src=common_components_js_src)
            sui.attrs["defer"] = "defer"
            head_tag.append(sui)
        if soup.find("script", {"src": common_js_src}) is None and soup.find("script", {"src": "common.js"}) is None:
            sui = soup.new_tag("script", src=common_js_src)
            sui.attrs["defer"] = "defer"
            head_tag.append(sui)
        
        # Only add chapter-specific CSS/JS for chapter, appendix, preface, and bibliography files
        if self._should_include_chapter_styling(html_filename):
            chapter_css_href = f"{self._asset_rel_prefix}chapter.css" if self._asset_rel_prefix else "chapter.css"
            chapter_js_src = f"{self._asset_rel_prefix}chapter.js" if self._asset_rel_prefix else "chapter.js"
            if head_tag.find("link", {"href": chapter_css_href}) is None and head_tag.find("link", {"href": "chapter.css"}) is None:
                head_tag.append(soup.new_tag("link", rel="stylesheet", href=chapter_css_href, type="text/css"))
            if soup.find("script", {"src": chapter_js_src}) is None and soup.find("script", {"src": "chapter.js"}) is None:
                cjs = soup.new_tag("script", src=chapter_js_src)
                cjs.attrs["defer"] = "defer"
                head_tag.append(cjs)



    def _remove_legacy_head_relations(self, soup: BeautifulSoup) -> None:
        """Remove LaTeXML relational <link> elements we do not use.

        Also removes rel="prev"/"next"; we no longer emit these in <head>.
        """
        head_tag = ensure_tag(soup.find("head"), "head")
        if not head_tag:
            return
        try:
            removal_rels = {"up", "start", "chapter", "part", "appendix", "prev", "next", "bibliography"}
            for link_any in list(head_tag.find_all("link")):
                link_tag = ensure_tag(link_any)
                if not link_tag:
                    continue
                rel_attr = cast(Any, link_tag.get("rel"))
                if not rel_attr:
                    continue
                # BeautifulSoup typically parses rel into a list; be robust to string
                if isinstance(rel_attr, list):
                    rels = [str(r).lower() for r in rel_attr]
                else:
                    rels = [s.lower() for s in str(rel_attr).split()]  # handles "up up" cases
                if any(r in removal_rels for r in rels):
                    link_tag.decompose()
        except Exception:
            # Defensive: never let cleanup break the build
            pass

    def _remove_bibliography_backlinks(self, soup: BeautifulSoup) -> None:
        """Remove appendix-only backlinks to the bibliography (links to bib.html).

        We strip any anchors with href exactly 'bib.html' found in ToC blocks and footer/header areas,
        but leave in-text citation links (which point to bib.html#bibxNNN) intact.
        """
        try:
            # Remove direct links whose href is exactly 'bib.html'
            for a_any in list(soup.find_all("a", href=True)):
                a = ensure_tag(a_any)
                if not a:
                    continue
                href_val = cast(Optional[str], a.get("href"))
                if href_val and href_val.strip().lower() == "bib.html":
                    a.decompose()
            # Remove any empty ToC bibliography entries
            for li_any in list(soup.select("li.ltx_tocentry_bibliography")):
                li = ensure_tag(li_any)
                if li and not li.get_text(strip=True):
                    li.decompose()
        except Exception:
            pass

    def _collapse_excess_blank_lines(self, html: str) -> str:
        """Collapse runs of blank lines to a single blank line."""
        try:
            # Replace 2+ consecutive blank lines (optionally with spaces) with a single \n
            html = re.sub(r"\n[\t \r\f\v]*\n+", "\n", html)
            return html
        except Exception:
            return html

    def _ensure_body_top_anchor(self, soup: BeautifulSoup) -> None:
        body_tag = ensure_tag(soup.find("body"), "body")
        if body_tag and not body_tag.get("id"):
            body_tag["id"] = "top"

    # ---------- figures & images ----------
    def _strip_tex_comments(self, text: str) -> str:
        try:
            # Remove LaTeX comments: '%' to end-of-line (ignore escaped \%)
            lines = []
            for line in text.splitlines():
                # crude but effective: split on unescaped %
                pos = 0
                cut = len(line)
                while True:
                    idx = line.find('%', pos)
                    if idx == -1:
                        break
                    if idx > 0 and line[idx - 1] == '\\':
                        pos = idx + 1
                        continue
                    cut = idx
                    break
                lines.append(line[:cut])
            return "\n".join(lines)
        except Exception:
            return text

    def _extract_includegraphics_paths(self, tex_path: Path) -> List[str]:
        try:
            raw = tex_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        cleaned = self._strip_tex_comments(raw)
        try:
            pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
            return [m.group(1) for m in pattern.finditer(cleaned)]
        except Exception:
            return []

    def _extract_includegraphics_by_figure(self, tex_path: Path) -> List[List[str]]:
        r"""Return a list of includegraphics path lists, grouped per LaTeX figure.

        Heuristic parser: splits the .tex by \begin{figure}...\end{figure} (also figure*),
        and collects \includegraphics within each block. If no explicit figure env is found,
        falls back to a single block with all includegraphics in document order.
        """
        grouped: List[List[str]] = []
        try:
            raw = tex_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return grouped
        text = self._strip_tex_comments(raw)
        try:
            # Split into figure blocks (non-greedy dotall)
            fig_re = re.compile(r"\\begin\{figure\*?\}([\s\S]*?)\\end\{figure\*?\}", re.MULTILINE)
            inc_re = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
            blocks = list(fig_re.finditer(text))
            if not blocks:
                all_paths = [m.group(1) for m in inc_re.finditer(text)]
                if all_paths:
                    grouped.append(all_paths)
                return grouped
            for b in blocks:
                span_text = b.group(1)
                paths = [m.group(1) for m in inc_re.finditer(span_text)]
                grouped.append(paths)
            return grouped
        except Exception:
            return grouped

    def _find_source_tex_for_html_name(self, name: str) -> Optional[Path]:
        # Map ChN.html to chapters/chapterN/*.tex (english), A1/A2 to appendices, Chx1 to preface
        m_ch = re.match(r"^Ch(\d+)\.html$", name, flags=re.IGNORECASE)
        if m_ch:
            num = int(m_ch.group(1))
            chapter_dir = self.repo_root / "chapters" / f"chapter{num}"
            if chapter_dir.is_dir():
                tex_files = sorted([p for p in chapter_dir.glob("*.tex")])
                # Prefer non-Chinese, non-temporary/draft files
                def is_preferred(p: Path) -> bool:
                    n = p.name.lower()
                    undesirable = ("zh" in n) or ("tmp" in n) or ("draft" in n) or ("old" in n) or ("backup" in n)
                    return not undesirable
                preferred = [p for p in tex_files if is_preferred(p)]
                return preferred[0] if preferred else (tex_files[0] if tex_files else None)
        m_pref = re.match(r"^Chx\d+\.html$", name, flags=re.IGNORECASE)
        if m_pref:
            cand = self.repo_root / "chapters" / "preface" / "preface.tex"
            return cand if cand.exists() else None
        m_app = re.match(r"^A(\d+)\.html$", name, flags=re.IGNORECASE)
        if m_app:
            app_map = {1: "appendixA", 2: "appendixB"}
            num = int(m_app.group(1))
            app_dir_name = app_map.get(num)
            if app_dir_name:
                app_dir = self.repo_root / "chapters" / app_dir_name
                tex_files = sorted([p for p in app_dir.glob("*.tex")])
                def is_preferred_app(p: Path) -> bool:
                    n = p.name.lower()
                    undesirable = ("zh" in n) or ("tmp" in n) or ("draft" in n) or ("old" in n) or ("backup" in n)
                    return not undesirable
                preferred = [p for p in tex_files if is_preferred_app(p)]
                return preferred[0] if preferred else (tex_files[0] if tex_files else None)
        return None

    def _sanitize_filename(self, stem: str) -> str:
        # Replace spaces and unsafe chars with '-'
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", stem.strip())
        safe = re.sub(r"-+", "-", safe).strip("-._")
        return safe or "image"

    def _convert_pdf_to_png(self, src_pdf: Path, dest_png: Path) -> bool:
        try:
            dest_png.parent.mkdir(parents=True, exist_ok=True)
            # Prefer ImageMagick if available
            try:
                proc = subprocess.run(
                    ["magick", "convert", "-density", "300", str(src_pdf), "-quality", "92", str(dest_png)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0 and dest_png.exists():
                    return True
            except Exception:
                pass
            # Fallback to macOS sips
            try:
                proc2 = subprocess.run(
                    ["sips", "-s", "format", "png", str(src_pdf), "--out", str(dest_png)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if proc2.returncode == 0 and dest_png.exists():
                    return True
            except Exception:
                pass
            logging.warning("Could not convert PDF to PNG: %s", src_pdf)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("PDF→PNG conversion failed for %s: %s", src_pdf, exc)
            return False

    def _figure_caption_text(self, figure_tag: Tag) -> Optional[str]:
        try:
            cap = ensure_tag(figure_tag.find("figcaption"))
            return strip_whitespace(cap.get_text(" ", strip=True)) if cap else None
        except Exception:
            return None

    def _fix_images_in_document(self, soup: BeautifulSoup, html_name: str, output_dir: Path) -> None:
        try:
            # Replace generic alt text with caption; resolve missing images using source .tex
            src_tex = self._find_source_tex_for_html_name(html_name)
            include_groups: List[List[str]] = self._extract_includegraphics_by_figure(src_tex) if src_tex else []
            group_idx = 0
            image_idx_in_group = 0

            def get_next_include_path() -> Optional[str]:
                nonlocal group_idx, image_idx_in_group, include_groups
                while group_idx < len(include_groups) and image_idx_in_group >= len(include_groups[group_idx]):
                    group_idx += 1
                    image_idx_in_group = 0
                if group_idx < len(include_groups) and image_idx_in_group < len(include_groups[group_idx]):
                    path = include_groups[group_idx][image_idx_in_group]
                    image_idx_in_group += 1
                    return path
                return None

            # Compute base output image dir
            rel_img_dir: Optional[Path] = None
            m_ch = re.match(r"^Ch(\d+)\.html$", html_name, flags=re.IGNORECASE)
            m_app = re.match(r"^A(\d+)\.html$", html_name, flags=re.IGNORECASE)
            if m_ch:
                rel_img_dir = Path("chapters") / f"chapter{int(m_ch.group(1))}" / "figs"
            elif m_app:
                app_map = {1: "appendixA", 2: "appendixB"}
                dir_name = app_map.get(int(m_app.group(1)))
                if dir_name:
                    rel_img_dir = Path("chapters") / dir_name / "figs"

            # Process only top-level LaTeX figure containers (ids like F23),
            # and resolve all nested <img> within each container using the next
            # group of includegraphics() paths from the source .tex.
            all_figs = list(soup.find_all("figure"))
            container_figs: List[Tag] = []
            for fig in all_figs:
                ft = ensure_tag(fig)
                if not ft:
                    continue
                fig_id = cast(Optional[str], ft.get("id"))
                if fig_id and re.match(r"^F\d+$", fig_id):
                    container_figs.append(ft)

            for container in container_figs:
                fig_tag = container
                caption_text = self._figure_caption_text(fig_tag)
                # Paths for this LaTeX figure environment
                paths_for_group: List[str] = include_groups[group_idx] if group_idx < len(include_groups) else []
                path_idx_for_group = 0

                for img_any in fig_tag.find_all("img"):
                    img = ensure_tag(img_any)
                    if not img:
                        continue
                    # Improve alt text where possible
                    try:
                        alt_val = cast(Optional[str], img.get("alt"))
                        if alt_val and alt_val.strip().lower() == "refer to caption" and caption_text:
                            img["alt"] = caption_text
                    except Exception:
                        pass

                    # Attempt to resolve missing images
                    classes_attr_any: Any = img.get("class")
                    classes: List[str] = cast(List[str], classes_attr_any if isinstance(classes_attr_any, list) else [])
                    src_attr = cast(Optional[str], img.get("src"))
                    is_missing = (not src_attr) or ("ltx_missing" in classes or "ltx_missing_image" in classes)

                    # Determine includegraphics path for this image using the current figure's paths.
                    include_path: Optional[str] = None
                    if path_idx_for_group < len(paths_for_group):
                        include_path = paths_for_group[path_idx_for_group]
                    # Always advance within the group to align with nested subfigure order
                    path_idx_for_group += 1

                    if not is_missing:
                        continue
                    if not include_path:
                        continue

                    # Normalize the includegraphics path
                    inc = include_path.strip()
                    # Expand LaTeX macro \toplevelprefix to repo root
                    try:
                        inc = inc.replace(r"\toplevelprefix", str(self.repo_root))
                    except Exception:
                        pass
                    inc = inc.replace("\\", "/")  # normalize backslashes
                    inc = inc.replace("/./", "/")
                    inc_path = Path(inc)
                    if not inc_path.is_absolute():
                        base_dir = src_tex.parent if src_tex else self.repo_root
                        inc_path = (base_dir / inc).resolve()

                    if not inc_path.exists():
                        logging.debug("Include path not found on disk: %s", inc_path)
                        continue

                    # If it's a PDF, try to convert to PNG under output_dir/rel_img_dir
                    if inc_path.suffix.lower() == ".pdf" and rel_img_dir is not None:
                        out_dir = (output_dir / rel_img_dir).resolve()
                        sanitized = self._sanitize_filename(inc_path.stem) + ".png"
                        out_png = out_dir / sanitized
                        if not out_png.exists():
                            ok = self._convert_pdf_to_png(inc_path, out_png)
                            if not ok:
                                continue
                        # Point img src to the new PNG
                        img["src"] = str((rel_img_dir / sanitized).as_posix())
                        # Remove missing classes if present
                        try:
                            new_classes_list = [c for c in classes if c not in {"ltx_missing", "ltx_missing_image"}]
                            if new_classes_list:
                                # BeautifulSoup accepts list for class; cast to list[str]
                                img["class"] = cast(Any, new_classes_list)
                            else:
                                if img.has_attr("class"):
                                    del img["class"]
                        except Exception:
                            pass
                    else:
                        # For non-PDF, try to compute a relative path into output dir if not already set
                        if rel_img_dir is not None:
                            target = output_dir / rel_img_dir / inc_path.name
                            try:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                if not target.exists():
                                    shutil.copy2(inc_path, target)
                                img["src"] = str((rel_img_dir / inc_path.name).as_posix())
                                new_classes_list = [c for c in classes if c not in {"ltx_missing", "ltx_missing_image"}]
                                if new_classes_list:
                                    img["class"] = cast(Any, new_classes_list)
                                else:
                                    if img.has_attr("class"):
                                        del img["class"]
                            except Exception:
                                pass
                # After finishing this container, move to the next source figure group
                group_idx += 1
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to fix images for %s: %s", html_name, exc)

    # ---------- navigation ----------
    def _compute_collections(self, file_by_name: dict[str, Path]) -> Tuple[Optional[str], List[str], List[str]]:
        def sort_key_numeric(name: str, prefix: str) -> int:
            try:
                return int(re.match(rf"{prefix}(\d+)\\.html$", name, flags=re.IGNORECASE).group(1))  # type: ignore[union-attr]
            except Exception:
                return 10**9

        preface_candidates = sorted(
            [n for n in file_by_name if re.match(r"^Chx\d+\.html$", n, flags=re.IGNORECASE)],
            key=lambda n: sort_key_numeric(n, "Chx"),
        )
        if preface_candidates:
            preface_file = preface_candidates[0]
        else:
            preface_alt = [n for n in file_by_name if "preface" in n.lower()]
            preface_file = sorted(preface_alt)[0] if preface_alt else None

        chapter_files = sorted(
            [n for n in file_by_name if re.match(r"^Ch\d+\.html$", n, flags=re.IGNORECASE)],
            key=lambda n: sort_key_numeric(n, "Ch"),
        )
        appendix_files = sorted(
            [n for n in file_by_name if re.match(r"^A\d+\.html$", n, flags=re.IGNORECASE)],
            key=lambda n: sort_key_numeric(n, "A"),
        )
        return preface_file, chapter_files, appendix_files

    def _remove_center_header_navigation(self, soup: BeautifulSoup) -> None:
        """Remove the secondary top bar (centered header navigation) for ALL pages.
        
        Sidebar already provides navigation; keep header minimal.
        """
        header_tag = ensure_tag(soup.find("header", class_="ltx_page_header"))
        if not header_tag:
            return

        try:
            center_div = ensure_tag(header_tag.find("div", class_="ltx_align_center"))
            if center_div:
                center_div.decompose()
        except Exception:
            pass



    # ---------- cleanups ----------
    def _remove_preface_toc_if_needed(self, soup: BeautifulSoup, name: str) -> None:
        for toc_nav in list(soup.select("nav.ltx_TOC")):
            toc_tag = cast(Tag, toc_nav)
            # Check if TOC is empty (no meaningful content)
            if not toc_tag.get_text(strip=True):
                toc_tag.decompose()

    def _clean_footer_prev_next(self, soup: BeautifulSoup) -> None:
        footer_tag = ensure_tag(soup.find("footer", class_="ltx_page_footer"))
        if not footer_tag:
            return
        align_center = ensure_tag(footer_tag.find("div", class_="ltx_align_center"))
        if not align_center:
            return
        removed_any = False
        for a in list(align_center.find_all("a")):
            rel_val = cast(Optional[List[str]], cast(Tag, a).get("rel"))
            if rel_val and ("prev" in rel_val or "next" in rel_val):
                cast(Tag, a).decompose()
                removed_any = True
        if removed_any and not align_center.text.strip() and not align_center.find(True):
            align_center.decompose()

    # ---------- mini ToC ----------
    def _build_static_mini_toc(self, soup: BeautifulSoup, name: str) -> None:
        is_chapter= re.match(r"^Chx?[0-9]+\.html$", name, flags=re.IGNORECASE) is not None
        is_appendix = re.match(r"^Ax?[0-9]+\.html$", name, flags=re.IGNORECASE) is not None

        container_any = soup.select_one(".ltx_appendix" if is_appendix else ".ltx_chapter")
        container = ensure_tag(container_any)
        if not container:
            return
        if container.find("div", class_="mini-toc"):
            return

        sections: List[Tag] = [cast(Tag, s) for s in container.select("section.ltx_section")]  # type: ignore[list-item]
        if not sections:
            return

        def heading_text_excluding_number(heading_tag: Any, default_text: str) -> str:
            if not ensure_tag(heading_tag):
                return default_text
            parts: List[str] = []
            try:
                for child in cast(Tag, heading_tag).children:
                    tag_child = ensure_tag(child)
                    if tag_child:
                        classes_attr = tag_child.get("class")
                        classes_list: List[str] = cast(List[str], classes_attr) if isinstance(classes_attr, list) else []
                        if "ltx_tag" in classes_list:
                            continue
                        parts.append(tag_child.get_text(strip=True))
                    else:
                        text_part = str(child).strip()
                        if text_part:
                            parts.append(text_part)
                text = " ".join(filter(None, parts)).strip()
                if not text:
                    raw = cast(Tag, heading_tag).get_text(strip=True)
                    text = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", raw)
                return text
            except Exception:
                raw = cast(Tag, heading_tag).get_text(strip=True) if ensure_tag(heading_tag) else default_text
                return re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", raw)

        toc_div = soup.new_tag("div")
        toc_div["class"] = "mini-toc"
        title_div = soup.new_tag("div")
        title_div["class"] = "mini-toc-title"
        title_div.string = "In this section" if is_appendix else "In this chapter"
        list_ul = soup.new_tag("ul")

        for sec in sections:
            h2 = sec.select_one(".ltx_title_section")
            sec_id = cast(Optional[str], cast(Tag, sec).get("id"))
            if not sec_id and ensure_tag(h2) and cast(Tag, h2).has_attr("id"):
                sec_id = cast(str, cast(Tag, h2)["id"])  # type: ignore[index]
            text = heading_text_excluding_number(h2, "Section")

            li = soup.new_tag("li")
            a = soup.new_tag("a")
            a["href"] = f"#{sec_id}" if sec_id else "#"
            a.string = text
            li.append(a)

            subs_result = sec.select("section.ltx_subsection")
            if subs_result:
                sub_container = soup.new_tag("div")
                sub_container["class"] = "mini-toc-sub"
                for sub_any in subs_result:
                    sub = ensure_tag(sub_any)
                    if not sub:
                        continue
                    h3 = sub.select_one(".ltx_title_subsection")
                    sub_id: Optional[str] = cast(Optional[str], sub.get("id"))
                    if not sub_id and ensure_tag(h3) and cast(Tag, h3).has_attr("id"):
                        sub_id = cast(str, cast(Tag, h3)["id"])  # type: ignore[index]
                    stext = heading_text_excluding_number(h3, "Subsection")
                    sub_a = soup.new_tag("a")
                    sub_a["href"] = f"#{sub_id}" if sub_id else "#"
                    sub_a.string = stext
                    sub_container.append(sub_a)
                li.append(sub_container)

            list_ul.append(li)

        toc_div.append(title_div)
        toc_div.append(list_ul)

        title_selector = ".ltx_title_appendix" if is_appendix else ".ltx_title_chapter"
        container_title_any = container.select_one(title_selector)
        container_title = ensure_tag(container_title_any)
        if container_title:
            container_title.insert_after(toc_div)
        else:
            container.insert(0, toc_div)

    # ---------- tcolorbox conversion ----------
    def _convert_tcolorbox_svg_to_div(self, soup: BeautifulSoup) -> None:
        """Convert SVG tcolorbox environments to styled div.tcbox elements.
        
        This replicates the client-side JavaScript functionality that finds SVG pictures
        containing foreignObject with tcolorbox content and converts them to properly
        styled HTML div elements at build time.
        """
        try:
            # Find all SVG pictures that might contain tcolorbox content
            svg_pictures = soup.find_all("svg", class_="ltx_picture")
            converted_count = 0
            
            for svg in svg_pictures:
                # Look for the specific structure: svg > g > foreignobject > span.ltx_inline-block.ltx_minipage
                # The foreignobject might be nested inside <g> elements
                svg_tag = ensure_tag(svg)
                if not svg_tag:
                    continue
                # Use CSS selector for better typing compatibility
                minipage = svg_tag.select_one("span.ltx_inline-block.ltx_minipage")
                if not minipage:
                    continue
                
                # Extract plain text content to match original JavaScript behavior
                # The original JS used innerText/textContent which preserves inter-element whitespace
                raw_text = minipage.get_text()  # Don't strip=True, preserves spaces between elements
                if not raw_text:
                    continue
                
                # Normalize whitespace (equivalent to JS text.replace(/\s+/g, ' '))
                # This collapses multiple spaces/newlines but preserves single spaces
                normalized_text = re.sub(r'\s+', ' ', raw_text).strip()
                
                # Create the replacement div.tcbox structure
                wrapper = soup.new_tag("div")
                wrapper["class"] = "tcbox"
                
                p_elem = soup.new_tag("p")
                p_elem["class"] = "ltx_p"
                p_elem.string = normalized_text
                
                wrapper.append(p_elem)
                
                # Replace the SVG with the new div
                svg.replace_with(wrapper)
                converted_count += 1
            
            if converted_count > 0:
                logging.info("Converted %d tcolorbox SVG elements to div.tcbox", converted_count)
                
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to convert tcolorbox SVG elements: %s", exc)


# ------------------------------
# Search index generation
# ------------------------------

class SearchIndexBuilder:
    def generate(self, output_dir: Path) -> None:
        try:
            html_files = sorted([p for p in output_dir.glob("*.html") if p.name != "index.html"])
            entries: List[dict] = []

            for html_path in html_files:
                try:
                    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
                except Exception:
                    continue

                # Page label from <title>
                title_tag = ensure_tag(soup.find("title"))
                page_label = title_tag.get_text(strip=True) if title_tag else html_path.name
                if "‣" in page_label:
                    page_label = page_label.split("‣", 1)[0].strip()

                h1_chapter = ensure_tag(soup.select_one("h1.ltx_title_chapter"))
                h1_appendix = ensure_tag(soup.select_one("h1.ltx_title_appendix"))
                if h1_chapter:
                    entries.append({
                        "page": page_label,
                        "href": f"{html_path.name}#top",
                        "title": strip_whitespace(h1_chapter.get_text(" ", strip=True)),
                        "snippet": "",
                        })
                elif h1_appendix:
                    entries.append({
                        "page": page_label,
                        "href": f"{html_path.name}#top",
                        "title": strip_whitespace(h1_appendix.get_text(" ", strip=True)),
                        "snippet": "",
                        })

                targets: List[Tag] = []
                targets.extend(cast(List[Tag], soup.select("section.ltx_section[id]")))
                targets.extend(cast(List[Tag], soup.select("section.ltx_subsection[id]")))
                # Capture paragraph-level containers. LaTeXML emits either
                # section.ltx_paragraph (older) or div.ltx_para (newer) for paragraphs.
                # Include both to maximize recall.
                targets.extend(cast(List[Tag], soup.select("section.ltx_paragraph[id]")))
                targets.extend(cast(List[Tag], soup.select("div.ltx_para[id]")))
                targets.extend(cast(List[Tag], soup.select(".ltx_theorem[id]")))
                targets.extend(cast(List[Tag], soup.select(".ltx_equation[id]")))
                targets.extend(cast(List[Tag], soup.select(".ltx_equationgroup[id]")))
                targets.extend(cast(List[Tag], soup.select("h1[id], h2[id], h3[id], section[id]")))

                seen_ids: set[str] = set()
                for el in targets:
                    if not ensure_tag(el):
                        continue
                    el_id = cast(Optional[str], el.get("id"))
                    if not el_id or el_id in seen_ids:
                        continue
                    seen_ids.add(el_id)

                    title_el = el.select_one(
                        ".ltx_title, .ltx_title_section, .ltx_title_subsection, .ltx_title_theorem"
                    )
                    if not ensure_tag(title_el):
                        title_el = None
                        for tag_name in ["h1", "h2", "h3"]:
                            cand = el.find(tag_name)
                            if ensure_tag(cand):
                                title_el = cand
                                break
                    raw_title = cast(Tag, title_el).get_text(" ", strip=True) if title_el else el.get_text(" ", strip=True)
                    title = strip_whitespace(raw_title)[:200]
                    if not title:
                        continue
                    # Allow a longer snippet to improve search recall without bloating the index too much.
                    snippet = strip_whitespace(el.get_text(" ", strip=True))[:600]
                    entries.append({
                        "page": page_label,
                        "href": f"{html_path.name}#{el_id}",
                        "title": title,
                        "snippet": snippet,
                    })

            payload = {
                "generated": datetime.now(timezone.utc).isoformat(),
                "count": len(entries),
                "entries": entries,
            }
            (output_dir / "search-index.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            logging.info("Wrote search index with %d entries to %s", len(entries), output_dir / "search-index.json")
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to generate search index: %s", exc)


# ------------------------------
# Orchestration
# ------------------------------

def convert_latex_to_html(source_filepath: Path, output_dir: Path, verbose: bool = False) -> bool:
    """Orchestrate the full conversion + post-process pipeline."""
    resource_cfg = ResourceConfig()
    latexml_cfg = LaTeXMLConfig()
    runner = LaTeXMLRunner(resource_cfg, latexml_cfg)
    post = HTMLPostProcessor(PROJECT_ROOT)
    search = SearchIndexBuilder()

    ok = runner.run(source_filepath, output_dir, verbose)
    if not ok:
        return False

    # Assets (CSS/JS) are already in website/html/, no need to copy
    post.post_process_all(output_dir)
    search.generate(output_dir)
    return True


def postprocess_only(source_filepath: str, output_dir_arg: Optional[str]) -> Tuple[bool, Path]:
    """Run only the HTML post-processing and search indexing phases."""
    if output_dir_arg:
        output_dir = Path(output_dir_arg).resolve()
    else:
        output_dir = HTML_OUTPUT_DIR

    logging.info("Post-processing HTML in: %s", output_dir)
    # Assets (CSS/JS) are already in website/html/, no need to copy
    post = HTMLPostProcessor(PROJECT_ROOT)
    post.post_process_all(output_dir)
    SearchIndexBuilder().generate(output_dir)
    return True, output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LaTeX files to HTML using LaTeXML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nExamples:\n"
            "  python latex_to_html_converter.py ./my_paper/book-main.tex\n"
            "  python latex_to_html_converter.py ./my_paper/book-main.tex ./output\n"
            "  python latex_to_html_converter.py ./my_paper/book-main.tex --verbose\n"
        ),
    )
    parser.add_argument("source_filepath", help="Path to the LaTeX file to convert")
    parser.add_argument("output_dir", nargs="?", help="Output directory for HTML files (default: source_filepath/html)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Skip LaTeXML and only run HTML post-processing on output_dir (or inferred html dir)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.postprocess_only:
        success, output_dir = postprocess_only(args.source_filepath, args.output_dir)
    else:
        if not check_latexml_available():
            logging.error("LaTeXML (latexmlc) is not available on this system.")
            logging.error("Please install LaTeXML first:")
            logging.error("  - Ubuntu/Debian: sudo apt-get install latexml")
            logging.error("  - macOS: brew install latexml")
            logging.error("  - Docs: https://dlmf.nist.gov/LaTeXML/get.html")
            raise SystemExit(1)

        source = Path(args.source_filepath).resolve()
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            # Default to website/html/ directory
            output_dir = HTML_OUTPUT_DIR
        logging.info("Source directory: %s", source.parent)
        logging.info("Output directory: %s", output_dir)
        success = convert_latex_to_html(source, output_dir, args.verbose)

    if success:
        logging.info("Conversion completed successfully!")
        html_files = list(Path(output_dir).glob("*.html"))
        if html_files:
            logging.info("Generated HTML files: %s", [str(f) for f in html_files])
        raise SystemExit(0)
    else:
        logging.error("Conversion failed!")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
