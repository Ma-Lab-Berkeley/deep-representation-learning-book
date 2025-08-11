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

LATEXML_TIMEOUT_SEC: int = 1080

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
        "magnify=1.2,zoomout=1.2,tokenlimit=2499999999,iflimit=3599999,"
        "absorblimit=1299999,pushbacklimit=599999]latexml.sty"
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
# Asset management
# ------------------------------

class AssetManager:
    """Ensure repository-shipped shared assets are present in output directories."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.src_css = repo_root / "html" / "chapter.css"
        self.src_js = repo_root / "html" / "chapter.js"
        self.src_shared_ui = repo_root / "html" / "shared-ui.js"

    def ensure_shared_assets(self, output_dir: Path) -> None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not self.src_css.exists() or not self.src_js.exists():
                raise FileNotFoundError("Shared assets 'chapter.css'/'chapter.js' not found under repo html/")

            out_css = output_dir / "chapter.css"
            out_js = output_dir / "chapter.js"
            # Always copy; files are small and this avoids stale assets
            shutil.copy2(self.src_css, out_css)
            shutil.copy2(self.src_js, out_js)
            # Copy shared UI helpers if present (search, topbar wiring)
            if self.src_shared_ui.exists():
                shutil.copy2(self.src_shared_ui, output_dir / "shared-ui.js")
            logging.info("Shared assets ensured (chapter.css, chapter.js, shared-ui.js if present)")
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(f"Failed to ensure shared assets: {exc}")


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
            else (Path(__file__).parent,)
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
                timeout=self.latexml_cfg.timeout_sec + 5,
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

    # ---------- top-level driver ----------
    def post_process_all(self, output_dir: Path) -> None:
        self._remove_unwanted_pages(output_dir)
        html_files = sorted(output_dir.glob("*.html"))
        file_by_name = {f.name: f for f in html_files}

        preface_file, chapter_files, appendix_files = self._compute_collections(file_by_name)
        for html_file in html_files:
            try:
                soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")
                self._ensure_document_skeleton(soup)
                self._inject_head_basics(soup)
                self._inject_stacked_header_css(soup)
                # Remove LaTeXML-emitted document relation links that we don't use
                # (e.g., rel="up", rel="start", rel="chapter", rel="part", rel="appendix", "prev", "next").
                self._remove_legacy_head_relations(soup)
                self._ensure_body_top_anchor(soup)
                self._rebuild_header_navigation(soup, html_file.name, file_by_name, preface_file, chapter_files, appendix_files)
                self._remove_hardcoded_top_bar(soup)
                self._remove_preface_toc_if_needed(soup, html_file.name)
                self._clean_footer_prev_next(soup)
                self._build_static_mini_toc(soup, html_file.name)
                html_serialized = str(soup)
                html_serialized = self._collapse_excess_blank_lines(html_serialized)
                html_file.write_text(html_serialized, encoding="utf-8")
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("Post-processing failed for %s: %s", html_file, exc)

    # ---------- helpers: structure ----------
    def _remove_unwanted_pages(self, output_dir: Path) -> None:
        try:
            unwanted = [output_dir / "book-main.html"]
            unwanted.extend(output_dir.glob("Ptx*.html"))
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

    def _inject_head_basics(self, soup: BeautifulSoup) -> None:
        head_tag = ensure_tag(soup.find("head"), "head")
        if not head_tag:
            return
        if head_tag.find("meta", {"name": "viewport"}) is None:
            head_tag.insert(0, soup.new_tag("meta", attrs={"name": "viewport", "content": "width=device-width, initial-scale=1"}))
        if head_tag.find("link", {"href": "chapter.css"}) is None:
            head_tag.append(soup.new_tag("link", rel="stylesheet", href="chapter.css", type="text/css"))
        # Ensure external JS in preferred order: shared-ui.js, then chapter.js
        if soup.find("script", {"src": "shared-ui.js"}) is None:
            sui = soup.new_tag("script", src="shared-ui.js")
            sui.attrs["defer"] = "defer"
            head_tag.append(sui)
        if soup.find("script", {"src": "chapter.js"}) is None:
            cjs = soup.new_tag("script", src="chapter.js")
            cjs.attrs["defer"] = "defer"
            head_tag.append(cjs)

    def _inject_stacked_header_css(self, soup: BeautifulSoup) -> None:  # pragma: no cover - inline CSS moved to chapter.css
        head_tag = ensure_tag(soup.find("head"), "head")
        if not head_tag:
            return
        # Remove legacy inline style blocks we previously injected
        for legacy_id in ["stacked-topbars", "navbar-removed-fix"]:
            legacy = ensure_tag(head_tag.find(id=legacy_id))
            if legacy:
                legacy.decompose()

    def _remove_legacy_head_relations(self, soup: BeautifulSoup) -> None:
        """Remove LaTeXML relational <link> elements we do not use.

        Also removes rel="prev"/"next"; we no longer emit these in <head>.
        """
        head_tag = ensure_tag(soup.find("head"), "head")
        if not head_tag:
            return
        try:
            removal_rels = {"up", "start", "chapter", "part", "appendix", "prev", "next"}
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

    def _compute_prev_next_for(
        self,
        name: str,
        file_by_name: dict[str, Path],
        preface_file: Optional[str],
        chapter_files: List[str],
        appendix_files: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        lower = name.lower()
        if lower == "bib.html":
            return (None, None)

        m_pref = re.match(r"^Chx(\d+)\.html$", name, flags=re.IGNORECASE)
        if m_pref:
            next_target = "Ch1.html" if "Ch1.html" in file_by_name else (appendix_files[0] if appendix_files else None)
            return (None, next_target)

        m_ch = re.match(r"^Ch(\d+)\.html$", name, flags=re.IGNORECASE)
        if m_ch:
            num = int(m_ch.group(1))
            if num == 1:
                prev_target = preface_file
            else:
                prev_target = f"Ch{num-1}.html" if f"Ch{num-1}.html" in file_by_name else None
            if f"Ch{num+1}.html" in file_by_name:
                next_target: Optional[str] = f"Ch{num+1}.html"
            else:
                next_target = appendix_files[0] if appendix_files else None
            return (prev_target, next_target)

        m_app = re.match(r"^A(\d+)\.html$", name, flags=re.IGNORECASE)
        if m_app:
            num = int(m_app.group(1))
            if num == 1:
                prev_target = chapter_files[-1] if chapter_files else (preface_file or None)
            else:
                prev_target = f"A{num-1}.html" if f"A{num-1}.html" in file_by_name else None
            next_target = f"A{num+1}.html" if f"A{num+1}.html" in file_by_name else None
            return (prev_target, next_target)

        return (None, None)

    def _rebuild_header_navigation(
        self,
        soup: BeautifulSoup,
        this_name: str,
        file_by_name: dict[str, Path],
        preface_file: Optional[str],
        chapter_files: List[str],
        appendix_files: List[str],
    ) -> None:
        header_tag = ensure_tag(soup.find("header", class_="ltx_page_header"))
        if not header_tag:
            return

        center_div_any = header_tag.find("div", class_="ltx_align_center")
        center_div = ensure_tag(center_div_any)
        if not center_div:
            center_div = soup.new_tag("div")
            center_div["class"] = "ltx_align_center"
            header_tag.append(center_div)
        center_div.clear()
        assert isinstance(center_div, Tag)
        center_div_tag: Tag = center_div

        def add_link(href: Optional[str], text: str) -> None:
            if not href:
                return
            a = soup.new_tag("a")
            a["class"] = "ltx_ref"
            a["href"] = str(href)
            span = soup.new_tag("span")
            span["class"] = "ltx_text ltx_ref_title"
            span.string = text
            a.append(span)
            center_div_tag.append(a)

        add_link("#top", "Top of Page")
        desired_prev, desired_next = self._compute_prev_next_for(this_name, file_by_name, preface_file, chapter_files, appendix_files)

        # Ensure any legacy <head> rel prev/next are removed; we no longer emit these in <head>
        head_tag = ensure_tag(soup.find("head"), "head")
        if head_tag:
            for rel_name in ["prev", "next"]:
                for el in list(head_tag.find_all("link", rel=rel_name)):
                    cast(Tag, el).decompose()

        add_link(desired_prev, "Previous Chapter")
        add_link(desired_next, "Next Chapter")

    # ---------- topbar ----------
    def _remove_hardcoded_top_bar(self, soup: BeautifulSoup) -> None:
        """Remove any previously injected hard-coded topbar.

        The runtime navbar from html/shared-ui.js will render on page load.
        """
        try:
            existing = ensure_tag(soup.find(id="book-topbar"))
            if existing:
                existing.decompose()
        except Exception:
            pass

    # ---------- cleanups ----------
    def _remove_preface_toc_if_needed(self, soup: BeautifulSoup, name: str) -> None:
        if re.match(r"^Chx\d+\.html$", name, flags=re.IGNORECASE):
            for toc_nav in list(soup.select("nav.ltx_TOC")):
                cast(Tag, toc_nav).decompose()

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
        is_preface = re.match(r"^Chx\d+\.html$", name, flags=re.IGNORECASE) is not None
        is_chapter = re.match(r"^Ch\d+\.html$", name, flags=re.IGNORECASE) is not None
        is_appendix = re.match(r"^A\d+\.html$", name, flags=re.IGNORECASE) is not None
        if is_preface or not (is_chapter or is_appendix):
            return

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
                targets.extend(cast(List[Tag], soup.select("section.ltx_paragraph[id]")))
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
                    snippet = strip_whitespace(el.get_text(" ", strip=True))[:280]
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
    repo_root = Path(__file__).resolve().parents[1]
    resource_cfg = ResourceConfig()
    latexml_cfg = LaTeXMLConfig()
    runner = LaTeXMLRunner(resource_cfg, latexml_cfg)
    assets = AssetManager(repo_root)
    post = HTMLPostProcessor(repo_root)
    search = SearchIndexBuilder()

    ok = runner.run(source_filepath, output_dir, verbose)
    if not ok:
        return False

    assets.ensure_shared_assets(output_dir)
    post.post_process_all(output_dir)
    search.generate(output_dir)
    return True


def postprocess_only(source_filepath: str, output_dir_arg: Optional[str]) -> Tuple[bool, Path]:
    """Run only the HTML post-processing and search indexing phases."""
    repo_root = Path(__file__).resolve().parents[1]
    default_repo_html = repo_root / "html"
    if output_dir_arg:
        output_dir = Path(output_dir_arg).resolve()
    else:
        src_path = Path(source_filepath).resolve() if source_filepath else default_repo_html

        # If user points to an HTML file or an existing HTML directory, use that directory directly.
        try:
            if src_path.exists():
                if src_path.is_file():
                    # If it's already an .html file, operate on its parent directory
                    if src_path.suffix.lower() == ".html":
                        output_dir = src_path.parent
                    else:
                        # Assume LaTeX source; default to sibling html directory
                        output_dir = src_path.parent / "html"
                else:
                    # It's a directory. If it looks like an HTML output dir, use it as-is.
                    looks_like_html_dir = (
                        src_path.name.lower() == "html"
                        or any(p.suffix.lower() == ".html" for p in src_path.glob("*.html"))
                    )
                    output_dir = src_path if looks_like_html_dir else (src_path / "html")
            else:
                # Fallback to repo html/
                output_dir = default_repo_html
        except Exception:
            # Defensive fallback
            output_dir = default_repo_html

    logging.info("Post-processing HTML in: %s", output_dir)
    assets = AssetManager(repo_root)
    assets.ensure_shared_assets(output_dir)
    post = HTMLPostProcessor(repo_root)
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
        output_dir = Path(args.output_dir).resolve() if args.output_dir else (source.parent / "html")
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