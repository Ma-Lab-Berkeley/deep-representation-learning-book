#!/usr/bin/env python3
"""
Portable LaTeX to HTML converter using LaTeXML.

This script converts LaTeX files in a specified directory to HTML format using LaTeXML.
It can be placed in any folder and run after installing the required dependencies.

Usage:
    python latex_to_html_converter.py <source_directory> [output_directory]

Requirements:
    - LaTeXML installed on the system (latexmlc command available)
    - Python packages: beautifulsoup4, requests
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: beautifulsoup4 is required. Install with: pip install beautifulsoup4")
    sys.exit(1)

# Configuration constants
LATEXML_TIMEOUT_SEC = 1080
MISSING_PACKAGE_RE = re.compile(
    r"^Warning:missing_file:(\S+)\s(?:Can't\sfind\s(package|binding for class))?", flags=re.MULTILINE
)

# CSS and JS resources for HTML output
CSS_RESOURCES = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
    "https://cdn.jsdelivr.net/gh/arXiv/arxiv-browse@master/arxiv/browse/static/css/ar5iv.0.8.2.min.css",
    "https://cdn.jsdelivr.net/gh/arXiv/arxiv-browse@master/arxiv/browse/static/css/ar5iv-fonts.0.8.2.min.css",
    "https://cdn.jsdelivr.net/gh/arXiv/arxiv-browse@master/arxiv/browse/static/css/latexml_styles.0.8.2.css",
]

JS_RESOURCES = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.3.3/html2canvas.min.js",
]


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def check_latexml_available() -> bool:
    """Check if LaTeXML (latexmlc) is available on the system."""
    try:
        subprocess.run(["latexmlc"])
        # , 
        #               stdout=subprocess.DEVNULL, 
        #               stderr=subprocess.DEVNULL, 
        #               check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def format_missing_dependency(name: str, message_fragment: str) -> Optional[str]:
    """Format missing dependency name for reporting."""
    if name.endswith((".sty", ".cls")):
        return name
    # Ignore some common low-level issues
    elif name.endswith((".css", ".js", ".tex", ".ltx", ".def")):
        return None
    else:
        ext = "cls" if message_fragment == "binding for class" else "sty"
        return f"{name}.{ext}"


def list_missing_packages(log_path: Path) -> List[str]:
    """Parse LaTeXML log file to extract missing packages."""
    matches = []
    if log_path.exists():
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as log_file:
                for line in log_file:
                    match = re.search(MISSING_PACKAGE_RE, line)
                    if match:
                        matches.append(match)
        except Exception as e:
            logging.warning(f"Could not read log file {log_path}: {e}")
    
    return list(filter(None, map(lambda match: format_missing_dependency(match[1], match[2]), matches)))


def find_main_tex_file(source_dir: Path) -> Optional[Path]:
    """Find the main LaTeX file to convert."""
    # Look for common main file names
    main_candidates = ["book-main.tex", "main.tex", "paper.tex", "article.tex", "document.tex"]
    
    for candidate in main_candidates:
        main_file = source_dir / candidate
        if main_file.exists():
            return main_file
    
    # If no common names found, look for any .tex file
    tex_files = list(source_dir.glob("*.tex"))
    if len(tex_files) == 1:
        return tex_files[0]
    elif len(tex_files) > 1:
        # If multiple .tex files, try to find one with \documentclass
        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if r'\documentclass' in content:
                        return tex_file
            except Exception:
                continue
        # If no \documentclass found, return the first one
        return tex_files[0]
    
    return None


def convert_latex_to_html(source_filepath: Path, bib_filepath: Optional[Path] = None, output_dir: Path = Path("html"), verbose: bool = False) -> bool:
    """Convert LaTeX files in source_dir to HTML in output_dir."""
    if not source_filepath.parent.exists():
        logging.error(f"Source directory does not exist: {source_filepath.parent}")
        return False

    if not source_filepath.is_file():
        logging.error(f"Source file does not exist: {source_filepath}")
        return False
    
    source_dir = source_filepath.parent
    logging.info(f"Found main LaTeX file: {source_filepath}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up output paths
    output_html = output_dir / f"{source_filepath.stem}.html"
    log_file = output_dir / "latexml.log"
    
    # Build LaTeXML command
    latexml_config = [
        "latexmlc",
        "--preload=[nobibtex,nobreakuntex,localrawstyles,mathlexemes,magnify=1.2,zoomout=1.2,tokenlimit=2499999999,iflimit=3599999,absorblimit=1299999,pushbacklimit=599999]latexml.sty",
        "--pmml",
        "--mathtex",
        "--noinvisibletimes",
        f"--timeout={LATEXML_TIMEOUT_SEC}",
        "--splitat=chapter",
        "--nodefaultresources",
        "--format=html5",
        "--navigationtoc=context",
        "--whatsin=directory",
        f"--sourcedirectory={source_dir}",
        f"--source={source_filepath}",
        f"--log={log_file}",
        f"--dest={output_html}",
    ]
    cur_path = Path(__file__).parent
    latexml_config.append(f"--path={str(cur_path / 'ar5iv-bindings' / 'bindings')}")
    # for stub in ["bindings"]:#, "supported_originals"]:
    #     latexml_config.append(f"--path={str(cur_path / 'ar5iv-bindings' / stub)}")
    # latexml_config.append(f"--preload={str(cur_path / 'ar5iv-bindings' / 'bindings' / 'ar5iv.sty')}")
    # latexml_config.append(f"--preload={str(cur_path / 'ar5iv-bindings' / 'bindings' / 'biblatex.sty.ltxml')}")
    
    # Add CSS resources
    for css in CSS_RESOURCES:
        latexml_config.append(f"--css={css}")
    
    # Add JavaScript resources
    for js in JS_RESOURCES:
        latexml_config.append(f"--javascript={js}")
    
    logging.info(f"Converting {source_filepath} to HTML...")
    if verbose:
        logging.debug(f"LaTeXML command: {' '.join(latexml_config)}")
    
    try:
        # Run LaTeXML conversion
        completed_process = subprocess.run(
            latexml_config,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=LATEXML_TIMEOUT_SEC + 5,
        )
        
        if completed_process.returncode == 0:
            logging.info(f"Conversion successful! Output: {output_html}")
            
            # Check for missing packages
            missing_packages = list_missing_packages(log_file)
            if missing_packages:
                logging.warning(f"Missing packages detected: {', '.join(missing_packages)}")
                logging.warning("Consider installing these LaTeX packages for better results")
            
            # Post-process HTML if needed
            post_process_html(output_html)
            
            return True
        else:
            logging.error(f"LaTeXML conversion failed with return code {completed_process.returncode}")
            if completed_process.stdout:
                logging.error(f"Output: {completed_process.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"LaTeXML conversion timed out after {LATEXML_TIMEOUT_SEC + 5} seconds")
        return False
    except Exception as e:
        logging.error(f"LaTeXML conversion failed with error: {e}")
        return False


def post_process_html(html_file: Path) -> None:
    """Post-process the generated HTML file."""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Add viewport meta tag for better mobile display
        if soup.head:
            viewport_meta = soup.new_tag("meta", attrs={"name": "viewport", "content": "width=device-width, initial-scale=1"})
            soup.head.insert(0, viewport_meta)
        
        # Write back the modified HTML
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(str(soup))
            
        logging.info("HTML post-processing completed")
        
    except Exception as e:
        logging.warning(f"HTML post-processing failed: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert LaTeX files to HTML using LaTeXML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python latex_to_html_converter.py ./my_paper
    python latex_to_html_converter.py ./my_paper ./output
    python latex_to_html_converter.py ./my_paper --verbose
        """
    )
    
    parser.add_argument("source_filepath", 
                       help="Path to the LaTeX file to convert")
    parser.add_argument("output_dir", 
                       nargs="?", 
                       help="Output directory for HTML files (default: source_filepath/html)")
    parser.add_argument("-v", "--verbose", 
                       action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Check if LaTeXML is available
    if not check_latexml_available():
        logging.error("LaTeXML (latexmlc) is not available on this system.")
        logging.error("Please install LaTeXML first:")
        logging.error("  - Ubuntu/Debian: sudo apt-get install latexml")
        logging.error("  - macOS: brew install latexml")
        logging.error("  - Or visit: https://dlmf.nist.gov/LaTeXML/get.html")
        sys.exit(1)
    
    # Set up paths
    source_filepath = Path(args.source_filepath).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = source_filepath.parent / "html"
    
    logging.info(f"Source directory: {source_filepath.parent}")
    logging.info(f"Output directory: {output_dir}")
    
    # Perform conversion
    success = convert_latex_to_html(source_filepath, None, output_dir, args.verbose)
    
    if success:
        logging.info("Conversion completed successfully!")
        html_files = list(output_dir.glob("*.html"))
        if html_files:
            logging.info(f"Generated HTML files: {[str(f) for f in html_files]}")
        sys.exit(0)
    else:
        logging.error("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()