#!/usr/bin/env python3
"""
Figure Migration Script for LaTeX Book

This script takes a TeX file, creates a local 'figs' directory in the same directory,
copies all referenced figures to that directory, and updates the TeX file to use
the new local paths.

Usage:
    python migrate_figures.py <tex_file_path>

Example:
    python migrate_figures.py chapters/chapter1/introduction.tex
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Set


def find_includegraphics_references(tex_content: str) -> List[Tuple[str, str]]:
    """
    Find all \\includegraphics references in the TeX content.
    
    Returns:
        List of tuples (full_match, figure_path)
    """
    # Pattern to match \\includegraphics commands with various options
    # This handles cases like:
    # \\includegraphics[width=0.5\\linewidth]{figs_chap1/DNAs.png}
    # \\includegraphics[height=0.3\\linewidth]{figures/neuron.png}
    pattern = r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}'
    
    matches = []
    for match in re.finditer(pattern, tex_content):
        full_match = match.group(0)
        figure_path = match.group(1)
        matches.append((full_match, figure_path))
    
    return matches


def resolve_figure_path(figure_path: str, tex_file_dir: Path, repo_root: Path) -> Path:
    """
    Resolve the actual file path of a figure referenced in the TeX file.
    
    Args:
        figure_path: The path as written in the TeX file
        tex_file_dir: Directory containing the TeX file
        repo_root: Root directory of the repository
        
    Returns:
        Absolute path to the figure file
    """
    # Remove any leading/trailing whitespace
    figure_path = figure_path.strip()
    
    # Try different possible locations for the figure
    possible_paths = [
        repo_root / figure_path,  # Relative to repo root
        tex_file_dir / figure_path,  # Relative to TeX file
        tex_file_dir / ".." / ".." / figure_path,  # Relative to book root
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    
    # If not found, raise an error
    raise FileNotFoundError(f"Could not find figure: {figure_path}")


def create_figs_directory(chapter_dir: Path) -> Path:
    """
    Create a 'figs' directory in the chapter directory.
    
    Returns:
        Path to the created figs directory
    """
    figs_dir = chapter_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    return figs_dir


def copy_figure_to_figs(source_path: Path, figs_dir: Path) -> Path:
    """
    Copy a figure file to the figs directory.
    
    Returns:
        Path to the copied file
    """
    destination = figs_dir / source_path.name
    
    # Only copy if the file doesn't exist or is different
    if not destination.exists() or not files_are_identical(source_path, destination):
        shutil.copy2(source_path, destination)
        print(f"Copied: {source_path} -> {destination}")
    else:
        print(f"Already exists: {destination}")
    
    return destination


def files_are_identical(file1: Path, file2: Path) -> bool:
    """
    Check if two files are identical by comparing their sizes and modification times.
    """
    try:
        stat1 = file1.stat()
        stat2 = file2.stat()
        return stat1.st_size == stat2.st_size and stat1.st_mtime == stat2.st_mtime
    except OSError:
        return False


def update_tex_content(tex_content: str, figure_updates: List[Tuple[str, str]]) -> str:
    """
    Update the TeX content with new figure paths.
    
    Args:
        tex_content: Original TeX content
        figure_updates: List of (old_match, new_path) tuples
        
    Returns:
        Updated TeX content
    """
    updated_content = tex_content
    
    for old_match, new_path in figure_updates:
        # Extract the options part (everything between \\includegraphics and {)
        options_match = re.search(r'\\includegraphics(\s*(?:\[[^\]]*\])?\s*)\{[^}]+\}', old_match)
        if options_match:
            options_part = options_match.group(1)
            new_match = f"\\includegraphics{options_part}{{{new_path}}}"
            updated_content = updated_content.replace(old_match, new_match)
    
    return updated_content


def migrate_figures(tex_file_path: str) -> None:
    """
    Main function to migrate figures for a given TeX file.
    """
    tex_file = Path(tex_file_path)
    
    if not tex_file.exists():
        raise FileNotFoundError(f"TeX file not found: {tex_file_path}")
    
    # Read the TeX file
    with open(tex_file, 'r', encoding='utf-8') as f:
        tex_content = f.read()
    
    # Find all figure references
    figure_references = find_includegraphics_references(tex_content)
    
    if not figure_references:
        print(f"No figures found in {tex_file_path}")
        return
    
    print(f"Found {len(figure_references)} figure references in {tex_file_path}")
    
    # Get directories
    chapter_dir = tex_file.parent
    repo_root = tex_file.parent.parent.parent  # Assuming chapters/chapterX/file.tex structure
    
    # Create figs directory
    figs_dir = create_figs_directory(chapter_dir)
    
    # Process each figure
    figure_updates = []
    processed_figures = set()  # To avoid duplicate copying
    
    for full_match, figure_path in figure_references:
        try:
            # Resolve the actual figure file
            source_path = resolve_figure_path(figure_path, chapter_dir, repo_root)
            
            # Copy to figs directory (only if not already copied)
            if figure_path not in processed_figures:
                copied_path = copy_figure_to_figs(source_path, figs_dir)
                processed_figures.add(figure_path)
            
            # Create the new relative path from root directory
            # Get the relative path from repo root to the chapter directory
            chapter_relative_path = chapter_dir.relative_to(repo_root)
            new_figure_path = f"{chapter_relative_path}/figs/{source_path.name}"
            
            # Record the update for THIS specific instance
            figure_updates.append((full_match, new_figure_path))
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    if not figure_updates:
        print("No figures were successfully processed")
        return
    
    # Update TeX content
    updated_content = update_tex_content(tex_content, figure_updates)
    
    # Create backup of original file
    backup_path = tex_file.with_suffix('.tex.bak')
    shutil.copy2(tex_file, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Write updated content
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Successfully updated {tex_file_path}")
    print(f"Migrated {len(figure_updates)} unique figures to {figs_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate figures in a TeX file to a local figs directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_figures.py chapters/chapter1/introduction.tex
  python migrate_figures.py chapters/Chapter2/classic-models.tex
        """
    )
    parser.add_argument(
        'tex_file',
        help='Path to the TeX file to process'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dry_run:
            print("DRY RUN MODE - No files will be modified")
            # TODO: Implement dry run functionality
            print("Dry run mode not yet implemented")
        else:
            migrate_figures(args.tex_file)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
