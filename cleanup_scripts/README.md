# Figure Migration Script

This directory contains a Python script for migrating figures in LaTeX book chapters to local directories.

## migrate_figures.py

### Purpose
This script processes a TeX file to:
1. Find all `\includegraphics` references
2. Create a local `figs` directory in the same directory as the TeX file
3. Copy all referenced figures to the local `figs` directory
4. Update the TeX file to use the new local paths

### Usage

```bash
python cleanup_scripts/migrate_figures.py <tex_file_path>
```

### Examples

```bash
# Migrate figures for chapter 1
python cleanup_scripts/migrate_figures.py chapters/chapter1/introduction.tex

# Migrate figures for chapter 2
python cleanup_scripts/migrate_figures.py chapters/Chapter2/classic-models.tex
```

### What it does

1. **Scans the TeX file** for `\includegraphics` commands with various option patterns
2. **Resolves figure paths** by checking multiple possible locations:
   - Relative to repository root
   - Relative to TeX file directory
   - Relative to book root
3. **Creates local figs directory** in the same directory as the TeX file
4. **Copies figures** to the local directory (avoids duplicates)
5. **Updates TeX references** to use local `figs/` paths
6. **Creates backup** of original TeX file with `.bak` extension

### Features

- ✅ Handles various `\includegraphics` option formats
- ✅ Resolves figures from multiple source directories
- ✅ Avoids duplicate copies
- ✅ Preserves original file formatting
- ✅ Creates automatic backups
- ✅ Handles multiple references to the same figure
- ✅ Cross-platform path handling

### Example transformation

**Before:**
```latex
\includegraphics[width=0.5\linewidth]{figs_chap1/DNAs.png}
\includegraphics[height=0.4\linewidth]{figures/Cybernetics1.jpg}
```

**After:**
```latex
\includegraphics[width=0.5\linewidth]{figs/DNAs.png}
\includegraphics[height=0.4\linewidth]{figs/Cybernetics1.jpg}
```

### Safety

- Always creates a backup (`.tex.bak`) before modifying files
- Only copies files if they don't exist or are different
- Preserves original file permissions and timestamps
- Handles file not found errors gracefully

### Requirements

- Python 3.6+
- Standard library modules: `os`, `re`, `shutil`, `argparse`, `pathlib` 