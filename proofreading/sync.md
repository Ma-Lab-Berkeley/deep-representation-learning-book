# Sync English Edits to Chinese Translation

Synchronize recent English (`*.tex`) changes in `chapters/` to the corresponding Chinese (`*_zh.tex`) files by re-translating only the affected regions.

## Determine the diff range

Accept two optional commit SHAs as arguments. Defaults:
- If uncommitted changes exist: diff working tree vs `HEAD`
- Otherwise: diff from `git merge-base origin/main HEAD` to `HEAD`

## Identify changed English content

1. Run `git diff <base> <target>` with the resolved range.
2. Filter to files under `chapters/` whose names do **not** end in `_zh.tex` or `_ro.tex`.
3. Collect the changed line ranges (added, modified, or deleted lines) per file.

## Locate label-bounded regions

For each changed line range in an English file:

1. Search upward from the first changed line to find the nearest `\label{...}` line **above** (or beginning of file).
2. Search downward from the last changed line to find the nearest `\label{...}` line **below** (or end of file).
3. These two labels define the **sync region**. Extract all English content between them (inclusive of the label lines).

Labels follow prefixes such as `ch:`, `sec:`, `sub:`, `eqn:`, `fig:`, `thm:`, `ex:` and are identical across EN/ZH files.

## Re-translate into the Chinese file

1. Derive the Chinese filename (e.g., `chapter1.tex` → `chapter1_zh.tex`).
2. Open the `_zh.tex` file and locate the same two bounding labels.
3. Extract the old Chinese text between those labels.
4. Re-translate the current English content between the labels into Chinese, replacing the old Chinese content between the same labels.
5. Follow all translation conventions in `prompts/translate.md`:
   - Preserve all LaTeX commands, math environments, citations, and labels exactly.
   - Translate "representation learning" as 表征学习.
   - Maintain formal, academic Chinese tone.
   - Output raw LaTeX; do not add or remove structural elements.

## Edge cases

- If a changed region has no label above, use the start of file as the upper bound.
- If a changed region has no label below, use the end of file as the lower bound.
- If multiple changed regions share the same bounding labels, merge them into one sync region.
- If the `_zh.tex` file does not exist, skip that file and warn.

## Output

For each synced region, edit the `_zh.tex` file in place. After all edits, list the files and label pairs that were updated.
