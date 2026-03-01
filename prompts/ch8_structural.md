# Chapter 8 Structural Consistency

## Table Formatting

- ALWAYS use booktabs: `\toprule`, `\midrule`, `\bottomrule`.
- NEVER use `\hline`.
- Table captions should come BEFORE the `\centering` / table body.
- Table captions should follow the pattern: `\caption{\small\textbf{Short bold title.} Longer explanation text.}`

## Figure Path Prefix

- ALWAYS use `\toplevelprefix/chapters/chapter8/figs/...` for figure paths.
- Do NOT use bare `chapters/chapter8/figs/...`.

## Float Placement

- Prefer `[t]` for figures and tables (top of page).
- Do not use `[h]` or `[!htbp]` unless there is a specific reason.
- Standardize to `[t]` where possible.

## Caption Formatting

- All captions should begin with `\small\textbf{Title in bold.}` followed by normal-weight explanation.
- Ensure `\small` and `\textbf` are consistent across all captions.

## Image Dimension Ordering

- The book establishes `(C, H, W)` ordering (channels first): `\R^{c \times h \times w}`.
- Standardize to `(C, H, W)` throughout (channels-first).
- Convert any `(H, W, C)` orderings (e.g., `\x \in \R^{H \times W \times 3}`) to channels-first.
