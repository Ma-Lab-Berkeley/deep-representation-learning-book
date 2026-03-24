# Cross-Reference Style

## Figure/Table/Section/Chapter References

- ALWAYS use `\Cref{...}` (capital C) for all references to figures, tables, sections, chapters, theorems, remarks, etc.
- NEVER use bare `\ref{...}`, or manual constructions like `Figure~\ref{...}`, `Table~\ref{...}`, `Section~\ref{...}`, `Chapter~\ref{...}`.
- NEVER use lowercase `\cref{...}` --- always `\Cref{...}`.
- Our `\Cref{...}` formatting produces outputs like `\Cref{ch:foo}` -> "Chapter ##" (etc), so be sure to correct surrounding referrers.

## Equation References

- ALWAYS use `\eqref{...}` for equation references.
- NEVER use `\ref{...}` or `Equation~\ref{...}` for equations.

## Citation Style

- Use `\citep{...}` for parenthetical citations: "...as shown previously \citep{smith2020}."
- Use `\citet{...}` for textual citations: "As \citet{smith2020} showed..."
- Avoid bare `\cite{...}` where possible.
