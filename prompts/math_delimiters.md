# Math Delimiters and Environments

## Inline Math Delimiters

- ALWAYS use `\(...\)` for inline math. NEVER use `$...$`.
- Special exception: within `tikzpicture` environments, use `$...$` for all math.

## Equation Environments

- ALWAYS use `equation` environment (not `equation*` or `\[...\]`).
- ALWAYS use `align` environment (not `align*`).
- Convert all `\[...\]` display math to `\begin{equation}...\end{equation}`.
- Convert `eqnarray` and `aligned` to `align` where possible.
- Number every equation/line in align (don't use `\nonumber` or `\notag` and remove if they exist).
- Use `align` only when the environment has multiple lines (and otherwise use `equation`). Do not use `aligned`, `eqnarray`, etc. Only change an align-like environment to equation-like when it already uses only one line (e.g., no linebreaks `\\`). If there are linebreaks, DO NOT change the `align` environment to an `equation` environment.
