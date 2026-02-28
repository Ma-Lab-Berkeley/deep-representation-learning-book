Can you go through this excerpt and standardize the formatting (especially w.r.t. math)? Here are some guidelines:
- Use macros as much as possible. For example, \x and \boldsymbol{x} should be replaced by \vx
    - \v is the prefix for bold (e.g. \vx, \vy, \vA, \vB, \vphi, \vPhi, etc.), \c is the prefix for calligraphy (e.g., \cA, \cB, \cN, etc.), \b is the prefix for blackboard (e.g. \bA, \bB, etc.). There is no font for matrices (e.g. no \mA), just use \vA = \boldsymbol{A}, etc.
--- Special exceptions: \R = \bR = \mathbb{R}, \C = \bC = \mathbb{C}, \N = \bN = \mathbb{N} are real/complex/natural numbers respectively; in these cases always use the short form \R, \C, \N
--- Special exceptions: Use \Ex for an expectation (often \mathbb{E}) and \Pr for a probability (often \mathbb{P}).
--- Special exceptions: \I is equivalent to \vI, please use \vI in place of \I.
- Try to tell if the bracket spacing (i.e., \Big, \left, \right) is needed; if not needed, remove it
    - You can use \bp{...} in place of \left(...\right), \bs{...} in place of \left[...\right], and \bc{...} in place of \left\{...\right\}.  For brackets used right after another symbol (such as function application) where we don't want added space, use \rp{...}, \rs{...}, and \rc{...} respectively. But only use these if you want to use \left/\right, i.e., the inner object is particularly tall (like a display-style fraction).
- Always use equation environment instead of equation* or \[...\]
- Always use align environment instead of align*
- Number every equation/line in align (don't use \nonumber or \notag and remove if they exist)
- Always use \(...\) instead of $...$
--- special exception: Within tikzpictures, use $...$ for every math
- Use \mat{...} instead of \begin{bmatrix}...\end{bmatrix} or similar
- Use punctuation at the end of displaystyle equations wherever appropriate.
- Always use \Cref{...}  and \eqref{...} instead of \ref{...} (do not use \cref)
- Use align only when the environment has multiple lines (and otherwise use equation). Do not use aligned, eqnarray, etc. Only change an align-like environment to equation-like when it already uses only one line (e.g., no linebreaks \\).
- For subscripts and superscripts please use {}, e.g., always use x_{n} instead of x_n, and always use x^{n} instead of x^n, etc (same for all other "bases"). Also same with \hat{\vx}, \tilde{\vx}, etc.


In addition, please keep to the following notation table (some of it is redundant with the above):

<!-- include: chapters/notation.tex -->


