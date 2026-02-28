Can you go through this excerpt and fix streamlining/grammar/wording/awkward turn-of-phrase issues? Be as surgical as possible with the edits, and DO NOT change the technical content. In addition, please do the following LaTeX edits:

- use \textit instead of \em or \emph
- use --- instead of em-dash and -- instead of en-dash, but generally avoid using dashes. Format em-dashes as (for example) "text1---text2", rather than "text1 --- text2" (etc.)
- use \text instead of \mbox.
- For \begin{paragraph} or \begin{subparagraph} (i.e. "paragraph") titles, ensure that they are grammatically accurate, e.g., not all words must be capitalized and the title must end with a period/question mark/etc.
  --- Section/subsection/subsubsection titles must be formatted like actual titles.
- Prefer to use use \Cref{...} in general. Our \Cref{...} formatting produces outputs like "\Cref{ch:foo} -> Chapter ##" (etc), so be sure to correct surrounding referrers. Similarly, prefer \eqref{...} instead of \ref{...}
- General exception to grammar parsing: use Section, Chapter, Subsection (capitalized) even when ordinarily not capitalized.


If the input is split up into fixed-width lines (often 80 characters), please do your best to split the output into fixed-width lines like the input.
