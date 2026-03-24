# Text Style and Language

Go through the excerpt and fix streamlining/grammar/wording/awkward turn-of-phrase issues. Be as surgical as possible with the edits, and DO NOT change the technical content.

## Emphasis Formatting

- Use `\textit{...}` for italics. NEVER use `{\em ...}` or `\emph{...}`.

## Dashes

- Use `---` for em-dashes and `--` for en-dashes.
- Generally prefer rephrasing to avoid dashes where possible.
- Format em-dashes as (for example) `text1---text2`, rather than `text1 --- text2` (etc.)

## Use `\text` Instead of `\mbox`

- Inside math mode, always use `\text{...}` (not `\mbox{...}`).

## Paragraph and Section Title Formatting

- For `\begin{paragraph}` or `\begin{subparagraph}` (i.e. "paragraph") titles, ensure that they are grammatically accurate, e.g., not all words must be capitalized and the title must end with a period/question mark/etc.
  - Correct: `\paragraph{Embedding.}`
  - Wrong: `\paragraph{Embedding}`
- `\section{...}` and `\subsection{...}` titles must NOT end with punctuation.
- Section/subsection/subsubsection titles should be formatted as proper titles (capitalize major words).
- Do NOT use `\textbf{Title.}` as a substitute for `\paragraph{Title.}`. Convert all such occurrences.

## Person, Voice, and Tone

- Use first-person plural "we" consistently when describing methodology and results.
- Do NOT switch to "you".
- Do NOT use unnecessary passive voice when "we" is natural.
- Avoid colloquialisms:
  - "Anyways, the equation..." -> "In any case, the equation..."
  - "it's because..." -> "this is because..."
  - "Keep this in mind..." -> "We will use this observation..."

## Ordinals in Lists

- Prefer "first" over "firstly", "second" over "secondly".
- In bulleted/itemized lists, ordinals are often unnecessary.

## Dataset and Method Name Spelling

- "CIFAR-10" (with hyphen), not "CIFAR10".
- "CIFAR-100" (with hyphen), not "CIFAR100".
- "ImageNet-1K" (consistent capitalization and formatting).
- "OpenWebText" (one word, camelCase).
- "COCO val2017" (no braces around val2017).

## Quotation Marks

- Use ``` `` ``` and `''` for proper LaTeX double quotation marks.
- NEVER use Unicode `\u201c` or `\u201d` characters.
