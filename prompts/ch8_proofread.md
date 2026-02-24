# Chapter 8 Proofreading Instructions

Please apply the following fixes to the section you are given of the chapter. They are output by another LLM which is trying to unify the writing style and notation of the sections. Please try to fix every listed issue in the given paragraph.

---

## 1. Math Notation Rules

### 1.1 Inline Math Delimiters

- ALWAYS use `\(...\)` for inline math. NEVER use `$...$`.
- Many paragraphs (especially in sections on VAE, conditional generation, Michelangelo, Cupid, EgoAllo) use `$...$` and must be converted.

### 1.2 Bold Vector/Matrix Macros

- Use `\vx` (not `$\x$`, `$\boldsymbol{x}$`, `\mathbf{x}`, or `\bm{x}`) for bold lowercase vectors.
- Use `\vX` (not `$\boldsymbol{X}$`) for bold uppercase matrices.
- Use `\vphi`, `\vtheta`, `\vmu`, `\vSigma`, `\vepsilon` (not `$\boldsymbol{\phi}$`, `$\boldsymbol{\theta}$`, etc.) for bold Greek letters.
- Use `\vzero` for the zero vector (not `$\boldsymbol{0}$` or `$\Zero$`), `\vone` for the ones vector, `\vI` for identity (not `$\boldsymbol{I}$` or `\I`).
- The `\v` prefix means bold: `\vx`, `\vy`, `\vA`, `\vB`, `\vphi`, `\vPhi`, etc.
- There is no special matrix font; just use `\vA = \boldsymbol{A}`, etc.

Known violations to watch for:
- Bare `$\x$`, `$\z$` (undefined shorthand macros used in VAE/RAE sections).
- Raw `$\boldsymbol{\mu}$`, `$\boldsymbol{\sigma}$`, `$\boldsymbol{\epsilon}$`, `$\boldsymbol{0}$`, `$\boldsymbol{I}$` instead of `\vmu`, `\vsigma`, `\vepsilon`, `\vzero`, `\vI`.
- `$\bm{0}$`, `$\bm{\mu}$`, `$\bm{\Sigma}$` (using `\bm` instead of `\v` prefix macros).
- Local `\newcommand` aliases that produce raw `\boldsymbol{...}` output (e.g., `\pose`, `\object`, `\latent`, `\noise`, `\velocity`, `\point`, `\feat` defined in the Cupid section). These should use the book's standard macros.

### 1.3 Calligraphic and Blackboard-Bold Macros

- `\c` prefix for calligraphic: `\cA`, `\cB`, `\cD`, `\cL`, `\cN`, `\cT`, `\cX`, etc. (not `$\mathcal{A}$`).
- `\R` for real numbers (not `$\mathbb{R}$`). Similarly `\C` for complex, `\N` for naturals.
- `\Ex` for expectation (not `$\mathbb{E}$`).
- `\Pr` for probability (not `$\mathbb{P}$`).

Known violations: The VAE, conditional generation, Michelangelo, Cupid, and EgoAllo sections frequently use `$\mathbb{R}$`, `$\mathbb{E}$`, `$\mathcal{N}$`, `$\mathcal{L}$`, `$\mathcal{D}$`, `$\mathcal{E}$`, `$\mathcal{S}$` instead of the short-form macros.

### 1.4 Subscripts and Superscripts

- ALWAYS use braces: `x_{n}` not `x_n`, `x^{n}` not `x^n`.
- Same rule applies to hats, tildes, etc.: `\hat{\vx}` not `\hat\vx`, `\tilde{\vx}` not `\tilde\vx`.

### 1.5 Equation Environments

- ALWAYS use `equation` environment (not `equation*` or `\[...\]`).
- ALWAYS use `align` environment (not `align*`).
- Convert all `\[...\]` display math to `\begin{equation}...\end{equation}`.
- Convert `eqnarray` to `align` where possible.
- Special exception: within `tikzpicture` environments, use `$...$` for math.

### 1.6 Bracket Sizing

- Remove unnecessary `\left`/`\right`, `\Big`, `\bigg` when the enclosed content does not require grown brackets.
- Only use explicit sizing when the content is genuinely tall (e.g., fractions, sums with limits).

### 1.7 Inner Products, Norms, and Distances

- Use `\ip{\vx}{\vy}` for inner products (not `\langle \vx, \vy \rangle` or `\left\langle ... \right\rangle`).
- Use `\norm{\vx}_{2}` for norms (not `\|\vx\|_2` or `\left\| ... \right\|`).
- The CLIP and Cupid sections frequently use raw `\|...\|` and `\langle...\rangle` instead of macros.

### 1.8 Specific Operator Macros

- Use `\KL` for KL divergence (not `$D_{KL}$` or `$D_{\mathrm{KL}}$`).
- Use `\CE` for cross-entropy.
- Use `\hada` for element-wise (Hadamard) multiplication (not `$\odot$`).
- Use `\softmax`, `\ReLU`, `\MHSA`, `\MSSA`, `\MLP`, `\LN`, `\SA`, `\ISTA` as operator macros where defined.
- Use `\doteq` for definitional equality (not `:=` or `\triangleq`). Note: `\triangleq` appears in the Cupid section.

### 1.9 Function Notation

- Use `\colon` for function typing: `f \colon X \to Y` (not `f : X \to Y`).

---

## 2. Cross-Reference Style

### 2.1 Figure/Table/Section/Chapter References

- ALWAYS use `\Cref{...}` (capital C) for all references to figures, tables, sections, chapters, theorems, remarks, etc.
- NEVER use bare `\ref{...}`, or manual constructions like `Figure~\ref{...}`, `Table~\ref{...}`, `Section~\ref{...}`, `Chapter~\ref{...}`.
- NEVER use lowercase `\cref{...}` --- always `\Cref{...}`.

Known violations:
- Introduction: uses `Chapter \ref{ch:intro}`, `Section \ref{sec:SimDINO}` mixed with `\Cref{ch:future}`.
- CLIP section: uses lowercase `\cref{...}` in many places.
- Conditional generation section: uses `Figure~\ref{...}`, `Section~\ref{...}`, `Chapter \ref{...}`.
- Michelangelo section: uses `Figure~\ref{...}`, `Table~\ref{...}`, `Section~\ref{...}`.
- Cupid section: uses `Section~\ref{...}`, `Figure \ref{...}`, `Equation~\ref{...}`.
- EgoAllo section: uses `Chapter~\ref{...}`, `Section~\ref{...}`, `Figure~\ref{...}`.

### 2.2 Equation References

- ALWAYS use `\eqref{...}` for equation references.
- NEVER use `\ref{...}` or `Equation~\ref{...}` for equations.

### 2.3 Citation Style

- Use `\citep{...}` for parenthetical citations: "...as shown previously \citep{smith2020}."
- Use `\citet{...}` for textual citations: "As \citet{smith2020} showed..."
- Avoid bare `\cite{...}` where possible.
- Some sections use `\cite{...}` exclusively; others mix all three styles.

---

## 3. Text Style and Language

### 3.1 Emphasis Formatting

- Use `\textit{...}` for italics. NEVER use `{\em ...}` or `\emph{...}`.
- Known violations: Introduction uses `{\em the visual data}`, `{\em our body motions}`, `{\em natural languages}`. Michelangelo, Cupid, NLP, and EgoAllo sections also use `{\em ...}`.

### 3.2 Dashes

- Use `---` for em-dashes and `--` for en-dashes.
- Generally prefer rephrasing to avoid dashes where possible.

### 3.3 Use `\text` Instead of `\mbox`

- Inside math mode, always use `\text{...}` (not `\mbox{...}`).

### 3.4 Paragraph and Section Title Formatting

- `\paragraph{...}` titles MUST end with a period (or question mark/exclamation mark).
  - Correct: `\paragraph{Embedding.}`
  - Wrong: `\paragraph{Embedding}`
- `\section{...}` and `\subsection{...}` titles must NOT end with punctuation.
- Section/subsection titles should be formatted as proper titles (capitalize major words).
- Do NOT use `\textbf{Title.}` as a substitute for `\paragraph{Title.}`. Convert all such occurrences (found in the conditional generation section: "Architecture and Connectivity.", "Training Dynamics and Experiments.", "Identifier and Data Strategy.", "Prior Preservation Training.", "Multi-Concept Personalization.").

Known violations of missing periods on `\paragraph`:
- "3D Shape Representations" (should be "3D Shape Representations.")
- "Ablation Studies" (should be "Ablation Studies.")
- "Object 3D Shape and Poses from a Single 2D View" (should end with period)

### 3.5 Person, Voice, and Tone

- Use first-person plural "we" consistently when describing methodology and results.
- Do NOT switch to "you" (e.g., "apply what you have learned" in the introduction should be rephrased).
- Do NOT use unnecessary passive voice when "we" is natural.
- Maintain a consistent formality level throughout. The target tone is the measured, precise style of the DINO/SimDINO and CRATE sections.
- Avoid colloquialisms:
  - "Anyways, the equation..." → "In any case, the equation..."
  - "it's because..." → "this is because..."
  - "Keep this in mind..." → "We will use this observation..."
- The conditional generation section has a more tutorial-like voice; the Michelangelo section has a more paper-like tone; the EgoAllo section is conversational-textbook. These should be harmonized to match the style of the DINO/CRATE sections.

### 3.6 Ordinals in Lists

- Prefer "first" over "firstly", "second" over "secondly".
- In bulleted/itemized lists, ordinals are often unnecessary.

### 3.7 Dataset and Method Name Spelling

- "CIFAR-10" (with hyphen), not "CIFAR10".
- "CIFAR-100" (with hyphen), not "CIFAR100".
- "ImageNet-1K" (consistent capitalization and formatting).
- "OpenWebText" (one word, camelCase).
- "COCO val2017" (no braces around val2017).

### 3.8 Quotation Marks

- Use ``` `` ``` and `''` for proper LaTeX double quotation marks.
- NEVER use Unicode `"` or `"` characters. (Found in the CLIP section around line 1297.)

---

## 4. Structural Consistency

### 4.1 Table Formatting

- ALWAYS use booktabs: `\toprule`, `\midrule`, `\bottomrule`.
- NEVER use `\hline`. (The Michelangelo tables use `\hline`.)
- Table captions should come BEFORE the `\centering` / table body (most do; verify).
- Table captions should follow the pattern: `\caption{\small\textbf{Short bold title.} Longer explanation text.}`

### 4.2 Figure Path Prefix

- ALWAYS use `\toplevelprefix/chapters/chapter8/figs/...` for figure paths.
- Do NOT use bare `chapters/chapter8/figs/...` (found in RAE section, Cupid section).

### 4.3 Float Placement

- Prefer `[t]` for figures and tables (top of page).
- Do not use `[h]` or `[!htbp]` unless there is a specific reason.
- Be consistent: the chapter uses `[t]`, `[h]`, `[th]`, `[!htbp]` --- standardize to `[t]` where possible.

### 4.4 Caption Formatting

- All captions should begin with `\small\textbf{Title in bold.}` followed by normal-weight explanation.
- Some captions omit `\small` or `\textbf` --- make these consistent.

---

## 5. Image Dimension Ordering

- The book establishes `(C, H, W)` ordering (channels first) in the DINO section: `\R^{c \times h \times w}`.
- The VAE section uses `(H, W, C)` ordering: `\x \in \R^{H \times W \times 3}`.
- The RAE section uses `(C, H, W)`: `\x \in \R^{3 \times H \times W}`.
- **Standardize to `(C, H, W)` throughout** (channels-first, matching the convention established in the chapter).

---

## 6. Notation Conflicts to Resolve

### 6.1 Parameter Variable `\theta`

- Throughout the chapter, `\theta` is used for neural network parameters.
- The Cupid section redefines `\pose = \boldsymbol{\theta}` for camera pose --- this conflicts with network parameters `\theta`.
- The EgoAllo section uses `\theta` for SMPL joint rotations AND for network parameters.
- Resolution: use a different symbol for pose/joint rotations (e.g., `\vxi` or keep as-is but clearly distinguish with subscripts).

### 6.2 Encoder/Decoder Notation

- The chapter establishes `f_{\theta}` for encoder and `g_{\eta}` for decoder.
- The VAE section uses `\mathcal{E}` and `\mathcal{D}` for encoder and decoder.
- The RAE section uses `f` and `g` (without subscripts).
- The Michelangelo section uses `\mathcal{E}_{\mathrm{s}}`, `\mathcal{D}_{\mathrm{s}}`, `\mathcal{E}_{\mathrm{i}}`, `\mathcal{E}_{\mathrm{t}}`.
- The conditional generation section uses `\mathcal{E}` and `\mathcal{D}`.
- When the VAE encoder/decoder are architecturally different from the transformer-based `f_{\theta}`/`g_{\eta}`, the calligraphic notation `\cE`/`\cD` is acceptable, but should be acknowledged as distinct from the chapter's primary notation. At minimum, convert `\mathcal{E}` → `\cE` and `\mathcal{D}` → `\cD`.

### 6.3 Dataset Variable `\cD`

- In the overall setup, `\cD` is the set of possible data.
- In the VAE section, `\mathcal{D}` is used for both the dataset AND the decoder, creating ambiguity.
- Resolution: use `\cD` only for datasets; use a different symbol or subscripted version for the decoder.

### 6.4 Definitional Equality

- The chapter uses `\doteq` throughout.
- The Cupid section uses `\triangleq` in one place.
- Standardize to `\doteq`.

---

## 7. Specific Typos and Errors Found

These are concrete errors to fix (the proofreader should watch for similar patterns):

- "the the red block" → "the red block" (duplicated "the").
- "a fastly growing interest" → "a rapidly growing interest" ("fastly" is not standard English).
- "and and pose" → "and pose" (duplicated "and").
- "distribution of out 3D environment" → "distribution of our 3D environment" ("out" → "our").
- "it needs to convert" → "one needs to convert" or "we need to convert".
- "which is expressed as:" followed by display math should use a colon or no punctuation, consistently.

---

## 8. Commented-Out / Review Artifacts to Remove

The proofreader should flag (or the pre-processing pipeline should strip) the following before proofreading:

- All `% {\color{blue} Update: ...}` notes.
- All `% \yima{...}` review comments.
- Large blocks of commented-out text (many in the Cupid, EgoAllo, and Michelangelo sections).
- Manual spacing hacks: `\vspace{-5mm}`, `\vspace{-0.15in}`, `\vspace{-0.05in}`.
- `\xspace` usage in macro definitions (discouraged; use `{}` or `\ ` for spacing).

---

## 9. Punctuation in and around Math

- Inline math that ends a sentence: place the period OUTSIDE `\)`. Example: `...we obtain \(\vx\).`
- Display-math equations that end a sentence: place the period at the end of the equation, inside the environment.
- Equations followed by "where" clauses: place a comma at the end of the equation.
- Equations in the middle of a sentence (followed by more text): use no punctuation or a comma as grammatically appropriate.
- This is currently inconsistent across the chapter.

---

## 10. Summary of Section-by-Section Issues

| Section | Primary Issues |
|---|---|
| Introduction (8.0) | `{\em ...}`, mixed `\ref`/`\Cref`, "you" voice, "firstly/secondly" |
| DINO/SimDINO (8.1) | **Reference style** (well-written otherwise); uses `eqnarray` in places |
| CLIP (8.2) | Lowercase `\cref`, `\[...\]` display math, `\mathbb{R}`, `\|...\|` norms, Unicode quotes, some `$...$` |
| Classification/CRATE (8.3) | Minor: some `\cite` instead of `\citep` |
| MAE (8.4) | Minor: some notation slips |
| VAE (8.5.1) | `$...$`, `\mathbb{R/E}`, `\boldsymbol{...}`, `\odot`, `D_{KL}`, `(H,W,C)` ordering, `\mathcal{D}` overload |
| RAE (8.5.2) | `$...$`, `\cite` not `\citep`, mixed macros, bare figure paths |
| Conditional Gen (8.6) | `$...$`, `\boldsymbol{...}`, `\textbf{Title.}` for paragraphs, `Figure~\ref`, `\mathbb{R}`, `\bm{...}`, `\I`, tutorial tone |
| Michelangelo (8.7) | `$...$`, `{\em ...}`, `\hline` tables, `Figure~\ref`, `\mathbb{R}`, `\boldsymbol{...}`, `$$...$$` display math, paper-like tone, `\cite` |
| Cupid (8.8) | Local `\newcommand` conflicts, `$...$`, `\triangleq`, `Section~\ref`, bare figure paths, `\boldsymbol{...}`, `\[...\]`, `{\em ...}`, "the the" typo |
| EgoAllo (8.9) | `$...$` (partial), `\boldsymbol{...}`, `Chapter~\ref`, conversational tone |
| NLP/CRATE-GPT (8.10) | `{\em ...}` (minor), "Anyways" colloquialism, mostly well-written |
| Scaling (8.11) | Minor inconsistencies, mostly well-written |
