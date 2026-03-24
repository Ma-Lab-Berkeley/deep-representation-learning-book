# Math Macro Substitution

Replace raw LaTeX commands with the book's standard macros. This is purely mechanical find-and-replace.

## Bold Vector/Matrix Macros

- Use `\vx` (not `$\x$`, `$\boldsymbol{x}$`, `\mathbf{x}`, or `\bm{x}`) for bold lowercase vectors.
- Use `\vX` (not `$\boldsymbol{X}$`) for bold uppercase matrices.
- Use `\vphi`, `\vtheta`, `\vmu`, `\vSigma`, `\vepsilon` (not `$\boldsymbol{\phi}$`, `$\boldsymbol{\theta}$`, etc.) for bold Greek letters.
- Use `\vzero` for the zero vector (not `$\boldsymbol{0}$` or `$\Zero$`), `\vone` for the ones vector, `\vI` for identity (not `$\boldsymbol{I}$` or `\I`).
- The `\v` prefix means bold: `\vx`, `\vy`, `\vA`, `\vB`, `\vphi`, `\vPhi`, etc.
- There is no special matrix font; just use `\vA = \boldsymbol{A}`, etc.

## Calligraphic and Blackboard-Bold Macros

- `\c` prefix for calligraphic: `\cA`, `\cB`, `\cD`, `\cL`, `\cN`, `\cT`, `\cX`, etc. (not `$\mathcal{A}$`).
- `\bb` prefix for calligraphic: `\bbA`, `\bbB`, `\bbD`, `\bbL`, `\bbN`, `\bbT`, `\bbX`, etc. (not `$\mathbb{A}$`).
- Some exceptions:
    - `\R` for real numbers (not `$\mathbb{R}$`). Similarly `\C` for complex, `\N` for naturals.
    - `\Ex` for expectation (not `$\mathbb{E}$`).
    - `\Pr` for probability (not `$\mathbb{P}$`).

## Operator Macros

- Use `\KL` for KL divergence (not `$D_{KL}$` or `$D_{\mathrm{KL}}$`).
- Use `\CE` for cross-entropy.
- Use `\hada` for element-wise (Hadamard) multiplication (not `$\odot$`).
- Use `\softmax`, `\ReLU`, `\MHSA`, `\MSSA`, `\MLP`, `\LN`, `\SA`, `\ISTA` as operator macros where defined.
- Use `\doteq` for definitional equality (not `:=` or `\triangleq`).

## Inner Products, Norms, and Distances

- Use `\ip{\vx}{\vy}` for inner products (not `\langle \vx, \vy \rangle` or `\left\langle ... \right\rangle`).
- Use `\norm{\vx}_{2}` for norms (not `\|\vx\|_2` or `\left\| ... \right\|`).

## Matrix Macro

- Use `\mat{...}` instead of `\begin{bmatrix}...\end{bmatrix}` or similar.

## Probability 

- Wherever there is a non-subscripted probability density function of a random variable `p(\vx)` (say), replace it with a subscripted version `p_{\vx}`. 
    - Exception: only if the PDF has a subscript already, e.g., `p_{t}` is the density of `x_{t}`.
    - Remember to ONLY do this when the density/mass is EXPLICITLY related to a random variable. DO NOT do this otherwise.

---

Please also ensure that the notation keeps with the following notation table:

<!-- include: chapters/notation.tex -->
