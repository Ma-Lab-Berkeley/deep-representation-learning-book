# Math Style and Formatting

## Subscripts and Superscripts

- ALWAYS use braces: `x_{n}` not `x_n`, `x^{n}` not `x^n`.
- Same rule applies to hats, tildes, etc.: `\hat{\vx}` not `\hat\vx`, `\tilde{\vx}` not `\tilde\vx`.

## Bracket Sizing and Shortcuts

- Use `\bp{...}` in place of `\left(...\right)`, `\bs{...}` in place of `\left[...\right]`, and `\bc{...}` in place of `\left\{...\right\}`. For brackets used right after another symbol (such as function application) where we don't want added space, use `\rp{...}`, `\rs{...}`, and `\rc{...}` respectively. ONLY use these to replace `\left`/`\right`. 
- Remove unnecessary `\left`/`\right`, `\Big`, `\bigg` when the enclosed content does not require grown brackets.
- Only use explicit sizing when the content is genuinely tall (e.g., fractions, sums with limits).

## Function Notation

- Use `\colon` for function typing: `f \colon X \to Y` (not `f : X \to Y`).

## Punctuation in Display Equations

- Use punctuation at the end of displaystyle equations wherever appropriate.
