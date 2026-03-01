# Chapter 8 Notation Conflicts

## Parameter Variable `\theta`

- Throughout the chapter, `\theta` is used for neural network parameters.
- The Cupid section redefines `\pose = \boldsymbol{\theta}` for camera pose --- this conflicts with network parameters `\theta`.
- The EgoAllo section uses `\theta` for SMPL joint rotations AND for network parameters.
- Resolution: use a different symbol for pose/joint rotations (e.g., `\vxi` or keep as-is but clearly distinguish with subscripts).

## Encoder/Decoder Notation

- The chapter establishes `f_{\theta}` for encoder and `g_{\eta}` for decoder.
- The VAE section uses `\mathcal{E}` and `\mathcal{D}` for encoder and decoder.
- The RAE section uses `f` and `g` (without subscripts).
- The Michelangelo section uses `\mathcal{E}_{\mathrm{s}}`, `\mathcal{D}_{\mathrm{s}}`, `\mathcal{E}_{\mathrm{i}}`, `\mathcal{E}_{\mathrm{t}}`.
- The conditional generation section uses `\mathcal{E}` and `\mathcal{D}`.
- When the VAE encoder/decoder are architecturally different from the transformer-based `f_{\theta}`/`g_{\eta}`, the calligraphic notation `\cE`/`\cD` is acceptable, but should be acknowledged as distinct from the chapter's primary notation. At minimum, convert `\mathcal{E}` -> `\cE` and `\mathcal{D}` -> `\cD`.

## Dataset Variable `\cD`

- In the overall setup, `\cD` is the set of possible data.
- In the VAE section, `\mathcal{D}` is used for both the dataset AND the decoder, creating ambiguity.
- Resolution: use `\cD` only for datasets; use a different symbol or subscripted version for the decoder.

## Definitional Equality

- The chapter uses `\doteq` throughout.
- Standardize to `\doteq` (not `:=` or `\triangleq`).
