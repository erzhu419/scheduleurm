# Scheduleurm Figure 1 Draft Comparison

Generated on 2026-06-14 for the OR-focused manuscript revision.

## Version A: Claude prompt + GPT image

File: `md/figures/scheduleurm_core_claude_gptimage2.png`

Prompt source: `md/figures/scheduleurm_core_figure_work/scheduleurm_core_claude_prompt.txt`

Strengths:

- Closest to the previous BAPR-HRO Figure 1 style: restrained, two-panel, white background.
- Shows both the robust candidate MaxWeight certificate and the four workload quadrants.
- Makes the external-policy-as-finite-actions boundary explicit.

Weaknesses:

- More caption-heavy inside the bitmap.
- Some mathematical text is acceptable for a concept draft but not as reliable as TeX-rendered text for final camera-ready submission.

## Version B: Codex prompt + GPT image

File: `md/figures/scheduleurm_core_codex_gptimage2.png`

Prompt source: `md/figures/scheduleurm_core_figure_work/scheduleurm_core_codex_prompt.txt`

Strengths:

- Best fit for the revised OR positioning: the visual hierarchy is
  `full action space -> measured four-quadrant candidate set -> robust MaxWeight certificate`.
- External policies are visually subordinate and correctly framed as diagnostic actions on the same service cache.
- The four quadrants are clearer than Version A.

Weaknesses:

- Still a raster image with AI-rendered text; final publication should ideally recreate this layout as SVG/TikZ/matplotlib so all labels and formulas are exact.

## Recommendation

Use Version B as the design direction.  For submission quality, recreate Version B as a clean vector figure with controlled labels:

- title: `Robust Candidate MaxWeight: finite actions, audited slack`;
- quadrants: `q00 Light/control`, `q01 GPU-heavy`, `q10 CPU-heavy`, `q11 Hybrid CPU+GPU`;
- score: `Q^T \underline{\mu}(a) - p(a)`;
- slack: `\delta > L\rho + \epsilon_{\mathrm{est}} + \beta + \alpha_1`;
- bottom boundary: `External policies enter as diagnostic actions on the same service cache, not universal full-stack superiority`.

## Final Vector Version

Files:

- `md/figures/scheduleurm_core_vector.pdf`
- `md/figures/scheduleurm_core_vector.svg`
- `md/figures/scheduleurm_core_vector.png`

Source:

- `scripts/make_scheduleurm_core_figure.py`

Manuscript integration:

- inserted in `paper/main.tex` as Figure 1 after the introduction's full-action-space motivation;
- uses the PDF vector version in LaTeX;
- avoids introducing a new lower-service symbol in the figure by writing `lower-service(a)` instead of a second mathematical symbol for `\underline{\mu}(a)`.
