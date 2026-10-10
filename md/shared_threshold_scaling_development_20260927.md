# Four-catalogue R4/R8/R16 scaling

The frozen panel completed all 12 cells: 1,500 states and 7,500 cold search calls.
Queues use `(0,1,3,4,10)^3 × R/2`; comparison budgets are 9 and 36R, with a separate
K4096 reference. All references closed; the 500 R4 states also passed independent
Fraction enumeration. Twenty-three runner tests passed.

At budget 36R, both methods certify the same full-catalogue action in every state:

| R | LP calls, full → shared | Matrix coefficients, full → shared |
| --- | ---: | ---: |
| 4 | 8,445 → 7,373 | 549,503 → 405,650 |
| 8 | 7,500 → 6,480 | 462,347 → 332,019 |
| 16 | 9,803 → 7,701 | 676,004 → 414,669 |

Pooled reductions are 16.3% in LP calls and 31.7% in matrix coefficients. Every
node/R cell improves both totals; no state increases LP calls. At K9, shared
search still changes 176 actions and loses up to 20% of optimal score; retain
this limited-budget result. [Frozen result](experiment_artifacts/aligned_hardware_shared_threshold_scaling_screen_development_20260927.json): `MULTI_CATALOGUE_SCALING_ADVANCE`.

Next freeze independent queue samples to confirm action identity and compute
savings. These are replicated cached point-model lanes, not new physical GPU
measurements or wall-time/flow-time evidence. The paper remains unchanged.
