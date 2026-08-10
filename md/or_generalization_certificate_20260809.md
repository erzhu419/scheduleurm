# OR Generalization Certificate

- Status: `OR_GENERALIZATION_CERTIFICATE_PASS_MIXED_EVIDENCE`
- Protocol completeness and performance superiority are separate gates.
- Numeric dispositions and substantive counterexamples are kept separate.

| Domain | Prospective protocol | Status | Pareto-nondominated | Strict every-baseline | Fixed CP-SAT dominates |
|---|---|---|---:|---:|---:|
| FJSP | first public-family holdout | `FJSP_FAMILY_HOLDOUT_FAIL` | 20/20 | 3/20 | 7/20 |
| FJSP | Hurink confirmation | `FJSP_HURINK_HOLDOUT_PASS` | 20/20 | 0/20 | 15/20 |
| MMRCPSP | PSPLIB multi-family confirmation | `MMRCPSP_FAMILY_HOLDOUT_PASS` | 25/25 | 11/25 | n/a |
| Port | BACASP-S R87/R88 | `PORT_PUBLIC_EXTERNAL_HOLDOUT_PASS` | 12/54 | 3/54 | n/a |
| Port | factor-policy R85/R86 | `PORT_FACTOR_POLICY_HOLDOUT_PASS` | 16/54 | 1/54 | n/a |

## Retained negative evidence

- The immutable FJSP source artifacts each reported 19/20 under mixed floating-point precision; the exact-ledger disposition reclassifies one same-action row in each artifact and yields 20/20 baseline-union nondominance.
- Fixed-budget CP-SAT trajectories still dominate the generated family on 7/20 first-family and 15/20 Hurink rows; this is a substantive candidate-family gap, not a numerical artifact.
- The R89/R90 BACASP-S adapter-failure artifact is retained; R87/R88 contains dominated cases.
- The original MMRCPSP failure remains bound to its nonprospective repair regression.

## Claim boundary

- Supports: hash-bound external-instance protocol portability across FJSP, MMRCPSP, and BACASP-S
- Supports: prospective multi-family MMRCPSP evidence with retained earlier failure
- Supports: exact-ledger FJSP baseline-union nondominance with immutable source artifacts
- Supports: fixed-budget CP-SAT and port counterexamples that delimit the finite selector
- Supports: finite trajectory-family oracle and bounded-penalty accounting
- Does not support: global FJSP, MMRCPSP, berth-allocation, or quay-crane optimality
- Does not support: dominance over arbitrary exact or state-of-the-art domain solvers
- Does not support: physical terminal deployment or measured industrial migration cost
- Does not support: automatic transfer of server stochastic stability without a domain arrival and service model
