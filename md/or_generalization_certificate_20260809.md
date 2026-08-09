# OR Generalization Certificate

- Status: `OR_GENERALIZATION_CERTIFICATE_PASS_MIXED_EVIDENCE`
- Protocol completeness and performance superiority are separate gates.
- Counterexamples are retained and prevent a cross-domain dominance claim.

| Domain | Prospective protocol | Status | Pareto-nondominated | Strict every-baseline |
|---|---|---|---:|---:|
| FJSP | first public-family holdout | `FJSP_FAMILY_HOLDOUT_FAIL` | 19/20 | 3/20 |
| FJSP | Hurink confirmation | `FJSP_HURINK_HOLDOUT_PASS` | 19/20 | 0/20 |
| MMRCPSP | PSPLIB multi-family confirmation | `MMRCPSP_FAMILY_HOLDOUT_PASS` | 25/25 | 11/25 |
| Port | BACASP-S R87/R88 | `PORT_PUBLIC_EXTERNAL_HOLDOUT_PASS` | 12/54 | 3/54 |
| Port | factor-policy R85/R86 | `PORT_FACTOR_POLICY_HOLDOUT_PASS` | 16/54 | 1/54 |

## Retained negative evidence

- The first FJSP family holdout failed its complete-reference protocol and contains a dominated instance.
- The Hurink confirmation also contains one dominated instance.
- The R89/R90 BACASP-S adapter-failure artifact is retained; R87/R88 contains dominated cases.
- The original MMRCPSP failure remains bound to its nonprospective repair regression.

## Claim boundary

- Supports: hash-bound external-instance protocol portability across FJSP, MMRCPSP, and BACASP-S
- Supports: prospective multi-family MMRCPSP evidence with retained earlier failure
- Supports: prospective FJSP and port counterexamples that delimit the finite selector
- Supports: finite trajectory-family oracle and bounded-penalty accounting
- Does not support: global FJSP, MMRCPSP, berth-allocation, or quay-crane optimality
- Does not support: dominance over arbitrary exact or state-of-the-art domain solvers
- Does not support: physical terminal deployment or measured industrial migration cost
- Does not support: automatic transfer of server stochastic stability without a domain arrival and service model
