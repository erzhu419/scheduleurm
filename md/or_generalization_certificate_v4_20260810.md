# OR Generalization Certificate v4

- Status: `OR_GENERALIZATION_V4_PASS_MIXED_EVIDENCE`
- Protocol validity, finite-family oracle exactness, and performance superiority remain separate claims.

| Domain | New evidence | Protocol | Performance disposition | Boundary |
|---|---|---|---|---|
| FJSP | 20 unused Hurink-family instances | `FJSP_EXACT_LEDGER_CONFIRMATION_V2_PASS` | candidate family nondominated 20/20; CP-SAT dominates 14/20 | no global optimum |
| MMRCPSP | 25-class renewal stream, 105 runs | `MMRCPSP_RENEWAL_STREAM_PROTOCOL_PASS` | ours nondominated 0/21 scenario/seeds | finite drift only |
| Port | statewise synthetic transition model | `PORT_STATEWISE_TRAJECTORY_V3_PASS` | oracle gap 0.0; P0=2.54 | no industrial deployment |

## Reproducibility and retained counterexamples

- FJSP frozen candidate ledgers reproduce 20/20.
- CP-SAT qualitative relations reproduce 20/20, but exact numeric gaps reproduce only 17/20 because the comparator is wall-clock limited.
- The MMRCPSP stream does not convert empirical tail drift into a recurrence theorem.
- The BACASP source-core and synthetic migration certificates remain non-substitutable.

## Claim Boundary

- Supports: a common duration-normalized finite-trajectory interface across FJSP, MMRCPSP, and port scheduling
- Supports: pathwise queue recurrence, workload conservation, and drift-identity audits for the registered MMRCPSP stream
- Supports: persistent-resource and generalized transition-cost semantics in the synthetic port statewise model
- Supports: exact integer-ledger FJSP candidate-family comparisons with an independent fixed-budget CP-SAT reference
- Does not support: global FJSP or MMRCPSP optimality
- Does not support: dominance over unrestricted exact or state-of-the-art domain solvers
- Does not support: bitwise repeatability of wall-clock-limited CP-SAT incumbents
- Does not support: industrial terminal deployment or physical port safety certification
- Does not support: positive recurrence inferred from finite replay
