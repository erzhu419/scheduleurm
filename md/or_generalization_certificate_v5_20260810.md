# OR Generalization Certificate v5

- Status: `OR_GENERALIZATION_V5_PASS_MIXED_REPRODUCIBLE_EVIDENCE`
- This certificate treats protocol validity, registered-family oracle exactness, reproducibility, and performance as separate claims.

| Domain | Evidence | Result | Retained boundary |
|---|---|---|---|
| FJSP | 20 frozen Hurink-family ledgers | candidate family nondominated 20/20; fixed-budget CP-SAT dominates 14/20 | no global FJSP optimum |
| MMRCPSP total queue | 21 scenario/seed cells | baseline dominates 21/21 | retained negative result |
| MMRCPSP class balance | 35 prospectively frozen arrival tapes | ours nondominated 35/35; strict-all 0/35 | known instance/action library |
| Port | BACASP-S source-core plus separate synthetic migration | source robust MaxWeight dominated by reconfiguration_greedy, spt_static; synthetic oracle gap 0.0, P0=2.54 | no industrial deployment |

## Mathematical interpretation

- The renewal-frame MaxWeight theorem controls drift under its slack assumptions; it is not a theorem of total-delay or makespan optimality.
- The shortest registered trajectory baseline dominates the one-project-at-a-time abstraction on aggregate delay costs.
- On prospectively frozen arrival tapes, the normalized robust policy is nondominated on total queue plus worst-class queue costs, but never strictly dominates every baseline.
- The two results identify an objective tradeoff and do not contradict the stability theorem or imply global delay superiority.

## Reproducibility

- MMRCPSP semantic reruns: 2/2 exact.
- Raw gzip container hashes differ because the FNAME header stores the output basename; decompressed results do not differ.

## Claim boundary

- Supports: a duration-normalized finite-trajectory scheduling interface across FJSP, MMRCPSP, and port models.
- Supports: exact registered-family oracle audits in all 35 MMRCPSP class-balance scenario/seed cells.
- Supports: a prospective arrival-tape MMRCPSP class-balance Pareto result with 35/35 nondominated cells.
- Supports: semantic reproduction of both deterministic MMRCPSP result matrices.
- Supports: statewise port transition feasibility with a measured finite-family oracle gap and penalty bound.
- Does not support: global FJSP, MMRCPSP, or port optimality.
- Does not support: universal delay, makespan, or performance superiority.
- Does not support: erasing the 21/21 MMRCPSP aggregate-delay counterexamples.
- Does not support: calling the known MMRCPSP instance/action library a new structural holdout.
- Does not support: positive recurrence inferred from finite replay.
- Does not support: industrial terminal deployment or physical port safety certification.
- Does not support: path-independent bitwise gzip identity.
