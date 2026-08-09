# OR Generalization Certificate

- Status: `OR_GENERALIZATION_CERTIFICATE_PASS`
- Common action: complete finite configuration trajectory.
- Common selector: exact generated-family maximizer of lower service minus bounded penalty.

| Domain | Protocol | Instances | Ours Pareto-nondominated | Strong global claim |
|---|---|---:|---:|---:|
| FJSP | post-freeze Kacem holdout | 4 | 4/4 | false |
| MMRCPSP | disjoint post-repair PSPLIB holdout | 56 | 56/56 | false |
| Port | public BACASP-S plus registered synthetic four-resource instances | 4 | 4/4 | false |

## MMRCPSP prospective repair protocol

- v1 holdout: `FAIL`, first counterexample `j102_10`.
- Opened-row repair regression: `56/56`; not prospective.
- Disjoint v2 holdout: `56/56` Pareto-nondominated.

## Claim boundary

- Supports: finite trajectory-action portability across compute, port, FJSP, and MMRCPSP models
- Supports: post-freeze external holdout evidence for the registered FJSP and repaired MMRCPSP suites
- Supports: hash-bound public-source and synthetic four-resource port evidence
- Supports: exact generated-family oracle and bounded-penalty certificates
- Does not support: global FJSP, MMRCPSP, or continuous-port optimality
- Does not support: dominance over arbitrary exact or state-of-the-art domain solvers
- Does not support: physical port deployment or measured industrial reconfiguration cost
- Does not support: automatic transfer of the server stochastic-stability certificate without a domain arrival/service model
