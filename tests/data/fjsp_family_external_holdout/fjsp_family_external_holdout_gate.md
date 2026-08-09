# Prospective FJSP family holdout

- Protocol gate: `FAIL`
- Registered instances: `20`
- Families: `{"BehnkeGeiger2012": 5, "ChambersBarnes1996": 5, "DauzerePeresPaulli1994": 5, "FattahiMehrabadJolai2007": 5}`
- Ours Pareto-nondominated: `19/20`
- Ours strictly dominates every registered baseline: `3/20`
- Geometric mean makespan/public UB: `1.106731`

| Family | Instance | Ours makespan/UB | Nondominated | Strict all-baseline dominance |
|---|---|---:|---:|---:|
| BehnkeGeiger2012 | behnke1 | 1.111111 | true | false |
| BehnkeGeiger2012 | behnke9 | 1.152000 | true | false |
| BehnkeGeiger2012 | behnke15 | 1.140351 | true | true |
| BehnkeGeiger2012 | behnke20 | 1.075377 | true | false |
| BehnkeGeiger2012 | behnke60 | 1.022388 | true | false |
| ChambersBarnes1996 | mt10c1 | 1.070119 | true | false |
| ChambersBarnes1996 | mt10xxx | 1.098039 | true | false |
| ChambersBarnes1996 | setb4xx | 1.157838 | false | false |
| ChambersBarnes1996 | seti5x | 1.135225 | true | false |
| ChambersBarnes1996 | seti5xyz | 1.151111 | true | false |
| DauzerePeresPaulli1994 | 01a | 1.200399 | true | false |
| DauzerePeresPaulli1994 | 05a | 1.153251 | true | false |
| DauzerePeresPaulli1994 | 09a | 1.036875 | true | false |
| DauzerePeresPaulli1994 | 13a | 1.254919 | true | false |
| DauzerePeresPaulli1994 | 18a | 1.064734 | true | false |
| FattahiMehrabadJolai2007 | fattahi1 | 1.000000 | true | false |
| FattahiMehrabadJolai2007 | fattahi5 | 1.000000 | true | true |
| FattahiMehrabadJolai2007 | fattahi10 | 1.000000 | true | false |
| FattahiMehrabadJolai2007 | fattahi15 | 1.173152 | true | true |
| FattahiMehrabadJolai2007 | fattahi20 | 1.183946 | true | false |

The candidate union contains the registered heuristic trajectories. Its
nondominance guarantee is therefore a construction property, not evidence of
superiority over arbitrary FJSP solvers. CP-SAT is a separate time-bounded
makespan reference; mean flow remains descriptive.
