# Prospective Hurink FJSP holdout

- Protocol status: `FJSP_HURINK_HOLDOUT_PASS`
- Instances: `20`
- Subfamilies: `{"edata": 5, "rdata": 5, "sdata": 5, "vdata": 5}`
- Pareto-nondominated: `19/20`
- Strict every-baseline dominance: `0/20`
- Complete CP-SAT reference rows: `20/20`
- Geometric mean makespan/public UB: `1.116705`

| Subfamily | Instance | Makespan/UB | Nondominated | CP-SAT reference |
|---|---|---:|---:|---:|
| edata | car2 | 1.189031 | true | FEASIBLE |
| edata | la10 | 1.072748 | true | OPTIMAL |
| edata | la20 | 1.126021 | true | OPTIMAL |
| edata | la27 | 1.239627 | true | FEASIBLE |
| edata | abz9 | 1.277950 | true | FEASIBLE |
| rdata | car2 | 1.216876 | true | FEASIBLE |
| rdata | la10 | 1.055970 | true | FEASIBLE |
| rdata | la20 | 1.072751 | true | OPTIMAL |
| rdata | la27 | 1.168664 | true | FEASIBLE |
| rdata | abz9 | 1.235075 | true | FEASIBLE |
| sdata | car2 | 1.084426 | true | OPTIMAL |
| sdata | la10 | 1.000000 | true | OPTIMAL |
| sdata | la20 | 1.058758 | true | OPTIMAL |
| sdata | la27 | 1.194332 | true | FEASIBLE |
| sdata | abz9 | 1.258112 | true | FEASIBLE |
| vdata | car2 | 1.054141 | true | FEASIBLE |
| vdata | la10 | 1.033582 | false | FEASIBLE |
| vdata | la20 | 1.000000 | true | OPTIMAL |
| vdata | la27 | 1.017528 | true | FEASIBLE |
| vdata | abz9 | 1.050302 | true | FEASIBLE |

Source/policy protocol validity is independent of empirical dominance and
time-bounded CP-SAT coverage. UNKNOWN solver rows remain missing references,
not protocol failures or evidence of optimality.
