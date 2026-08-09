# Prospective PSPLIB MMRCPSP family holdout

- Protocol gate: `PASS`
- Registered instances: `25`
- Families: `{"j12": 5, "j14": 5, "j16": 5, "j18": 5, "j20": 5}`
- Ours Pareto-nondominated: `25/25`
- Ours strictly dominates every registered heuristic: `11/25`
- Ours attains public optimal makespan: `7/25`
- Geometric mean makespan/optimum: `1.166743`

| Family | Instance | Makespan/optimum | Nondominated | Strict all-heuristic dominance |
|---|---|---:|---:|---:|
| j12 | j1212_3 | 1.200000 | true | true |
| j12 | j1212_6 | 1.000000 | true | false |
| j12 | j1244_1 | 1.416667 | true | false |
| j12 | j1215_9 | 1.000000 | true | false |
| j12 | j122_8 | 1.000000 | true | false |
| j14 | j1422_1 | 1.000000 | true | false |
| j14 | j1413_10 | 1.083333 | true | true |
| j14 | j1463_5 | 1.000000 | true | false |
| j14 | j1432_3 | 1.000000 | true | false |
| j14 | j1438_8 | 1.222222 | true | true |
| j16 | j1653_9 | 1.029412 | true | true |
| j16 | j1661_5 | 1.103448 | true | true |
| j16 | j1635_9 | 1.266667 | true | false |
| j16 | j1638_2 | 1.125000 | true | true |
| j16 | j1612_4 | 1.533333 | true | false |
| j18 | j1810_7 | 1.100000 | true | true |
| j18 | j1834_2 | 1.384615 | true | true |
| j18 | j1842_10 | 1.384615 | true | false |
| j18 | j1838_8 | 1.170732 | true | true |
| j18 | j1846_2 | 1.576923 | true | false |
| j20 | j2041_2 | 1.100000 | true | false |
| j20 | j2056_3 | 1.208333 | true | true |
| j20 | j2011_4 | 1.172414 | true | true |
| j20 | j2064_8 | 1.000000 | true | false |
| j20 | j2045_1 | 1.393939 | true | false |

Protocol validity does not imply performance superiority. The registered
heuristic union is contained in the generated candidate family; its
nondominance property is therefore constructional. Public optima and the
time-bounded CP-SAT sidecar concern makespan only, while mean flow remains
a descriptive second objective.
