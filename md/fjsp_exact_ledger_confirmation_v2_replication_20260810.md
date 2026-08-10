# Prospective FJSP exact-ledger confirmation v2

- Status: `FJSP_EXACT_LEDGER_CONFIRMATION_V2_PASS`
- Registered instances: `20`
- Exact baseline-union nondominance: `20/20`
- Feasible fixed-budget CP-SAT references: `20/20`
- Fixed-budget CP-SAT dominates generated family: `14/20`
- Performance does not determine protocol validity.
- A feasible or makespan-optimal CP-SAT row is not a proof of multiobjective global optimality.

| Family | Instance | Baseline-union nondominated | CP-SAT relation | CP-SAT optimal |
|---|---|---:|---|---:|
| edata | edata/ft06.txt | true | tradeoff | true |
| edata | edata/la09.txt | true | comparator_dominates | true |
| edata | edata/la19.txt | true | comparator_dominates | true |
| edata | edata/la26.txt | true | comparator_dominates | false |
| edata | edata/abz8.txt | true | comparator_dominates | false |
| rdata | rdata/ft06.txt | true | tradeoff | true |
| rdata | rdata/la09.txt | true | comparator_dominates | false |
| rdata | rdata/la19.txt | true | comparator_dominates | true |
| rdata | rdata/la26.txt | true | comparator_dominates | false |
| rdata | rdata/abz8.txt | true | comparator_dominates | false |
| sdata | sdata/ft06.txt | true | comparator_dominates | true |
| sdata | sdata/la09.txt | true | same_action | true |
| sdata | sdata/la19.txt | true | tradeoff | true |
| sdata | sdata/la26.txt | true | comparator_dominates | false |
| sdata | sdata/abz8.txt | true | comparator_dominates | false |
| vdata | vdata/ft06.txt | true | same_action | true |
| vdata | vdata/la09.txt | true | comparator_dominates | false |
| vdata | vdata/la19.txt | true | comparator_dominates | true |
| vdata | vdata/la26.txt | true | comparator_dominates | false |
| vdata | vdata/abz8.txt | true | same_action | false |

{'scope': 'only preregistered instances and the frozen generated family', 'candidate_family_and_cp_sat_reference_are_distinct_evidence': True, 'cp_sat_feasible_without_optimal_status_is_not_an_optimality_proof': True, 'cp_sat_makespan_optimality_is_not_multiobjective_optimality': True, 'global_fjsp_optimality_claimed': False}
