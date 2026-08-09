# BACASP-S factor-conditioned policy development

- Status: `PORT_FACTOR_POLICY_DEVELOPMENT_PASS`
- Candidate plans: `31`
- Factor cells: `27`
- Distinct selected plans: `6`
- Development replications: `R87/R88/R89/R90`
- Confirmation replications used: `false`

| Factor cell | Selected plan | Maximum regret | Mean regret |
|---|---|---:|---:|
| `Q10|S160-0.2|D1` | `plan|reconfiguration_greedy` | 0.066773 | 0.012169 |
| `Q10|S160-0.2|D1.5` | `plan|reconfiguration_greedy` | 0.067963 | 0.011591 |
| `Q10|S160-0.2|D2` | `plan|reconfiguration_greedy` | 0.064795 | 0.010462 |
| `Q10|S200-0.15|D1` | `plan|reconfiguration_greedy+reconfiguration_greedy+robust_maxweight` | 0.097964 | 0.019296 |
| `Q10|S200-0.15|D1.5` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.155599 | 0.025020 |
| `Q10|S200-0.15|D2` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.169051 | 0.026029 |
| `Q10|S240-0.1|D1` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.053867 | 0.007774 |
| `Q10|S240-0.1|D1.5` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.025860 | 0.006323 |
| `Q10|S240-0.1|D2` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.027100 | 0.006634 |
| `Q15|S160-0.2|D1` | `plan|spt_static+robust_maxweight+robust_maxweight` | 0.065501 | 0.021037 |
| `Q15|S160-0.2|D1.5` | `plan|spt_static+robust_maxweight+robust_maxweight` | 0.077891 | 0.022150 |
| `Q15|S160-0.2|D2` | `plan|spt_static+robust_maxweight+robust_maxweight` | 0.081276 | 0.022243 |
| `Q15|S200-0.15|D1` | `plan|robust_maxweight+reconfiguration_greedy+reconfiguration_greedy` | 0.020976 | 0.003448 |
| `Q15|S200-0.15|D1.5` | `plan|robust_maxweight+reconfiguration_greedy+reconfiguration_greedy` | 0.009087 | 0.001847 |
| `Q15|S200-0.15|D2` | `plan|robust_maxweight+reconfiguration_greedy+reconfiguration_greedy` | 0.009192 | 0.001868 |
| `Q15|S240-0.1|D1` | `plan|robust_maxweight+robust_maxweight+reconfiguration_greedy` | 0.023483 | 0.007291 |
| `Q15|S240-0.1|D1.5` | `plan|robust_maxweight+robust_maxweight+reconfiguration_greedy` | 0.029578 | 0.008154 |
| `Q15|S240-0.1|D2` | `plan|robust_maxweight+robust_maxweight+reconfiguration_greedy` | 0.026754 | 0.007751 |
| `Q5|S160-0.2|D1` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.149600 | 0.023969 |
| `Q5|S160-0.2|D1.5` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.148912 | 0.023789 |
| `Q5|S160-0.2|D2` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.203821 | 0.027755 |
| `Q5|S200-0.15|D1` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.036547 | 0.016659 |
| `Q5|S200-0.15|D1.5` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.035369 | 0.013874 |
| `Q5|S200-0.15|D2` | `plan|reconfiguration_greedy+reconfiguration_greedy+robust_maxweight` | 0.036184 | 0.011463 |
| `Q5|S240-0.1|D1` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.086709 | 0.025759 |
| `Q5|S240-0.1|D1.5` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.120039 | 0.028834 |
| `Q5|S240-0.1|D2` | `plan|reconfiguration_greedy+robust_maxweight+reconfiguration_greedy` | 0.185104 | 0.034509 |

This artifact is development evidence only. It makes no performance claim
on R85/R86 and no claim about unrestricted berth allocation, physical
terminal deployment, or stochastic port stability.
