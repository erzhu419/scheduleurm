# Critical GPU Statewise Slack Certificate

- Status: `PASS`
- Load fraction: `0.8`
- Minimum eta: `0.04999999999999999`

| Node | Classes | delta | error budget | eta | B | Ready |
|---|---:|---:|---:|---:|---:|:---:|
| `jtl110gpu` | 4 | 0.05 | 0 | 0.05 | 4 | true |
| `jtl110gpu2` | 4 | 0.05 | 0 | 0.05 | 4 | true |
| `node007` | 4 | 0.05 | 0 | 0.05 | 4 | true |

PASS is a constructive theorem-condition certificate for three separate hardware-local, four-class stochastic models at the declared 0.8 load. The stationary mix and Bernoulli arrivals are explicit; diagonal normalization resolves heterogeneous step/iter units; exact measured actions give Lrho=0 and the admitted one-sided lower service gives epsilon_est=0 on each gate's coverage event. It is not an estimate of organic production arrival rates, does not certify mixed-workload co-location actions, and does not include the pending jtl311linux class. Per-node coverage is 0.9; a joint claim over all three nodes has only the stated 0.7 union-bound lower bound.
