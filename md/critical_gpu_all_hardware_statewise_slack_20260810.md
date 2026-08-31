# Critical GPU All-Hardware Statewise Slack Certificate

- Status: `PASS`
- Load fraction: `0.8`
- Minimum eta: `0.04999999999999999`
- Four-node union-bound coverage: `0.6`

| Node | Classes | delta | eta | B | Per-node coverage | Ready |
|---|---:|---:|---:|---:|---:|:---:|
| `jtl110gpu` | 4 | 0.05 | 0.05 | 4 | 0.9 | true |
| `jtl110gpu2` | 4 | 0.05 | 0.05 | 4 | 0.9 | true |
| `node007` | 4 | 0.05 | 0.05 | 4 | 0.9 | true |
| `jtl311linux` | 4 | 0.05 | 0.05 | 4 | 0.9 | true |

PASS certifies four separate hardware-local, four-class stochastic models at the declared load, conditional on the hash-linked one-sided LCB events. Diagonal normalization resolves heterogeneous service units. The simultaneous four-node value is only the displayed union-bound lower bound; the primary statistical statement remains per-node. This does not estimate organic production arrivals, certify loaded co-location dynamics, or pool rates across physical nodes.
