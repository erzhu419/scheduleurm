# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_READY`
- Allow launch: `true`
- Pass: `true`
- Admitted rows: `2`
- Boundary rows: `0`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `node003_cpu_full_resident_add_cpu` | `node003` | `cpu_resident` | `cpu_heavy_local_bench` | 12.5307 | true | `` |
| `node005_cpu_half_resident_add_cpu` | `node005` | `cpu_resident` | `cpu_heavy_local_bench` | 16.3924 | true | `` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
