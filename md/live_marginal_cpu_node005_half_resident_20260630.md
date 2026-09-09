# Live Marginal Under-Load Matrix

- Status: `LIVE_MARGINAL_UNDER_LOAD_OPEN`
- Allow launch: `true`
- Pass: `false`
- Admitted rows: `0`
- Boundary rows: `1`

| Scenario | Node | State | Workload | Stable rate | Valid | Reason |
|---|---|---|---|---:|---:|---|
| `node005_cpu_half_resident_add_cpu` | `node005` | `cpu_resident` | `cpu_heavy_local_bench` | 15.6313 | false | `background_progress_not_ready` |

Controlled add-one probes under resident load. These rows certify marginal ETA/service for matching resource_state only; they do not replace empty-resource rows.
