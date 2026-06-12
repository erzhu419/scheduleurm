# Live Trace Dry-Run Probe

| Quantity | Value |
|---|---:|
| pass | true |
| trace_slot_count | 8 |
| candidate_count_total | 50 |
| placed_count | 8 |
| unplaced_count | 0 |
| algorithm | `theorem_maxweight_v1` |

## Scope

synthetic task records against current live node probe; queue is not modified and no task is launched

## Placements

| task | kind | placement |
|---|---|---|
| `dryrun-oracle-000` | gpu | `{"gpu_idx": 0, "node": "local"}` |
| `dryrun-oracle-001` | gpu | `{"gpu_idx": 0, "node": "gpu_b"}` |
| `dryrun-oracle-002` | cpu | `{"gpu_idx": null, "node": "windows_b"}` |
| `dryrun-oracle-003` | cpu | `{"gpu_idx": null, "node": "windows_a"}` |
| `dryrun-oracle-004` | cpu | `{"gpu_idx": null, "node": "windows_b"}` |
| `dryrun-oracle-005` | cpu | `{"gpu_idx": null, "node": "windows_a"}` |
| `dryrun-oracle-006` | cpu | `{"gpu_idx": null, "node": "windows_b"}` |
| `dryrun-oracle-007` | cpu | `{"gpu_idx": null, "node": "windows_a"}` |
