# CPU allocation-worker lower-service certificate

- Gate: `PASS`
- Source: `/home/erzhu419/mine_code/scheduleurm/md/experiment_artifacts/native_cpu_allocation_worker_lcb_gate_20260727.json`
- Simultaneous split-conformal margin: `0.171893`
- Maximum fresh-holdout point error: `3.4943%`

| workers | point ETA (s) | holdout (s) | conservative ETA (s) | lower service (step/s) | covered |
|---:|---:|---:|---:|---:|:---:|
| 1 | 5.808 | 5.977 | 6.806 | 23.508016 | yes |
| 2 | 6.404 | 6.547 | 7.505 | 21.319005 | yes |
| 4 | 8.843 | 8.591 | 10.363 | 15.439992 | yes |
| 8 | 8.717 | 8.854 | 10.216 | 15.661973 | yes |
| 16 | 9.110 | 9.226 | 10.676 | 14.986754 | yes |
| 32 | 9.713 | 10.045 | 11.382 | 14.057088 | yes |
| 64 | 12.034 | 11.627 | 14.102 | 11.345878 | yes |
| 96 | 17.059 | 16.865 | 19.991 | 8.003561 | yes |
| 128 | 19.740 | 20.171 | 23.134 | 6.916313 | yes |
| 180 | 26.354 | 25.849 | 30.884 | 5.180655 | yes |
| 192 | 27.497 | 27.709 | 32.224 | 4.965257 | yes |

The gate covers the declared node001-node006 homogeneous CPU class, the measured empty dispatch regime, and allocation-worker profiles listed here. It does not relabel light/moderate external-load rows as empty and does not extrapolate to unmeasured worker counts.
