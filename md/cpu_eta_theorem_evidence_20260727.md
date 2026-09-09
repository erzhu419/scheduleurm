# CPU ETA and lower-service evidence table

The two profile axes are intentionally separate: `colocation_count` counts independent tasks, whereas `allocation_workers` is the CPU allocation of one task.

| evidence | workload/node | axis/profile | state | point ETA | holdout | conservative ETA | lower service | valid |
|---|---|---|---|---:|---:|---:|---:|:---:|
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p1 | empty | 5.808 | 5.977 | 6.806 | 23.508016 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p2 | empty | 6.404 | 6.547 | 7.505 | 21.319005 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p4 | empty | 8.843 | 8.591 | 10.363 | 15.439992 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p8 | empty | 8.717 | 8.854 | 10.216 | 15.661973 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p16 | empty | 9.110 | 9.226 | 10.676 | 14.986754 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p32 | empty | 9.713 | 10.045 | 11.382 | 14.057088 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p64 | empty | 12.034 | 11.627 | 14.102 | 11.345878 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p96 | empty | 17.059 | 16.865 | 19.991 | 8.003561 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p128 | empty | 19.740 | 20.171 | 23.134 | 6.916313 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p180 | empty | 26.354 | 25.849 | 30.884 | 5.180655 | yes |
| allocation_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p192 | empty | 27.497 | 27.709 | 32.224 | 4.965257 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p1 | half_loaded | 7.168 | 6.779 | 8.972 | 17.832984 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p1 | full_loaded | 13.275 | 13.468 | 16.616 | 9.629082 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p2 | half_loaded | 8.160 | 8.516 | 10.214 | 15.665148 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p2 | full_loaded | 14.143 | 13.870 | 17.703 | 9.038114 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p4 | half_loaded | 10.053 | 10.320 | 12.583 | 12.715786 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p4 | full_loaded | 14.687 | 14.182 | 18.384 | 8.703356 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p8 | half_loaded | 13.167 | 13.600 | 16.481 | 9.708367 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p8 | full_loaded | 18.447 | 16.944 | 23.090 | 6.929389 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p16 | half_loaded | 14.275 | 14.388 | 17.868 | 8.954499 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p32 | half_loaded | 15.395 | 14.981 | 19.270 | 8.303172 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p64 | half_loaded | 17.459 | 18.253 | 21.854 | 7.321448 | yes |
| controlled_loaded_worker_curve | cpu_heavy_local_bench / node001-node006 homogeneous class | allocation_workers=p96 | half_loaded | 23.456 | 21.629 | 29.360 | 5.449595 | yes |
| native_colocation | freqduet_cpu_native / node001 | colocation_count=p1 | empty | 149.971 | 141.703 | 193.219 | 0.031053 | yes |
| native_colocation | freqduet_cpu_native / node001 | colocation_count=p2 | controlled_colocation | 161.315 | 149.737 | 194.581 | 0.061671 | yes |
| native_colocation | freqduet_cpu_native / node001 | colocation_count=p4 | controlled_colocation | 176.454 | 177.325 | 193.301 | 0.124159 | yes |
| native_colocation | sumo_eval_cpu_native / node001 | colocation_count=p1 | empty | 30.735 | 31.267 | 38.940 | 0.154082 | yes |
| native_colocation | sumo_eval_cpu_native / node001 | colocation_count=p2 | controlled_colocation | 31.793 | 30.455 | 36.860 | 0.325555 | yes |
| native_colocation | sumo_eval_cpu_native / node001 | colocation_count=p4 | controlled_colocation | 32.969 | 33.739 | 37.250 | 0.644302 | yes |
| native_colocation | freqduet_cpu_native / node002 | colocation_count=p1 | empty | 149.541 | 145.377 | 196.177 | 0.030585 | yes |
| native_colocation | freqduet_cpu_native / node002 | colocation_count=p2 | controlled_colocation | 153.899 | 146.787 | 173.849 | 0.069025 | yes |
| native_colocation | freqduet_cpu_native / node002 | colocation_count=p4 | controlled_colocation | 176.897 | 179.993 | 199.428 | 0.120344 | yes |
| native_colocation | sumo_eval_cpu_native / node002 | colocation_count=p1 | empty | 30.520 | 29.906 | 39.196 | 0.153078 | yes |
| native_colocation | sumo_eval_cpu_native / node002 | colocation_count=p2 | controlled_colocation | 31.417 | 33.574 | 38.647 | 0.310502 | yes |
| native_colocation | sumo_eval_cpu_native / node002 | colocation_count=p4 | controlled_colocation | 33.137 | 32.988 | 37.166 | 0.645750 | yes |
| native_colocation | freqduet_cpu_native / node003 | colocation_count=p1 | empty | 153.144 | 155.729 | 192.902 | 0.031104 | yes |
| native_colocation | freqduet_cpu_native / node003 | colocation_count=p2 | controlled_colocation | 157.113 | 154.799 | 181.060 | 0.066276 | yes |
| native_colocation | freqduet_cpu_native / node003 | colocation_count=p4 | controlled_colocation | 173.875 | 185.566 | 196.036 | 0.122426 | yes |
| native_colocation | sumo_eval_cpu_native / node003 | colocation_count=p1 | empty | 30.779 | 30.460 | 40.239 | 0.149110 | yes |
| native_colocation | sumo_eval_cpu_native / node003 | colocation_count=p2 | controlled_colocation | 31.083 | 33.428 | 36.488 | 0.328875 | yes |
| native_colocation | sumo_eval_cpu_native / node003 | colocation_count=p4 | controlled_colocation | 32.869 | 32.887 | 37.059 | 0.647622 | yes |
| native_colocation | freqduet_cpu_native / node004 | colocation_count=p1 | cpu_resident_external | 161.881 | 163.167 | 213.918 | 0.028048 | yes |
| native_colocation | freqduet_cpu_native / node004 | colocation_count=p2 | controlled_colocation_external | 172.238 | 173.572 | 204.536 | 0.058669 | yes |
| native_colocation | freqduet_cpu_native / node004 | colocation_count=p4 | controlled_colocation_external | 181.310 | 178.699 | 210.693 | 0.113910 | yes |
| native_colocation | sumo_eval_cpu_native / node004 | colocation_count=p1 | cpu_resident_external | 32.602 | 33.621 | 41.639 | 0.144095 | yes |
| native_colocation | sumo_eval_cpu_native / node004 | colocation_count=p2 | controlled_colocation_external | 33.297 | 31.970 | 40.575 | 0.295746 | yes |
| native_colocation | sumo_eval_cpu_native / node004 | colocation_count=p4 | controlled_colocation_external | 35.796 | 37.221 | 38.882 | 0.617246 | yes |
| native_colocation | freqduet_cpu_native / node005 | colocation_count=p1 | empty | 146.816 | 139.689 | 200.873 | 0.029870 | yes |
| native_colocation | freqduet_cpu_native / node005 | colocation_count=p2 | controlled_colocation | 158.338 | 170.897 | 180.419 | 0.066512 | yes |
| native_colocation | freqduet_cpu_native / node005 | colocation_count=p4 | controlled_colocation | 173.420 | 187.443 | 193.791 | 0.123845 | yes |
| native_colocation | sumo_eval_cpu_native / node005 | colocation_count=p1 | empty | 30.720 | 28.901 | 37.581 | 0.159655 | yes |
| native_colocation | sumo_eval_cpu_native / node005 | colocation_count=p2 | controlled_colocation | 31.754 | 31.960 | 37.152 | 0.322997 | yes |
| native_colocation | sumo_eval_cpu_native / node005 | colocation_count=p4 | controlled_colocation | 33.227 | 33.288 | 37.494 | 0.640098 | yes |
| native_colocation | freqduet_cpu_native / node006 | colocation_count=p1 | cpu_resident_external | 160.595 | 169.150 | 189.478 | 0.031666 | yes |
| native_colocation | freqduet_cpu_native / node006 | colocation_count=p2 | controlled_colocation_external | 174.809 | 176.842 | 205.180 | 0.058485 | yes |
| native_colocation | freqduet_cpu_native / node006 | colocation_count=p4 | controlled_colocation_external | 175.313 | 181.431 | 197.655 | 0.121424 | yes |
| native_colocation | sumo_eval_cpu_native / node006 | colocation_count=p1 | cpu_resident_external | 30.910 | 31.509 | 38.294 | 0.156683 | yes |
| native_colocation | sumo_eval_cpu_native / node006 | colocation_count=p2 | controlled_colocation_external | 31.567 | 33.175 | 37.699 | 0.318309 | yes |
| native_colocation | sumo_eval_cpu_native / node006 | colocation_count=p4 | controlled_colocation_external | 34.006 | 33.621 | 37.903 | 0.633203 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p2 | cpu_external_moderate | 6.458 | 6.590 | 6.987 | 22.900645 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p4 | cpu_external_moderate | 8.290 | 8.002 | 8.970 | 17.837943 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p8 | cpu_external_moderate | 8.121 | 8.552 | 8.787 | 18.209409 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p16 | cpu_external_moderate | 8.486 | 8.221 | 9.181 | 17.427055 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p32 | cpu_external_moderate | 9.680 | 9.639 | 10.473 | 15.277027 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p64 | cpu_external_moderate | 14.997 | 16.095 | 16.226 | 9.860926 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p96 | cpu_external_moderate | 18.127 | 17.709 | 19.612 | 8.158253 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node004 | allocation_workers=p128 | cpu_external_moderate | 20.955 | 20.936 | 22.671 | 7.057325 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p2 | cpu_external_light | 6.362 | 6.200 | 6.883 | 23.246124 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p4 | cpu_external_light | 8.729 | 8.594 | 9.444 | 16.942145 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p8 | cpu_external_light | 8.372 | 7.426 | 9.058 | 17.664804 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p16 | cpu_external_light | 8.755 | 8.882 | 9.472 | 16.891763 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p32 | cpu_external_light | 9.764 | 9.847 | 10.564 | 15.146364 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p64 | cpu_external_light | 13.519 | 13.172 | 14.626 | 10.939227 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p96 | cpu_external_light | 17.323 | 17.488 | 18.742 | 8.536993 | yes |
| organic_external_worker_curve | cpu_heavy_local_bench / node006 | allocation_workers=p128 | cpu_external_light | 20.857 | 20.346 | 22.566 | 7.090268 | yes |

Only rows with task-native progress, natural completion, state-certified dispatch, and fresh holdout coverage are theorem-facing.
