# Module59 SimpleSAC SUMO Eval Completed-History Certificate

```text
workload_key = sumo_eval_simple_sac_c_le2
record_count = 54
profile_domain = [1]
min_realized_eval_s = 0.001525164
```

| Quantity | Min | P10 | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| `duration_s` | 340.797087 | 367.355070 | 492.909650 | 631.247417 | 655.667244 |
| `realized_eval_s` | 0.001525 | 0.001584 | 0.002029 | 0.002694 | 0.002934 |

## Slowest Records

| Task | Method | Seed | OD | Duration s | Eval/s |
|---|---|---:|---:|---:|---:|
| `t2875` | `rlpd_nosnap_s42` | 1001 | 1.0 | 655.667 | 0.001525164 |
| `t2890` | `rlpd_nosnap_s123` | 1003 | 1.0 | 652.001 | 0.001533740 |
| `t2862` | `rlpd_nosnap_s789` | 1003 | 1.0 | 648.632 | 0.001541707 |
| `t2878` | `rlpd_nosnap_s42` | 1002 | 1.0 | 647.724 | 0.001543867 |
| `t2832` | `wsrl_nosnap_s42` | 1002 | 1.0 | 638.417 | 0.001566373 |
| `t2884` | `rlpd_nosnap_s123` | 1001 | 1.0 | 631.323 | 0.001583975 |
| `t2887` | `rlpd_nosnap_s123` | 1002 | 1.0 | 631.247 | 0.001584165 |
| `t2829` | `wsrl_nosnap_s42` | 1001 | 1.0 | 631.221 | 0.001584232 |
| `t2859` | `rlpd_nosnap_s789` | 1002 | 1.0 | 619.579 | 0.001613999 |
| `t2847` | `wsrl_nosnap_s789` | 1001 | 1.0 | 619.346 | 0.001614606 |
| `t2835` | `wsrl_nosnap_s42` | 1003 | 1.0 | 618.797 | 0.001616038 |
| `t2838` | `wsrl_nosnap_s123` | 1001 | 1.0 | 612.246 | 0.001633329 |

Strict completed-history lower-service certificate for clean SimpleSAC run_multiseed_eval.sh tasks. Only profile 1 is loaded; higher co-location profiles are not claimed.
