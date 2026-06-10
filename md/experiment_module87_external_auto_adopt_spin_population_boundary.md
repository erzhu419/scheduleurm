# Module87 External Auto-Adopt Spin Population Boundary

Date: 2026-06-11

Module87 is a production-population boundary correction, not a service-rate
certificate.  It excludes external `freqduet_autoadopt_spin.py` helper
processes from the controlled-arrival theorem population only when the record is
auto-adopted from an external origin and has no scheduler id, no scheduler log,
and no reproducible progress unit.

```text
excluded command shape = freqduet_autoadopt_spin.py
population label = excluded_external_auto_adopted_unobservable
completed-active records removed from cpu_sumo_transit_eval_or_control = 2
service cache change = none
theorem role = controlled-arrival boundary, not capacity/service measurement
```

This follows the same operational semantics as the earlier exclusion of
external stdin/wait-for processes: Scheduleurm cannot claim stability for
arbitrary external helper processes that it did not launch and cannot assign a
progress-bearing service unit to.  Normal FreqDuet ablation, runner, baseline,
preflight, analysis, and merge records remain mapped through their existing
service certificates.

After this correction and the current queue-state refresh, the regenerated
Module53 CPU/SUMO/Transit residual is:

```text
cpu_sumo_transit_eval_or_control = 7 / 3280 completed-active production records
first_probe_order =
  transit_freqhrl_cpu_validation|c_33_64
  transit_freqhrl_cpu_validation|c_9_16
  transit_freqhrl_cpu_validation|c_le2
  bamor_cpu_training|c_3_8
```
