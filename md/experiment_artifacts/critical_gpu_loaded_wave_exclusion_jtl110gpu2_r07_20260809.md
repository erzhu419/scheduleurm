# jtl110gpu2 Loaded Wave 7 Exclusion

- Gate: `critical_gpu_loaded_wave_exclusion`
- Status: `EXCLUDED_HOST_CPU_CONTAMINATION`
- Protocol: `critical_gpu_loaded_natural_completion_v2`
- Split: calibration wave 7
- Excluded before LCB construction: `true`
- Performance outcome used for exclusion: `false`
- Ordinary running tasks touched: `false`

The independent host-process audit observed an unapproved `python3` process during
`rl_after_cnn`.  Its conservative exposure was `0.0013422150572821772`, above
the preregistered `0.001` threshold.  No unapproved GPU process was observed.
The completed benchmark rows are therefore retained as invalid evidence but are
not eligible for the stochastic lower confidence bound (LCB) sample.

The four campaign/raw trees and both campaign and runner reports were moved to
paths ending in `_host_cpu_contamination_excluded`.  Their SHA-256 values are
recorded in the JSON companion artifact.  The replacement reruns the same wave,
with the same frozen code manifest and registered configuration, and is admitted
only if its independent audits pass.  This rule is exogenous to the measured
performance values and prevents outcome-based sample selection.
