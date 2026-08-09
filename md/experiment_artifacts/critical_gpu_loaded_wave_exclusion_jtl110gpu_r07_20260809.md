# jtl110gpu Loaded Wave 7 Exclusion

- Gate: `critical_gpu_loaded_wave_exclusion`
- Status: `EXCLUDED_EXTERNAL_CPU_CONTAMINATION`
- Protocol: `critical_gpu_loaded_natural_completion_v2`
- Split: calibration wave 7
- Excluded before LCB construction: `true`
- Performance outcome used for exclusion: `false`
- Ordinary running tasks touched: `false`

The independent continuous audit observed one `python` process outside the two
allowed campaign process groups during `rl_after_cnn`.  Its conservative CPU
exposure was `0.001240516989646551`, above the registered `0.001` bound.  No
unapproved GPU compute process was observed, and the other registered GPU
remained idle.  The process source cannot be established from the recorded
command name alone, so the wave is excluded rather than reclassified.

All reports and raw trees are retained at paths ending in
`_external_cpu_contamination_excluded`, with hashes in the JSON companion.  A
replacement may run only after five consecutive one-minute idle checks.  No
remote progress-log diagnostics will be launched while future theorem-facing
waves are active; monitoring is restricted to local durable-state evidence.
