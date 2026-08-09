# jtl110gpu2 Loaded Wave 7 Retry 1 Exclusion

- Gate: `critical_gpu_loaded_wave_exclusion`
- Status: `EXCLUDED_EXTERNAL_PRODUCTION_CONTAMINATION`
- Protocol: `critical_gpu_loaded_natural_completion_v2`
- Split: calibration wave 7
- Excluded before LCB construction: `true`
- Performance outcome used for exclusion: `false`
- Ordinary running tasks touched: `false`

Both registered GPUs were idle before each scenario.  During the long
`rl_after_cnn` and `cnn_after_rl` scenarios, independent monitoring observed
new process groups outside the campaign.  The latter scenario included 34 GPU
compute observations from process groups `1778481` and `1778482`, while its
other registered GPU was no longer idle.  Conservative unapproved CPU exposure
also exceeded the registered `0.001` bound.

The wave is therefore retained as invalid external-contamination evidence and
cannot enter the stochastic lower confidence bound (LCB).  Its reports and raw
trees were moved to paths ending in
`_external_production_contamination_excluded_retry1`; hashes are recorded in
the JSON companion artifact.  The campaign will not terminate, pause, or alter
the external jobs.  It may repeat this same preregistered wave only after both
GPUs and those process groups have cleared for an idle guard interval.
