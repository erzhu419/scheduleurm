# node007 Loaded Wave 5 Exclusion and Controlled Abort

- Gate: `critical_gpu_loaded_wave_exclusion`
- Status: `EXCLUDED_EXTERNAL_PRODUCTION_CONTAMINATION_CONTROLLED_ABORT`
- Protocol: `critical_gpu_loaded_natural_completion_v2`
- Split: calibration wave 5
- Excluded before LCB construction: `true`
- Performance outcome used for exclusion: `false`
- Ordinary running tasks touched: `false`

Sixteen BAPR production tasks (`t79247` through `t79262`) were launched across
all four GPUs after the wave began.  The campaign monitor recorded 160
unapproved GPU observations during `llm_after_cnn` and 21,104 during
`rl_after_cnn`; the other registered GPUs were not idle.  The wave was already
ineligible independently of its performance outcome.

To avoid wasting shared resources, only the process tree rooted at the exact
campaign control directory was terminated.  Both supervisor and command
cmdlines were identity-checked first.  The production tasks were neither
signaled nor modified.  The resulting third row records target return code 143
and the campaign correctly reports `INCOMPLETE`.

Reports and the three available raw trees are retained at paths ending in
`_external_production_contamination_controlled_abort`, with hashes in the JSON
companion.  Wave 5 may be rerun only after all four GPUs remain idle for five
consecutive one-minute checks.
