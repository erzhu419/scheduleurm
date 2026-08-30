# jtl311linux loaded r04 monitor epilogue decision

## Observed failure

The controlled loaded-state completion campaign for `jtl311linux` wave 4
finished all four registered resident/target trajectories naturally.  The
completion artifact passed, but the safe wave runner initially returned
`ROW_PREFLIGHT_AUDIT_FAILED` for `rl_after_cnn`.

The GPU monitor contained 1,947 samples and ended with one interrupted sample
header at nanosecond timestamp `1788131243748869770`.  That header had no GPU
rows because the monitor was terminated while opening its next sample.  The
interrupted header occurred after the target interval ended at
`1788130018823254151`; the preceding complete samples already covered the full
target interval.  The other three scenario audits passed unchanged.

## Resolution

The continuous-load parser now drops an incomplete sample only when all of the
following hold:

1. it is the only incomplete sample;
2. it is the final sample in the file; and
3. its timestamp is strictly after the audited target interval.

The dropped timestamp is recorded as
`dropped_interrupted_epilogue_sample_ns`.  An incomplete sample inside the
target interval, before the final sample, or accompanied by another incomplete
sample still fails closed with `GPU_MONITOR_SAMPLES_INVALID`.

After this change, the original wave-4 monitor logs pass all four prelaunch and
continuous-load audits.  No service observation, completion model, overlap
interval, or workload output was modified, and the wave was not rerun.
