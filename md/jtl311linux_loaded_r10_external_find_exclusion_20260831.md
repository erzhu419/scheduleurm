# jtl311linux loaded r10 external-process exclusion

Wave 10 completed all four registered loaded trajectories naturally, but the
continuous host-process audit rejected `rl_after_cnn`. One target-interval
snapshot observed an unapproved process group containing `bash`, `find`, and
`wc`; `find` used 150% CPU. Its conservative full-host exposure was
`0.001183120258570789`, above the preregistered `0.001` limit.

The process group could not be tied to the resident, target, or a registered
measurement-instrumentation group from the available hash-bound log. The row is
therefore treated as externally contaminated rather than allowlisted by command
name. The original completion and runner artifacts are retained with a
`preaudit_excluded_find` suffix and are excluded from canonical discovery.

Wave 10 must be remeasured under the same measurement manifest and natural-
completion protocol. A clean remeasurement may replace the canonical wave-10
artifact; the excluded observation remains diagnostic and does not enter the
split-conformal calibration population.
