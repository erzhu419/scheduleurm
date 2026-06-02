# scheduleurm algorithm layer

`scheduler.py` imports optional placement policies from this top-level package.
The default policy is `legacy`, which preserves existing scheduler behavior.
Non-legacy policies are kept here so experiments can change the action selection
surface without rewriting the scheduler.

Select a policy without editing scheduler code:

```bash
SCHEDULEURM_ALGORITHM=sweetspot_v1 python3 skill/scheduler.py dispatch
python3 skill/scheduler.py dispatch --algorithm sweetspot_v1
python3 skill/scheduler.py watch --algorithm sweetspot_v1
```

For the systemd watcher, write parameters to:

```text
~/.claude/scheduler/scheduler.env
```

then restart:

```bash
systemctl --user restart scheduler
```

Current policies:

```text
legacy          no behavior change
sweetspot_v1    finite-feature robust candidate scoring plus optional gates
```

Useful parameters for `sweetspot_v1`:

```text
SCHEDULEURM_ALGO_GPU_SWEET_SPOT_TASKS=3
SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU=4
SCHEDULEURM_ALGO_MAX_POST_VRAM_FRAC=0.82
SCHEDULEURM_ALGO_MAX_GPU_UTIL_PCT=95
SCHEDULEURM_ALGO_WEIGHT_VRAM=120
SCHEDULEURM_ALGO_WEIGHT_UTIL=8
SCHEDULEURM_ALGO_WEIGHT_COLOCATION=3
SCHEDULEURM_ALGO_WEIGHT_SWEET_GAP=2
SCHEDULEURM_ALGO_WEIGHT_OVER_SWEET=25
SCHEDULEURM_ALGO_WEIGHT_RUNTIME=0.20
SCHEDULEURM_ALGO_WEIGHT_RUNTIME_UNKNOWN=15
SCHEDULEURM_ALGO_WEIGHT_PRIORITY_REWARD=3
SCHEDULEURM_ALGO_WEIGHT_QUEUE_AGE_REWARD=0.20
SCHEDULEURM_ALGO_BETA_PENALTY=0
SCHEDULEURM_ALGO_BOUNDED_PENALTY_CAP=1000
```

Code map:

```text
action_model.py complete global-action schema for candidate logs
features.py     class/regime/candidate-bucket keys and finite-feature metric
scoring.py      bounded robust score components used by sweetspot_v1
candidates.py   active-bucket representative selection utilities
placement.py    scheduler-facing policy interface
experiments/    manifest and trace/bootstrap utilities
  fabric_metric.py  rho cover audit and L service-Lipschitz calibration
  slot_builder.py   queue recurrence and nonnegative arrival/service records
  action_model.py   per-slot full/candidate/chosen action-family validation
  service_model.py  lower-service LCB rows and epsilon_est residual
  penalty_fit.py    P0,beta finite envelope
  oracle_audit.py   alpha0,alpha1 approximate-oracle envelope
  capacity_lp.py    full-action capacity slack and drift margin eta,N
  report.py         calibration table / PASS-FAIL helpers
```

These are experimental controls plus theorem-calibration utilities.  Live
scheduler behavior is still selected by policy name; theorem claims must be
made from the calibration artifacts, not from raw throughput curves alone.
