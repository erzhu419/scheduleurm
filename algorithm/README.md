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
theorem_maxweight_v1  certified lower-service robust MaxWeight GPU placement hook
adaptive_trace  experiment-side queue-adaptive MaxWeight-penalty replay policy
```

Reviewer-facing boundary:

```text
sweetspot_v1 is an integration hook for the live scheduler: it scores one
pending task against node/GPU placements and optional admission gates.

theorem_maxweight_v1 is the theorem-facing live hook.  It only scores a GPU
candidate with robust MaxWeight semantics when the task and post-placement
profile bind to an exact positive non-boundary service-cache row.  Its audit
fields include queue_vector, lower_service, penalty_units,
robust_maxweight_score, score_semantics, and best_score_exact_over_candidate_set.
Set SCHEDULEURM_THEOREM_UNCERTIFIED_MODE=block for theorem-grade probes that
must exclude uncertified candidates from the emitted trace.

The theorem-facing robust MaxWeight object is stronger and lives in the
replay/oracle pipeline: each candidate row must expose queue_vector,
lower_service, penalty_units, selected action, score_semantics, and an
oracle-gap audit.  Do not claim that enabling sweetspot_v1 alone makes every
live dispatch a global robust MaxWeight decision.

The 2026-06-11 OR closure runner uses `adaptive_trace_policy.py` for replay:
large backlogs prioritize measured service support, while bounded profile
penalties choose lower-interference profiles in finite sublevel sets.  This is
an experiment/theorem-facing replay policy, not a direct edit to
`skill/scheduler.py`.

The ablation gate in `or_submission_closure.py` keeps the reviewer axes
separate: legacy caps, scalar sweetspot hook, support-only scoring, delay-only
scoring, statewise guard, no profile penalty, profile tie-break direction, and
the full robust lower-service scorer are separate replay policies.
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
adaptive_learning.py  deterministic active-bucket sampler plus bounded two-window detector
features.py     class/regime/candidate-bucket keys and finite-feature metric
scoring.py      bounded robust score components used by sweetspot_v1
candidates.py   active-bucket representative selection utilities
placement.py    scheduler-facing policy interface
experiments/    manifest and trace/bootstrap utilities
  fabric_metric.py  rho cover audit and L service-Lipschitz calibration
  global_fabric_cover_calibration.py  exact measured-slice L,rho table plus greedy candidate-cover Lrho curves
  slot_builder.py   queue recurrence and nonnegative arrival/service records
  action_model.py   per-slot full/candidate/chosen action-family validation
  service_model.py  lower-service LCB rows and epsilon_est residual
  penalty_fit.py    P0,beta finite envelope
  oracle_audit.py   alpha0,alpha1 approximate-oracle envelope
  capacity_lp.py    full-action capacity slack and drift margin eta,N
  adaptive_trace_policy.py  queue-adaptive MaxWeight-penalty replay policy
  live_trace_dryrun.py      live-node-state synthetic placement trace without dispatch
  natural_live_theorem_trace.py  theorem_maxweight_v1 live-node candidate trace without dispatch
  live_trace_realization_bridge.py  trace-to-queue/archive progress/completion boundary audit
  admission_population_gate.py  service-domain theorem population admission certificate
  direct_sota_baseline_scaffold.py  external SOTA repo/entrypoint scaffold plus replay fallback
  direct_sota_fullstack_readiness.py  read-only external full-stack readiness and blocker certificate
  production_live_theorem_trace_gate.py  real-queue gate for production-wide live theorem-trace claims
  production_shadow_theorem_trace.py  non-invasive active-production shadow theorem hook trace
  learning_regime_certificate.py  active-bucket union-bound and hidden-regime dwell/switching certificate
  adaptive_sampler_detector_certificate.py  concrete sampler/detector probability certificate
  or_submission_closure.py  online/holdout/ablation/live-trace/supplement gate runner
  sota_adapters/ Scheduleurm taskset/service-cache export seeds for Gavel, AdaptDL/Pollux, IADeep, Salus, and Decima
  progress_units.py task-native progress parser: step/s, iter/s, s/iter
  report.py         calibration table / PASS-FAIL helpers
  service_curve_validation.py  pinned live co-location service-curve runner
  workload_service_curve_validation.py  fixed-profile runner for real commands with strict placement and capacity-boundary reporting
```

These are experimental controls plus theorem-calibration utilities.  Live
scheduler behavior is still selected by policy name; theorem claims must be
made from the calibration artifacts, not from raw throughput curves or scalar
placement scores alone.
