# Module 3 Validation: Theorem Calibration Surface

Date: 2026-06-03

## Scope

This module validates the experiment-side calibration layer that turns logged
scheduler evidence into the constants used by the Lean theorem:

- statewise full/candidate/chosen action-family reconstruction
- fabric cover radius `rho`, service Lipschitz constant `L`, and `Lrho`
- queue recurrence `Q(k+1)=max(Q(k)-S(k),0)+A(k)`
- nonnegative progress/service censoring
- lower-service LCB rows and `epsilon_est`
- penalty envelope `P0 + beta ||Q||_1`
- approximate-oracle envelope `alpha0 + alpha1 ||Q||_1`
- full-action capacity slack `delta`
- final drift margin `eta = delta - (Lrho + epsilon_est + beta + alpha1)`
- reviewer-facing calibration table with `PASS` / `FAIL` / `EMPIRICAL-OPEN`

This is not a live scheduler performance A/B module.  It does not change
placement behavior.  It is the evidence pipeline required before any live
performance run can be theorem-grade rather than a raw throughput curve.

## Validation Commands

Syntax/bytecode validation:

```bash
python3 -m py_compile \
  algorithm/experiments/action_model.py \
  algorithm/experiments/capacity_lp.py \
  algorithm/experiments/fabric_metric.py \
  algorithm/experiments/oracle_audit.py \
  algorithm/experiments/penalty_fit.py \
  algorithm/experiments/report.py \
  algorithm/experiments/service_model.py \
  algorithm/experiments/slot_builder.py \
  skill/tests/test_experiment_calibration.py
```

Targeted validation harness:

```bash
python3 - <<'PY'
import importlib.util
from pathlib import Path
repo=Path('/home/erzhu419/mine_code/scheduleurm')
spec=importlib.util.spec_from_file_location(
    'test_experiment_calibration',
    repo/'skill'/'tests'/'test_experiment_calibration.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fail=[]
def check(name, cond, diag=''):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fail.append((name, diag))
for fn_name in [
    'test_fabric_cover_and_lipschitz_calibration',
    'test_slot_builder_queue_recurrence_and_censoring',
    'test_experiment_action_family_reconstruction',
    'test_service_lower_bound_and_epsilon_est',
    'test_penalty_and_oracle_envelopes',
    'test_capacity_lp_and_drift_margin',
]:
    getattr(mod, fn_name)(check, None)
if fail:
    raise SystemExit(f'{len(fail)} calibration checks failed')
PY
```

CLI smoke validation:

```bash
python3 -m algorithm.experiments.action_model build ...
python3 -m algorithm.experiments.fabric_metric calibrate ...
python3 -m algorithm.experiments.slot_builder step ...
python3 -m algorithm.experiments.service_model calibrate ...
python3 -m algorithm.experiments.penalty_fit fit ...
python3 -m algorithm.experiments.oracle_audit audit ...
python3 -m algorithm.experiments.capacity_lp solve ...
python3 -m algorithm.experiments.report build ...
```

All commands passed on synthetic theorem-shaped fixtures.

## Passed Checks

- Cover audit computes max-min distance and refuses uncovered candidate sets.
- Lipschitz calibration catches zero-metric service disagreement.
- Action-family report refuses missing chosen actions and accepts `all_feasible`
  as a full statewise family.
- Service residual estimation refuses insufficient/unusable lower-bound rows.
- Capacity LP now refuses zero-slack theorem certificates.
- Report generation marks `eta <= 0` as `FAIL`.
- Report generation marks missing constants as `EMPIRICAL-OPEN`, not zero.
- Report generation accepts per-class constants such as `Amax_i` and `Smax_i`.

## Theorem Alignment

The schema target is:

```text
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle
```

The calibrated constants match the proof-side condition:

```text
eta = delta - (L * rho + epsilon_est + beta + alpha1) > 0
B + P0 + alpha0 + alpha <= eta * N
```

This module is therefore ready to be used by the next live experiment module.
