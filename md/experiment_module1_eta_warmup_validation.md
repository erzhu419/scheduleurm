# Module 1 Validation: ETA Warmup Threshold

Date: 2026-06-02, Asia/Shanghai.

## Scope

Module under validation:

- `skill/eta_tracker.py`
- `skill/tests/test_eta_warmup_validation.py`

This module is intentionally tested before placement/admission changes because Scheduleurm's service-rate and load-balance experiments depend on ETA/progress signals. Early progress-rate ETA used to turn startup-heavy jobs into absurd multi-thousand-hour projections.

## Change

`compute_eta_seconds` and `runtime_projection` now require an adaptive minimum progress count before trusting cumulative `current / elapsed` rate math:

```text
threshold(total) = max(3, min(20, ceil(total / 100)))
```

Explicit tqdm ETA and inline ETA still take priority. If progress is below the threshold, the code uses historical EWMA remainder when available; otherwise it reports unknown ETA (`0`).

## Validation Metric

Metric:

```text
early_eta_inflation = old_rate_math_eta / new_eta
```

For cases where the new ETA is intentionally unknown (`0`), the improvement is treated as eliminating a false numeric ETA rather than producing a smaller numeric estimate.

Pass criteria:

- early `Iter 1/2000` does not produce a numeric multi-month ETA;
- early-but-not-trusted `Iter 19/2000` uses history instead of cumulative startup rate;
- once the threshold is crossed (`Iter 20/2000`), progress-rate ETA still works;
- tqdm ETA still overrides threshold/rate math.

## Targeted Results

Command:

```bash
python3 -m py_compile skill/eta_tracker.py skill/tests/test_eta_warmup_validation.py
python3 - <<'PY'
import importlib.util
from pathlib import Path
import skill.scheduler as sch
path = Path('skill/tests/test_eta_warmup_validation.py').resolve()
spec = importlib.util.spec_from_file_location('eta_warmup_validation', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
results=[]
def check(name, cond, diag=''):
    results.append((name, bool(cond), diag))
    print(('PASS' if cond else 'FAIL'), name)
for attr in sorted(dir(mod)):
    if attr.startswith('test_') and callable(getattr(mod, attr)):
        getattr(mod, attr)(check, sch)
failed=[r for r in results if not r[1]]
print('summary', len(results)-len(failed), 'pass', len(failed), 'fail')
raise SystemExit(1 if failed else 0)
PY
```

Result:

```text
summary 6 pass 0 fail
```

Comparison table:

| Case | Old ETA seconds | New ETA seconds | Result |
|---|---:|---:|---|
| `Iter 1/2000`, no history, elapsed 1h | 7,196,400 | 0 | false numeric ETA removed |
| `Iter 19/2000`, 6h history, elapsed 1h | 375,347 | 18,000 | 20.85x smaller, uses history remainder |
| `Iter 20/2000`, no history, elapsed 1h | 356,400 | 356,400 | threshold crossed; rate ETA restored |
| `Epoch 2/200`, 1h history, elapsed 10m | 59,399 | 3,000 | 19.80x smaller, uses history remainder |
| `Epoch 3/200`, no history, elapsed 10m | 39,400 | 39,400 | threshold crossed; rate ETA restored |
| tqdm `10/100 [00:42<03:21]` | 32,400 | 201 | tqdm ETA still wins |

## Full Regression Status

Full command attempted:

```bash
python3 skill/test_regression.py
```

It did not finish cleanly, but the blockers are not caused by this ETA module:

- existing crash-requeue stale launch artifact assertions failed early;
- existing history LRU assertion failed;
- the runner later calls `python` instead of `python3`, and this environment has no `python` executable.

Because of those unrelated blockers, the acceptance gate for this module is the targeted ETA suite plus `py_compile`.

## Performance Verdict

PASS.

The module improves the ETA signal used by Scheduleurm load accounting and later service-curve validation by suppressing startup-dominated projections until progress is sufficiently informative. It does not reduce theorem scope and does not enable any placement/admission policy yet.

