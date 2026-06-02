# Module 4 Validation: TUI Slurm Virtual Tasks

Date: 2026-06-03

## Scope

This module adds a display-only TUI bridge for Slurm jobs discovered from node
snapshots.  It does not change scheduler placement, dispatch, hard rules,
candidate selection, or any algorithm policy.  The rows are virtual task-table
rows for operator visibility.

## Validation

Syntax validation:

```bash
python3 -m py_compile skill/tui.py skill/tests/test_tui_slurm_virtual_tasks.py
```

Targeted functional validation:

```bash
python3 - <<'PY'
import importlib.util, sys
from pathlib import Path
repo=Path('/home/erzhu419/mine_code/scheduleurm')
sys.path.insert(0, str(repo/'skill'))
spec=importlib.util.spec_from_file_location(
    'test_tui_slurm_virtual_tasks',
    repo/'skill'/'tests'/'test_tui_slurm_virtual_tasks.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fail=[]
def check(name, cond, diag=''):
    print(('PASS' if cond else 'FAIL'), name)
    if not cond:
        fail.append((name, diag))
for fn_name in [
    'test_tui_slurm_status_mapping',
    'test_tui_virtual_slurm_tasks_from_nodes',
]:
    getattr(mod, fn_name)(check, None)
if fail:
    raise SystemExit(f'{len(fail)} TUI checks failed')
PY
```

## Passed Checks

- `RUNNING` and `COMPLETING` map to TUI `running`.
- pending/configuring Slurm states map to TUI `queued`.
- terminal Slurm states stay terminal.
- unknown Slurm states map to `launching`.
- only live Slurm-cluster node jobs become virtual rows.
- rows already tracked by `slurm_job_id` are not duplicated.
- rows colliding with an existing `slurm:<job_id>` task id are not duplicated.
- GPU Slurm jobs expose table fields including owner, project, vram marker,
  external origin, and Slurm job id.

## Performance Impact

No performance A/B is applicable for this module.  The code is display-only and
does not affect scheduling decisions.
