# OR Submission Gap Closure Plan

Date: 2026-06-05

This note records the remaining gaps before the Scheduleurm line can be written
as an OR/Stochastic Systems paper. It is intentionally stricter than an
engineering experiment checklist: each item states what a reviewer can ask, what
must be shown, and what artifact should answer it.

## Current Assessment

The mathematical spine is close to OR-grade:

```text
finite-feature fabric-cover candidate approximation
-> robust candidate MaxWeight with explicit slack accounting
-> bounded conditional second moment / finite-set Foster recurrence
-> approximate-oracle and statewise feasible-family variants
```

The main theorem should be the approximate-oracle calibrated version:

```text
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle
```

For the actual scheduler with dynamic feasible families, the stronger aligned
statement is:

```text
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle
```

The paper is not yet submission-ready because the remaining work is mostly
artifact, calibration, and empirical closure, not abstract theorem invention.

## Gap 1: Theorem-Condition Calibration

Reviewer question:

```text
Your theorem needs δ > Lρ + ε_est + β + α1. Do your measured Scheduleurm data
actually satisfy this inequality?
```

Required output:

```text
L, rho, Lrho
epsilon_est
P0, beta
alpha0, alpha1
B
delta
eta = delta - (Lrho + epsilon_est + beta + alpha1)
finite-set threshold N from B + P0 + alpha0
```

Required artifact:

```text
algorithm/experiments/slack_accounting.py
md/experiment_module30_slack_accounting.md
```

Status:

```text
Infrastructure exists for separate components:
  fabric_metric.py        -> L, rho
  penalty_fit.py          -> P0, beta
  oracle_audit.py         -> alpha0, alpha1
  capacity_lp.py          -> delta, eta

Missing before this note:
  one consolidated slack-accounting table and pass/fail certificate.
```

## Gap 2: q10 Real CPU/Data-Loader Trace and q00 Control Bucket Closure

Reviewer question:

```text
Why is q10_cpu_host_bound based on a real workload measured under the same
benchmark discipline, and what bucket does that claim cover?
```

Required output:

```text
real CPU-heavy or data-loader-heavy progress-bearing workload;
profiles measured on a declared stable bucket;
legacy-comparable cap measured on that same bucket;
candidate profile measured or certified;
q10 taskset promoted only after those conditions hold.
```

Current status:

```text
modules25+33 have a real local CPU-heavy curve;
profiles 1-9 are measured;
profile 10 is a measured local capacity boundary;
active q10 taskset uses cpu_heavy_local_bench;
legacy cap is profile 9 on the same local bucket;
candidate profile is profile 8.
```

q00 status:

```text
modules23+24+38 have the exact local light-control curve;
profiles 1-13 are measured;
profile 14 is a measured local scheduling-capacity boundary;
active q00 taskset uses light_control_local;
legacy cap is profile 1 on the same local bucket;
candidate profile is profile 13.
```

Required artifact:

```text
md/experiment_module31_q10_real_cpu_trace.md
md/experiment_module38_q00_light_control_extended_curve.md
```

Remaining breadth item:

```text
Remote CPU-node or data-loader-heavy q10 replication is still useful, but it is
not required for the declared local CPU-bucket q10 comparison. Remote/light
control-plane replication is likewise useful for q00 generalization but not
required for the declared local q00 comparison.
```

## Gap 3: SOTA Wording

Reviewer question:

```text
Did you directly run Gavel/Pollux/Sia/IADeep, or are these stylized policy
baselines on the Scheduleurm service cache?
```

Required wording:

```text
We compare against SOTA-style replay baselines that reproduce policy semantics
on the same measured Scheduleurm service cache. We do not claim direct binary
execution of external schedulers unless a later experiment explicitly runs or
faithfully ports their full allocation semantics.
```

Current status:

```text
module29 has per-taskset SOTA comparisons and fixed-policy matrix;
no individual fixed SOTA-style policy Pareto-dominates Scheduleurm candidate.
```

## Gap 4: Replay-to-Live Validation

Reviewer question:

```text
Your replay result is plausible, but does it agree with small real live runs?
```

Required output:

```text
small q01 live run; completed by module39;
small q11 live run; completed by module46 after profile-10 robust boundary correction;
small portfolio live sanity run; completed by module47;
predicted all-job completion / mean-flow from replay;
observed live completion / progress-window JCT;
relative error and confidence interval;
explicit statement that long production jobs are not required to run to natural
completion for service-curve calibration.
```

Required artifact:

```text
algorithm/experiments/live_validation.py
algorithm/experiments/portfolio_live_proxy.py
md/experiment_module32_live_replay_sanity.md
md/experiment_module39_46_live_replay_sanity_and_q11_robust_boundary.md
md/experiment_module47_portfolio_live_sanity.md
```

Current status:

```text
q01 module39: usable_for_live_sanity = true
q11 module46: usable_for_live_sanity = true
portfolio module47: usable_for_live_sanity = true
portfolio replay-to-live relative errors:
  makespan  = 0.007349
  mean-flow = 0.005417
  p90-flow  = 0.007964
```

## Gap 5: Lean Artifact Repackaging

Reviewer question:

```text
Does the uploaded Lean artifact contain exactly the theorem names cited in the
paper, with a fresh build log and no sorry/admit/axiom?
```

Required output:

```text
fresh ScheduleurmUpload.lean
sha256 hash
git commit of proof repo
lake build Scheduleurm
lake env lean ScheduleurmUpload.lean
grep for sorry/admit/axiom
theorem-name search log
```

Required artifact:

```text
md/lean_verification_submission.md
```

## Gap 6: OR Related Work Framing

Reviewer question:

```text
Is this a scheduling-systems paper with some math, or a queueing/control paper
with a system-backed calibration?
```

Required framing:

```text
stochastic processing networks
MaxWeight / backpressure
capacity regions and support functions
approximate MaxWeight / restricted candidate controls
queueing networks with learning / unknown service
restless or regime-switching service extensions
```

Systems work such as Gavel/Pollux/Sia/IADeep should be used mainly for
experiment design and baseline semantics, not as the theoretical related-work
center.

Required artifact:

```text
md/or_related_work_outline.md
```

## Submission Readiness Standard

The paper can move from "research prototype" to "submission draft" only after:

```text
1. slack accounting table has eta > 0 or clearly explains why the current
   workload/load point is outside certified stability;
2. q10 and q00 are real for declared buckets, or explicitly removed from theorem-grade claims;
3. SOTA claims use the honest SOTA-style replay wording;
4. at least one small live validation connects replay to real completion/JCT;
5. Lean artifact is freshly repackaged and theorem names match the paper;
6. the main manuscript exists as a coherent OR paper, not a pile of md notes.
```
