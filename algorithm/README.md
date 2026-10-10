# scheduleurm algorithm layer

`scheduler.py` imports optional placement policies from this top-level package.
The default policy is `legacy`, which preserves existing scheduler behavior.
Non-legacy policies are kept here so experiments can change the action selection
surface without rewriting the scheduler.

## Current main result (2026-10-03)

The adopted research policy is **BG** (`BackgroundCompetitionContinuousLaneRollout`).
Independent confirmation used 96 paired tapes: 12 fresh arrival-seed blocks,
four measured catalogues and loads 0.8/0.95. Relative to exact enumerated
single-lane MaxWeight, mean flow decreased **8.33%** and the mean of paired
common-completion P90 values decreased **17.03%**. See the
[confirmation evidence](../md/background_competition_continuous_20260930.md)
and [current result record](../md/main_result_BG_20261003.md).

Evidence covers the frozen asynchronous independent-lane model with fixed-rate
Poisson arrivals, deterministic measured-cache service and zero computation
delay in the service clock. BG is a research replay policy; live placement
integration and an asynchronous stability proof remain incomplete.

The finite branch is closed as exploratory negative evidence, retaining its
original full24 **HOLD** and subsequent results. Its pending later-action
intervention design is parked. The experimental history below remains archived.

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

## Aligned complete-job frame experiment

`experiments/aligned_hardware_frame_model.py` registers the two-GPU q01
`gpu_b` scenario with three canonical job classes and nine lane options.
Each option completes its admitted jobs, including resident initialization and
finalization for loaded options. Selection, realized departures, elapsed time,
capacity and conditional drift all use this same frame model.

`budgeted_frame_candidates.py` constructs candidates lazily. The budget includes
idle, one maximal feasible homogeneous seed per local option, and rejected
construction attempts. Every decision returns the chosen score and an upper
bound on the unexplored family. Local scoring and bound probes have separate
counters. The minimum budget is 10; the seed support bound and verified capacity
projections give a uniform drift bound for each fixed budget at least this large.
Ordering reuses paid seed feasibility, moving unavailable profiles after the
available profiles while keeping their weighted-count order. Upper bounds cap
each service coordinate by the remaining queue and reuse precomputed threshold
and suffix maxima. Table construction probes and threshold queries are counted
separately from candidate attempts.

Run locally from the repository root:

```bash
python3 -m pip install -r algorithm/experiments/requirements-aligned-frame.txt
python3 -m algorithm.experiments.aligned_hardware_frame_experiment
python3 -m algorithm.experiments.aligned_hardware_frame_comparison
python3 -m pytest -q skill/tests/test_budgeted_frame*.py skill/tests/test_aligned_hardware_frame*.py
```

The default experiment uses a 20,000-second Poisson arrival window followed by
draining, loads 0.50/0.80/0.95, seeds 41/42/43, and budgets 10/14/26/120.
Budget 120 covers the exact reference search; budgets share each arrival tape.
The current output is `md/experiment_artifacts/aligned_hardware_frame_experiment_v4_20260909.json`.
The comparison command pairs it with the original `aligned_hardware_frame_experiment_20260909.json`
and writes `aligned_hardware_frame_comparison_v4_20260909.json` in the same directory.
It checks that the model, protocol, certificates and job populations agree,
then reports seed-level ratios and ranges. Each version uses its own exact
reference, since search ordering can change which equal-score action is selected.
The v2 rate-order and v3 seed-score experiment outputs remain as development
comparators; `--baseline`, `--current` and `--output` select a different comparison.
It reads the local service cache and loaded-action ledger and generates no paper
assets. `--horizon`, `--loads`, `--seeds`, `--budgets` and `--output` allow smaller
runs; the exact reference is always included.

Limitations: this is a deterministic cached phase model with random arrivals.
Its certificate does not extend to physical execution or the earlier asynchronous
replay with lognormal job sizes. Four- and eight-lane catalogue probes measure
search construction only. Candidate budgets bound construction attempts, not
wall time or upper-bound probe counts. These reused seeds support development
comparisons; independent arrival-seed confirmation is a separate experiment.

Register and run the frozen v4 confirmation:

```bash
python3 -m algorithm.experiments.aligned_hardware_frame_confirmation --register
python3 -m algorithm.experiments.aligned_hardware_frame_confirmation --run
```

This fixes seeds 1041–1050 and preserves the original loads, budgets, horizon,
model and completion rules. The historical v1 selector and v4 use the same
runner and arrival tapes. The registration and exact source texts are saved
before running; existing confirmation results are preserved.
Outputs are in `md/experiment_artifacts/aligned_hardware_frame_confirmation_v4_20260909/`.
`confirmation.json` separates execution checks from four primary claims:
K26 improvement over v1 at load 0.50, and K26 mean flow within 0.2% of v4 exact
at each load. Seed-paired arithmetic ratio means use four Bonferroni-adjusted
Student t intervals; all other budgets remain descriptive comparisons.

The 2026-09-09 confirmation completed all 240 runs on the ten registered seeds.
At load 0.50, K26 reduced mean flow relative to v1 by 14.77%; the adjusted
interval for the reduction was 14.02–15.52%. The three-load 0.2% equivalence claim
was not confirmed: at load 0.50, K26 was faster than its exact reference beyond
that band; at loads 0.80 and 0.95, equivalence remained inconclusive. Exact search
maximizes each frame's support score, which does not make it a mean-flow oracle.

The development mechanism diagnosis runs separately:

```bash
python3 -m algorithm.experiments.aligned_hardware_frame_mechanism
python3 -m pytest -q skill/tests/test_aligned_frame_flow_decomposition.py skill/tests/test_aligned_hardware_frame_mechanism.py
```

Its 45 runs use seeds 41–43 and reconstruct each job's arrival-barrier wait,
subsequent queue wait and dispatch-to-completion time. Same-queue shadow decisions,
strict-loss-only and equal-score-only substitutions, and one first-difference
substitution separate action effects from mere action frequencies. The frozen
selector, simulator and confirmation outputs remain unchanged. Output is
`md/experiment_artifacts/aligned_hardware_frame_mechanism_development_20260909.json`.
The four improved K26/exact pairs obtain their gains through reduced subsequent
queue wait, principally for CNN jobs. Equal-score substitutions reproduce exact
mean flow, while strict-loss substitutions reproduce K26 mean flow in these nine
pairs. One-action substitutions improve three pairs and worsen six; their effects
are trajectory-dependent and do not establish a generally superior action.

The seed-constrained holding-cost experiment is a separate development selector:

```bash
python3 -m algorithm.experiments.aligned_hardware_waiting_cost
python3 -m pytest -q skill/tests/test_waiting_cost_frame_candidates.py skill/tests/test_aligned_hardware_waiting_cost.py
```

`waiting_cost_frame_candidates.py` retains v4's paid candidate generation and
support pruning. Among candidates with support at least the best homogeneous
seed, it minimizes `H = sum(Q)-sum(D)+sum(completion_offsets)/tau+Lambda*tau/2`.
This is the expected unfinished-job area per frame duration under constant
Poisson arrivals. The seed floor preserves the original drift bound. Cost
evaluations and completion-offset reads are additional reported work; the final
support gap and search completion flag are recorded separately.

The 72 development runs retain seeds 41–43, all three loads and four budgets.
All execution checks passed and all 36 v4 controls reproduced their stored
metrics. At the predeclared K26, paired mean-flow changes relative to v4 were
−2.53%, +27.62% and +25.10% at loads 0.50, 0.80 and 0.95 respectively, with all
three seeds agreeing in each direction. At the two higher loads, subsequent
queue waiting increased by 9.11 s and 12.91 s per job, overwhelming the smaller
arrival-barrier and completion times. Results and the saved protocol are in
`md/experiment_artifacts/aligned_hardware_waiting_cost_development_20260909.json`
and its `_protocol.json` companion. The seed-only rule is retained as a
development control.

The subsequent support-cap comparison fixes a 1% maximum loss relative to the
best paid candidate, with a 0% control for changes among equal-support actions:

```bash
python3 -m algorithm.experiments.aligned_hardware_support_cap
python3 -m pytest -q skill/tests/test_waiting_cost_support_cap.py skill/tests/test_aligned_hardware_support_cap.py
```

The selector's `max_relative_support_loss=r` sets the eligibility floor to
`max(seed_support, (1-r)*support_incumbent_score)`. The default `r=1` reproduces
the seed-only rule; `r=0.01` is a fixed conservative choice, without a parameter
search. Candidate generation is unchanged, the seed drift bound still holds,
and the final support certificate obeys
`U-S_selected <= (U-S_incumbent)+r*S_incumbent`.

The 2026-09-10 comparison completed all 108 runs, with all constraints satisfied
and all 36 v4 controls reproducing their stored metrics. At K26, the 1% rule's
paired mean-flow changes versus v4 were 0.00%, −0.11% and −0.36% for loads
0.50, 0.80 and 0.95; respectively zero, one and two of the three seeds improved.
No lower-cost action was selected at load 0.50. Thus the cap removed the large
seed-only regressions, while also removing its low-load gain. The zero-cap
control matched v4 at the first two loads and was 0.11% slower at load 0.95.
Results and the pre-run protocol are in
`md/experiment_artifacts/aligned_hardware_support_cap_development_20260910.json`
and its `_protocol.json` companion. These small, mixed development gains
motivated the subsequent independent-seed comparison.

The fixed 1% rule's independent confirmation runs separately:

```bash
python3 -m algorithm.experiments.aligned_hardware_support_cap_confirmation --register
python3 -m algorithm.experiments.aligned_hardware_support_cap_confirmation --run
```

Registration fixes seeds 2041–2060, all original loads and budgets, three
policies, and the complete source texts. It replays load 0.80/seed 41 before
using new arrivals. The 720 cases retain the fixed holding objective and 1%
cap. The primary family has two K26 mean-flow improvement comparisons against
v4, at loads 0.80 and 0.95. Seed-paired ratio means use Bonferroni-adjusted
97.5% two-sided Student t intervals; both upper endpoints must fall below 1.
Other loads, budgets and the zero-cap policy remain descriptive. Registration,
per-run JSONL, paired comparisons and the compact confirmation summary are in
`md/experiment_artifacts/aligned_hardware_support_cap_confirmation_20260910/`.

All 720 registered cases completed with execution checks passing. At K26,
load 0.80 mean flow increased by 1.03% (adjusted interval +0.65% to +1.42%);
load 0.95 changed by −0.07% (−0.78% to +0.64%). Neither improvement claim was
confirmed. At load 0.80, subsequent queue waiting grew by 0.440 s per job,
outweighing the 0.152 s combined reduction in arrival-barrier and completion
time. The 1% holding rule is not adopted; v4 remains the main selector.
This finding motivates an explicit next-boundary queue value in development.

The one-step lookahead experiment runs on the original development roster:

```bash
python3 -m algorithm.experiments.aligned_hardware_lookahead
python3 -m pytest -q skill/tests/test_lookahead_frame_candidates.py skill/tests/test_aligned_hardware_lookahead.py
```

`lookahead_frame_candidates.py` minimizes `A+E[V(Q_next)]`, where A is the current
holding area in job-seconds and `V(q)=sum(q_i^2/(2*mu_i))` is a fluid clearing
proxy. The fixed mu comes from the model's feasible boundary time allocation.
With `m=Q-D+lambda*tau`, the terminal expectation includes the Poisson variance:
`E[V]=sum((m_i^2+lambda_i*tau)/(2*mu_i))`. The same paid pool, seed floor and 1%
support cap apply. An area-only control removes the terminal objective term;
both variants compute and count the queue moments and terminal diagnostics.
The 144-run protocol and results are stored as
`md/experiment_artifacts/aligned_hardware_lookahead_development_20260910_protocol.json`
and `aligned_hardware_lookahead_development_20260910.json`.

All 144 runs completed with execution checks passing, and the 72 frozen control
runs reproduced their recorded results. At K26, lookahead changed paired mean
flow versus v4 by +0.21%, −0.42% and −2.45% at loads 0.50, 0.80 and 0.95.
All three high-load seeds improved, with subsequent queue waiting reduced by
1.142 s per job on average. The area-only control improved high-load mean flow
by 0.36%, so the terminal term adds 2.10% improvement relative to that control.
The low-load regression matches the area-only control. These results motivated
independent confirmation, including assessment of the low-load regression.

Register and run the frozen lookahead confirmation:

```bash
python3 -m algorithm.experiments.aligned_hardware_lookahead_confirmation --register
python3 -m algorithm.experiments.aligned_hardware_lookahead_confirmation --run
```

This fixes seeds 3041–3060, all four policies and budgets, the original loads,
the terminal service vector, and full source texts. A high-load development
case is reproduced before the 960 new cases begin. Three K26 paired mean-flow
claims against v4 share Bonferroni-adjusted 98.33% two-sided t intervals:
improvement at 0.80 and 0.95 requires an upper endpoint below 1; no worsening
at 0.50 requires an upper endpoint at most 1, with no tolerance margin.
Other budgets and comparisons with the area-only control remain descriptive.
Artifacts are in
`md/experiment_artifacts/aligned_hardware_lookahead_confirmation_20260910/`.

All 960 cases completed with execution checks passing. At K26/load 0.95, all
20 seeds improved and mean flow fell 2.77%; the adjusted reduction interval was
1.86–3.69%, confirming the registered high-load improvement. Subsequent queue
waiting fell by 1.314 s per job. At load 0.80, mean flow changed by +0.47%
(−0.12% to +1.06%); at load 0.50 it changed by +0.04% (−0.14% to +0.23%).
The medium-load improvement and low-load no-worsening claims remain
inconclusive, so the rule does not replace v4 across all loads. Comparison
with the area-only policy is descriptive and has no separate confirmation
claim. The next development step is to test queue-backlog-based activation of
lookahead while retaining the same support constraints.

The fixed backlog gate runs on development seeds 41–43:

```bash
python3 -m algorithm.experiments.aligned_hardware_backlog_gate
python3 -m pytest -q skill/tests/test_backlog_gated_frame_candidates.py skill/tests/test_aligned_hardware_backlog_gate.py
```

For this catalogue, any two-lane frame serves at most 48 weighted jobs under
class weights `(8,8,3)`, obtained from maximum lane batches `(3,3,8)`. The gate
uses lookahead only when `8*Q_CNN + 8*Q_JAX + 3*Q_LLM > 48`; otherwise it uses
v4. The test compares these branches with always-on lookahead over the original
three loads and four budgets. Each frame performs one search; gate setup, the
queue scan and active prediction work are recorded separately. Both branches
retain the seed floor and the 1% support-loss limit. The protocol and results
use the `aligned_hardware_backlog_gate_development_20260910` artifact prefix.

All 108 cases passed execution checks and the 72 controls reproduced their
stored results; nine selector and integration tests passed. Gated and always-on
lookahead had identical metrics and flow decompositions in all 36 paired
cases. At K26 the changes versus v4 remain +0.21%, −0.42% and −2.45% for
loads 0.50, 0.80 and 0.95. Gating reduced low-load prediction evaluations by
91.2%, but did not remove the flow regression. The diagnostic
`python3 -m algorithm.experiments.aligned_hardware_backlog_gate_diagnostic`
exhaustively compared all 169 inactive queues under the 12 rate/budget settings:
all 2,028 comparisons chose identical actions. Thus this gate preserves the
always-on action policy for the registered catalogue, rates and budgets; only
its computation changes. The separately counted diagnostic searches are in
`aligned_hardware_backlog_gate_inactive_states_20260910.json`.
It is not adopted as a scheduling improvement. Actual substitutions are
examined by paired continuation replays:

```bash
python3 -m algorithm.experiments.aligned_hardware_lookahead_continuation
python3 -m pytest -q skill/tests/test_aligned_lookahead_continuation.py skill/tests/test_aligned_hardware_lookahead_continuation.py
```

The diagnostic includes every K26 substitution on the nine original lookahead
development trajectories. Each pair shares its prefix and arrival tape,
forces v4 or lookahead once, then uses v4 through the complete population's
drain. Prediction and realized differences use job-seconds. Their discrepancy
is split into cutoff correction, current-arrival residual, next-queue residual,
and the subsequent v4 area minus the realized terminal proxy. Results and the
pre-run protocol use the `aligned_hardware_lookahead_continuation_development_20260910`
artifact prefix; memoized searches, action audits and replay frames are counted
separately.

All nine source trajectories reproduced, and all 290 pairs (580 replays) passed
execution and area checks; eight tests passed. At load 0.50 the only substitution,
at queue `(1,4,8)`, changed a JAX-3/LLM-8 frame to two LLM-3 lanes. It predicted
a 39.17 job-second saving but incurred 784.78 extra job-seconds: the first frame
saved 70.36, while subsequent area grew by 855.14. The terminal continuation
residual accounts for this reversal, despite a favorable realized next-queue
value. At loads 0.80 and 0.95, respectively 87/154 and 71/135 local interventions
worsened flow; mean effects were −30.70 and −400.49 job-seconds because larger
benefits offset those losses. The next development step is to test a terminal
estimate using executable subsequent frames, with its extra searches explicitly
charged to the computation budget.

The paid two-frame backlog estimate runs with an allocation control:

```bash
python3 -m algorithm.experiments.aligned_hardware_batch_rollout
python3 -m pytest -q skill/tests/test_batch_rollout_frame_candidates.py skill/tests/test_aligned_hardware_batch_rollout.py
```

The current search gets K−4 attempts when four successor attempts fit beyond
the seed reserve; otherwise it gets K and returns its support incumbent. At
K26 this is K22 plus two clearing frames for each of two contenders: the
analytic challenger and current support incumbent. Each successor serves the
nonempty class with largest q_i/mu_i, using its largest fitting measured batch
and maximal homogeneous lane copies. With r=Q−D, the new objective is
`Jbatch=Janalytic+Vbatch2(r)−Vfluid(r)`. This replaces the known-backlog clearing
term while preserving the original Poisson arrival correction. The four-policy
comparison includes analytic lookahead at the identical reduced current budget.
Both contenders retain the seed floor and 1% cap against that current pool.
Results and the pre-run protocol use the
`aligned_hardware_batch_rollout_development_20260910` artifact prefix.

All 144 cases passed execution checks, all 72 full-K controls reproduced, and
ten tests passed. K26 paired mean-flow changes versus v4 were −0.23%, −0.72%
and −0.94% at loads 0.50, 0.80 and 0.95. Against the allocation control, changes
were −0.05%, −0.54% and +1.64%; all three high-load seeds became slower. K14
reserved four attempts but performed no rollouts, matching the allocation
control exactly and worsening flow by 17.76–31.58% versus v4 K14. The prior
low-load counterexample also remains accepted: its predicted saving falls from
39.17 to 1.90 job-seconds but keeps the wrong sign. Reproduce that separate,
previous-state check with
`python3 -m algorithm.experiments.aligned_hardware_batch_rollout_known_state`.
This variant is not adopted. Next, return unused successor reserves to current
search, then address the remaining proxy errors separately; budget recovery
cannot fix the counterexample, whose current search already finishes in 17 attempts.

Unused rollout reserves now continue the original current-search frontier:

```bash
python3 -m algorithm.experiments.aligned_hardware_recycled_rollout
python3 -m pytest -q skill/tests/test_recycled_rollout_frame_candidates.py skill/tests/test_aligned_hardware_recycled_rollout.py
```

The initial split and active two-contender rollout remain fixed. When the initial
analytic challenger is physically identical to the support incumbent, the four
reserved attempts return to the same frontier, retaining its seeds, bounds and
candidate pool. The final support incumbent is selected after continuation;
new challengers without a paid rollout remain diagnostic. Both the support
floor and upper bound use the final current pool. Analytic costs are cached by
candidate object, and stage counters retain all actual pricing work, including
initial candidates subsequently excluded by a higher floor. Results and the
pre-run protocol use the
`aligned_hardware_recycled_rollout_development_20260910` artifact prefix.

All 144 development cases passed, 108 frozen control runs reproduced, and nine
tests passed. At K14, all nine mean-flow and p90 results equal v4 K14, eliminating
the previous 17.76–31.58% regression. K26 paired mean-flow changes versus v4 are
+0.21%, −0.17% and −1.54% at loads 0.50, 0.80 and 0.95; versus fixed allocation
batch rollout they are +0.45%, +0.55% and −0.61%. High-load K26 remains 0.93%
slower than full-K analytic lookahead. K10 and K120 mean flow match the previous
batch variant, with no additional current attempts at those budgets. Total
attempts stay within K, both pricing stages are accounted for, and the seed
floor, 1% cap and drift checks hold. The known `(1,4,8)` rollout decision is
unchanged. V4 remains the main policy. Next, revise the terminal approximation
to account for subsequent arrivals interacting with discrete batches, keeping
successor work within the same total budget and using the original development
protocol before any fresh confirmation.

The next arrival-aware rollout uses one common planning population:

```bash
python3 -m algorithm.experiments.aligned_hardware_arrival_rollout
python3 -m pytest -q skill/tests/test_arrival_rollout_terminal.py skill/tests/test_arrival_rollout_frame_candidates.py skill/tests/test_aligned_hardware_arrival_rollout.py
```

Class i arrivals occur at `(n-0.5)/lambda_i` through the shared horizon
`H=3*max(effective catalogue duration)`. Each contender executes its first frame
and two paid successor frames on that integer path. Remaining arrivals and
backlog are integrated under the frozen per-class fluid service rates through
H, followed by clearing without further arrivals. Thus both costs cover the
same jobs, with actual batch completion offsets and barrier waiting. This
entire path cost replaces the rollout objective; the original Poisson formulas
only select the analytic challenger. The current search, recovery and support
floors remain fixed. Arrival generation, reads and fluid integration are
reported alongside the unchanged total candidate budget. Results and the
pre-run protocol use the `aligned_hardware_arrival_rollout_development_20260910`
artifact prefix.

All 144 development cases passed, all 108 controls reproduced, and 11 tests
passed. K26 paired mean-flow changes versus v4 are 0.00%, +0.13% and −1.35%
at loads 0.50, 0.80 and 0.95; versus recycled rollout they are −0.21%, +0.30%
and +0.22%. High-load K26 remains 1.14% slower than analytic lookahead. K10
and K14 match the preceding variant; at K120, medium-load flow is 0.67% worse
than v4, with all three seeds worse. At the known `(1,4,8)` state, the shared
31-job path gives the old challenger cost 498.13 versus incumbent 456.02,
so the new selector chooses the JAX-3/LLM-8 incumbent within 21 total attempts.
This corrects that predicted ordering; the old realized continuation effect is
not a fresh measurement of this policy. The variant is not adopted. Next,
test a common Poisson arrival path in place of fixed-spacing arrivals, holding
the two successor steps and total budget fixed to isolate the arrival
approximation before considering independent confirmation.

The sampled Poisson variant retains the same path evaluator and total budget:

```bash
python3 -m algorithm.experiments.aligned_hardware_poisson_rollout
python3 -m pytest -q skill/tests/test_poisson_rollout_sampling.py skill/tests/test_aligned_hardware_poisson_rollout.py
```

Each active comparison samples independent per-class exponential gaps through
the same H, then shares that one path between its two contenders. Planning RNGs
use `aligned-planning:{planning_seed}:{coordinate}`, independently of the real
arrival streams, and are recreated for each full run. Inactive decisions make
no draws; repeated active states advance the streams. The first gap beyond H
is charged even though it produces no retained event. RNG setup, draws and
the existing event/integration costs are reported separately.
Planning seeds 0, 1 and 2 are fully crossed with the original arrival seeds,
loads and budgets. The protocol specifies 108 frozen control runs plus 108
sampled-policy runs. Main summaries average planning repetitions within each
real arrival seed first; all three planning-seed summaries are retained.
Results and the pre-run protocol use the
`aligned_hardware_poisson_rollout_development_20260910` artifact prefix.

All 216 development runs passed, all 108 controls reproduced, and ten tests
passed. Averaging the three planning repetitions within each real seed, K26
mean-flow changes versus v4 are +0.14%, −0.78% and −1.88% at loads 0.50, 0.80
and 0.95. Versus the midpoint variant they are +0.14%, −0.91% and −0.54%.
High-load K26 remains 0.59% slower than analytic lookahead. Individual K26
repetitions still worsen flow versus v4 by up to 1.78% at medium load and
1.82% at high load. At the known low-load state, fresh planning seeds 0 and 1
accept the previous bad action; seed 2 rejects it. These diagnostic streams
start fresh and do not assert a full-run stream position. K10/K14 retain the
preceding decisions; K120 mean-flow changes versus v4 are −0.04%, −0.25% and
−1.83%. All candidate and random-draw accounting checks hold. This variant
is not adopted. Next, test the known analytic expectation as a control variate
for sampled current-area and next-queue fluid costs, retaining the same search
and successor budget; any variance reduction must be measured.

The control-variate variant keeps the Poisson samples and fixes coefficient 1:

```bash
python3 -m algorithm.experiments.aligned_hardware_control_variate
python3 -m pytest -q skill/tests/test_control_variate_rollout_frame_candidates.py skill/tests/test_aligned_hardware_control_variate.py
```

With raw path cost X and `Y=sampled current area+Vfluid(sampled next queue)`,
the corrected estimate is `X-Y+Janalytic`. Both candidates are fixed before
sampling and the old analytic cost equals E[Y]. Implementation cancels the
sampled current-area terms directly; negative estimates remain untruncated.
Each active comparison adds only 2d fluid-coordinate evaluations, included in
the terminal-work total, with the same samples and four successor attempts.
The 288-run roster contains 180 reproduced controls and 108 corrected runs,
with Poisson controls paired on the same planning seed. A separate diagnostic
uses ten fixed queue states at each load and 64 common samples per active
state; inactive states retain one charged probe. It reports within-state
raw/corrected cost-difference variance and paired mean-shift Monte Carlo error,
with diagnostic search and sampling work counted separately. Results and the
pre-run protocol use the `aligned_hardware_control_variate_development_20260910`
artifact prefix.

All 288 development runs passed, all 180 controls reproduced, and ten tests
passed. K26 mean-flow changes versus v4 are +0.07%, −0.63% and −1.46% at
loads 0.50, 0.80 and 0.95; versus the matched uncorrected Poisson runs they
are −0.08%, +0.16% and +0.44%. High-load K26 is still 1.03% slower than
analytic lookahead, while high-load K120 worsens by 0.80% versus uncorrected Poisson.
In the known-state diagnostic, two of the three planning samples now reject
the previous bad action; planning seed 0 retains it. Among the 30 fixed
queue/load cells, only the three `(1,4,8)` cells activate both contenders.
Their corrected/raw cost-difference variance ratios are 1.160, 1.089 and 0.974;
the coefficient therefore increases sampled variance at low and medium load.
All 192 active samples and 27 inactive probes are retained, costing 4,506
candidate attempts and 5,589 draws separately from the full-run budgets.
This fixed-coefficient variant is not adopted. Next, inspect actual raw/CV
action disagreements on development trajectories with paired continuations
and repeated local samples to distinguish sampling error from continuation
model error before further tuning.

The actual-disagreement diagnostic is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_control_variate_continuation
python3 -m pytest -q skill/tests/test_aligned_control_variate_continuation_replay.py skill/tests/test_aligned_hardware_control_variate_continuation.py
```

At K26, it reproduced all 54 frozen Poisson/CV source trajectories and native
counters. Opposite-policy decisions use copied pre-call RNG states; every
physical disagreement receives two replays with the same prefix and arrival
tape, one forced raw/CV action, then common v4 continuation until drain.
All 174 disagreements completed 348 replays, and all nine targeted tests passed.
CV-first improved 97 interventions and worsened 77; by increasing load the
counts are 2/0, 54/27 and 41/50. These are correlated single-action effects.

Fresh planning samples cover all 63 distinct load/queue states, 64 per state.
The raw and corrected fixed-contender mean predictions each disagree with the
realized effect on 95 of 174 occurrences. At high load, corrected cost-difference
variance exceeds raw variance in all 41 visited states (median ratio 1.112).
The mean absolute raw sampling adjustment is 97 job-seconds, versus 1,703 for
the realized-minus-proxy residual; high-load medians are much closer, 141 and
167, so the large mean residual is not uniform across cases.
All 4,032 fresh samples cost 81,984 candidate attempts and 141,405 draws;
1,443 shadow comparisons and 721,117 replay frames are counted separately.
The result and pre-run protocol use the
`aligned_hardware_control_variate_continuation_development_20260910` prefix.
The fixed CV coefficient remains unadopted. The following variant tests a single
successor-rule change on the raw Poisson branch: replace class priority `q_i/mu_i` by
`q_i*nu_i`, where `nu_i=lanes*max(pure batch/duration)` is fixed from the model.
Retain the current pool, samples, fluid tail and four paid successor attempts;
charge static profile reads and priority arithmetic separately. This tests a
service-pressure continuation without queue-dependent free candidate searches.

The service-pressure successor variant is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_pressure_rollout
python3 -m pytest -q skill/tests/test_pressure_rollout_terminal.py skill/tests/test_pressure_rollout_frame_candidates.py skill/tests/test_aligned_hardware_pressure_rollout.py skill/tests/test_aligned_hardware_pressure_rollout_diagnostic.py
```

The fixed pure-class rates are 0.40074, 0.87970 and 1.07749 jobs/s for CNN,
JAX and LLM. Each active decision reads and prices the seven pure profiles once;
both contenders share those rates. The class-pressure evaluation count is a
subtotal of the existing class-evaluation count. Current search, the shared
Poisson path, four successor attempts and the original-mu fluid tail retain
their previous definitions. The full roster contains 180 frozen controls and
108 pressure runs. A separate diagnostic reuses all 63 old disagreement states,
their 64 planning samples and 174 fixed continuation targets. Its model margins
are oriented to each retained action pair, with no new real-trajectory replay.
Artifacts and the pre-run protocol use the
`aligned_hardware_pressure_rollout_development_20260910` prefix.

All 288 runs passed, all 180 controls reproduced and eleven targeted tests
passed. K26 flow changes versus raw Poisson are -0.07%, +0.28% and +0.08%
at increasing load; versus v4 they are +0.07%, -0.51% and -1.81%.
The medium-load regression occurs in all three real seeds after planning-seed
averaging. K10/K14 retain raw-Poisson decisions; K120 worsens by 0.02%, 0.06%
and 0.15% versus raw Poisson. The fixed-state model direction mismatch increases
from 95 to 104 of 174 occurrences, including 41 to 52 at high load.
Mean absolute target residual rises from 1,703 to 1,724 job-seconds, while its
median falls from 712 to 701. Only 13 of 63 states have larger sample variance,
so reduced variance alone does not explain or resolve the ordering error.
The 4,032 diagnostic calls retain 81,984 candidate attempts and 141,405 draws,
adding 28,224 static profile reads and rate evaluations each. This priority
variant is not adopted; v4 remains the main policy.
Next, return to raw Poisson's `q_i/mu_i` priority and change only successor
construction: fill lanes sequentially, updating temporary visible queues after
each pure profile so a frame can serve multiple classes. Keep four complete
successor constructions, and charge per-lane ranking, profile scans and temporary
queue updates along with runtime under the same development protocol.

The mixed-successor variant is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_mixed_successor_rollout
python3 -m pytest -q skill/tests/test_mixed_successor_rollout_terminal.py skill/tests/test_mixed_successor_rollout_frame_candidates.py skill/tests/test_aligned_hardware_mixed_successor_rollout.py skill/tests/test_aligned_hardware_mixed_successor_rollout_diagnostic.py
```

All 288 development runs passed, all 180 controls reproduced and eleven targeted
tests passed. At K26, mean-flow changes versus matched raw Poisson are -0.14%,
-0.29% and -0.40% at increasing load; versus v4 they are 0.00%, -1.08% and
-2.28%. The medium-load change improves all three real-seed means; high load
improves two of three. P90 changes versus raw Poisson are -0.15%, -0.72% and
-0.54%. Selector time rises by about 0.7%, 3.1% and 4.2%, respectively.
K10/K14 retain all raw-Poisson outcomes; K120 changes are +0.04%, +0.23% and
-0.11% versus raw Poisson, so improvement is not monotone in search budget.

On the retained 63-state diagnostic, point-estimate direction mismatches fall
from 95 to 92 of 174 occurrences. Mean and median absolute target residuals fall
from 1,703/712 to 1,693/664 job-seconds. Mixed construction lowers cost-difference
variance in 62 states and leaves one unchanged. The 4,032 paired samples retain
81,984 candidate attempts and 141,405 draws, adding 32,256 lane visits, 95,730
priority evaluations, 33,636 profile probes and 32,255 temporary queue updates.
The result and pre-run protocol use the
`aligned_hardware_mixed_successor_rollout_development_20260910` prefix.
This is the current development candidate, while v4 remains the main policy.
Next, inspect actual raw/mixed action disagreements under same-state planning
samples and common-v4 paired continuations before assigning fresh confirmation
seeds. This will test whether the full-run gain follows the intended mechanism.

The actual raw/mixed disagreement diagnosis is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_mixed_successor_continuation
python3 -m pytest -q skill/tests/test_aligned_mixed_successor_continuation_replay.py skill/tests/test_aligned_hardware_mixed_successor_continuation_sampling.py skill/tests/test_aligned_hardware_mixed_successor_continuation.py
```

All 54 frozen K26 source trajectories reproduce their metrics, decompositions
and deterministic counters. Their 1,414 active
comparisons contain 554 physical raw/mixed action disagreements at 100 distinct
load/queue states, all inside the arrival window. Each disagreement has raw-first
and mixed-first replays with the same source prefix, real arrival tape, complete
population and common v4 continuation. The 1,108 replays execute 2,321,714 frames;
continuation-cache searches, shadow calls and all replay work are reported
separately. Nineteen targeted tests pass.

Mixed-first improves 281 interventions and worsens 273, with mean delta -97
job-seconds. By increasing load, improved/worsened counts are 4/1, 138/159 and
139/113; mean deltas are -615, +61 and -273 job-seconds. The high-load effect
supports the intended mechanism, while the medium-load single-action result does
not explain the preceding full-policy improvement under its different subsequent
state feedback.

Fresh same-path sampling runs both estimators 64 times at every distinct state.
The mixed proxy lowers direction mismatches from 312 to 272 of 554 occurrences
and lowers cost-difference variance in 99 of 100 states. Mean absolute sampling
adjustment falls from 166 to 99 job-seconds. Mean target residual falls slightly
from 1,993 to 1,984 job-seconds, while its median rises from 911 to 933; proxy
accuracy therefore improves mainly in ordering and variance, not uniformly in
realized-error magnitude. These 12,800 selector calls cost 264,320 candidate
attempts and 440,214 draws in total, plus the reported mixed lane work.
Results, protocol and full-text source snapshot use the
`aligned_hardware_mixed_successor_continuation_development_20260910` prefix.
The implementation is now frozen as the candidate for an independent K26
confirmation with fresh real-arrival and planning seeds; no development tuning
is introduced before that test.

The frozen mixed-successor confirmation was registered and run in two stages:

```bash
python3 -m algorithm.experiments.aligned_hardware_mixed_successor_confirmation --register
python3 -m algorithm.experiments.aligned_hardware_mixed_successor_confirmation --run
python3 -m pytest -q skill/tests/test_aligned_hardware_mixed_successor_confirmation.py skill/tests/test_aligned_hardware_mixed_successor_confirmation_stats.py
```

It uses 20 new real-arrival seeds 4041--4060, planning seeds 101--103, K26 and
H=20000. All 480 registered runs and all 15 execution checks passed; the 20
targeted registration, roster and inference tests also passed. Planning repeats
are averaged within each real seed before three Bonferroni-adjusted 98.33%
two-sided t intervals are formed. At loads 0.50, 0.80 and 0.95, the mean
mixed/raw flow ratios are 0.999898, 0.999378 and 0.993077, with simultaneous
intervals [0.999562, 1.000235], [0.996752, 1.002005] and
[0.988057, 0.998097]. Thus high-load improvement is confirmed, while low-load
no-worsening and medium-load improvement are not confirmed.

The predeclared point guardrails agree with that decision. Mixed/v4 mean-flow
ratios are 1.000069, 0.999548 and 0.986421; mixed/raw P90 ratios are 1.000041,
0.998038 and 0.986446. Both low-load thresholds fail, while the medium- and
high-load thresholds pass. The registered outcome is therefore `NOT_ADOPT`;
v4 remains the main policy, and the manuscript is unchanged. Artifacts use the
`aligned_hardware_mixed_successor_confirmation_20260910` prefix. Any subsequent
mixed/raw activation rule must return to development data and receive a new
independent seed roster rather than tune on these confirmation outcomes.

The catalogue-derived two-frame depth gate is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_rollout_depth_gate --register
python3 -m algorithm.experiments.aligned_hardware_rollout_depth_gate --run
python3 -m pytest -q skill/tests/test_rollout_depth_gated_frame_candidates.py skill/tests/test_aligned_hardware_rollout_depth_gate.py
```

The largest pure batches are `(3,3,8)` with work weights `(8,8,3)`, so two
lanes provide weighted capacity `C=48` in one complete frame. The gate uses the
strict, catalogue-fixed condition `8 Q_CNN + 8 Q_JAX + 3 Q_LLM > 2C = 96`:
active frames call the frozen mixed-successor selector, while all other frames
call frozen v4. It was registered before execution on the original development
real seeds 41--43 and planning seeds 0--2; it reads neither the independent
confirmation roster nor its outcomes. Filtering the old continuation data by
the same gate retained 354 of 554 disagreement occurrences, with 189 improving
and 165 worsening mixed-first interventions and a mean mixed-minus-raw effect
of -207.31 job-s. That intervention summary is descriptive and was not used to
change the gate.

All 99 registered runs passed all 15 execution checks, all 72 frozen controls
reproduced, and nine targeted tests passed. Mean-flow ratios over v4 are
1.000000, 0.991136 and 0.976207 at loads 0.50, 0.80 and 0.95; paired P90 ratios
are 1.000000, 0.989632 and 0.944712. The predeclared seed-consistency condition
nevertheless fails: the three medium-load real-seed flow ratios are 1.001579,
0.979700 and 0.992130. Therefore `development_advance=false`; no new
confirmation is run, v4 remains the main policy, and the manuscript is
unchanged. Artifacts use the
`aligned_hardware_rollout_depth_gate_development_20260910` prefix. The
first-divergence diagnosis below uses only that development tape.

The fixed seed-41 first-divergence diagnosis is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_rollout_depth_gate_first_divergence --register
python3 -m algorithm.experiments.aligned_hardware_rollout_depth_gate_first_divergence --run
python3 -m pytest -q skill/tests/test_rollout_depth_gated_frame_candidates.py skill/tests/test_aligned_hardware_rollout_depth_gate.py skill/tests/test_aligned_hardware_rollout_depth_gate_first_divergence.py
```

Registration verifies that load 0.80, real seed 41 is the unique medium/high
real-seed failure, then retains all planning seeds 0--2. Each source trajectory
is reproduced before its first physical gate/v4 difference is replayed twice:
once with v4 and once with the gate action, followed in both branches by frozen
v4 K26 until the identical 12,697-job population drains. All three sources, six
replay branches, 14 execution checks and 14 targeted tests pass.

Planning seed 0 first differs at frame 270, time 2253.36 s and queue `(5,5,9)`
with weighted backlog 107. Seeds 1 and 2 first differ at frame 31, time 230.79 s
and queue `(6,6,10)` with weighted backlog 126. In every case, v4 serves
JAX-3 plus LLM-8 in 14.8493 s, while the gate selects CNN-3 plus LLM-8 in
14.9724 s. The sampled mixed-successor cost favors that gate action by 249.09,
125.62 and 69.94 job-s; the analytic one-frame-plus-fluid proxy also favors it
by 14.34, 25.91 and 25.91 job-s. Under common-v4 continuation, however, the
single gate action increases total flow by 5168.92, 4378.42 and 4378.42 job-s,
or 0.40710, 0.34484 and 0.34484 s per job. Only 26--27 job-s arise in the first
frame; 4352--5142 job-s arise after its boundary.

The full gated-policy effects are +926.16, +2188.21 and -1401.58 job-s. The
post-first gated suffix contributes -4242.76, -2190.20 and -5780.00 job-s, so
later gate decisions mitigate the harmful first action and fully offset it only
for planning seed 2. Total flow and all three flow components satisfy
`full = first + suffix`. Thus the strict backlog condition certifies that a
second successor sees work, but it does not certify the sampled action ordering.
The threshold-only mixed-successor gate family is closed without tuning or new
confirmation; v4 remains the main policy and the manuscript remains unchanged.
Artifacts use the
`aligned_hardware_rollout_depth_gate_first_divergence_development_20260910`
prefix.

The frozen v4 candidate-search trace is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_candidate_search_diagnostic --register
python3 -m algorithm.experiments.aligned_hardware_candidate_search_diagnostic --run
python3 -m pytest -q skill/tests/test_aligned_budgeted_frame_search_trace.py skill/tests/test_aligned_hardware_candidate_search_diagnostic.py skill/tests/test_budgeted_frame_candidates.py skill/tests/test_budgeted_frame_queue_caps.py
```

The registered diagnostic replays the 18 consumed K26/K120 development
trajectories to recover their complete queue distributions, then traces each of
5,003 distinct queues once through K120 and recovers K26 from the same search
prefix. All 18 source trajectories, 15 execution checks and 18 targeted tests
pass. On the primary K26 trajectories, loads 0.50, 0.80 and 0.95 have 113/10,209,
466/6,773 and 22/5,633 strict support-loss frames. A further 1,132, 1,301 and
244 frames have already found the exact support score by K26 but retain an open
certificate. The occurrence-weighted mean finite-budget support regrets are
0.001190, 0.009891 and 0.000469, while mean frontier-bound slacks are 0.009245,
0.092809 and 0.024841. Strict-loss winners first appear at paid attempts 28--40,
28--51 and 31--51, respectively.

Thus the certificate gap is mostly relaxation slack, while actual support misses
are concentrated at load 0.80. Across the 22,615 primary-policy frame visits,
601 find the exact score after K26; 388 of these visits contain an exact-action
component that the static order places behind an equal-weight, longer-duration
option. Because the bound also determines heap expansion,
this trace does not assign either effect causally to the static option order or
the bound. The next controlled step is one preregistered, parameter-free
duration-aware tie-order shadow on this fixed state roster; only a successful
state-level result can proceed to full trajectory development. No scheduler or
paper result changes in this diagnostic. Artifacts use the
`aligned_hardware_candidate_search_diagnostic_development_20260910` prefix.

The registered duration-aware fixed-state shadow is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_duration_tie_order_shadow --register
python3 -m algorithm.experiments.aligned_hardware_duration_tie_order_shadow --run
python3 -m pytest -q skill/tests/test_duration_tie_frame_candidates.py skill/tests/test_duration_tie_frame_search_trace.py skill/tests/test_aligned_hardware_duration_tie_order_shadow.py
```

The only changed rule resolves equal availability and weighted service by the
shorter effective duration. All 5,003 registered states, 18 source-population
cells and 11 execution checks pass; 17 targeted tests pass. At primary loads
0.50, 0.80 and 0.95, K26 strict support-loss frames fall from 113 to 72, 466 to
266 and 22 to 15. Mean finite-budget support regret falls from 0.001190 to
0.000629, 0.009891 to 0.004542 and 0.000469 to 0.000312. Nine states improve
support and none regress.

The prespecified certificate guardrail fails at every load: mean raw certified
gap rises from 0.010434 to 0.010718, 0.102700 to 0.103315 and 0.025310 to
0.025526. Consequently seven of eight advancement checks pass but
`shadow_advance=false`. The composite duration-aware order is closed without
tuning or full trajectories; its support gain is retained as a negative
trade-off finding, v4 remains the main policy, and the manuscript is unchanged.
Artifacts use the
`aligned_hardware_duration_tie_order_shadow_development_20260910` prefix.

The registered binary-price residual-capacity-bound shadow is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_residual_capacity_bound_shadow --register
python3 -m algorithm.experiments.aligned_hardware_residual_capacity_bound_shadow --run
python3 -m pytest -q skill/tests/test_residual_capacity_frame_candidates.py skill/tests/test_residual_capacity_frame_search_trace.py skill/tests/test_aligned_hardware_residual_capacity_bound_shadow.py
```

The v4 option order and K26/K120 budgets remain fixed; only the search upper
bound changes, taking the minimum over the eight binary prices
`lambda_i in {0,Q_i}`. The shadow reuses the compact 5,003-state registered
roster and reads no arrivals, trajectories or random draws. All 12 execution
checks and 17 targeted tests pass. The bound tightens on 1,783 states and changes
the retained physical action on six. At primary loads 0.50, 0.80 and 0.95, mean
certified gaps fall from 0.010434 to 0.010148, 0.102700 to 0.070262 and 0.025310
to 0.016492; exact-certified fractions rise from 0.878049 to 0.924087, 0.739111
to 0.804518 and 0.952778 to 0.963430. Mean candidate attempts also fall from
14.624 to 14.215, 17.073 to 15.981 and 13.042 to 12.757.

The finite-budget support guardrails fail. Strict support-loss frames change
from 113 to 167, 466 to 470 and 22 to 21, while mean support regret rises at all
three loads. Occurrence-weighted support improves on 8, 74 and 3 frames but
regresses on 62, 78 and 2. Because the tighter bound also changes heap expansion,
preserving the static v4 option order does not preserve its K26 search prefix.
Only eight of 15 advancement checks pass, so `shadow_advance=false` and
`certificate_only=false`. This integrated-bound rule is closed without tuning
or full trajectories; v4 remains the main policy and the manuscript is
unchanged. Artifacts use the
`aligned_hardware_residual_capacity_bound_shadow_development_20260910` prefix.

The corrected first-divergence diagnosis for the four residual-bound regressions
is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_residual_bound_first_divergence_v2 --register
python3 -m algorithm.experiments.aligned_hardware_residual_bound_first_divergence_v2 --run
python3 -m pytest -q skill/tests/test_residual_bound_first_divergence_trace.py skill/tests/test_aligned_hardware_residual_bound_first_divergence.py skill/tests/test_aligned_hardware_residual_bound_first_divergence_v2.py
```

The post-outcome diagnostic selects all and only the four source states with
lower K26 support and reuses their registered records without copying the full
roster. All 14 execution checks and ten targeted tests pass. In every case, the
legacy and residual searches share paid attempts 11--13 and first choose
different logical prefixes at attempt 14. Both competing prefixes are live in
both three-node frontiers, and no residual-only pruning has occurred. Their old
upper bounds are tied, so v4 chooses the earlier serial; the residual bound
tightens only that prefix and reverses the heap order.

The attempt-14 candidates are `[7,3]` and `[3,7]`, which have identical service,
duration and completion offsets because the lanes are interchangeable. Thus the
finding concerns the first logical search-order change, not the first physical
candidate difference or a complete attribution of all later search behavior.
The registered exact-support candidate `[6,7]` appears at legacy attempt 26 but
is delayed to residual attempts 38, 38, 32 and 50 for states 424, 425, 437 and
438. This supports testing a separately registered fixed-state sidecar that
retains the legacy heap priority and uses the tighter bound only for safe pruning
and certification; no trajectory experiment or paper change follows yet.

The preserved v1 artifact misreported the attempt-14 physical-equivalence field
because it included lane-ordered option indices in the comparison. Its frontier
classification and K26 checkpoints remain valid, but v2 supersedes it. Use the
`aligned_hardware_residual_bound_first_divergence_v2_development_20260910`
artifact prefix.

The corrected legacy-priority residual-capacity sidecar shadow is reproducible
with:

```bash
python3 -m algorithm.experiments.aligned_hardware_residual_sidecar_shadow_v2 --register
python3 -m algorithm.experiments.aligned_hardware_residual_sidecar_shadow_v2 --run
python3 -m pytest -q skill/tests/test_residual_sidecar_frame_candidates.py skill/tests/test_residual_sidecar_frame_search_trace_v2.py skill/tests/test_aligned_hardware_residual_sidecar_shadow_v2.py
```

The heap key remains `(-U_v4, stable_serial)`. The eight-mask residual bound is
used only to prune prefixes with `U_8 < incumbent` and to certify the maximum
over the complete surviving frontier; equality is not removed by the sidecar.
The screen reuses all 5,003 registered states uniformly and generates no
arrivals, trajectories or random draws. All 16 execution checks and 17 targeted
tests pass. Safe pruning activates on 266 states, support improves on one state
and regresses on none, the certificate improves on 57 states, and paid candidate
attempts fall on 222 states.

For primary loads 0.50, 0.80 and 0.95, strict support-loss frames fall from
113 to 107, 466 to 427 and 22 to 19. Mean certified gaps fall from 0.010434 to
0.009730, 0.102700 to 0.085483 and 0.025310 to 0.023186; mean paid attempts fall
from 14.624 to 14.521, 17.073 to 16.473 and 13.042 to 12.813. All six registered
nonworsening metrics pass at every primary and K120-policy sensitivity load, so
the preregistered decision is to advance this frozen sidecar to a separately
registered full development-trajectory experiment. This fixed-state screen is
post-outcome development evidence, and the manuscript remains unchanged.

The preserved v1 sidecar result omitted K26 certificate scans from its aggregate
work count when the trace continued beyond K26. V2 changes only trace-level work
accounting: it records 5,189 actual scans across 5,003 traces and 569 scanned
frontier entries. Its selections, population summaries and scientific decision
are identical to v1. Use the
`aligned_hardware_residual_sidecar_shadow_v2_development_20260911` artifact
prefix.

The separately registered full sidecar development trajectories are
reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_residual_sidecar_trajectories --register
python3 -m algorithm.experiments.aligned_hardware_residual_sidecar_trajectories --run
python3 -m pytest -q skill/tests/test_aligned_hardware_residual_sidecar_trajectories.py skill/tests/test_residual_sidecar_frame_candidates.py skill/tests/test_aligned_frame_flow_decomposition.py skill/tests/test_aligned_hardware_frame_experiment.py skill/tests/test_aligned_hardware_residual_sidecar_shadow_v2.py
```

The protocol adds only the nine K26 sidecar trajectories on the three reused
development loads and seeds. It freezes the 18 existing v4 K26/K120 rows and
their flow decompositions as controls; K120 remains a descriptive exact-support
reference rather than a new performance arm. All nine execution checks and 32
joint tests pass. Safe pruning occurs in 2,712 frames. Mean paid-attempt ratios
relative to v4 are 0.99337, 0.95275 and 0.98025 at loads 0.50, 0.80 and 0.95.

The scientific guardrails do not pass. The load-level paired mean-flow ratios
are 1.00060, 0.99439 and 1.00258; the paired P90 ratios are 0.99799, 0.99568 and
1.00176. Three individual seed pairs have worse mean flow. The improvement at
load 0.80 is driven mainly by 0.151 seconds less deferred-visible waiting per
job, whereas load 0.95 incurs 0.141 seconds more deferred-visible waiting per
job. Thus safe same-state pruning can alter later queues in both directions.
The preregistered decision is `TRAJECTORY_TRADEOFF_STOP`: close this sidecar
branch without tuning, added seeds or confirmation. The manuscript remains
unchanged. Use the
`aligned_hardware_residual_sidecar_trajectories_development_20260911` artifact
prefix.

The registered lane-permutation quotient-cache shadow is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_permutation_quotient_shadow --register
python3 -m algorithm.experiments.aligned_hardware_permutation_quotient_shadow --run-registered
python3 -m pytest -q skill/tests/test_permutation_quotient_frame_candidates.py skill/tests/test_aligned_hardware_permutation_quotient_shadow.py skill/tests/test_budgeted_frame_candidates.py
```

The selector preserves the v4 logical heap, support bound and raw lane-order
tie break. Within each decision it charges the first option-index multiset and
reuses later lane permutations as cache hits, reducing the two-lane catalogue
from 100 ordered assignments to 55 quotient orbits. The fixed-state screen
reuses the existing 5,003-state roster without copying it. All 11 execution
checks and all 20 targeted tests pass; K55 is exact and matches the frozen v4
K120 support on every state. K26 has no state-level support, upper-bound or gap
regression and records 4,043 free cache-hit logical pops.

At primary loads 0.50, 0.80 and 0.95, strict support-loss frames fall from
113 to 6, 466 to 134 and 22 to 9. Exact-support fractions rise to 0.99941,
0.98022 and 0.99840, while mean certified gaps fall from 0.01043 to 0.00280,
0.10270 to 0.04648 and 0.02531 to 0.00809. Mean paid attempts fall by 1.482,
1.120 and 0.626 candidates per frame. The same six nonworsening gates pass on
all K120-policy state-distribution sensitivity loads. Sixteen registered states
change to strictly higher-support actions; none regress.

The preregistered decision is `ADVANCE_TO_AUTHORED_TRANSITION_SCREEN`. The next
stage is limited to short drains authored from these changed states and an
explicit FIFO completion-offset pairing fixture. It must measure raw-lane
effects directly before any full development trajectories, because identical
option multisets can still pair job ages with completion offsets differently.
No trajectory or paper result changes in this screen. Artifacts use the
`aligned_hardware_permutation_quotient_shadow_development_20260911` prefix.

The registered authored-transition screen is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_permutation_quotient_transition --register
python3 -m algorithm.experiments.aligned_hardware_permutation_quotient_transition --run-registered
python3 -m pytest -q skill/tests/test_authored_transition_drain.py skill/tests/test_aligned_hardware_permutation_quotient_transition.py skill/tests/test_permutation_quotient_frame_candidates.py skill/tests/test_aligned_hardware_permutation_quotient_shadow.py skill/tests/test_budgeted_frame_candidates.py
```

The protocol drains the 16 changed states under coincident and unit-spaced FIFO
ages, with no future arrivals. Only the first action differs; both arms then use
v4 K26 until the initial population is empty. All ten execution checks and all
32 targeted tests pass. Of the 16 unique mean-flow effects, eight improve and
eight worsen, ranging from -0.909 to +0.634 seconds. Across the 32 state-profile
nearest-rank P90 effects, 12 improve, 16 worsen and four tie; the largest
regression is 5.365 seconds.

Frozen v4 occurrence weights give a negative flow-area delta in all three load
0.80 seed cells, but a positive delta in all six load 0.50 and 0.95 cells. The
range is -0.508 to +5.845 job-seconds per changed-state visit. Therefore the
state, tail and nine-cell nonworsening gates fail despite strict improvements in
some cases. The preregistered decision is `LOCAL_FLOW_TRADEOFF_STOP`: close the
quotient-cache branch without tuning or full development trajectories. The
manuscript remains unchanged. Artifacts use the
`aligned_hardware_permutation_quotient_transition_development_20260911` prefix.

The registered equal-requirement duration-dominance shadow is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_requirement_dominance_shadow --register
python3 -m algorithm.experiments.aligned_hardware_requirement_dominance_shadow --run-registered
python3 -m pytest -q skill/tests/test_requirement_dominance_frame_candidates.py skill/tests/test_aligned_hardware_requirement_dominance_shadow.py
```

This parameter-free preprocessing groups local options by identical job
requirements and removes only a strictly longer effective-duration member;
minimum-duration ties remain. In the registered catalogue, option 1 is dominated
for MaxWeight support by option 0 at requirements `(1,0,1)`, reducing the local
catalogue from nine options to eight, the two-lane ordered family from 100 to 81,
and the safe exhaustive search cap from K120 to K99. The returned lane tuple is
mapped back to original catalogue indices. This is support dominance only:
the removed option has a different completion-offset profile.

All 5,003 states, 18 population cells, 12 execution checks and 24 new targeted
tests pass. K99 matches the registered K120 maximum support on every state. At
primary loads 0.50, 0.80 and 0.95, K26 strict support-loss frames fall from
113 to 80, 466 to 333 and 22 to 16; mean paid attempts fall from 14.624 to
13.177, 17.073 to 16.009 and 13.042 to 11.988. Every pointwise and primary or
sensitivity population nonworsening gate passes.

K26 changes five physical actions, all exact queue/action pairs already covered
by the closed quotient transition. One additional state only reverses the raw
lane tuple while preserving service, duration and per-class completion-offset
multisets. The preregistered redundancy decision is
`KNOWN_QUOTIENT_ACTION_STOP`: retain the computational result but do not repeat
an authored drain or run trajectories. The manuscript remains unchanged.
Artifacts use the
`aligned_hardware_requirement_dominance_shadow_development_20260911` prefix.

The registered two-lane anchor-star candidate screen is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_anchor_star_shadow --register
python3 -m algorithm.experiments.aligned_hardware_anchor_star_shadow --run-registered
python3 -m pytest -q skill/tests/test_anchor_star_frame_candidates.py skill/tests/test_aligned_hardware_anchor_star_shadow.py
```

The selector retains all nine local options and the 100 ordered two-lane
actions. It pays for all-idle and nine maximal homogeneous seeds, then spends
the remaining budget only on complete directed heterogeneous pairs. Free self
and individually infeasible skips plus a safe suffix bound give a worst-case
support-exact cap of K82 rather than K120; equality pruning may close earlier,
so exactness does not claim a globally tied duration, raw tuple or physical
action. All 5,003 states, 18 population cells and 11 execution checks pass, and
K82 matches the registered K120 maximum support on every state. The 92-test
dependency chain also passes.

At primary loads 0.50, 0.80 and 0.95, K26 strict support-loss frames fall from
113 to 6, 466 to 134 and 22 to 9. Mean certified gaps fall from 0.010434 to
0.002279, 0.102700 to 0.038486 and 0.025310 to 0.005518; mean paid attempts
fall from 14.624 to 12.165, 17.073 to 14.529 and 13.042 to 11.107. Every
pointwise gate and all six primary and K120-distribution sensitivity metrics
are nonworse.

The action audit distinguishes sorted physical signatures from the actual FIFO
execution order. Of 1,397 raw-only changes, 912 preserve the per-class ordered
completion offsets and 485 can change FIFO tail pairing. Together with 16
physical changes, there are 501 FIFO-effective state/action pairs; 14 exactly
match the completed quotient transition and 487 are new. The preregistered
decision is `ADVANCE_AUTHORED_TRANSITION`: the next controlled stage is a
separately registered short drain for those new FIFO-effective pairs, reusing
the completed known cases. No trajectories or paper text change in this
screen. Artifacts use the
`aligned_hardware_anchor_star_shadow_development_20260911` prefix.

Before executing the 487 new pairs, the registered transition closure is
reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_anchor_star_transition --register
python3 -m algorithm.experiments.aligned_hardware_anchor_star_transition --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_hardware_anchor_star_transition.py \
  skill/tests/test_anchor_star_frame_candidates.py \
  skill/tests/test_aligned_hardware_anchor_star_shadow.py \
  skill/tests/test_authored_transition_drain.py \
  skill/tests/test_aligned_hardware_permutation_quotient_transition.py \
  skill/tests/test_permutation_quotient_frame_candidates.py \
  skill/tests/test_aligned_hardware_permutation_quotient_shadow.py \
  skill/tests/test_budgeted_frame_candidates.py \
  skill/tests/test_aligned_budgeted_frame_search_trace.py \
  skill/tests/test_aligned_hardware_candidate_search_diagnostic.py \
  skill/tests/test_aligned_hardware_frame_model.py \
  skill/tests/test_aligned_hardware_frame_experiment.py
```

The closure partitions all 1,413 changed actions into 912 FIFO-equivalent
aliases, 14 exact matches to completed quotient-transition cases and 487 novel
FIFO-effective pairs. The 14 reused cases already contain six positive
mean-flow deltas and 12 positive state-profile P90 deltas; the largest reused
P90 regression is 4.636 seconds. These witnesses falsify two mandatory
universal conditions in the frozen conjunctive transition gate, so outcomes
for the 487 novel pairs cannot restore an advance decision.

All 11 closure checks and the 92-test dependency chain pass. Registration and
execution call no selector or drain and use no arrivals, trajectories or RNG.
They reuse 56 completed drains and skip 1,948 new drains. The scientific
decision is `KNOWN_LOCAL_FLOW_TRADEOFF_STOP`; the unknown novel effects and
nine-cell complete weighted effects remain unestimated. The anchor-star branch
is closed without tuning, full trajectories or paper changes. Artifacts use
the `aligned_hardware_anchor_star_transition_development_20260911` prefix.

The registered FIFO-execution quotient-cache screen and its evidence-only
closure are reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_fifo_execution_quotient_shadow --register
python3 -m algorithm.experiments.aligned_hardware_fifo_execution_quotient_shadow --run-registered
python3 -m algorithm.experiments.aligned_hardware_fifo_execution_quotient_evidence_reuse --register
python3 -m algorithm.experiments.aligned_hardware_fifo_execution_quotient_evidence_reuse --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_hardware_fifo_execution_quotient_evidence_reuse.py \
  skill/tests/test_aligned_hardware_fifo_execution_quotient_shadow.py \
  skill/tests/test_fifo_execution_quotient_frame_candidates.py \
  skill/tests/test_authored_transition_drain.py \
  skill/tests/test_aligned_hardware_permutation_quotient_transition.py \
  skill/tests/test_permutation_quotient_frame_candidates.py \
  skill/tests/test_budgeted_frame_candidates.py
```

The selector retains the v4 heap, option order, support bound and raw tie rule,
but reuses a candidate only when aggregate service, effective duration and every
class's FIFO-ordered completion-offset sequence are identical. This distinguishes
lane reversals that the earlier option-multiset cache merged. The frozen
two-lane catalogue has 100 raw actions and 71 exact FIFO execution signatures;
K71 reproduces the registered K120 raw, physical and FIFO witness on all 5,003
states. All 16 shadow execution checks and the 58-test dependency chain pass.

At primary loads 0.50, 0.80 and 0.95, K26 strict support-loss frames fall from
113 to 104, 466 to 415 and 22 to 21. Exact-certification fractions rise from
0.87805 to 0.95435, 0.73911 to 0.79625 and 0.95278 to 0.97302; mean paid
attempts fall from 14.624 to 13.326, 17.073 to 16.238 and 13.042 to 12.457.
Every pointwise, primary and K120-distribution sensitivity gate passes. Four
states change physical and FIFO execution, and all four exactly match completed
permutation-quotient transition cases; there are no novel transition pairs.

The evidence closure therefore executes no selector or drain. It reuses four
case outcomes, representing 16 completed arm/profile drains and 61 primary
changed-state visits. All four unique-state mean-flow effects improve; the
occurrence-weighted flow-area delta is -437.885 job-seconds, or -7.178
job-seconds per matched visit, and all nine weighted-cell totals are nonpositive.
However, state 582 worsens nearest-rank P90 by 4.636 seconds under both age
profiles, and state 753 worsens unit-FIFO P90 by 2 seconds. The inherited
conjunctive tail gate therefore fails and the scientific decision is
`KNOWN_TAIL_TRADEOFF_STOP`. Close this branch without tuning, new drains, full
trajectories or paper changes. Artifacts use the
`aligned_hardware_fifo_execution_quotient_shadow_development_20260911` and
`aligned_hardware_fifo_execution_quotient_evidence_reuse_development_20260911`
prefixes.

The action-preserving FIFO certificate sidecar is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_fifo_execution_certificate_sidecar --register
python3 -m algorithm.experiments.aligned_hardware_fifo_execution_certificate_sidecar --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_hardware_fifo_execution_certificate_sidecar.py \
  skill/tests/test_fifo_execution_quotient_frame_candidates.py \
  skill/tests/test_aligned_hardware_fifo_execution_quotient_shadow.py \
  skill/tests/test_aligned_hardware_fifo_execution_quotient_evidence_reuse.py \
  skill/tests/test_budgeted_frame_candidates.py \
  skill/tests/test_authored_transition_drain.py \
  skill/tests/test_aligned_hardware_permutation_quotient_transition.py \
  skill/tests/test_permutation_quotient_frame_candidates.py
```

The sidecar fixes the registered legacy K26 raw action, service, duration and
score, and intersects its upper bound with the FIFO-execution quotient K26
upper bound. It reuses the registered K71/K120 oracle result rather than
repeating K71. All 5,003 actions are unchanged; 163 unique states receive a
strictly tighter certificate, and exact certification rises from 4,813 to
4,890 states. All 15 execution checks, 16 certificate gates and 73 dependency
tests pass, with no arrivals, trajectories, drains or RNG.

On the primary K26-policy populations at loads 0.50, 0.80 and 0.95, mean
certified gaps fall by 40.49%, 35.20% and 62.19%, while exact-certified frame
fractions rise by 7.56, 5.54 and 2.01 percentage points. Fixed-action support
and support regret are identical throughout; the K120-policy sensitivity
populations also pass every non-regression gate. The formal timing pass adds
1.170 seconds of FIFO certificate work to 0.688 seconds of legacy selection,
for 1.858 seconds total over 5,003 states (0.371 ms/state, 2.701 times the
legacy-only wall time). Timing has no registered scientific threshold, so this
is retained as an offline certificate audit rather than an online latency
claim. The decision is `CERTIFICATE_TIGHTENING_ONLY_CLOSE`; unchanged actions
make flow and tail trajectories exactly the legacy trajectories, so this
branch closes without another drain or paper change. Artifacts use the
`aligned_hardware_fifo_execution_certificate_sidecar_development_20260912`
prefix.

The registered finite-box budget/support frontier is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_budget_support_frontier --register
python3 -m algorithm.experiments.aligned_hardware_budget_support_frontier --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_budgeted_frame_search_curve.py \
  skill/tests/test_aligned_hardware_budget_support_frontier.py
```

One continuous legacy-v4 search is checkpointed at every integer K from 10
through 120 on the complete 41 by 26 by 25 queue box (26,650 states). An
independent 100-raw-action oracle audits support only. The returned selector
gap defines the computable coefficient
`alpha_D(K)=max_q (upper-score)/L1(q)`; oracle regret and frontier slack remain
diagnostics. All 20 execution gates pass, including 2,958,150 checkpoints,
2,665,000 oracle action checks, full-box anchor comparisons and exact K26/K120
reconstruction on the registered 5,003 states.

At K26, alpha is 0.0864864: the residual margin is positive at load 0.50 but
negative at loads 0.80 and 0.95. K73 is the first subexact budget with positive
residual margins at all three loads: 0.0997602, 0.0339740 and 0.00108094. Every
state already has the oracle-optimal support score at K73, while 26,648 of
26,650 states have an exact returned certificate; the remaining alpha is bound
slack. K76 certifies all states exactly. K73 uses 288,806 paid attempts versus
288,812 at K120, a saving of only 6 attempts (0.0021%), because most searches
close far below either nominal cap.

The K73 algebraic backlog thresholds are 5,125.42, 15,401.65 and 491,231.03,
all above the box maximum L1 backlog of 89. Thus the box contains no state in
the strict negative-drift region at any load, and it is not invariant under
Poisson arrivals. The decision
`POSITIVE_FINITE_BOX_MARGIN_ADVANCE_TO_GLOBAL_EXTENSION` therefore advances
only to an analytic all-queue or polyhedral-cone alpha extension. It is not a
trajectory gate or a global-stability result, and it does not improve the
stronger K-independent homogeneous-seed certificate. Artifacts use the
`aligned_hardware_budget_support_frontier_development_20260912` prefix.

The registered all-integer-queue extension is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_global_support_extension --register
python3 -m algorithm.experiments.aligned_hardware_global_support_extension --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_hardware_global_support_extension.py
```

Exact binary-rational comparisons show that the maximum local support is
always attained by the CNN-p3, JAX-p3 or LLM-p8 pure option. If a maximizing
option fits on both lanes, its mandatory homogeneous seed meets the root upper
bound and closes after 10 attempts. Every queue where this argument can fail
lies in `[0,40] x [0,18] x [0,15]`, which is contained in the already-audited
finite box. The complete K10--K120 curve therefore extends to every
nonnegative integer queue without another selector or oracle run.

K51 has zero global actual support regret under the offline oracle, K73 has a
global returned-gap coefficient of 0.00988342, and K76 has a zero returned gap
globally. The K73 residual margins at loads 0.50, 0.80 and 0.95 are 0.0997602,
0.0339740 and 0.00108094. Its generic high-load drift threshold is 491,231.03,
versus 48,428.62 for the stronger K-independent homogeneous-seed certificate,
and it saves only 6 of 288,812 paid attempts over K120 on the source box. The
decision is
`GLOBAL_K73_RETURNED_GAP_CERTIFIED_K76_EXACT_COMPUTE_GAIN_NEGLIGIBLE_CLOSE`;
the next experiment measures scaling as the Cartesian lane-action family grows.
Artifacts use the
`aligned_hardware_global_support_extension_development_20260912` prefix.

The registered Cartesian scaling screen is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_cartesian_scaling --register
python3 -m algorithm.experiments.aligned_hardware_cartesian_scaling --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_hardware_cartesian_scaling.py
```

The fixed nine-option catalogue is replicated across 2, 4, 8 and 16 synthetic
lanes, inducing ordered raw action counts from `10^2` through `10^16`. One
continuous lazy search is evaluated at 11 frozen budgets on each of 1,000
threshold-aware queue points. The formal run executes 4,000 traces and 12
capacity calculations in 22.43 seconds, with no raw-action enumeration,
oracle, arrival tape or trajectory.

Four lanes close exactly by K2048 while using 0.326% of raw candidate attempts
and 2.739% of the registered logical-work proxy. At K4096, eight lanes use
0.000523% of raw candidate attempts but retain alpha 0.352410 versus the
high-load margin 0.0438574; sixteen lanes use about `1.04e-11%` but retain
alpha 1.001842 versus margin 0.0877149. The decision is
`COMPUTE_SAVING_WITH_LOOSE_CERTIFICATE_STOP`: lazy construction avoids the
Cartesian count, but the current upper bound does not preserve the registered
finite-grid capacity margin as lane count grows. These lane replicas are an
algorithmic scaling model, not measurements of additional GPUs. Artifacts use
the `aligned_hardware_cartesian_scaling_development_20260912` prefix.

The fixed-witness action-regret veto is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_scaling_action_regret_veto --register
python3 -m algorithm.experiments.aligned_hardware_scaling_action_regret_veto --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_hardware_scaling_action_regret_veto.py
```

This post-result mechanism audit replays only the two K4096 alpha witnesses
from the scaling screen. At eight lanes, a fixed feasible challenger gives a
normalized action-regret lower bound of 0.146327, exceeding the smallest
capacity margin 0.0438574. At sixteen lanes the corresponding values are
0.686110 and 0.0877149. Both selected actions therefore fail the capacity gate
even if an action-preserving sidecar tightens the upper bound to the true
optimum. The decision is `ACTION_REGRET_VETO_CERTIFICATE_ONLY_STOP`; the next
development branch must improve scalable candidate discovery. The formal run
uses two selector calls and six small capacity calculations, with no grid,
oracle, trajectory or RNG work. Artifacts use the
`aligned_hardware_scaling_action_regret_veto_development_20260912` prefix.

The registered local LCB complete-frame envelope is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_lcb_frame_envelope_shadow --register
python3 -m algorithm.experiments.aligned_hardware_lcb_frame_envelope_shadow --run-registered
python3 -m pytest -q \
  skill/tests/test_aligned_hardware_lcb_frame_envelope_shadow.py
```

Seven dedicated actions use their empty-state simultaneous drain upper bounds;
the two loaded actions use the loaded target upper bound plus a fresh resident
p1 drain upper bound. With the original point-model arrival vectors fixed, the
effective LCB margins at loads 0.50, 0.80 and 0.95 are 0.0862782, 0.0202398 and
-0.0127793. The largest certifiable point-model load factor is 0.891946. Loaded
duration sensitivity does not move this boundary because loaded actions have
zero capacity-LP weight. Shadow-relative margins remain positive but are only a
renormalized diagnostic. The decision is `HIGH_LOAD_LCB_CAPACITY_FAIL_STOP`.

The empty and loaded source gates each have nominal simultaneous coverage 0.9;
without an independence assumption their intersection has only a 0.8 union-
bound lower guarantee. Nine calibration waves per source cannot certify the
requested combined 0.95 level. The formal run reads 26 local wave JSON files
and runs small LPs only; it performs no server pull, trajectory or RNG work.
Artifacts use the
`aligned_hardware_lcb_frame_envelope_shadow_development_20260912` prefix.

The residual-fill scaling screen is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_residual_fill_scaling_screen --register
python3 -m algorithm.experiments.aligned_hardware_residual_fill_scaling_screen --run
python3 -m pytest -q skill/tests/test_residual_fill_frame_candidates.py
```

The repaired fractional upper bound improves the fixed K4096 Cartesian result:
alpha falls from 0.352410 to 0.148421 at eight lanes and from 1.001842 to
0.187877 at sixteen lanes. The corresponding 0.80-load residual margins are
0.0270087 and 0.162983, but the 0.95-load residuals remain negative at
-0.104564 and -0.100162. Increasing the budget to K8192 leaves both alpha
values unchanged. The decision is
`CERTIFIED_IMPROVEMENT_WITH_HIGH_LOAD_FAIL_STOP`; this branch is retained as a
negative high-load result. Artifacts use the
`aligned_hardware_residual_fill_scaling_screen_development_20260912` prefix.

The count-space branch-and-bound development screen and its independent queue
confirmation are reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_scaling_screen --register
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_scaling_screen --run
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_confirmation --register
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_confirmation --run
python3 -m pytest -q \
  skill/tests/test_count_space_bnb_frame_candidates.py \
  skill/tests/test_aligned_hardware_count_space_bnb_scaling_screen.py \
  skill/tests/test_aligned_hardware_count_space_bnb_confirmation.py
```

The method searches integer option counts and certifies every unresolved count
box with an exactly checked repaired LP dual or an inherited analytic bound.
On the consumed 1,000-state grid, R8/K36 and R16/K72 pass the quality and work
gates. R4 stops the aggregate development decision because its matrix work is
5.97568% of exhaustive ordered search, above the frozen 5% limit. No gate was
changed after seeing that result.

The independent 512-queue roster is disjoint from that development grid. Its
fixed primary cases both pass: R8/K36 has alpha 0.0314303 and a 0.95-load
residual margin of 0.0124271; R16/K72 has alpha zero and residual margin
0.0877149. All 1,024 primary queue/case certificates are valid, with no ordered
action or count-vector enumeration. The decision is
`PRIMARY_LARGE_ACTION_CONFIRMATION_PASS`; it supports a separately registered
trajectory transition while remaining a finite-roster algorithmic result, not
a physical multi-GPU measurement or an all-queue theorem. Artifacts use the
`aligned_hardware_count_space_bnb_scaling_screen_development_20260913` and
`aligned_hardware_count_space_bnb_confirmation_20260913` prefixes.

The registered trajectory transition is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_trajectory_transition --register
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_trajectory_transition --run
```

All 13 integrity checks pass on 32 fresh common tapes, but the scientific gate
does not. R8 fails mean-flow non-inferiority at loads 0.80 and 0.95, while R16
fails it at 0.80. At 0.95, the observed support loss exceeds the certified
margin for both R8 and R16. The decision is
`COUNT_SPACE_TRAJECTORY_TRANSITION_FAIL_STOP`; these consumed tapes must not be
used to tune budgets or gates.

Fresh-root exactness escalation is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_exactness_escalation_development --register
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_exactness_escalation_development --run
```

Every escalated frame is exact and its action and per-job flow sequence match
the cold K4096 count-space reference. The work gate nevertheless fails in all
four cases: restarts increase aggregate LP calls from 70,204 to 114,484 and
matrix coefficients from 6,482,497 to 11,253,706. The decision is
`EXACTNESS_ESCALATION_WORK_FAIL_STOP`. A fresh restart pays the complete final
exact search plus every failed smaller search, so this branch is stopped rather
than reporting nominal budget limits as work savings.

A separate temporal-incumbent development run projects the previous exact
action into the current visible queue and uses it only as a certified warm
lower bound. All 13 integrity checks pass, including equal-score tie handling,
and warm and cold K4096 trajectories select identical actions and per-job flow
sequences. Both searches still use exactly 68,284 LP calls and 6,235,173 matrix
coefficients; warm start adds 2,313 incumbent evaluations and 81,693 projection
probes. The decision is `TEMPORAL_INCUMBENT_WORK_FAIL_STOP`, so the temporal
incumbent is retained as a negative result and is not adopted.

The independent cold exact compute confirmation is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_cold_exact_compute_confirmation --register
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_cold_exact_compute_confirmation --run
```

On 32 fresh point-model trajectories, one cold K4096-capped search per frame
certifies the optimal support score in all 2,314 frames. The searches execute 72,976 LPs,
0.770% of the cap envelope; the empirical p95 is 139 LPs per frame and the
maximum is 1,052. Matrix work is 6,827,226 coefficients, 0.616% of its declared
non-executed cap envelope. All 15 integrity and compute gates pass, with no
restart, warm start, exhaustive ordered-action enumeration, or exhaustive
count-vector enumeration.
The decision is `COLD_EXACT_COMPUTE_CONFIRMATION_PASS`; it confirms empirical
point-model compute and exact support scores, not a performance advantage or
the pending joint-95 physical model. A subsequent audit found two equal-score
R8 frames whose returned action was not the selector's canonical tie-break
action and whose completion offsets differed. The artifact is therefore not
evidence of canonical exact policy behavior; a versioned equality-pruning fix
and independent confirmation are required for that stronger claim.

That versioned confirmation is reproducible with:

```bash
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_canonical_exact_compute_confirmation --register
python3 -m algorithm.experiments.aligned_hardware_count_space_bnb_canonical_exact_compute_confirmation --run
```

The canonical selector retains an equal-score count box unless an optimistic
lower bound on its `(duration, option-index tuple)` key cannot beat the current
incumbent. On 32 new trajectories, all 2,347 frames are both score-exact and
canonical-exact. Aggregate LP cap utilization is 0.807%, empirical p95 work is
131 LPs per frame, and the observed maximum is 796/4096. The run performs
43,329 canonical-key bound checks and prunes 5,267 certified tie regions; all
17 execution and compute checks pass. The decision is
`CANONICAL_EXACT_COMPUTE_CONFIRMATION_PASS`. It supersedes the older artifact
for canonical-policy claims while leaving that artifact's exact-score and work
accounting intact.

The first joint-95 hardware campaign is terminally fail-stopped. Waves 14--22
provide nine valid first-attempt calibration units, but an external scheduler
task overlapped an empty-cell measurement in wave 23's first attempt. Wave 23
is invalid and cannot be retried; 9/19 units cannot define the frozen rank-19
margin, and the holdout was never released. Consequently v1 supports no
joint-95 margin, holdout, capacity, selector, or baseline claim.

An independent v2 campaign was frozen and registered before its measurements:
fresh waves 34--52 were assigned to calibration and wave 53 to the locked
holdout, with v1 contributing zero eligible units or scores. Wave 34 attempt01
stopped after seven empty cells: it has no loaded rows or scored observations.
The terminal decision is `JOINT95_V2_INTERVAL_OVERLAP_FAIL_STOP`, leaving 0/19
complete calibration units and 0/1 holdout units. The implicated queue task
`t93454` was auto-adopted from that visit's preceding profile-1 cell; its
scheduler timestamps record discovery rather than physical execution. That
cell's children had exited before the next measurement began, so the recorded
overlap does not establish external interference. The frozen v2 failure is
retained; a corrected interval audit requires a separately registered campaign.

The v2 capacity closure, canonical static screen, and corrected-v3 baseline
comparison were also registered outcome-blind. Their runners enforce the order
capacity, then canonical, then baseline after the complete v2 19+1 collection;
all three downstream result artifacts are still absent.

The v2 supervisor `t93366` terminated on that fail-stop. Postcampaign task
`t93367` could never receive its wave-53 file gate and was cancelled before
execution. The capacity, canonical, and baseline protocols remain registered,
but none has a v2 result. The superseded v2 supervisor `t93363` was cancelled
during pre-run audit and has no scientific effect.

The provenance-aware interval audit now recognizes only exact auto-adopted
measurement children with completed execution evidence; other target-node
tasks still cause fail-stop. A read-only replay of the wave-34 interval passes
and identifies `t93454` as the preceding completed child, without changing the
v2 artifact. The independently registered v3 campaign assigned waves 54--72
to calibration and wave 73 to holdout, with v1/v2 contributing zero units.
Waves 54--58 produced five complete calibration units. Wave 59 attempt01
fail-stopped on `UNDECLARED_HOST_PROCESS_DURING_TARGET`: a separate process
group used roughly 6.4--6.9 CPU cores during the loaded target measurement.
The campaign retains 5/19 calibration units, has no holdout or v3 margin, and
cannot run the downstream protocols. Supervisor `t93466` terminated on the
fail-stop; file-gated postcampaign task `t93468` was cancelled because wave 73
will not be produced.
Future joint95 calls now check material same-user host CPU load as part of each
clean preflight; the stricter in-target exposure audit remains in force. An
independent v4 campaign was registered with fresh waves 74--92 for calibration
and wave 93 for the locked holdout. Its capacity, canonical-selector, baseline,
and completion-triggered postcampaign protocols were frozen before measurement.
Waves 74--84 produced eleven complete calibration units. The wave-85 controller
was evicted by the local scheduler's RAM-headroom guard before any remote
measurement began and left its first-attempt artifact at `WAIT_SCHEDULER_ACK`.
Automatic retries were refused because that attempt path already existed. The
v4 collection is therefore `FAIL_RESULT_STATE` at 11/19 calibration units,
without a released holdout, joint margin, or downstream result. Its stale
`gpu_b` scheduler reservation was manually released after all v4
controllers and supervisor retries had terminated.

The independent v5 campaign completed with dispatcher status `COMPLETE`.
Calibration waves 94--112 and the locked holdout wave 113 each produced exactly
one dispatcher-valid first-attempt `PASS` terminal. Every visit contains nine
observations and nine passing scheduler-interval audits; v1--v4 contribute zero
units or scores. The rank-19 calibration margin is 0.9372136082. The holdout
joint score is 0.5406866012, so the registered one-future-visit coverage audit
passes. Supervisor `t94247` completed and released the exclusive `gpu_b`
campaign reservation.

The capacity closure executed validly (`status=PASS`) but reached the negative
scientific decision `JOINT95_NO_POSITIVE_CAPACITY_MARGIN_STOP`. The margin gives
a time multiplier of 1.9372136082 and a common capacity scale of 0.5079464127.
Effective slack is positive only at load 0.5: 0.0069969 for eight lanes and
0.0139938 for sixteen lanes. It is negative at loads 0.8 and 0.95
(-0.2571565/-0.3892332 for eight lanes and -0.5143129/-0.7784663 for sixteen
lanes). Consequently `capacity_established=false` and `closure_passed=false`.
The frozen postcampaign order advances to the canonical and baseline stages
only after a positive capacity decision, so their result artifacts are
correctly absent. Earlier candidate-support evidence is not numerically
comparable because its action-duration catalogue and visited-state population
differ; no selector call, trajectory run, RNG draw, or server pull was made.
The postcampaign runner now JSON-normalizes returned and stored payloads before
comparison and treats this registered negative decision as a successful
terminal outcome. Finalizer `t94682` retained the capacity artifact unchanged
and completed with scheduler status `done`.

Limitations: the cost only optimizes the paid pool and is short-sighted. When
all admitted jobs finish at the frame end, the departure count cancels from H,
favoring shorter frames. Lambda stays fixed during drain as a stationary-arrival
proxy. These development results do not provide independent confirmation.
The cap bounds support loss within the paid pool, not mean-flow loss. With the
cap active, more candidate attempts can raise the support floor, so holding
cost need not decrease with the search budget.
The lookahead terminal proxy assumes fluid clearing without further arrivals
after the next boundary; it is not the value of actual future synchronous frames.
The backlog gate detects sufficient overload for one frame, not every queue
that requires multiple frames, and does not itself guarantee lower flow time.
Continuation interventions share tapes and overlapping futures; their effects
are descriptive, correlated, and cannot be added to infer whole-policy gains.
The continuation residual includes later arrivals and v4 behavior. Each branch
partitions current and future area at its own first-frame boundary.
The batch estimate uses two feasible frames followed by a fluid tail; new
arrivals do not influence the two successor choices. Its current pool is
smaller than a full-K search, and only two contenders receive the new estimate.
The arrival-aware variant uses a fixed-phase rate path, which omits Poisson
fluctuations and can align with batch thresholds. Its planning cutoff H is
distinct from the experiment's arrival cutoff; only the first three frames
are discrete, and the remaining continuation uses the fluid approximation.
The Poisson variant uses one sampled path per decision, so taking the smaller
of two noisy costs can be optimistic. Planning repetitions reuse the same real
arrival seeds and are not additional independent environments.
The fixed control variate preserves each pre-sampling candidate's hybrid-model
expectation; it does not guarantee smaller variance, an unbiased selected
minimum, or improved full-policy flow time.
Fresh 64-sample means retain Monte Carlo uncertainty and cover only visited
disagreements. Their residual against realized flow mixes arrival noise,
horizon cutoff and continuation differences; pure model bias is not identified.
Static homogeneous throughput pressure does not equal finite-queue v4 joint
action selection. Changing this successor rule changes the proxy expectation,
and the retained diagnostic targets cover earlier raw/CV disagreement states.
Sequential lane filling is greedy and does not enumerate all joint successor
actions. The positive result uses development seeds, while the retained target
diagnostic covers states selected by the earlier raw/CV disagreement mechanism.
