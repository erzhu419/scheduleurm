# Scheduleurm OR mathematical-depth review and 10/10 upgrade path

Date: 2026-07-11

## Bottom line

The current proof package is approximately **9/10 in formal correctness and
technical completeness**, but approximately **8--8.5/10 in unmistakable
methodological originality**.  It is credible for an *Operations Research*
submission under a tightly scoped claim: robust candidate MaxWeight stability
on declared finite measured service domains, with explicit candidate,
estimation, penalty, and oracle losses.  It is not yet a 10/10 theory paper if
the claimed object is the unrestricted production scheduler, arbitrary future
workloads, or migration-aware heterogeneous scheduling.

The current Lean tree builds successfully (8054 jobs at the time of this
review), `ScheduleurmUpload.lean` checks, and a comment-aware search finds no
proof-term use of `sorry`, `admit`, or `axiom` in the Scheduleurm artifact.
Those facts establish proof integrity; they do not by themselves establish
methodological novelty or empirical validity.

## What is already mathematically strong

The main theorem has a coherent end-to-end implication:

\[
\text{full-action capacity slack}
\Longrightarrow
\text{candidate support margin}
\Longrightarrow
\text{lower-service margin}
\Longrightarrow
\text{penalized approximate-MaxWeight margin}
\Longrightarrow
\text{quadratic drift}
\Longrightarrow
\text{finite-set recurrence}.
\]

The residual slope is correctly accounted for as

\[
\eta
=
\delta-
\left(
\epsilon_{\mathrm{cand}}
+\epsilon_{\mathrm{est}}
+\beta
+\alpha_1
\right)>0,
\]

with additive term \(B+P_0+\alpha_0\).  The artifact contains fixed-family,
statewise, bounded-sample, bounded-second-moment, approximate-oracle,
diagonal-scaling, operational-capacity, active-bucket, and regime-switching
components.  This breadth is substantially stronger than a conventional
systems-paper proof sketch.

## Hard mathematical risks

### 1. Conditioning and filtration are not yet uniform

The policy observes queue state, node availability, pinned jobs, measured
capacity boundaries, and co-location state.  The stochastic assumptions are
currently written mainly conditional on \(Q(t)\).  The rigorous object should
be the pre-decision information filtration

\[
\mathcal F_t
=
\sigma\!\left(
\text{history through }t,
Q(t),
\chi(t)
\right),
\]

with \(a_t\), the feasible families, lower-service rows, and penalties
\(\mathcal F_t\)-measurable.  Arrival, service, and second-moment assumptions
must be conditional on the same \(\mathcal F_t\).  Otherwise the lower-service
bound may cease to be conservative after observing the scheduler-visible
state used to choose the action.

### 2. Statewise capacity must index the full family as well

The paper explicitly indexes the candidate family and lower-service map, while
the strongest Lean theorem indexes the full family, candidate family, true
service map, lower-service map, feature map, and penalty.  The paper should use

\[
\mathcal A^{\mathrm{full}}_t,
\quad
\mathcal A^{\mathrm{cand}}_t,
\quad
\mu_t,
\quad
\underline\mu_t,
\quad
\Phi_t,
\quad
P_t
\]

consistently.  A uniform pointwise slack assumption is a robust
intersection-of-state-capacity condition.  It must not be confused with
average-regime slack, which needs dwell-time or switching analysis.

### 3. Offered service and completed departures are ambiguous

The queue equation is correct when \(S_i(t)\) is potential or offered service,
which may be wasted when the queue is empty.  If \(S_i(t)\) is described as
actual completed work, a positive lower-service bound cannot hold at an empty
queue.  The manuscript should define

\[
D_i(t)=\min\{Q_i(t),S_i(t)\}
\]

as completed departure and reserve \(S_i(t)\) for potential service.

### 4. The Lean recurrence result has a precise semantic boundary

The Lean artifact proves `PositiveRecurrentViaFiniteSet` through a custom
transition-expectation interface: uniformly bounded truncated hitting times to
a finite set and finite expected returns to that set.  It does not formalize
the complete measure-theoretic chain from a Markov kernel to an invariant
distribution for every implementation.  The manuscript should continue to
call the formal object a finite-set recurrence certificate and invoke the
standard countable-state Foster theorem only under the explicit Markov,
countability, closed-class, and local-return conditions.

### 5. Candidate approximation is still mostly assumed empirically

The general theorem accepts a support gap or a Lipschitz cover.  Most current
theorem-facing slices use exact enumeration and \(\rho=0\).  This validates the
finite-slice theorem but does not demonstrate nontrivial compression of the
actual combinatorial action space.  A 10/10 route should construct the
candidate family from a finite feature partition and prove both

\[
|\mathcal A^{\mathrm{cand}}|
\le
\text{number of active feature cells}
\]

and

\[
H_{\mathrm{full}}(q)
\le
H_{\mathrm{cand}}(q)+L\rho\|q\|_1.
\]

At least one nonzero \(L\rho\) calibration is still required before claiming
that this compression has been empirically validated.

### 6. Heterogeneous LCB scaling should be a main numbered result

The diagonal-scaling Lean theorem is mathematically useful because one global
absolute error is dimensionally inappropriate across CPU, CNN, LLM, and RL
service units.  The paper currently describes the result mostly through the
artifact crosswalk and diagnostics.  It should state and prove the support
bound

\[
H_{\mu}(q)
\le
H_{\mu^{\mathrm L}}(q)
+2\varepsilon\sum_i s_iq_i
\le
H_{\mu^{\mathrm L}}(q)
+2\varepsilon C_s\|q\|_1.
\]

This result must be distinguished from the empirical question of whether the
confidence event holds for every admitted row.

### 7. The current slack certificate is scoped, not production-wide

Loads such as \(\lambda=0.8\mu(a_{\mathrm{selected}})\) produce a valid
controlled capacity check, but positive slack is partly induced by the
experimental load construction.  It does not estimate an unrestricted
production arrival process.  Production stability therefore remains outside
the claim unless a load-certified arrival model and its conservation law are
measured for that population.

### 8. Migration is not yet a theorem-level action model

The implementation records checkpoint, synchronization, staging, warmup, lost
work, and risk costs for controlled tasks.  A time cost in seconds cannot be
subtracted directly from \(Q^\top\underline\mu\) without a common unit map.
A theorem-level migration model should use either a semi-Markov frame or a net
service action that includes downtime and lost work.  Until then, migration is
controlled supplementary evidence, not part of the main stability claim.

### 9. Learning and hidden regimes are not yet one end-to-end theorem

The proof library contains confidence-event, active-bucket, forced-exploration,
dwell/switching, and detector-delay components.  They are currently modular
extensions.  A 10/10 theorem would combine adaptive sampling, confidence
validity under selection, regime detection delay, switching loss, and queue
drift in one stochastic statement.

## ETA matrix boundary

The incomplete ETA matrix is not a defect in the abstract theorem.  The theorem
consumes a conservative service map \(\underline\mu\); task-native tqdm rates
are the empirical mechanism used to estimate that map.  Missing matrix rows
limit which workload-environment-node-load states can enter the theorem-facing
population.  They do not invalidate the implication from certified assumptions
to drift and recurrence.

No unmeasured row, history fallback, or unstable warmup estimate should enter a
theorem-facing lower-service certificate.  The remaining ETA work can therefore
wait for resources without blocking the mathematical revision.

## 10/10 upgrade sequence

1. Use one filtration for all stochastic assumptions and make potential service
   explicit.
2. State the fully indexed statewise theorem with full and candidate families,
   true and lower service, features, and penalties.
3. Add a constructive finite-cell candidate theorem with an explicit cardinality
   and nonzero \(L\rho\) support guarantee.
4. Promote diagonal-scaled LCB support loss to a numbered proposition with a
   conventional proof and Lean crosswalk.
5. Add the finite-horizon expected-backlog bound implied by the drift theorem.
6. Keep the recurrence formalization boundary explicit.
7. In a later mathematical round, model migration through frame-level net
   service and combine adaptive learning/regime detection with drift.
8. After resources are available, calibrate nonzero \(L\rho\), the complete
   node/load-aware lower-service matrix, and migration setup distributions.

## Score after the first upgrade round

If items 1--6 are completed and the paper maintains its current bounded claim,
the theoretical presentation should reach approximately **9--9.3/10**.  A
credible **9.5--10/10** claim requires at least one of the two deeper closures:

- a nontrivial candidate-construction/calibration result on the real
  combinatorial fabric; or
- an end-to-end adaptive-learning or migration-aware stochastic stability and
  performance theorem.

Lean proof volume is supporting evidence, not the source of the novelty score.
The decisive question is which currently assumed objects are derived from the
model and which are only supplied as certificates.

## First revision completed on 2026-07-11

Completed in the first 10/10-oriented revision:

- unified the manuscript stochastic assumptions under the pre-decision
  filtration \(\mathcal F_t\);
- distinguished potential service \(S_i(t)\) from completed departure
  \(D_i(t)\);
- indexed the full family, candidate family, true service, lower service,
  feature map, and penalty in the statewise corollary;
- made the augmented-state recurrence conditions explicit;
- added a numbered finite-feature-cell candidate construction proposition;
- formally proved `finite_feature_cell_candidate_certificate` in Lean, including
  full-family containment, an active-cell cardinality bound, and \(L\rho\)
  support loss;
- added a numbered diagonal-scaled LCB proposition and conventional proof;
- added finite-horizon and asymptotic expected-backlog bounds;
- synchronized the q11 feasible family to profiles 1--5 with boundary 6 and the
  selected-profile LCB slack to \(0.0341241\);
- rebuilt the manuscript and checked the split and consolidated Lean artifacts.

Still required for a defensible 9.5--10/10 methodological claim:

- empirical nonzero \(L\rho\) calibration on a genuinely compressed real action
  family;
- an augmented-state measure-theoretic Lean bridge if the paper wants to claim
  full formalization beyond the queue-indexed specialization;
- an end-to-end adaptive-learning/regime theorem or a frame/semi-Markov
  migration theorem;
- the remaining node/workload/load-state ETA matrix and migration distributions
  once resources are available.
