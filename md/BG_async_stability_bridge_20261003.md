# BG: a physical-time stability bridge for the cached lane model

This argument concerns the infinite-horizon extension of the registered causal dispatch rule, with a fixed deterministic catalogue and independent Poisson arrivals of rates \(\lambda_i\). Each of \(R\) independent lanes immediately starts another action, or a one-second idle poll, after its reservation ends. Computation does not delay service.

## Model and uniform support bounds

For action \(a\), let \(n_a\) be its admitted job counts and let \(\tau_a=\max\{1,\text{cached duration}_a\}\). Include the idle action \(n_0=0,\tau_0=1\). Put
\[
B=\max_a\|n_a\|_1,\quad K=RB,\quad M_i=\max_a n_{a,i},\quad r_i=\max_a n_{a,i}/\tau_a,\quad r_*=\max_i r_i.
\]
The effective minimum duration, including idle polling, is \(\tau_{\min}=1\), and \(T=\max_a\tau_a<\infty\). Define \(H(q)=\max_a q^\top n_a/\tau_a\) and its feasible counterpart \(H_f(q)\), which requires \(n_a\le q\). Both include idle.

**Projection lemma.** In all four frozen catalogues, each cross-class action has counts \((1,0,1)\). Each of its single-coordinate rate projections is strictly dominated by a dedicated action for that coordinate. Dedicated actions already have single-coordinate support. Consequently, the projection of every action onto every coordinate subset is dominated by an action supported on that subset.

For an integer queue, set \(J=\{i:q_i\ge M_i\}\). Every action supported on \(J\) is feasible. The projection property gives \(H_f(q)\ge\max_a\sum_{i\in J}q_i n_{a,i}/\tau_a\); omitted coordinates satisfy \(q_i\le M_i-1\). Hence
\[
H(q)-H_f(q)\le C_{\rm av}:=\sum_i(M_i-1)r_i.
\]
Singleton existence alone would not imply this bound; the measured projection domination is the necessary additional fact verified here.

**BG gap.** BG uses exact feasible MW when \(\|q\|_1>K\). Inside the envelope, any selected feasible action has nonnegative support and \(H_f(q)\le Kr_*\). Therefore, for every realised planning history and random sample,
\[
H_f(q)-q^\top n_{\rm BG}/\tau_{\rm BG}\le C_{\rm BG}:=Kr_*,\qquad
q^\top n_{\rm BG}/\tau_{\rm BG}\ge H(q)-C,\quad C=C_{\rm av}+C_{\rm BG}.
\]
Overflow fallbacks also satisfy this bound. No accuracy or stationarity assumption on count/time rate estimates, FIFO ages, planning RNG or S2 predictions is used.

Static constants from the frozen confirmation catalogue (single-lane support units):

| Catalogue | \(K\) | \(T\) | \(C_{\rm av}\) | \(C_{\rm BG}\) |
|---|---:|---:|---:|---:|
| cpu_a | 10 | 88.105419 | 1.251048 | 2.839517 |
| node007 | 40 | 60.772729 | 2.037051 | 8.290096 |
| gpu_a | 16 | 23.567414 | 3.707810 | 6.117243 |
| gpu_b | 16 | 14.972417 | 5.051665 | 8.619941 |

## Fixed physical-window drift

Assume strictly interior offered load, with a uniform support slack \(\delta>0\):
\[
RH(q)-q^\top\lambda\ge\delta\|q\|_1\qquad(q\ge0).
\]
Let \(Q(t)\) count all born, unadmitted jobs, excluding inflight jobs; at dispatch it equals the released queue supplied to the policy. Between dispatches it also includes births awaiting the next release. Use consistent half-open windows \([t,t+L)\), measuring boundary queues before boundary admissions. Let \(A_L,D_L\) be arrivals and admissions in that window, so \(Q(t+L)=Q(t)+A_L-D_L\). Assume \(\mathbb E V(Q(0))<\infty\), with \(V(q)=\frac12\sum_iq_i^2\).

Take a fixed \(L>2T\), rather than treating an option duration as the next global event interval. Each lane has at most two partially overlapping reservations at the window endpoints, each of duration at most \(T\). Full reservations wholly inside the window thus cover at least \(R(L-2T)\) lane-time, including idle polls; their admissions are a subset of \(D_L\).

Each lane starts at most \(\lceil L/\tau_{\min}\rceil+2\) reservations. Put \(N_L=RB(\lceil L/\tau_{\min}\rceil+2)\), \(\Lambda=\sum_i\lambda_i\), and \(U_L=\|A_L\|_1+N_L\). Then \(\|D_L\|_1\le N_L\) and every within-window pre-decision queue \(q_s\) obeys \(\|q_s-Q(t)\|_1\le U_L\).

The support function and every action rate have Lipschitz constant \(r_*\) in the queue \(\ell_1\) norm. Combining this fact twice with the BG bound gives \(Q(t)^\top n_a/\tau_a\ge H(Q(t))-C-2r_*U_L\) for every full reservation selected inside the window. Their nonnegative durations total at most \(RL\), so
\[
Q(t)^\top D_L\ge R(L-2T)H(Q(t))-RL(C+2r_*U_L).
\]
Future physical Poisson increments are independent of past actual and planning information. Thus \(\mathbb E[U_L\mid\mathcal F_t]\le\Lambda L+N_L\), and the queue identity yields the quadratic bound \(\Delta V\le\frac12\sum_i(A_{L,i}^2+D_{L,i}^2)+Q(t)^\top(A_L-D_L)\). A sufficient finite constant is
\[
B_L=\tfrac12\left[\sum_i\{(\lambda_iL)^2+\lambda_iL\}+N_L^2\right]+RL\left[C_{\rm av}+C_{\rm BG}+2r_*(\Lambda L+N_L)\right].
\]
Using the support slack and \(H(q)\le r_*\|q\|_1\), we obtain
\[
\mathbb E[V(Q(t+L))-V(Q(t))\mid\mathcal F_t]\le B_L-\underbrace{(L\delta-2RT r_*)}_{c_L}\|Q(t)\|_1.
\]

## Consequence and scope

Choose \(L>\max\{2T,2RT r_*/\delta\}\). Summing drift over the fixed skeleton gives \(\limsup_m m^{-1}\sum_{k<m}\mathbb E\|Q(kL)\|_1\le B_L/c_L\). Between boundaries, expected additional backlog is at most \(\Lambda L\), giving physical-time mean-backlog strong stability. The full unfinished backlog differs from this queue by at most \(RB\) inflight jobs, each with remaining completion offset at most \(T\).

This establishes a conditional stability bridge for the stated infinite-horizon model. It does not turn the finite confirmation into an infinite-horizon test, assert recurrence of the complete history-dependent controller state, or prove actual P90 improvement, physical service confidence, or BG's performance advantage over MW. The constants are conservative stability constants, not predicted performance gains.

## Extension to the registered periodic arrival model (2026-10-04)

For the new experiment's ideal infinite-horizon extension, let each class have intensity \(\lambda_i m_\phi(t)\), with the independent initial phase \(\phi\) and common periodic multiplier \(m_\phi\) equal to 1.5 and 0.5 for 50 seconds each. Its cumulative clock satisfies \(C_\phi(t+100k)-C_\phi(t)=100k\) for every physical start \(t\) and integer \(k\ge1\).

Choose \(L=100k>\max\{2T,2RT r_*/\delta\}\), and include the phase in the theoretical filtration. Conditional on phase and causal past actual/planning information, independent physical Poisson increments give independent full-window counts \(A_{L,i}\sim\operatorname{Poisson}(\lambda_i L)\). Their laws are the same for either phase, so conditioning only on the observable history gives the same first and second moments. Here \(\lambda\) is the average arrival vector, required to lie strictly inside the same capacity region.

The queue-change bound, lane reservation coverage and BG support-loss constants above are unchanged. Thus the same \(B_L\) and \(c_L\) give negative drift on this skeleton. Within a skeleton window, births are bounded by the entire window's count, with expectation \(\Lambda L\); physical-time mean-backlog stability follows as before.

This extension concerns continued periodic NHPP arrivals and the fixed deterministic service model. It supplies no finite-horizon, transient-burst, actual-P90 or BG-over-MW performance conclusion; those remain measured outcomes of the registered experiment.

Sources: [frozen confirmation catalogue](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_background_competition_continuous_confirmation_20260930_protocol.json), [bounded BG selection](/home/researcher/mine_code/scheduleurm/algorithm/background_competition_continuous_lane_rollout.py:200), [duration floor and queue envelope](/home/researcher/mine_code/scheduleurm/algorithm/arrival_aware_lane_rollout.py:109), [asynchronous admission clock](/home/researcher/mine_code/scheduleurm/algorithm/experiments/observed_rate_contextual_lane_refill_replay.py:29), [registered periodic arrival design](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_bursty_arrivals_20261004_design.json).
