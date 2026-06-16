# Module30 Slack Accounting Certificate

Date: 2026-06-05

This module adds the consolidated theorem-condition certificate for the main
Scheduleurm stability theorem. It does not invent new constants; it combines
the calibration outputs required by the proof.

Reviewer-facing condition:

```text
delta > Lrho + epsilon_est + beta + alpha1
eta = delta - (Lrho + epsilon_est + beta + alpha1)
```

where:

```text
Lrho        candidate fabric-cover support loss
epsilon_est lower-service estimation loss
beta        queue-scaled penalty slope
alpha1      queue-scaled approximate-oracle error
```

The fixed terms `B`, `P0`, and `alpha0` do not consume linear slack; they set
the finite-set Foster threshold.

## Artifact

```text
algorithm/experiments/slack_accounting.py
skill/tests/test_experiment_calibration.py
```

Component inputs:

```text
algorithm/experiments/fabric_metric.py     -> L, rho, Lrho
algorithm/experiments/service_model.py     -> epsilon_est
algorithm/experiments/penalty_fit.py       -> P0, beta
algorithm/experiments/oracle_audit.py      -> alpha0, alpha1
algorithm/experiments/capacity_lp.py       -> delta, eta helper
```

## Command

```bash
python3 -m algorithm.experiments.slack_accounting build \
  --fabric <fabric_metric_certificate.json> \
  --service <service_lower_certificate.json> \
  --penalty <penalty_envelope.json> \
  --oracle <oracle_audit.json> \
  --capacity <capacity_slack.json> \
  --moment <moment_bound.json> \
  --N <finite_set_threshold_claim> \
  --output <slack_certificate.json> \
  --markdown-output <slack_certificate.md>
```

The command exits with code `0` only when:

```text
eta > 0;
N is large enough for B + P0 + alpha0 + alpha;
all component inputs are marked theorem-usable.
```

## Current Status After Module48

The accounting layer is implemented and tested. Module48 now produces the first
positive numeric table for the measured `hybrid_research_portfolio` finite
service-action slice:

```text
md/experiment_module48_theorem_condition_calibration.md
md/experiment_artifacts/module48_portfolio_slack_certificate.json
md/experiment_artifacts/module48_portfolio_slack_certificate.md
```

The module48 certificate has:

```text
delta = 0.078777710
Lrho = 0
epsilon_est = 0
beta = 0
alpha1 = 0
eta = 0.078777710
B = 34499765.485
finite_set_threshold_N = 437938174
```

This closes the arithmetic slack-accounting gap for a declared finite-support
load model on the measured service cache. It still does not claim a production
arrival process or unmeasured bucket generalization.

Modules31 and 38 improve the inputs beyond the portfolio certificate. q10 now
has a declared local CPU bucket with profiles 1-9 measured and profile 10
closed by a measured capacity boundary. q00 now has a declared local
light-control bucket with profiles 1-13 measured and profile 14 closed by a
measured capacity boundary. These promotions supply theorem-relevant
service-map slices
`\widehat{\mu}_b(k)` and infeasible-boundary evidence for exact replay. They do
not by themselves certify production stability; that still requires a
load-certified arrival model and live oracle-error audit for the scheduler mode
being claimed.

This is an improvement over the previous state because the missing quantities
are now operationally precise. A future experiment cannot merely report
"candidate improves replay"; it must either produce a positive `eta` certificate
or explicitly state that the tested load point is outside the certified theorem
region.

## Validation

Targeted tests check:

```text
positive eta but too-small N is rejected;
positive eta with sufficient N is accepted;
nonpositive eta is rejected;
reviewer-facing markdown table is emitted.
```
