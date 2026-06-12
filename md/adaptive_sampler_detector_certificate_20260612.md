# Adaptive Sampler / Detector Probability Certificate

## Summary

| Quantity | Value |
|---|---:|
| `pass` | true |
| `concrete_probability_model_ready` | true |
| `live_scheduler_integrated` | false |
| `active_bucket_count` | 64 |
| `scenario_count` | 120 |
| `sampler_certified` | true |
| `detector_certified` | true |

## Scope

concrete probability certificate for a deployable sampler/detector model. It closes the stochastic inputs conditionally for this model, but does not claim the current production scheduler has already enabled the sampler or change detector.

## Sampler

| Quantity | Value |
|---|---:|
| `model` | `deterministic_round_robin_forced_active_bucket_sampler` |
| `total_decisions` | 99938 |
| `active_bucket_count` | 64 |
| `min_forced_samples_per_bucket` | 1561 |
| `required_min_samples_per_bucket` | 128 |
| `coverage_probability_lower_bound` | 1.000000000 |
| `bounded_unit_radius_at_forced_min` | 0.052304086 |

## Detector

| Quantity | Value |
|---|---:|
| `model` | `bounded_two_window_mean_shift_detector` |
| `window_size` | 512 |
| `threshold` | 0.200000000 |
| `min_shift` | 0.400000000 |
| `switching_count` | 39 |
| `per_change_miss_bound` | 0.000071426 |
| `per_window_false_alarm_bound` | 0.000071426 |
| `union_tail_bound` | 0.005642630 |
| `detection_delay_bound_decisions` | 1024 |
