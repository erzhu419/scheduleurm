# OR Gate: Online Arrival Experiments

## Summary

| Quantity | Value |
|---|---:|
| pass | true |
| scoped_claim_ready | true |
| strong_claim_ready | false |
| scenario_count | 192 |
| candidate_completes_all | true |
| candidate_not_pareto_dominated_all | false |
| candidate_vs_legacy_geomean_makespan | 1.66963 |
| candidate_vs_legacy_geomean_mean_flow | 4.96904 |
| worst_candidate_vs_legacy_makespan | 1.0034 |
| worst_candidate_vs_legacy_mean_flow | 0.989192 |
| max_candidate_backlog_jobs | 90 |

## Scenario Rows

| taskset | trace | jobs | candidate profiles | cand/legacy makespan | cand/legacy flow | SOTA nondominated | mean backlog |
|---|---|---:|---|---:|---:|---:|---:|
| q00_light_control | q00_light_control_poisson_load0.50_seed41 | 512 | `{"light_control_local": 13}` | 5.70534 | 196.059 | true | 6.17653 |
| q00_light_control | q00_light_control_poisson_load0.50_seed42 | 512 | `{"light_control_local": 13}` | 5.92423 | 198.969 | true | 6.41921 |
| q00_light_control | q00_light_control_poisson_load0.50_seed43 | 512 | `{"light_control_local": 13}` | 6.14787 | 198.301 | true | 6.68465 |
| q00_light_control | q00_light_control_poisson_load0.70_seed41 | 512 | `{"light_control_local": 13}` | 7.94353 | 204.752 | true | 8.72555 |
| q00_light_control | q00_light_control_poisson_load0.70_seed42 | 512 | `{"light_control_local": 13}` | 8.21249 | 206.006 | true | 9.09037 |
| q00_light_control | q00_light_control_poisson_load0.70_seed43 | 512 | `{"light_control_local": 13}` | 8.56279 | 201.424 | true | 9.66909 |
| q00_light_control | q00_light_control_poisson_load0.85_seed41 | 512 | `{"light_control_local": 13}` | 9.60741 | 200.534 | true | 11.0427 |
| q00_light_control | q00_light_control_poisson_load0.85_seed42 | 512 | `{"light_control_local": 13}` | 9.86732 | 198.159 | true | 11.6278 |
| q00_light_control | q00_light_control_poisson_load0.85_seed43 | 512 | `{"light_control_local": 13}` | 10.3494 | 180.074 | true | 13.3722 |
| q00_light_control | q00_light_control_poisson_load0.95_seed41 | 512 | `{"light_control_local": 13}` | 10.702 | 181.012 | true | 13.7897 |
| q00_light_control | q00_light_control_poisson_load0.95_seed42 | 512 | `{"light_control_local": 13}` | 10.8817 | 177.319 | true | 14.4956 |
| q00_light_control | q00_light_control_poisson_load0.95_seed43 | 512 | `{"light_control_local": 13}` | 11.3653 | 127.047 | true | 21.0433 |
| q00_light_control | q00_light_control_bursty_load0.50_seed41 | 512 | `{"light_control_local": 13}` | 6.46813 | 169.105 | true | 8.3623 |
| q00_light_control | q00_light_control_bursty_load0.50_seed42 | 512 | `{"light_control_local": 13}` | 6.87059 | 131.413 | true | 11.5107 |
| q00_light_control | q00_light_control_bursty_load0.50_seed43 | 512 | `{"light_control_local": 13}` | 5.83311 | 178.445 | true | 7.08441 |
| q00_light_control | q00_light_control_bursty_load0.70_seed41 | 512 | `{"light_control_local": 13}` | 8.98009 | 150.79 | true | 13.6831 |
| q00_light_control | q00_light_control_bursty_load0.70_seed42 | 512 | `{"light_control_local": 13}` | 9.53884 | 98.907 | true | 22.1664 |
| q00_light_control | q00_light_control_bursty_load0.70_seed43 | 512 | `{"light_control_local": 13}` | 8.1151 | 169.97 | true | 10.8704 |
| q00_light_control | q00_light_control_bursty_load0.85_seed41 | 512 | `{"light_control_local": 13}` | 10.5485 | 119.256 | true | 20.7574 |
| q00_light_control | q00_light_control_bursty_load0.85_seed42 | 512 | `{"light_control_local": 13}` | 11.1836 | 75.2978 | true | 34.7711 |
| q00_light_control | q00_light_control_bursty_load0.85_seed43 | 512 | `{"light_control_local": 13}` | 9.80782 | 151.165 | true | 15.0857 |
| q00_light_control | q00_light_control_bursty_load0.95_seed41 | 512 | `{"light_control_local": 13}` | 11.3387 | 98.0886 | true | 27.4062 |
| q00_light_control | q00_light_control_bursty_load0.95_seed42 | 512 | `{"light_control_local": 13}` | 11.8011 | 57.9377 | true | 48.112 |
| q00_light_control | q00_light_control_bursty_load0.95_seed43 | 512 | `{"light_control_local": 13}` | 10.9272 | 113.986 | true | 22.5172 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.50_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.08101 | 4.15004 | true | 1.09946 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.50_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.11297 | 3.64781 | true | 1.51597 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.50_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.07529 | 4.06549 | true | 1.14249 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.70_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.13773 | 4.36453 | true | 1.69877 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.70_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.16841 | 4.16002 | true | 2.53542 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.70_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.12946 | 5.12328 | true | 1.69656 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.31154 | 5.38305 | true | 2.34179 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.37475 | 4.17574 | true | 3.93674 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.85_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.32964 | 6.18984 | true | 2.34958 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.95_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.43594 | 5.33382 | true | 3.04374 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.95_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.48088 | 3.64806 | true | 5.46026 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_poisson_load0.95_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.44628 | 5.01605 | true | 3.65164 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.50_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.03962 | 2.21909 | true | 3.71864 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.50_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.21928 | 2.23493 | true | 6.66755 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.50_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.04097 | 2.37554 | true | 4.38334 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.70_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.18237 | 2.42873 | true | 5.9097 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.70_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.38302 | 2.11532 | true | 10.1276 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.70_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.33711 | 2.5889 | true | 7.6271 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.3933 | 2.68407 | true | 7.6669 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.42189 | 2.02494 | true | 11.8355 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.85_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.47283 | 2.29349 | true | 10.8178 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.95_seed41 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.46207 | 2.47002 | true | 9.48486 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.95_seed42 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.42231 | 2.00622 | true | 12.8435 |
| q01_gpu_bound_compute | q01_gpu_bound_compute_bursty_load0.95_seed43 | 48 | `{"gpu_heavy_jax_matmul": 1}` | 1.47971 | 2.13707 | true | 12.3071 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.50_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.02096 | 1.21812 | true | 2.63982 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.50_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.02237 | 1.21759 | true | 2.91703 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.50_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.01776 | 1.34901 | true | 2.14361 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.70_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.03922 | 1.06176 | true | 4.35048 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.70_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.0316 | 1.12145 | false | 4.80158 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.70_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.02597 | 1.16093 | true | 3.50247 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.85_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02542 | 1.00887 | false | 5.66084 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.85_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02157 | 1.00791 | false | 7.23491 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.85_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02967 | 1.0606 | true | 4.79408 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.95_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.0361 | 0.989192 | false | 6.82202 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.95_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.0034 | 1.0114 | false | 8.82158 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.95_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.03375 | 1.04308 | true | 6.0151 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.50_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.0162 | 1.04744 | true | 5.10331 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.50_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.01798 | 1.0282 | true | 6.4326 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.50_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.00352 | 1.03203 | true | 5.97202 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.70_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.04341 | 1.0419 | true | 7.82014 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.70_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02347 | 1.01251 | true | 10.6004 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.70_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02741 | 1.01775 | true | 9.8999 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.85_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02457 | 1.01525 | true | 10.0141 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.85_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02414 | 1.03769 | true | 13.9701 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.85_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02623 | 0.998809 | true | 13.9066 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.95_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.0231 | 1.00562 | true | 12.2089 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.95_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.05411 | 1.03544 | true | 15.8895 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.95_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.02711 | 1.00063 | true | 16.0913 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.50_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.40509 | 4.58815 | true | 8.80749 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.50_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.426 | 4.81952 | true | 9.0049 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.50_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.42297 | 4.15404 | true | 9.05268 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.70_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.91456 | 6.60944 | true | 12.6546 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.70_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.95154 | 6.76064 | true | 12.9308 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.70_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.91891 | 6.18188 | true | 13.0227 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.85_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.28094 | 7.42399 | true | 15.4255 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.85_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.30878 | 7.48407 | true | 15.7924 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.85_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.2615 | 6.95965 | true | 15.8654 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.95_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.50544 | 7.69451 | true | 17.4117 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.95_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.54574 | 7.59317 | true | 18.2035 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.95_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.4411 | 7.24072 | true | 17.635 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.50_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.62522 | 5.07307 | true | 11.3469 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.50_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.08012 | 5.29161 | true | 18.8435 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.50_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 1.6995 | 6.17405 | true | 11.6791 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.70_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.13424 | 6.33043 | true | 16.3555 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.70_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.47207 | 5.04903 | true | 28.1868 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.70_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.30613 | 7.03193 | true | 17.5085 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.85_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.44631 | 6.69859 | true | 19.9077 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.85_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.63366 | 4.69437 | true | 34.6912 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.85_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.69025 | 6.45403 | true | 24.3167 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.95_seed41 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.61068 | 6.64822 | true | 22.5428 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.95_seed42 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.68078 | 4.32558 | true | 39.7093 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.95_seed43 | 160 | `{"gpu_llm_distilgpt2": 10}` | 2.72773 | 5.54189 | true | 29.88 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.50_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.12777 | 3.88045 | true | 1.33216 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.50_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 3}` | 1.12852 | 3.52281 | true | 1.82336 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.50_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.1208 | 3.74717 | true | 1.42604 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.70_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.29972 | 3.97121 | true | 2.06533 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.70_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.43667 | 3.78269 | true | 3.31338 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.70_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.31677 | 4.63734 | true | 2.17188 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.85_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.45053 | 4.29792 | true | 2.7851 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.85_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.56777 | 3.16858 | true | 5.10785 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.85_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.57421 | 4.59465 | true | 3.22097 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.95_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.52298 | 4.11917 | true | 3.40218 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.95_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.59016 | 2.76539 | true | 6.33843 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.95_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.61842 | 3.54158 | true | 4.7215 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.50_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.12754 | 2.24383 | true | 3.84977 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.50_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.44394 | 1.94915 | true | 7.55716 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.50_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.32338 | 2.31978 | true | 5.35035 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.70_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.34532 | 2.1815 | true | 6.61972 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.70_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.49657 | 1.86172 | true | 9.30718 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.70_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.62163 | 2.29789 | true | 8.22346 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.85_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.60692 | 2.22973 | true | 8.67486 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.85_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.54289 | 1.9196 | true | 10.203 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.85_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.62666 | 2.08329 | true | 9.86235 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.95_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.61285 | 2.10105 | true | 9.69638 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.95_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.57575 | 1.9547 | true | 10.6758 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.95_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10}` | 1.62558 | 1.99975 | true | 10.6883 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.50_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.00963 | 1.75877 | true | 3.79099 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.50_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.01178 | 1.79516 | true | 3.86285 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.50_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.01018 | 1.90445 | true | 4.00294 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.70_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.03011 | 2.91997 | true | 5.36677 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.70_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.0463 | 3.30596 | true | 5.55791 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.70_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.09782 | 2.94316 | true | 6.01775 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.24394 | 6.35715 | true | 6.8514 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.26328 | 6.49532 | true | 7.21467 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.85_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.30064 | 5.43135 | true | 8.13991 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.95_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.37979 | 7.41272 | true | 8.39832 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.95_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.40847 | 7.15917 | true | 9.17879 |
| q10_cpu_host_bound | q10_cpu_host_bound_poisson_load0.95_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.421 | 5.83056 | true | 10.7388 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.50_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.04328 | 2.17175 | true | 5.69764 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.50_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.03319 | 2.73069 | true | 9.61268 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.50_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.01556 | 2.19583 | true | 6.06864 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.70_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.20458 | 3.85508 | true | 9.16553 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.70_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.37912 | 3.49826 | true | 19.8144 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.70_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.26049 | 5.08734 | true | 9.94243 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.414 | 4.82717 | true | 13.8153 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.50998 | 3.09814 | true | 31.3058 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.85_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.50614 | 4.8246 | true | 17.5135 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.95_seed41 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.49477 | 4.3624 | true | 19.1076 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.95_seed42 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.52934 | 2.7499 | true | 39.5332 |
| q10_cpu_host_bound | q10_cpu_host_bound_bursty_load0.95_seed43 | 256 | `{"cpu_heavy_local_bench": 8}` | 1.53067 | 3.4625 | true | 28.2479 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.01699 | 3.88798 | true | 0.941319 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.02406 | 3.52478 | true | 1.09194 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.04231 | 4.09699 | true | 1.01487 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.04168 | 3.99511 | true | 1.60415 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.02767 | 3.6016 | true | 1.95846 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.07337 | 3.21145 | true | 2.17969 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.18842 | 7.56974 | true | 2.46442 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.17822 | 7.01279 | true | 2.9834 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.12131 | 4.37865 | true | 3.21172 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.29605 | 8.47667 | true | 3.45931 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.30911 | 7.56797 | true | 4.22562 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.22705 | 5.84635 | true | 4.14461 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.06434 | 2.05504 | true | 3.86882 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.15675 | 2.14183 | true | 10.8159 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.03994 | 1.9833 | true | 4.30408 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.11773 | 2.19207 | true | 6.65409 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.2688 | 1.99282 | true | 19.6446 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.14242 | 3.55087 | true | 7.30037 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.23323 | 3.30197 | true | 8.91326 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.35356 | 2.17342 | true | 24.6563 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.36105 | 3.1689 | true | 14.2088 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed41 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.32902 | 3.4173 | true | 11.6257 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed42 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.38108 | 2.08446 | true | 28.9557 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed43 | 160 | `{"hybrid_rl_resac_ant": 2}` | 1.39739 | 2.49395 | true | 21.2477 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.02904 | 2.91398 | true | 1.75919 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.02346 | 2.81819 | true | 1.92251 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.02173 | 2.95723 | true | 1.61268 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.02995 | 3.90341 | true | 2.81422 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.03468 | 3.542 | true | 3.19816 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.02812 | 3.83296 | true | 2.5326 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.15389 | 6.34831 | true | 4.15567 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.18258 | 5.6887 | true | 4.72477 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.08264 | 5.81464 | true | 3.6613 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.32056 | 6.45608 | true | 5.95135 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.32743 | 5.63316 | true | 6.9352 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.19996 | 6.50935 | true | 4.99504 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.02685 | 1.93458 | true | 4.20523 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.15519 | 1.87913 | true | 12.0822 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.03191 | 2.06466 | true | 6.37771 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.0459 | 2.30655 | true | 6.61758 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.23432 | 2.03138 | true | 18.2733 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.23644 | 3.42606 | true | 11.1161 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.18168 | 3.54417 | true | 8.63823 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.35622 | 2.27829 | true | 22.5861 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.38764 | 2.93301 | true | 19.519 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.30175 | 3.60949 | true | 11.5502 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.38404 | 2.2098 | true | 26.7561 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 3, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 10, "hybrid_rl_resac_ant": 2}` | 1.38798 | 2.43687 | true | 26.3343 |

## Scope

rate-controlled replay on the same measured service cache; this is an online-arrival stress certificate, not a direct execution of external scheduler binaries.  Per-scenario SOTA Pareto-frontier language is reported by the separate strict/frontier gate.
