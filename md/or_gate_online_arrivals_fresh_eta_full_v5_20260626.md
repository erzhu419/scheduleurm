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
| candidate_vs_legacy_geomean_makespan | 1.61834 |
| candidate_vs_legacy_geomean_mean_flow | 4.16428 |
| worst_candidate_vs_legacy_makespan | 1.00276 |
| worst_candidate_vs_legacy_mean_flow | 1.00048 |
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
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.50_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.03075 | 1.61328 | true | 2.33701 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.50_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.0169 | 1.45457 | true | 2.93159 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.50_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 1}` | 1.02627 | 1.76655 | true | 1.90307 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.70_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 2}` | 1.04637 | 1.36394 | true | 4.13943 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.70_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 2}` | 1.03988 | 1.29233 | true | 5.3343 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.70_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 2}` | 1.0345 | 1.42576 | true | 3.34945 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.85_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.08234 | 1.24952 | true | 6.30973 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.85_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.12921 | 1.29127 | true | 8.29747 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.85_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.04568 | 1.38187 | true | 5.13012 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.95_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.15205 | 1.35341 | true | 7.96368 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.95_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 7}` | 1.19215 | 1.35393 | true | 10.5737 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_poisson_load0.95_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 3}` | 1.11572 | 1.523 | true | 6.99052 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.50_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.03939 | 1.1507 | true | 5.70738 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.50_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.04038 | 1.1629 | true | 7.06858 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.50_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.0346 | 1.11765 | true | 6.85884 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.70_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.06372 | 1.1704 | true | 8.71215 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.70_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.04406 | 1.18696 | true | 11.5571 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.70_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.10341 | 1.23161 | true | 11.3738 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.85_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.1358 | 1.33015 | true | 11.0357 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.85_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 6}` | 1.15962 | 1.27392 | true | 14.9587 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.85_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 5}` | 1.24139 | 1.3148 | false | 15.4057 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.95_seed41 | 64 | `{"gpu_cnn_torch_resnet50": 4}` | 1.22016 | 1.35293 | true | 13.5825 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.95_seed42 | 64 | `{"gpu_cnn_torch_resnet50": 6}` | 1.23724 | 1.28047 | true | 17.3005 |
| q01_gpu_bound_cnn_resnet50 | q01_gpu_bound_cnn_resnet50_bursty_load0.95_seed43 | 64 | `{"gpu_cnn_torch_resnet50": 6}` | 1.22367 | 1.28171 | false | 17.4036 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.50_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.36666 | 4.84267 | true | 7.58396 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.50_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.3921 | 5.1445 | true | 7.76461 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.50_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.38341 | 4.35708 | true | 7.78917 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.70_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.87357 | 7.56082 | true | 10.4402 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.70_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.90435 | 7.71079 | true | 10.7483 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.70_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.84575 | 6.97074 | true | 10.7956 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.85_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.25066 | 8.64622 | true | 12.7428 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.85_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.26008 | 8.66923 | true | 13.0545 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.85_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.13856 | 7.97363 | true | 12.8645 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.95_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.4496 | 8.87726 | true | 14.4248 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.95_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.49639 | 8.81888 | true | 15.1016 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_poisson_load0.95_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.33846 | 8.33893 | true | 14.4619 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.50_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.57437 | 5.46551 | true | 9.76368 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.50_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.9982 | 5.74436 | true | 16.2963 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.50_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 1.65055 | 6.90202 | true | 9.86224 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.70_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.07064 | 6.99734 | true | 14.0577 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.70_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.45159 | 5.29218 | true | 26.3315 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.70_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.24728 | 8.07457 | true | 14.649 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.85_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.35907 | 7.42593 | true | 17.0479 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.85_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.58559 | 4.85462 | true | 32.6003 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.85_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.641 | 7.46914 | true | 20.3898 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.95_seed41 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.5263 | 7.3231 | true | 19.5536 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.95_seed42 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.63965 | 4.43609 | true | 37.7067 |
| q01_gpu_bound_llm_inference | q01_gpu_bound_llm_inference_bursty_load0.95_seed43 | 160 | `{"gpu_llm_distilgpt2": 8}` | 2.72663 | 6.03153 | true | 27.2014 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.50_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.1258 | 3.93398 | true | 1.32683 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.50_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.12614 | 3.57083 | true | 1.81602 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.50_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.11738 | 3.78707 | true | 1.42882 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.70_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 2, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.2971 | 4.05125 | true | 2.0685 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.70_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.41583 | 3.78802 | true | 3.33627 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.70_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.30013 | 4.67696 | true | 2.18573 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.85_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.45077 | 4.30934 | true | 2.81911 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.85_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.59503 | 3.1949 | true | 5.1355 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.85_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.57428 | 4.61839 | true | 3.2639 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.95_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.5201 | 4.1066 | true | 3.46763 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.95_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.60907 | 2.78232 | true | 6.39142 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_poisson_load0.95_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.62929 | 3.56637 | true | 4.7812 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.50_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.12754 | 2.25765 | true | 3.89005 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.50_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.42211 | 1.94813 | true | 7.60246 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.50_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.32429 | 2.32073 | true | 5.39224 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.70_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.34678 | 2.20583 | true | 6.67215 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.70_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.46859 | 1.86487 | true | 9.36309 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.70_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.60968 | 2.30548 | true | 8.27432 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.85_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.61934 | 2.25387 | true | 8.73983 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.85_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.55517 | 1.92631 | true | 10.2638 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.85_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.59963 | 2.08997 | true | 9.92254 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.95_seed41 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.61633 | 2.12365 | true | 9.75569 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.95_seed42 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.58841 | 1.96248 | true | 10.7373 |
| q01_gpu_model_portfolio | q01_gpu_model_portfolio_bursty_load0.95_seed43 | 176 | `{"gpu_cnn_torch_resnet50": 5, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8}` | 1.62229 | 2.00987 | true | 10.749 |
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
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01248 | 1.89699 | true | 1.34251 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00607 | 1.66254 | true | 1.58901 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.50_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00894 | 1.59241 | true | 1.63745 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01176 | 1.4382 | true | 2.60221 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01171 | 1.34229 | true | 2.95748 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.70_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.0045 | 1.28589 | true | 3.24306 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01315 | 1.15201 | true | 4.2226 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01423 | 1.18012 | true | 4.54106 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.85_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01229 | 1.14523 | true | 4.7282 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.0118 | 1.06997 | true | 5.68431 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01012 | 1.04694 | true | 6.51838 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_poisson_load0.95_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01585 | 1.10768 | true | 5.83534 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00628 | 1.03499 | true | 4.66277 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00869 | 1.00504 | true | 11.8869 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.50_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00541 | 1.03608 | true | 5.10292 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00276 | 1.01593 | true | 7.69811 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00763 | 1.00048 | true | 20.9378 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.70_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01073 | 1.01675 | true | 8.52296 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00524 | 1.01672 | true | 10.2689 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00843 | 1.00993 | true | 26.0445 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.85_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.02083 | 1.00204 | true | 15.5944 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed41 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00912 | 1.00317 | true | 13.0078 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed42 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.00733 | 1.01171 | true | 30.5566 |
| q11_cpu_gpu_coupled | q11_cpu_gpu_coupled_bursty_load0.95_seed43 | 160 | `{"hybrid_rl_resac_ant": 5}` | 1.01157 | 1.00079 | true | 22.556 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01762 | 1.87714 | true | 2.19903 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01566 | 1.70832 | true | 2.47799 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.50_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01069 | 1.90457 | true | 2.02947 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.02143 | 2.273 | true | 3.7731 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01836 | 1.92628 | true | 4.18551 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.70_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00803 | 2.30911 | true | 3.42961 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.02087 | 2.51362 | true | 5.9785 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01107 | 2.18308 | true | 6.32397 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.85_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01678 | 2.81395 | true | 5.0934 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 1, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.0226 | 2.45518 | true | 8.21183 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01053 | 2.12583 | true | 9.06612 |
| hybrid_research_portfolio | hybrid_research_portfolio_poisson_load0.95_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01869 | 2.79981 | true | 6.74466 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00814 | 1.28811 | true | 4.95967 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00924 | 1.18242 | true | 13.1418 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.50_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00659 | 1.30979 | true | 7.28903 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00483 | 1.4109 | true | 7.64206 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00862 | 1.26216 | true | 19.5611 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.70_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01745 | 1.76673 | true | 12.6887 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01743 | 1.63217 | true | 10.1158 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.02057 | 1.31897 | true | 23.9598 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.85_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.00428 | 1.67081 | true | 20.9614 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed41 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01561 | 1.68911 | true | 12.9632 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed42 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.03208 | 1.30862 | true | 28.5245 |
| hybrid_research_portfolio | hybrid_research_portfolio_bursty_load0.95_seed43 | 496 | `{"cpu_heavy_local_bench": 8, "gpu_cnn_torch_resnet50": 4, "gpu_heavy_jax_matmul": 1, "gpu_llm_distilgpt2": 8, "hybrid_rl_resac_ant": 5}` | 1.01056 | 1.50898 | true | 27.9626 |

## Scope

rate-controlled replay on the same measured service cache; this is an online-arrival stress certificate, not a direct execution of external scheduler binaries.  Per-scenario SOTA Pareto-frontier language is reported by the separate strict/frontier gate.
