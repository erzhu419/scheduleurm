# 2026-06-14 registered-adapter and Decima execution closure update

这段是最新状态，覆盖下面旧计划里关于 registered SOTA 只停留在
runtime inventory、Decima 只停留在 smoke / single heuristic benchmark 的描述。

新增闭合：

- `md/registered_sota_adapter_closure_gate_20260614.md`：
  11/11 registered non-adjacent SOTA systems 已有 policy/service-unit adapter
  row，进入同一个 finite measured-cache Scheduleurm+SOTA action union；
  `registered_policy_adapter_universe_ready=true`。同时
  `direct_external_binary_adapter_ready_count=5/11`，
  `registered_direct_external_binary_superiority_ready=false`。因此可以写
  registered finite adapter universe closed，不能写 registered external-binary
  full-stack universe closed。
- `md/decima_same_domain_benchmark_gate_20260614.md`：Decima Spark-DAG
  same-domain paired benchmark 在 fixed seeds 下全部执行完成；
  `spark_dag_same_domain_benchmark_ready=true`。结果是
  wall-time noninferior，但 mean-completion mixed，所以
  `spark_dag_same_domain_performance_superiority_ready=false`。这只能作为
  Decima adjacent-domain executable benchmark evidence，不能写成 GPU
  co-location full-stack superiority。
- `md/non_future_claim_closure_gate_20260614.md` 已改为读取上述两个更强
  gate：排除 arbitrary future workload 后，5/5 finite non-future scoped rows
  仍 closed，但 registered row 的依据现在是 policy/service-unit adapter
  closure，Decima row 的依据现在是 smoke/metric bridge + same-domain
  execution closure。
- `md/gate_status_dashboard.md` 已新增
  `registered_sota_adapter_closure_gate` 和
  `decima_same_domain_benchmark_gate` 两行，便于 reviewer 逐项检查 scoped
  claim 与 strong claim。

仍不能写：

- arbitrary / unregistered SOTA superiority；
- 对没有 direct executable adapter row 的 registered systems 写 external
  binary full-stack superiority；
- 把 Decima Spark-DAG same-domain execution 写成 GPU co-location SOTA row；
- external SOTA original multi-node control-plane superiority；
- arbitrary future workload positive-service theorem readiness。

# 2026-06-14 final non-future closure update

这段是最新状态，覆盖下面旧计划里关于 organic production completion
仍为 false / pending 的描述。

已闭合：

- `md/organic_history_completion_gate_20260614.md`：strict scheduler-history
  organic production population 已闭合。4,058 launched，4,047 completed，
  completion fraction 0.9973，35 workload domains，13 nodes，0 strict
  unadmitted launched rows。admission mode 是
  `strict_exact_or_signature_service_certificate`，不是 token-overlap fuzzy
  admission。
- `md/production_wide_organic_trace_gate_20260614.md`：
  `production_wide_history_completion_closed=true`，
  `large_scale_organic_launched_completion_ready=true`。同时
  `production_wide_live_trace_closed=false` 继续单独保留，不能写成每个 live
  slot 都有 oracle trace。
- `md/production_launch_completion_gate_20260612.md`：重新生成后为
  `HISTORY_COMPLETION_AND_ACTIVE_PROGRESS_SHADOW_TRACE_PASS`，有 12 个 active
  progress observations，并由 strict history certificate 闭合 launched
  completion。
- `md/multinode_history_completion_gate_20260614.md`：Scheduleurm-native
  multi-node history 闭合，13 nodes、3 GPU nodes、35 domains、0 strict
  unadmitted。这是 Scheduleurm 自己的原生多节点 launched/completed history，
  不是外部 Gavel/Pollux/Sia/IADeep/Salus 的 original multi-node control-plane
  superiority。
- `md/non_future_claim_closure_gate_20260614.md`：排除 arbitrary future
  workload 后，5/5 finite non-future scoped rows closed：named external SOTA
  full-stack、registered SOTA policy-semantics universe、production organic
  completion、Scheduleurm-native multi-node history、Decima Spark-DAG bridge。

仍不能写：

- arbitrary future workload positive-service theorem readiness；
- arbitrary / unregistered SOTA superiority；
- 对没有 full-stack adapter row 的 registered systems 写 direct external
  binary superiority；
- 外部 SOTA 系统的 original multi-node control-plane superiority；
- 把 Decima Spark-DAG bridge 写成 GPU co-location full-stack superiority。

## 2026-06-14 broad-claim gate execution update

这轮把“任意 SOTA、任意未来 workload、多节点原始部署、Decima
Spark-DAG、production-wide organic trace”五个方向全部补成了可执行 gate，
并新增总控 artifact：

- `md/sota_universe_registry_gate_20260614.md`：named five full-stack
  same-host same-workload 证据继续为 true；registered SOTA universe 和
  arbitrary SOTA superiority 仍为 false。
- `md/future_workload_protocol_gate_20260614.md`：未来任务通过 strict
  admission/probe 协议进入 theorem stream；未知 LLM/CPU synthetic rows 被
  强制路由到 `PROBE_REQUIRED`；arbitrary future workload theorem readiness 为
  false。
- `md/multinode_original_deployment_gate_20260614.md`：same-host rows 和
  original multi-node deployment rows 被严格分开；multi-node original
  superiority 仍为 false。
- `md/decima_spark_dag_gate_20260614.md`：Decima repo、Spark-DAG import、
  baseline entrypoint smoke 通过；Decima 仍是 Spark-DAG simulator 证据，
  不是 GPU co-location full-stack superiority。
- `md/production_wide_organic_trace_gate_20260614.md`：production organic
  recorder/admission contract 已闭合；live-trace closure、active progress、
  launched-completion 分开报告；large-scale organic launched completion 仍为
  false。
- `md/universal_claim_closure_gate_20260614.md`：5/5 scoped boundary gates
  ready，1/5 universal strong claims ready；strong row 来自 strict
  scheduler-history production completion。这个 gate 现在读取 registered
  SOTA policy/service-unit adapter closure 和 Decima same-domain execution
  closure，而不是只读 registry/smoke。它的意义是 reviewer 可复核的边界闭合，
  不是把五个 universal strong claim 都说成已经成立。

同步更新了 `md/gate_status_dashboard.md`、
`md/or_claim_scope_matrix_2026_06_11.md`、
`md/or_submission_closure_status_2026_06_11.md`、
`md/math_code_alignment_2026_06_11.md` 和 `paper/main.tex`。

### 2026-06-14 follow-on strengthening

继续攻击后又补了几项正进展：

- 扩展 SOTA registry 不再只是空 blocker：已 clone 并登记
  Tiresias、Shockwave、AlloX、Optimus；AlloX 的 `kubernetes-allox` 和
  `allox_sim` 子 repo 也已 clone。新增
  `md/registered_sota_runtime_gate_20260614.md`，记录 6 个 extension row：
  4 个 public runtime repo + 2 个 paper-only system。当前 `entrypoint_smoke`
  仍为 0/4：Tiresias/Optimus 缺 Python2 `numpy`/`jinja2`，Shockwave 缺
  protobuf stubs/Gurobi，AlloX 有 Java 但 simulator bin 不完整且本机无
  `javac` 重编译。
- Decima 从 import smoke 推进到真实 Spark-DAG heuristic benchmark：
  `spark_dag_heuristic_benchmark_ready=true`，小型 dynamic-partition run
  完成 3 个 DAG、29 个 step。learned policy 仍因 TensorFlow-1 runtime
  阻塞，且 Decima 仍不能混成 GPU co-location full-stack SOTA row。
- future workload protocol 从 3 条 synthetic stress row 扩到 10 条，覆盖
  measured q00/q01/q10/q11/CNN/LLM/CPU/control 和 unknown LLM/CPU/CUDA/Spark
  DAG/NUMA pipeline；unknown rows 均保持 `PROBE_REQUIRED`。
- multi-node read-only inventory 已跑：当前主机可达 `jtl110gpu` 和
  `jtl110gpu2`，各 24 CPU / 2 GPU；`node001`--`node007` 在当前主机 DNS
  不可解析。这个结果支持 infrastructure inventory，不支持原始 multi-node
  deployment superiority。
- production-wide organic trace 当前快照有 19 个 running production、0 个
  queued production，所以 strong launched-completion gate 正确保持 false。

## 2026-06-13 closure update

### Direct SOTA full-stack update

这份历史计划里关于 direct same-workload full-stack 仍为零或 scoped 4/5 的描述已经过时。最新 direct SOTA full-stack gate 是 **NAMED 5/5 PASS**：Gavel physical scheduler/worker/RPC/GavelIterator、Pollux/AdaptDL、Sia、IADeep extender/device-plugin、Salus server/zrpc rows 均已闭合，并且每个 named row 都有 paired native superiority。Salus 通过 local 镜像拉取、同步到 jtl110gpu、远端 Docker load 后完成了 native TensorFlow-Salus 与 Salus server-plus-client 同工作负载对照：native \(199.055\)s，Salus \(201.744\)s。因此可以 claim 的是 named five systems on scoped same-host same-workload probes；不能外推到任意 SOTA、未来 workload、多节点原始部署、Decima Spark-DAG 场景或 production-wide organic traces。

这轮继续推进了三个原本最容易被 OR reviewer 抓住的边界：

- `corner_case_lower_service_gate` 已把 controlled node007/node001 corner-case rows 纳入 theorem-facing 的独立 service cache：5/5 stable canonical rows admitted，row-level lower-service capacity slack 全部为正；默认 replay cache 没被污染，只有显式引用 `md/experiment_artifacts/corner_case_service_cache_20260613.json` 时才使用这些 row。
- `gavel_resident_delay_jct_holdout_gate` 已闭合 scoped Gavel-style finish-time/JCT holdout：在 3 条 co-location row 上，即使 defer baseline 使用 resident-alone 的最快观测速率，immediate co-location 仍然降低 mean JCT 23.9%--46.7%，降低 makespan 11.0%--24.5%。这提升了 Gavel-style 证据，但仍不是 direct full-stack SOTA superiority。
- `production_launch_completion_gate` 重新跑过；当前 rolling snapshot 是 `WAIT_RESOURCE_OR_QUEUE`：1 个 production running、0 个 queued、无可解析 progress subset，因此 large-scale organic launched completion 仍为 false。这不是代码缺口，而是安全 gate 正确拒绝无队列/无进度证据时的强 claim。

因此最新边界是：corner-case finite measured slice、Gavel-style resident-delay/JCT holdout、以及 named five direct full-stack same-workload gate 都已经补成 positive scoped evidence；organic production-wide launched completion 仍不能诚实 claim，named five 结果也不能泛化到任意系统/未来 workload。

## 2026-06-12 execution update

这份路线图里的两个关键 blocker 已经推进：

- `selected_profile_holdout_lcb_gate` 现在对 scoped stochastic lower-service capacity claim 为 true：gate 使用 profile-level aggregate service windows，11/11 selected targets 都达到 20-window threshold，LCB lower-service capacity slack 为 \(\delta_{\mathrm{LCB}}=0.0247121365\)。
- Lean 新增 `Scheduleurm.DiagonalScaling`，证明 diagonal-scaled LCB support loss，避免 CPU/CNN/LLM/JAX/RL 不同量纲被一个 absolute epsilon 混在一起。
- `gavel_service_unit_calibration_gate` 已拆成 scalar equivalence 和 profile-aware same-workload calibration。scalar service-unit equivalence 仍 false；profile-aware q01/q11 native-Gavel-simulator baseline 已 ready，bounded paired holdout p95 relative error 为 0。

仍不能 claim：

- 0.8 mean-service load 由 stochastic LCB 证明。absolute mean-service eta 和 diagonal-normalized mean-service eta 仍为负。
- Gavel simulator step 与 Scheduleurm measured service unit 的 scalar equivalence。
- 直接全栈击败 Gavel/Pollux/Sia/IADeep/Salus。

我的建议是：**不要把三个原始强 claim 硬推成 true，而是把每个 claim 的“可审稿版本”顶到 9/10，再把“强版本”的 10/10 门槛定义清楚。** 你现在的 repo 已经很接近这个形态了。下一步的核心不是再加很多散 artifact，而是把每个 gate 做成 reviewer 一眼能判断的 **claim ladder**：

```text
scope claim true
strong claim false / pending
blocker explicit
next threshold explicit
raw evidence reproducible
```

下面按模块给你一个可以执行的路线图。

---

# 总体目标

我建议把论文最终 claim 分成三层：

| 层级                         | 目标                                                   | 审稿含义                     |
| -------------------------- | ---------------------------------------------------- | ------------------------ |
| **9/10**                   | scoped claim 可审稿、可复现、不可误读                            | OR reviewer 会觉得边界清楚、证据扎实 |
| **10/10 in scope**         | scoped claim 不只是 gate pass，还有直接实验/统计/复现闭环            | 即使挑剔 reviewer 也很难攻击      |
| **Strong universal claim** | arbitrary all-state / all-SOTA / production-wide 全闭合 | 不建议现在追，成本高且容易偏离 OR 主线    |

现在你最值得做的是：**把 scoped claim 全部顶到 9/10–10/10 in scope，而不是追 universal claim。**

---

# 0. 先统一所有 gate 的语义

这是投稿前必须做的小改动。现在几个 gate 都有 `pass=true`，但强 claim flag 仍然是 false。逻辑没错，但 reviewer 可能误读。

比如 Gavel calibration gate 现在是 `gate_pass=true`、`profile_aware_model_calibration_ready=true`，同时 `gavel_service_unit_equivalence_ready=false`；意思是 scoped profile-aware native Gavel simulator baseline 已闭合，但 scalar service-unit equivalence 没闭合。Controlled production gate 现在是 `pass=true`、`controlled_32_task_completion_ready=true`，但 `large_scale_organic_launched_completion_ready=false`。

## 9/10 改法

所有 gate 统一加四个字段：

```json
{
  "gate_pass": true,
  "scoped_claim_ready": true,
  "strong_claim_ready": false,
  "status": "SCOPED_PASS_STRONG_CLAIM_FALSE"
}
```

然后每个 gate 的 markdown 第一屏都写：

```text
This gate passes the scoped certificate. It does not make the adjacent strong claim.
```

建议三个具体 status：

```text
Gavel:
ADAPTER_COMPATIBILITY_PASS_SERVICE_EQUIV_FALSE

Declared finite cover:
DECLARED_FINITE_POSITIVE_COVER_PASS_ALL_STATE_POSITIVE_FALSE

Controlled completion:
CONTROLLED_32_COMPLETION_PASS_ORGANIC_FALSE
```

## 10/10 改法

做一个总表 `md/gate_status_dashboard.md`，每行都包含：

```text
Gate
Scoped claim
Scoped claim ready
Strong claim
Strong claim ready
Blocker
Next threshold
Artifact path
Raw evidence path
```

这个 dashboard 可以成为 rebuttal 时的主证据。

---

# 1. Direct full-stack SOTA superiority

## 当前状态

这部分已经被 2026-06-13 的 named-system gate 进一步推进。现在你有：

* Gavel trace schema compatibility ready；
* throughput seed ready；
* Gavel native bounded microbaseline ready；
* 但 Gavel service-unit equivalence 仍 false；
* Gavel physical scheduler/worker/RPC/GavelIterator same-workload row ready；
* Pollux/AdaptDL、Sia、IADeep、Salus scoped same-host same-workload rows ready；
* `named_same_host_runtime_probe_ready=true`，但 `direct_fullstack_named_sota_superiority_ready=false`；仅限 named five measured probes。

artifact 也清楚解释了 blocker：Gavel native trace rows 使用 Gavel template job types 和 simulator throughput tables，而 Scheduleurm theorem rows 使用 Scheduleurm measured aggregate lower-service rates；schema、job counts、arrival times、resource counts、total_units 已经对齐，但没有证明 Gavel simulator step 和 Scheduleurm measured service unit 等价。

代码注释也把三件事拆开了：能导出 Gavel native trace、Gavel simulator 能跑、但 service units 没有等价证明。这个 blocker 仍然阻止 scalar simulator-unit equivalence claim，但不再阻止 named same-host physical full-stack row claim，因为后者由 `sota_fullstack_superiority_gate_20260613` 单独闭合。

## 当前分数

我会给 **10/10 within named same-workload scope**。它已经是 direct named SOTA full-stack evidence；剩余边界是不要外推成 universal/all-future claim。

## 推到 9/10：把 Gavel adapter compatibility 做成正式 scoped claim

目标 claim：

> We certify bounded same-trace Gavel adapter compatibility and native simulator metric extraction for q01/q11. We do not claim service-unit equivalence or full-stack superiority.

具体改动：

### 1.1 修正 arrival-time equivalence 表述

现在 Gavel certificate 里：

```text
trace_schema_compatibility_ready = true
exact_arrival_times_ready = false
```

同时表里还有 max arrival jitter：q01 是 0.047s，q11 是 0.159s。

这会让 reviewer 问：“exact arrival false 为什么 trace schema true？”

建议二选一：

**方案 A：让 converter 保留完全相同 arrival time。**

把 native trace converter 的输出精度提高，不要四舍五入；让：

```text
exact_arrival_times_ready = true
max_arrival_jitter_s = 0
```

**方案 B：显式改成 tolerance-based equivalence。**

字段改成：

```json
"arrival_times_within_tolerance_ready": true,
"arrival_time_tolerance_s": 0.2,
"exact_arrival_times_ready": false
```

论文里写：

> Arrival timestamps are equivalent up to the native Gavel trace precision tolerance.

9/10 更推荐方案 A，因为最干净。

### 1.2 把 Gavel microbaseline 单独成表

论文现在已经说明 Gavel q01/q11 native simulator microbaseline，但建议单独放一个小表：

| Taskset |     Jobs | Native rows | Completed | Avg JCT | Makespan | Service-unit equivalence |
| ------- | -------: | ----------: | --------: | ------: | -------: | ------------------------ |
| q01     |  4 or 48 |         ... |       yes |     ... |      ... | false                    |
| q11     | 8 or 160 |         ... |       yes |     ... |      ... | false                    |

注意最后一列必须写 false。这样 reviewer 不会误以为你在 claim direct superiority。

### 1.3 给 Gavel certificate 加 status

```json
"status": "GAVEL_NATIVE_ADAPTER_COMPATIBILITY_PASS_SERVICE_UNIT_EQUIV_FALSE"
```

### 1.4 把 `pass` 改名或补充

保留 `pass=true` 可以，但加：

```json
"pass_meaning": "adapter compatibility and native microbaseline readiness, not service-unit equivalence"
```

## 推到 10/10 in scope：闭合 Gavel service-unit equivalence，但只对 q01/q11

这不是必须，但如果你想把 SOTA 部分打到很强，可以做。

### 10/10 实验设计

做一个 **Gavel service-unit calibration experiment**：

1. 对 q01/q11 选定 trace window。
2. 同一个 job list，分别生成：

   * Scheduleurm measured service units；
   * Gavel native trace；
   * Gavel throughput table。
3. 用 calibration jobs 估计比例：
   [
   c = \frac{\text{Scheduleurm measured unit}}{\text{Gavel simulator unit}}
   ]
4. 在 holdout trace 上验证：
   [
   |\hat T_{\mathrm{gavel}\to\mathrm{scheduleurm}} - T_{\mathrm{scheduleurm}}|/T_{\mathrm{scheduleurm}} \le \epsilon
   ]
5. 给出 confidence interval。

新增 artifact：

```text
md/gavel_service_unit_calibration_q01_q11.md
md/experiment_artifacts/gavel_service_unit_calibration_q01_q11.json
```

通过条件建议：

```json
{
  "calibration_tasksets": ["q01", "q11"],
  "holdout_relative_error_p95": "<= 0.05",
  "service_unit_equivalence_ready": true,
  "direct_full_stack_same_workload_ready": true,
  "scope": "Gavel native simulator on q01/q11 bounded trace windows only"
}
```

### 不建议追的 10/10

不要现在追：

```text
direct superiority over Gavel/Pollux/Sia/IADeep/Salus
```

因为 Pollux/IADeep/Salus 需要 Kubernetes、Docker、Go、NVIDIA runtime 等全栈环境。你现在的 direct readiness gate 已经把这些 blocker 列明，这是对的。

---

# 2. Fabric cover / all-state cover

## 当前状态

这一块是你这次更新最成功的地方。

现在 declared finite-domain gate 给出：

* `declared_finite_positive_cover_ready=true`
* `positive_service_all_state_cover_ready=false`
* 193 个 declared universe buckets；
* 113 个 workload domains；
* 187 个 positive lower-service buckets；
* 6 个 capacity boundary buckets；
* 0 个 probe-required；
* 0 个 uncovered；
* coverage fraction 1.0。

declared universe 也定义清楚了：

```text
service_cache workload_key x exact measured profile x node_bucket x resource_kind x command_fingerprint
```



代码里也很明确：这个 gate 比 arbitrary all-state fabric cover 更窄；它定义的 positive population 是 declared finite service-cache domain；unknown future states 在测量前不进入 positive population。

## 当前分数

这块我会给 **9/10**。已经是可以写进 OR 稿的正式 certificate。

## 推到 9.5/10：避免 coverage fraction 误读

当前 `coverage_fraction=1.0` 是 classification coverage：187 positive + 6 boundary = 193 all classified。这个没错，但 reviewer 可能误以为 193/193 都是 positive service。

### 改法

新增字段：

```json
"classification_fraction": 1.0,
"positive_service_fraction": 187 / 193,
"boundary_fraction": 6 / 193,
"uncovered_fraction": 0.0
```

把 `coverage_fraction` 改名为：

```json
"declared_domain_classification_fraction": 1.0
```

论文里写：

> All declared buckets are classified; 187/193 have positive lower service and 6/193 are measured capacity boundaries.

你现在 limitation 里已经这么写了：193 个 exact service-cache buckets，其中 187 positive lower service，6 capacity boundaries，0 uncovered。 所以只需要让 artifact 字段更精确。

## 推到 10/10 in scope：把 declared finite domain 变成 fully auditable domain

### 2.1 给每个 row 加时间和样本信息

当前代码里 `first_seen`、`last_seen` 是：

```text
not_available_in_service_cache_v1
```



这会被 reviewer 问：“这个 service cache 是什么时候测的？跨天稳定吗？”

建议升级 service cache schema v2：

```json
{
  "first_seen": "...",
  "last_seen": "...",
  "measurement_window_s": ...,
  "sample_count": ...,
  "source_run_ids": [...],
  "node_bucket": "...",
  "resource_kind": "...",
  "command_fingerprint": "...",
  "capacity_boundary_reason": "...",
  "lower_service_method": "min_observed|LCB|conservative_cache"
}
```

### 2.2 theorem-facing admission 必须 strict

现在 `service_registry.infer_workload_key` 前半部分是规则匹配，后半部分会用 token overlap：如果 best_score >= 2，就返回 best_key。 这对工程好用，但对 theorem-facing gate 有点危险。

建议加：

```text
SCHEDULEURM_THEOREM_ADMISSION_MODE=strict
```

strict 模式只允许：

1. explicit workload_key；
2. exact signature prefix；
3. command fingerprint match；
4. declared admission manifest match。

token overlap 只允许输出：

```json
"route": "PROBE_REQUIRED",
"candidate_workload_key": "...",
"reason": "fuzzy_token_match_not_theorem_admissible"
```

这会把 declared finite cover 从 9/10 推到 10/10，因为 reviewer 不会担心 unmeasured future jobs 被 fuzzy admit。

### 2.3 对 capacity boundaries 给出 boundary proof

现在 6 个 boundary 已经计入 classification。建议给每个 boundary row 补：

```json
"boundary_reason": "oom|timeout|nonpositive_rate|runtime_failure",
"boundary_profile": k,
"first_invalid_profile": k,
"monotone_exclusion": "profiles >= k excluded"
```

这和你论文里 “capacity boundary” 的理论接口对齐。

## 不建议追的 claim

不要把这个改成：

```text
positive_service_all_state_cover_ready = true
```

除非你定义了有限 all-state universe 并测完。当前正确结论是：

> declared finite-domain positive cover true；all-state safety true；arbitrary all-state positive cover false。

这正是 OR reviewer 会认可的边界。

---

# 3. Production completion / live evidence

## 当前状态

这次新增 controlled completion gate 是实质改善。

现在 artifact 给出：

* bounded controlled completion ready = true；
* controlled launched task count = 32；
* controlled completed task count = 32；
* theorem slot count = 32；
* candidate count total = 56；
* (\alpha_0=0,\alpha_1=0)；
* controlled 32-task completion true；
* organic large-scale launched completion false；
* organic launch 当前仍不 promotion，因为 production-wide natural thresholds 还没闭合。

代码里也安全：这个 gate 不绕过 scheduler safety，只识别已有 bounded evidence；只有显式 allow launch 且 GPU utilization 低于 threshold 才会 launch。

32-task threshold 也写得很具体：launched ≥ threshold、completion fraction ≥ 0.95、theorem slots ≥ launched、candidate rows ≥ launched（非空 finite candidate family per launched slot）、(\alpha_0=\alpha_1=0)、unadmitted launched count = 0。

## 当前分数

我给 **8/10**。比之前强很多，但还不是 production-wide completion。

## 推到 9/10：闭合 controlled 32-task completion

这是最值得做的下一步，因为它完全在你可控范围内，不依赖自然 production queue。

目标 claim：

> Controlled launched theorem-dispatch completion is closed at 32-task scale under safety gating.

### 具体执行条件

等 GPU util 低于阈值，比如 25% 或你设定的 safe threshold，然后运行：

```bash
python -m algorithm.experiments.controlled_production_completion_gate build \
  --allow-launch \
  --controlled-threshold 32 \
  --completion-fraction-threshold 0.95 \
  --min-candidate-multiplier 2.0 \
  --max-gpu-util-for-launch 25 \
  --launch-task-count 32
```

通过标准：

```json
{
  "controlled_launched_task_count": 32,
  "controlled_completed_task_count": ">= 31",
  "completion_fraction": ">= 0.95",
  "theorem_slot_count": ">= 32",
  "candidate_count_total": ">= 64",
  "alpha0": 0,
  "alpha1": 0,
  "unadmitted_launched_count": 0,
  "resource_eviction_count": 0,
  "controlled_32_task_completion_ready": true
}
```

### 论文表述

不要叫 production-wide。写：

> Controlled launched theorem-dispatch completion closed at 32-task scale.

这可以把 live evidence 从 8/10 推到 9/10。

## 推到 10/10 in scope：organic production canary completion

目标 claim：

> Organic production theorem-traced launch/progress/completion evidence is closed for admitted production jobs.

这个需要自然 queued production 和安全资源，不要强行跑。

### 10/10 canary 方案

在 watcher / dispatch 层开启 trace-only 或 theorem canary mode：

```bash
export SCHEDULEURM_ORACLE_TRACE_PATH=~/.claude/scheduler/theorem_oracle_trace.jsonl
export SCHEDULEURM_THEOREM_UNCERTIFIED_MODE=block
export SCHEDULEURM_THEOREM_ADMISSION_MODE=strict
export SCHEDULEURM_ALGORITHM=theorem_maxweight_v1
```

对每个 production task：

```text
ADMIT_THEOREM_TRACE -> dispatch -> launch -> progress -> completion
PROBE_REQUIRED -> no theorem claim
```

### 通过标准

建议：

```json
{
  "organic_production_launched_count": ">= 64",
  "organic_production_completed_count": ">= 50",
  "completion_fraction": ">= 0.80",
  "workload_domain_count": ">= 3",
  "node_count": ">= 2",
  "theorem_trace_closed": true,
  "completion_bridge_closed": true,
  "probe_required_launched_count": 0,
  "alpha0": 0,
  "alpha1": 0,
  "large_scale_organic_launched_completion_ready": true
}
```

如果生产任务很长，completion 很慢，可以先做 10/10 progress version：

```json
"organic_production_progress_ready": true,
"organic_production_completion_ready": false
```

但不要把 progress 写成 completion。

## 命名建议

当前 gate 叫 `Controlled Production Completion Gate`，但证据是 controlled launched completion，不是 organic production-wide completion。建议标题改成：

```text
Controlled Launched Completion Gate
```

或：

```text
Controlled Launched Completion Gate for Production-Safe Evidence
```

这样 reviewer 不会挑 “production completion” 这个词。

---

# 4. Theorem-policy alignment

## 当前状态

你已经有 `TheoremMaxWeightPlacementPolicy`，它会绑定 measured service row，构造 `queue_vector`、`lower_service`、`penalty_units`、`robust_maxweight_score`，并把 theorem-ready candidate 的 score semantics 标成 `robust_maxweight_lower_service`。

它的 score 是：

```python
robust_score = q_weight * lower_service - penalty
```

并返回 `(-robust_score, penalty, -lower, profile) + legacy_tuple` 作为排序键。

这比以前的 scalar sweetspot hook 强很多。但它仍然是 **one-task placement scorer**，不是 full global batch MaxWeight optimizer。论文已经把这点写清楚：live placement hook 不是 deployed global MaxWeight implementation。

## 当前分数

**8/10**。作为 hook/certificate 很好；作为 theorem policy implementation 还不是 10/10。

## 推到 9/10：把 live hook 变成 “theorem-grade single-task candidate oracle”

目标不是 global optimizer，而是明确 claim：

> For admitted GPU candidates, the live hook emits theorem-grade lower-service MaxWeight candidate rows and exact candidate-set audit fields.

需要补：

1. 每个 dispatch trace 都包含：

   ```json
   queue_vector
   lower_service
   penalty_units
   robust_maxweight_score
   score_semantics
   selected_action
   candidate_count
   oracle_gap
   ```
2. `score_semantics=robust_maxweight_lower_service` 的 candidate subset 单独 audit。
3. CPU fallback / legacy semantics 明确 excluded，不混入 theorem subset。

你论文现在已经这么写了：CPU fallback slots 使用 legacy scheduler sort-key semantics，并从 theorem subset 排除。 保持这个边界。

## 推到 10/10：实现 global theorem dispatcher

这不是必须，但如果要真正顶到理论-系统完全闭合，需要做一个新入口：

```bash
python skill/scheduler.py dispatch --algorithm theorem_global_v1
```

### 设计

1. 读当前 queue vector。
2. 枚举 bounded global configurations：

   ```text
   action = set of placements for up to B queued tasks
   ```
3. 每个 action 绑定 lower_service vector。
4. 计算：
   [
   Q^\top \underline{\mu}(a)-K(a)-G(a)
   ]
5. 选择 best action。
6. 一次 launch 多个 tasks。
7. 记录 global action trace。

### 通过标准

```json
{
  "global_action_dispatch_ready": true,
  "global_action_candidate_count": ">= 2 per slot",
  "batch_launch_count": ">= 1",
  "oracle_gap_alpha0": 0,
  "oracle_gap_alpha1": 0,
  "score_semantics": "robust_maxweight_lower_service",
  "fallback_legacy_rows_excluded": true
}
```

### 为什么这不是当前必须

OR 稿现在可以安全地说：theory is global/statewise candidate MaxWeight；Scheduleurm live hook is integration/audit surface；theorem-facing replay/oracle pipeline implements the objective. Claim matrix 已经这么写。

所以 10/10 global dispatcher 是长期增强，不是投稿前必需。

---

# 5. Holdout lower-service / stochastic generalization

## 当前状态

论文已经诚实写了：sparse empirical-Bernstein holdout diagnostic 得到 (\epsilon_{\mathrm{est}}=43.1457)，adjusted (\eta) 为负，所以不 claim high-confidence stochastic generalization；theorem-facing empirical certificate 用 finite measured lower-service model，(\epsilon_{\mathrm{est}}=0) by construction。

这是诚实，但也说明 stochastic generalization 还没强。

## 当前分数

**7.5/10**。边界清楚，但泛化证据还弱。

## 推到 9/10：让 selected profiles 的 holdout eta 转正

目标：

```text
holdout_adjusted_eta_selected > 0
```

### 方法

对每个 selected high-backlog profile 增加样本：

| Taskset |              Profile |   目标样本数 |
| ------- | -------------------: | ------: |
| q00     |                   13 |    ≥ 20 |
| q01     |                    8 |    ≥ 20 |
| q10     |                    8 |    ≥ 20 |
| q11     |                    3 |    ≥ 20 |
| mixed   | cpu=8,gpu=4,hybrid=3 | 每个 ≥ 20 |

然后用 LCB 而不是 mean service 做 theorem certificate：

```json
{
  "lower_service_method": "empirical_bernstein_lcb",
  "confidence": 0.95,
  "epsilon_est_selected": "...",
  "holdout_adjusted_eta": "> 0",
  "usable_for_stochastic_holdout_theorem": true
}
```

### 关键

不要试图让 all profiles 都转正；先让 theorem-selected profiles 转正。这样就能把 finite measured certificate 从 deterministic cache 推向 stochastic lower-service certificate。

## 推到 10/10

做 cross-day / cross-load / cross-node holdout：

```text
train day 1 / holdout day 2
train node A / holdout node B
train low-load / holdout moderate-load
```

通过标准：

```json
{
  "cross_day_holdout_eta": "> 0",
  "cross_node_holdout_eta": "> 0 or scoped false with blocker",
  "lcb_violation_count": 0,
  "min_samples_per_selected_profile": ">= 30"
}
```

这会把实证从 “measured finite-slice” 推到 “stochastic service uncertainty under declared regime”。

---

# 6. Online replay / ablation

## 当前状态

这部分已经不错。论文写了 120 个 Poisson/bursty load-sweep scenarios，adaptive candidate all jobs completed，legacy makespan geometric improvement 1.643×，mean flow 4.640×；ablation 也说明没有 ablated policy Pareto-dominate full candidate。

## 当前分数

**8.5/10**。

## 推到 9/10

增加统计区间和 per-scenario failure table。

现在只有 aggregate ratios。建议加：

```text
median
geomean
min
max
5% / 95%
number of dominated scenarios
number of candidate losses > 0.5%
```

新增 artifact：

```text
online_arrival_experiments_summary_ci.json
ablation_pareto_by_scenario.csv
```

论文表格改成：

| Metric | Geomean | Median | Worst | 5% | 95% |
| ------ | ------: | -----: | ----: | -: | --: |

## 推到 10/10

加入 **load boundary sweep**：

```text
load = 0.50, 0.70, 0.85, 0.95, 0.98, 1.02
```

并报告：

```text
stable below capacity
backlog grows near/above capacity
candidate degrades gracefully
```

这会和 MaxWeight/capacity story 更一致。

---

# 7. Production/admission contract

## 当前状态

future production admission contract 已经很强：active production count 36，admitted traceable count 36，probe required 0，future production automatic theorem closure ready true，但 future jobs all theorem-grade without probe false。

## 当前分数

**8.5/10**。

## 推到 9/10

把 admission contract 接入 actual scheduler submission path：

```text
submit -> classify -> ADMIT_THEOREM_TRACE / PROBE_REQUIRED
```

每个 task record 增加：

```json
"theorem_admission_route": "ADMIT_THEOREM_TRACE|PROBE_REQUIRED",
"theorem_workload_key": "...",
"theorem_profile": ...,
"theorem_admission_reason": "...",
"theorem_service_source": "..."
```

## 推到 10/10

默认启用 strict theorem admission in trace-only mode：

```text
theorem admission always recorded
theorem launch policy optional
unknown jobs never enter theorem-facing positive stream
```

也就是即使不启用 theorem dispatcher，生产 telemetry 也会自动产生 theorem admission evidence。

---

# 8. Lean / reproducibility

## 当前状态

论文说 Lean supplement 包含 `ScheduleurmUpload.lean`、`lakefile.toml`、`lean-toolchain`、paper-to-Lean theorem crosswalk、path-independent build log；并且 comment-aware search 对 `sorry/admit/axiom` 没有 proof-term matches。

## 当前分数

**8.5/10**。

## 推到 9/10

给 reviewer 一个一键脚本：

```bash
./scripts/reproduce_or_submission.sh
```

输出：

```text
1. build Lean
2. regenerate gate dashboard
3. regenerate paper tables
4. verify artifact hashes
5. write reproduction_manifest.json
```

## 推到 10/10

做一个 clean container / Nix / Conda artifact：

```text
Dockerfile.reviewer
environment.yml
Makefile
```

命令：

```bash
make verify-lean
make verify-gates
make paper-tables
make artifact-manifest
```

如果不能 Docker，就用 `uv` / `conda-lock` / `requirements-lock.txt`。

---

# 9. 论文写法

## 当前状态

claim matrix 很好，但现在用了 `\tiny` 和压缩表格。 OR reviewer 有时会不喜欢首页大表太密。

## 推到 9/10

把 claim matrix 拆成两张：

### 主文 Table 1：短版 claim matrix

只保留：

```text
Claim
Ready evidence
Non-claim
```

### Appendix Table EC.1：完整版 gate matrix

放所有 artifact path、status、blocker、next threshold。

## 推到 10/10

每个 major result 都加一句 “Claim boundary”：

```text
Claim boundary. This result applies to the declared finite measured service-cache domain; it does not certify arbitrary future states.
```

这会让审稿人很难说你过度 claim。

---

# 最终优先级清单

如果你只想做最值的 10 件事，我建议按这个顺序：

## P0：投稿前必须做

1. **所有 gate 加 status 字段**，区分 scoped pass 和 strong claim false。
2. **Declared cover 的 coverage_fraction 改成 classification_fraction / positive_fraction。**
3. **Gavel certificate 加 pass_meaning/status，避免 `pass=true` 被误读成 service equivalence true。**
4. **Controlled completion gate 改标题为 Controlled Launched Completion Gate。**
5. **论文里把 `\tiny` claim matrix 拆短，完整版放 appendix。**

## P1：把论文顶到 9/10

6. **跑 32-task controlled launched completion**，等 GPU util 安全时再跑，不绕过 gate。
7. **selected profiles 增加 samples，让 holdout adjusted eta 转正。**
8. **theorem admission strict mode**，token fuzzy match 不允许直接 theorem admit。
9. **online replay 加 CI / worst-case / loss-count 表。**
10. **一键 reviewer reproduction script。**

## P2：冲 10/10 in scope

11. **Gavel q01/q11 service-unit calibration experiment。**
12. **declared finite domain service cache v2：first_seen / last_seen / sample_count / boundary_reason。**
13. **organic production canary recorder 长期运行，累计 ≥64 launched / ≥50 completed。**
14. **global theorem dispatcher 原型：batch global candidate action，而不是 one-task score hook。**
15. **clean-room artifact build：Lean + gates + tables 全自动复现。**

---

# 我会给你的最终目标状态

## 投稿 9/10 版本

你应该把最终 dashboard 做成这样：

```text
Stability theorem:
  scoped_claim_ready = true
  strong_claim_not_needed = true

Declared finite measured service domain:
  declared_finite_positive_cover_ready = true
  arbitrary_all_state_positive_cover_ready = false
  classification_fraction = 1.0
  positive_service_fraction = 187/193

SOTA:
  policy_semantics_replay_ready = true
  gavel_adapter_compatibility_ready = true
  gavel_service_unit_equivalence_ready = false
  direct_full_stack_superiority_ready = false

Live hook:
  theorem_candidate_trace_ready = true
  bounded_controlled_completion_ready = true
  controlled_32_task_completion_ready = true
  organic_large_scale_completion_ready = false

Production:
  admission_contract_ready = true
  active_progress_ready = true
  launched_completion_ready = scoped controlled only

Reproducibility:
  lean_build_ready = true
  gate_regeneration_ready = true
  paper_table_regeneration_ready = true
```

## 10/10 in-scope 版本

如果你继续往上打，目标是：

```text
selected-profile stochastic lower-service LCB eta > 0
32-task controlled launched completion ready = true
strict theorem admission default trace mode = true
Gavel q01/q11 service-unit calibration ready = true
declared finite domain service cache v2 with timestamps/sample counts = true
one-command artifact reproduction = true
```

---

# 最后一条判断

你现在不应该再问“这三个强 claim 能不能都变 true”。更好的问题是：

> 每个强 claim 的 scoped substitute 是否足够强、足够可复现、足够不可误读？

这次更新后，答案已经接近 yes。下一步把 **status 命名、32-task controlled completion、strict admission、holdout eta、reproduction script** 做掉，整篇稿子的审稿抗性会明显上一个档次。

---

# 2026-06-14 follow-up: three non-future broad-claim bridges

这轮继续补了除“任意未来 workload”之外的三个强 claim 边界。结论不是把原始无限强 claim 全部改成 true，而是把能闭合的 in-scope 替代证书闭合到可审稿状态。

## 1. SOTA

新增：

```text
algorithm/experiments/sota_admitted_universe_closure_gate.py
md/sota_admitted_universe_closure_gate_20260614.md
md/experiment_artifacts/sota_admitted_universe_closure_gate_20260614.json
```

结果：

```text
registered_sota_policy_semantics_universe_ready = true
registered_sota_admitted_policy_superiority_ready = true
registered_sota_universe_superiority_ready = false
arbitrary_sota_superiority_ready = false
```

解释：

11 个 non-adjacent registered SOTA systems 都被映射进 finite measured-cache policy-family action union：Gavel, Pollux/AdaptDL, Sia, IADeep, Salus, Tiresias, Themis, Gandiva, Shockwave, AlloX, Optimus。strict frontier 已闭合。因此可以 claim admitted policy-semantics universe，不可以 claim every external binary/full-stack SOTA universe。

## 2. Multi-node

新增：

```text
algorithm/experiments/multinode_theorem_shadow_gate.py
md/multinode_theorem_shadow_gate_20260614.md
md/experiment_artifacts/multinode_theorem_shadow_gate_20260614.json
```

结果：

```text
reachable_gpu_node_count = 2
candidate_row_count = 18
theorem_candidate_row_count = 18
candidate_configuration_count = 626
multinode_theorem_shadow_ready = true
multinode_original_launch_claim_ready = false
```

解释：

read-only SSH inventory 下 `jtl110gpu` 和 `jtl110gpu2` 可见。gate 构造 q00/q01/q10/q11 synthetic theorem actions，枚举跨节点 disjoint robust-MaxWeight configurations，并选出跨两个 GPU node 的配置。这个闭合的是 cross-node theorem candidate interface，不是 original multi-node launched full-stack superiority。

## 3. Production-wide organic

新增/修复：

```text
algorithm/experiments/production_queued_theorem_trace.py
algorithm/experiments/production_organic_readiness_bridge_gate.py
md/production_queued_theorem_trace_20260614.md
md/production_organic_readiness_bridge_gate_20260614.md
md/experiment_artifacts/production_queued_theorem_trace_20260614.json
md/experiment_artifacts/production_organic_readiness_bridge_gate_20260614.json
```

结果：

```text
organic_readiness_bridge_ready = true
production_queued_count = 3
admissible_production_queued_count = 3
queued_theorem_trace_ready = true
queued_theorem_trace_task_count = 3
queued_theorem_trace_slot_count = 3
queued_theorem_trace_candidate_count_total = 6
queued_theorem_trace_alpha0 = 0
queued_theorem_trace_alpha1 = 0
shadow_trace_closed = true
large_scale_organic_launched_completion_ready = false
```

解释：

当前 3 个 queued BAPR-BUS production rows 在 legacy 1/3 VRAM hard rule 下被挡住。只读 theorem trace probe 使用 opt-in `clean_bench` hard-rule mode 和 `SCHEDULEURM_AB_ALLOW_ALL_TASKS=1`，但不修改队列、不 launch 任务。它证明：硬规则关闭后的 theorem hook 能为当前真实 queued production 生成 robust lower-service oracle trace。这个闭合 production organic readiness bridge，但仍不是 launched organic completion。

## 更新过的主状态

同步更新：

```text
paper/main.tex
md/or_submission_closure_status_2026_06_11.md
md/or_claim_scope_matrix_2026_06_11.md
md/math_code_alignment_2026_06_11.md
algorithm/experiments/gate_status_dashboard.py
skill/tests/test_submission_boundary_gates.py
```

仍然禁止的写法：

```text
arbitrary external-binary SOTA superiority
original multi-node launched full-stack superiority
production-wide organic launched completion
```
