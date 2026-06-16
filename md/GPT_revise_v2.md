结论先说：**整体确实又进了一大步，但现在出现了一个新的风险：论文主文有几处 claim 已经跑到 artifact 前面了。** 证明包本身的数学 spine 看起来比之前扎实，repo 里的 gate/certificate 体系也更接近 OR 可审稿形态；但如果现在直接提交，我会要求先做一次 **claim–artifact consistency freeze**，否则 reviewer 很容易抓到“不一致”。

我会把当前状态评为：

| 模块                    |                            当前评分 | 主要判断                                                                       |
| --------------------- | ------------------------------: | -------------------------------------------------------------------------- |
| 数学证明 spine            |                          8.5/10 | 结构完整、无 `sorry/admit/axiom` 静态痕迹；但上传 zip 不是完整可 build supplement。            |
| 论文 claim 边界           |                            8/10 | 摘要很克制，但正文部分新增了一些更激进的 closure claim，需要 artifact 对齐。                         |
| gate / certificate 体系 |                          8.5/10 | declared finite cover、Gavel certificate、controlled completion gate 都是正确方向。 |
| SOTA 证据               |                          7.5/10 | policy-semantics 很强，strict full-stack gate 仍明确 false；正文有些 frontier 语言要小心。  |
| production evidence   | 7/10 或 8.5/10，取决于 artifact 是否补齐 | 主文声称 history completion / 32-task completion，但我看到的对应 artifact 仍是 false。    |
| OR 投稿成熟度              |                            8/10 | 可以继续冲 OR，但先修一致性，否则会被 reviewer 以 artifact mismatch 打掉。                      |

---

## 1. 证明包评估

你上传的 `Scheduleurm.zip` 解压后是一个模块化 Lean 源码包：

```text
Scheduleurm/
  Basic.lean
  SupportFunction.lean
  CandidateApprox.lean
  CapacityRegion.lean
  MaxWeightDrift.lean
  RobustPolicy.lean
  PenaltyGrowth.lean
  ConcreteStochasticModel.lean
  MainTheorems.lean
  ...
```

我做了静态检查：

```text
Lean files: 29
Total lines: 6142
MainTheorems.lean: 53649 bytes
No grep hit for: sorry / admit / axiom / unsafe
Local Scheduleurm imports: no missing local module
```

但有一个很重要的问题：**这个 zip 不是论文里描述的完整 reviewer proof supplement。** 论文现在说提交的是 consolidated `ScheduleurmUpload.lean`，并且 supplement 包含 `ScheduleurmUpload.lean`、`lakefile.toml`、`lean-toolchain`、crosswalk 和 build log；还给了 SHA-256 hash。 你上传的 zip 里没有：

```text
ScheduleurmUpload.lean
lakefile.toml
lean-toolchain
build log
theorem crosswalk
```

本环境也没有 `lean` / `lake`，所以我不能实际编译。也就是说：

> 数学源码看起来像一个完整 Lean module tree，但作为 reviewer supplement 还不合格；它缺少可复现构建外壳。

### 数学 spine 本身的优点

`MainTheorems.lean` 的主线是清楚的：

1. candidate-restricted capacity approximation；
2. robust candidate MaxWeight drift；
3. bounded second moment / finite-support stochastic stability；
4. approximate oracle version；
5. statewise calibrated fabric version；
6. active-bucket / hidden-regime extension。

尤其是你有这些 paper-facing theorem：

```text
main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric
main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound
main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle
main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle
```

这和论文主线是对得上的。证明不是只停在 support-function lemma，而是接到了 finite-support queue model、coordinate moment、Foster finite-set recurrence 这一层。

### 数学证明的边界

证明里的 recurrence 不是无条件“Markov chain positive recurrence”大定理，而是 `PositiveRecurrentViaFiniteSet` 这种 finite-small-set certificate。这个边界是对的，论文也应该一直这么说。你现在摘要里说 finite-set Foster recurrence certificate，而不是直接说 full positive recurrence，这个口径是安全的。

### 证明包需要补的东西

投稿前必须补：

```text
ScheduleurmUpload.lean
lakefile.toml
lean-toolchain
manifest.json
build.log
paper_theorem_to_lean_theorem_crosswalk.md
sha256 list
```

并且 zip 根目录最好长这样：

```text
ScheduleurmProof/
  lakefile.toml
  lean-toolchain
  Scheduleurm/
  ScheduleurmUpload.lean
  build.log
  manifest.json
  theorem_crosswalk.md
  README.md
```

否则 reviewer 无法复现你论文里 “lake build passes” 的 claim。

---

## 2. 论文总体写法：摘要很稳，但正文有局部过冲

摘要现在是稳的。它明确说贡献是 proof and certification framework，不是 exhaustive systems race；外部 scheduler 只作为 policy-semantics diagnostics 和 adapter audits，不用于 direct full-stack superiority claim。 这很好。

Scope table 也比之前强。它明确说 finite-slice evidence 包括 measured service cache、capacity boundaries、explicit replay、slack certificate、future-admitted gate、declared finite-domain positive-cover gate、conservative all-state safety gate；Not claimed 仍然是 arbitrary unseen workloads / hardware regimes / all-state positive service outside admission contract。

但正文后半部分变激进了不少，尤其是 SOTA frontier 和 production completion。不是不能写，而是要确保 artifact 同步。

---

## 3. SOTA 部分：边界 gate 很好，但 “strict frontier closed” 要谨慎

你现在有两个方向：

### 3.1 Direct full-stack gate 仍然安全

`SOTA Full-Stack Superiority Gate` 很清楚：

```text
direct_fullstack_sota_superiority_ready = false
full_stack_ready_count = 0
policy_semantics_comparison_ready = true
hard_blocker_certificate_ready = true
```

并且工具 blocker 也明确：没有 Docker、Go、kubectl；Gavel 有 native microbaseline，但没有 service-unit equivalence / full-stack production execution。

这部分是安全的。

### 3.2 Gavel service-unit certificate 也安全

Gavel certificate 现在说：

```text
trace_schema_compatibility_ready = true
throughput_seed_ready = true
gavel_native_bounded_microbaseline_ready = true
gavel_service_unit_equivalence_ready = false
direct_full_stack_same_workload_ready = false
```



这也是安全的。它把 adapter compatibility 和 direct superiority 分开了。很好。

### 3.3 风险在 “SOTA_STRICT_FRONTIER_CLOSED” 语言

论文现在写：

> final strict-frontier gate reports `SOTA_STRICT_FRONTIER_CLOSED`，每个 measured-cache scenario 都严格不劣于 external-policy envelope in makespan and mean flow。

又写 external-policy action-union gate 关闭 measured-cache external-policy envelopes，并引入 Scheduleurm+external Pareto-slack union。

这个可以作为 **measured-cache policy-semantics frontier**，但 reviewer 很容易把它误读成 external SOTA superiority。你需要在每次出现 `SOTA_STRICT_FRONTIER_CLOSED` 附近都加一句：

```text
This is a measured-cache external-policy-semantics frontier, not direct full-stack SOTA superiority.
```

现在 table caption 有类似说明，但正文里 `SOTA_STRICT_FRONTIER_CLOSED` 这个名字太强。我建议改名为：

```text
MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_CLOSED
```

而不是 `SOTA_STRICT_FRONTIER_CLOSED`。

否则你前面所有 direct full-stack gate 的克制，会被这个名字削弱。

---

## 4. Fabric cover：现在基本 9/10

这一块是最稳的。

Declared finite-domain positive-cover gate 现在给出：

```text
declared_finite_positive_cover_ready = true
positive_service_all_state_cover_ready = false
declared_universe_bucket_count = 193
workload_domain_count = 113
positive_bucket_count = 187
boundary_bucket_count = 6
probe_required_bucket_count = 0
uncovered_bucket_count = 0
coverage_fraction = 1.0
```



代码里也明确说：这不是 arbitrary all-state fabric cover，而是 declared finite service-cache domain；unknown future states 仍然必须测量后才能进 positive population。

论文 limitation 也写得对：declared finite-domain positive-cover gate formalizes 193 exact service-cache buckets，其中 187 positive lower service、6 capacity boundaries、0 uncovered；all-state safety gate 仍然是 measured-admitted 或 unknown probe/defer zero theorem service。

这里我只建议两个小改：

1. 把 `coverage_fraction` 改成 `classification_fraction`，同时新增 `positive_service_fraction=187/193`，避免 reviewer 误会 193/193 都是 positive service。
2. 补 service-cache row 的 `first_seen/last_seen` 或明确说明 cache v1 不含时间戳。

除此之外，这部分已经很接近 9/10。

---

## 5. Production / live evidence：这是当前最大一致性风险

这里我看到**论文主文和 artifact 明显不一致**。

### 5.1 论文现在声称 32-task controlled completion closed

论文写：

> controlled 32-task ScheduleurmBench run launched 32 theorem-admitted tasks, emitted 32 robust lower-service theorem slots and 56 candidate rows, completed all 32 tasks with (\alpha_0=\alpha_1=0)。

但当前我能看到的 artifact `controlled_production_completion_gate_20260612.json` 仍然是：

```text
controlled_32_task_completion_ready = false
controlled_launched_task_count = 6
controlled_completed_task_count = 6
theorem_slot_count = 7
candidate_count_total = 13
```



markdown 也是同样：6 launched / 6 completed，32-task false。

**这是投稿前必须修的 blocker。**

要么：

1. 把论文那段 32-task claim 删除/降回 6-task bounded controlled completion；
   要么：
2. 把真实 32-task artifact push 上来，并确保文件名、表格、JSON、runner 都一致。

如果 32-task run 已经真的做了，那现在 repo 里至少应该有：

```text
md/controlled_production_completion_gate_20260613.md
md/experiment_artifacts/controlled_production_completion_gate_20260613.json
md/experiment_artifacts/live_theorem_dispatch_controlled_32_*.json
md/experiment_artifacts/live_theorem_dispatch_controlled_32_*_trace.jsonl
```

现在我没看到对应公开 artifact，所以 reviewer 会认为正文 overclaim。

### 5.2 论文现在声称 strict scheduler-history large-scale completion closed

论文写 organic history completion gate 报告：

```text
4020 strict organic launched rows
3992 completed rows
completion fraction 0.9930
35 workload domains
13 nodes
zero strict unadmitted launched rows
large_scale_organic_launched_completion_ready = true
```



但我能看到的 `production_launch_completion_gate_20260612.md` 仍然说：

```text
large_scale_launched_completion_ready = false
large_scale_active_progress_ready = true
launch_status = WAIT_RESOURCE_OR_QUEUE
```



你可能另有 “history completion gate” artifact，但我没有在当前可见路径里看到。正文里如果要 claim 4020/3992，必须给出对应 artifact path 和 table。

建议：

```text
md/production_history_completion_gate_20260613.md
md/experiment_artifacts/production_history_completion_gate_20260613.json
```

并且 Scope 要写：

```text
strict scheduler-history completion, not live oracle-traced completion
```

论文已经试图区分 live trace false vs history true，这个方向对，但 artifact 必须能一眼查到。

### 5.3 Production claim 应该拆成四个独立 flag

建议统一成：

```json
{
  "controlled_6_task_completion_ready": true,
  "controlled_32_task_completion_ready": true/false,
  "strict_history_large_scale_completion_ready": true/false,
  "live_oracle_traced_large_scale_completion_ready": false
}
```

论文对应四句话，不要混在一个段落里。

---

## 6. OR closure runner 没有纳入最新三个 gate

这是另一个重要问题。

`or_submission_closure.py` 现在的 `build_all_closure_gates()` 仍然只收集：

```text
online
holdout
ablation
live_trace
launched_live_theorem_dispatch
natural_live_theorem_trace
realization_bridge_boundary
admission_population
direct_sota_scaffold
gavel_direct_native_smoke
direct_sota_fullstack_readiness
global_fabric_cover
production_wide_live_trace_gate
production_shadow_theorem_trace
active_bucket_hidden_regime_certificate
adaptive_sampler_detector_certificate
reviewer_supplement
```

 

它**没有纳入**你这次最重要的新 gates：

```text
gavel_service_unit_equivalence_certificate
declared_finite_domain_positive_cover_gate
controlled_production_completion_gate
sota_fullstack_superiority_gate
```

这会造成一个问题：论文主文已经引用这些 gate，但“一键 OR closure runner” 不能复现它们。投稿前一定要把这些 gate 加进 `build_all_closure_gates()` 和 dashboard。

否则 reviewer 运行总 closure 时，会发现主文 claims 不在总 manifest 里。

---

## 7. Holdout / stochastic LCB：这次写法更科学，但也更复杂

论文现在说 selected-profile stochastic LCB diagnostic 改成 profile-level aggregate service windows，11 个 selected targets 都达到 20 aggregate-window threshold，新增 921 raw progress rates；但 absolute mean-service holdout 和 diagonal-normalized mean-service holdout 仍然 false，theorem-facing stochastic claim 使用 LCB service 本身作为 lower-service action family，得到 (\delta_{\mathrm{LCB}}=0.024712)。

这个写法是严谨的。它不是说 measured mean-service generalizes，而是说 lower-service LCB 本身形成一个可稳定的 lower-service action family。这个方向是对的。

但 reviewer 会问：

> 如果 LCB ratios 低到 0.3137 / 0.3560，性能结论是否还是基于 mean service，而稳定结论基于 LCB service？

你需要明确区分：

```text
Performance replay: measured service cache / empirical means.
Stability certificate: finite measured lower-service model or stochastic LCB lower-service model.
```

论文现在有这层区分，但建议在 table caption 里再加一句，避免混淆。

---

## 8. 最新版本最大优点

### 8.1 摘要克制

摘要没有因为新结果变成 full-stack superiority claim，而是明确说是 proof/certification framework，外部 scheduler 只是 diagnostics / adapter audits。 这是非常好的一点。

### 8.2 Claim matrix 结构好

Scope table 现在包含：

* stability theorem；
* finite-slice instantiation；
* live scheduler hook；
* production bridge；
* external scheduler comparison；
* learning/regime extensions。

并且每行有 Not claimed。 这对 OR reviewer 很有帮助。

### 8.3 Proof spine 比普通系统论文强很多

proof 不只是“score rule lemma”，而是接到 support slack、candidate cover、lower service、approx oracle、finite support queue model、Foster recurrence。这是 OR 视角最有价值的部分。

### 8.4 Declared finite-domain cover 是真正可审稿贡献

193 bucket、187 positive、6 boundary、0 uncovered 这个 certificate 很清楚。 这部分我认为可以作为论文的强 empirical-theorem bridge。

---

## 9. 当前最大风险

### 风险 1：proof supplement 包装不完整

你上传的 zip 不是论文声明的完整 proof supplement。缺少 `ScheduleurmUpload.lean`、`lakefile.toml`、`lean-toolchain`、build log。这个必须补。

### 风险 2：paper–artifact mismatch

最明显的是 32-task controlled completion。论文说 true，artifact 说 false。 

另一个是 strict history completion：论文说 large-scale history completion true，但我看到的 production launch gate 仍 false；如果有另一个 history artifact，需要显式放进 repo 并在 paper 里引用路径。 

### 风险 3：SOTA frontier 命名太强

`SOTA_STRICT_FRONTIER_CLOSED` 容易被误读成 full-stack SOTA dominance。建议改成 `MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_CLOSED`。

### 风险 4：OR closure runner 落后于新 gates

新 gates 没进总 closure runner。投稿前必须整合。 

---

## 10. 投稿前必须做的清单

### P0：不做会被 reviewer 抓住

1. **补完整 Lean supplement**

   ```text
   ScheduleurmUpload.lean
   lakefile.toml
   lean-toolchain
   build.log
   theorem_crosswalk.md
   manifest.json
   ```

   并保证 hash 和论文一致。

2. **修 paper–artifact mismatch**

   * 如果 32-task run 没 artifact，就删掉论文 32-task closure。
   * 如果已经做了，就 push 对应 artifact。
   * production history 4020/3992 也同理，必须有明确 artifact。

3. **把新 gates 纳入 OR closure runner**

   * `gavel_service_unit_equivalence_certificate`
   * `declared_finite_domain_positive_cover_gate`
   * `controlled_production_completion_gate`
   * `sota_fullstack_superiority_gate`
   * 如果有 history completion gate，也加入。

4. **改 SOTA strict frontier 命名**

   * 从 `SOTA_STRICT_FRONTIER_CLOSED`
   * 改成 `MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_CLOSED`

5. **统一 gate status**
   每个 gate 加：

   ```json
   scoped_claim_ready
   strong_claim_ready
   status
   pass_meaning
   ```

---

## 11. 我现在的最终判断

如果你把上述 mismatch 修掉，我会把当前项目评为：

> **OR 可投，但定位必须是 “robust candidate MaxWeight certificate + Scheduleurm case study + measured-service certification framework”，不是 systems SOTA race。**

现在最强的贡献是：

1. **理论**：candidate restriction 不再是偷换 action space，而是显式 (L\rho) slack accounting。
2. **证明**：Lean spine 覆盖 support/candidate/drift/stochastic recurrence。
3. **工程证据**：declared finite measured domain、service-cache lower-service、oracle gaps、admission gates、trace bridges。
4. **审稿边界**：明确哪些 claim true，哪些 strong claim false。

但现在不能直接提交的原因是：

> 论文新增 claim 已经比部分 artifact 和 total closure runner 领先了一步。先冻结 claim，重跑 artifacts，统一 manifest，再投。

我的建议是：下一步不要再加新 claim。做一次 **release-candidate pass**：

```text
RC1:
  regenerate all gates
  regenerate paper tables from those gates
  regenerate proof supplement
  check every numeric claim appears in exactly one artifact
  check every artifact appears in dashboard
  check every false strong claim remains false in text
```

完成这个之后，这篇稿子的审稿抗性会非常强。
