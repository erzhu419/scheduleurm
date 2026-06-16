又看了一轮。**这次 push 的方向是对的：你把旧的 20260612 SOTA full-stack gate 标成 superseded，并把它的强 claim 改回 false；理论证明部分也已经补得很扎实。** 但现在还有一个核心问题没有完全消掉：

> 论文/claim matrix 的安全口径已经收住了；但 `20260613` 的 SOTA full-stack artifact 仍然有 `direct_fullstack_sota_superiority_ready=true` 这类强字段。
> 也就是说，**你最新 push 修了旧 snapshot，但还没有把所有后续 SOTA artifact 的强 claim 口径完全统一。**

我分块说。

---

## 1. 这次 push 的实际改善

最新 commit 看起来主要改了 5 个文件：`sota_fullstack_superiority_gate_20260612`、`or_claim_scope_matrix`、`or_submission_gap_closure` 和一个 reviewer collect json。也就是说这次不是大改理论/实验，而是**修 claim scope**。

最明显的改善是：`md/sota_fullstack_superiority_gate_20260612.md` 现在已经改成：

```text
# Superseded Named External Runtime Probe Snapshot
status = SUPERSEDED_BY_20260613_NAMED_RUNTIME_PROBE_DIRECT_SOTA_FALSE
direct_fullstack_sota_superiority_ready = false
direct_fullstack_named_sota_superiority_ready = false
registered_sota_universe_superiority_ready = false
arbitrary_sota_superiority_ready = false
```

并且 scope 明确说这个 20260612 snapshot 只应读作 scoped same-host same-workload runtime-probe evidence，不 claim direct full-stack SOTA superiority、registered-universe superiority、arbitrary SOTA superiority、future-workload superiority、original multi-node superiority 或 production-wide trace superiority。

这是非常正确的修法。它把旧 artifact 从 “可能被误读成 SOTA superiority” 改成了 “superseded runtime-probe snapshot”。

---

## 2. 理论证明部分：现在已经达标

这一块我现在基本满意。你补了：

1. proof roadmap；
2. assumptions；
3. deterministic slack lemmas；
4. selected lower-service margin；
5. quadratic drift proof；
6. Foster finite-set certificate；
7. statewise corollary；
8. appendix conventional proof；
9. Lean crosswalk；
10. artifact assumptions table。

主文的 proof roadmap 已经把完整链条写出来了：capacity support、candidate cover、lower service、penalty/oracle、drift margin、Foster certificate，并明确说 Lean 检查 algebraic implications，而 service/cover/oracle/moment assumptions 由 experiment gates 认证。

main theorem proof 现在也是人类可读的，不再只是“Lean 里有证明”。它先用 quadratic drift inequality，再用 stochastic domination 和 selected-margin lemma，最后得到

[
\E[\Delta V(t)\mid Q(t)=Q]
\le B+P_0+\alpha_0-\eta|Q|_1.
]

然后只在 recurrence-side assumption 下转成 closed communicating class 的 positive recurrence。

你之前 EC.3 的一个小问题也修了：现在 appendix 明确写的是 **all admitted candidate penalties** obey

[
0\le K_t(a)+G_t(a)\le P_0+\beta|Q|,\qquad a\in\Acand_t,
]

并且后面推 selected lower-service margin 时也写成 “by the oracle gap and the all-candidate penalty bound”。这和主文 lemma 一致了。

Appendix 也补了 Lean crosswalk，明确把 Theorem、approx-oracle version、statewise corollary、coordinate-scaled LCB、operational capacity boundary 分别映射到 Lean theorem identifiers。

**结论：理论部分现在可以给 9/10。** 论文不再需要继续大幅补数学证明；剩下主要是 proof artifact packaging 和 claim consistency。

---

## 3. 仍然最大的问题：20260613 SOTA gate 和最新 scope matrix 不一致

虽然 20260612 SOTA snapshot 已经修成 direct false，但 `md/sota_fullstack_superiority_gate_20260613.md` 仍然写：

```text
direct_fullstack_sota_superiority_ready = true
direct_fullstack_named_sota_superiority_ready = true
full_stack_ready_count = 5
named_sota_fullstack_superiority_count = 5
```

同时又说 registered/arbitrary/multinode/production-wide 都 false。

这就和你最新的 scope matrix 有冲突。`or_claim_scope_matrix` 现在把这部分安全地写成：

> named external runtime-probe gate records `named_same_host_runtime_probe_ready=true` and keeps `direct_fullstack_named_sota_superiority_ready=false`。

这两个口径不能同时存在：

* scope matrix：`direct_fullstack_named_sota_superiority_ready=false`
* 20260613 artifact：`direct_fullstack_named_sota_superiority_ready=true`

我的建议很明确：**把 20260613 artifact 也改成 runtime-probe wording，不要叫 direct full-stack SOTA superiority ready。**

推荐字段：

```json
{
  "named_same_host_runtime_probe_ready": true,
  "named_same_host_runtime_probe_native_better_ready": true,
  "direct_fullstack_named_sota_superiority_ready": false,
  "direct_fullstack_sota_superiority_ready": false,
  "registered_sota_universe_superiority_ready": false,
  "arbitrary_sota_superiority_ready": false,
  "status": "NAMED_RUNTIME_PROBE_PASS_DIRECT_SOTA_FALSE"
}
```

同时把 markdown 标题从：

```text
SOTA Full-Stack Superiority Gate
```

改成：

```text
Named External Runtime-Probe Gate
```

这样和 scope matrix 一致，也和摘要“external schedulers are diagnostics / adapter audits, not direct full-stack superiority claims”的口径一致。摘要现在是克制的，明确说外部 scheduler comparisons 是 policy-semantics diagnostics 和 adapter audits，不用于 direct full-stack superiority claims。

现在的问题是：**摘要和 scope matrix 很稳，但 20260613 SOTA gate 仍然太强。**

---

## 4. Dashboard 也需要同步

`md/gate_status_dashboard.md` 的行已经非常多。它开头说得对：

> scoped pass is not a universal claim.

但其中 SOTA 行仍然会造成歧义。比如它对 `sota_fullstack_superiority_gate` 写的是 scoped same-host same-workload full-stack rows for Gavel/Pollux/Sia/IADeep/Salus，strong claim 是 arbitrary SOTA 等 false，status 是 named same-host fullstack superiority pass。

这个比 20260613 artifact 稍微安全，但仍然建议改成：

```text
Gate: named_external_runtime_probe_gate
Scoped claim: scoped same-host same-workload runtime-probe rows for named external systems
Strong claim: direct full-stack SOTA superiority
Strong ready: false
Status: NAMED_RUNTIME_PROBE_PASS_DIRECT_SOTA_FALSE
```

原因：**“full-stack superiority” 这个词一出现，reviewer 就会把注意力转到系统 benchmark，而不是 OR theorem。**

---

## 5. Production 部分：scope matrix 已经更稳，但要确保 artifact 路径一致

scope matrix 现在把 live/production 写得比较清楚：

* dry-run / queued / shadow 不是 launched completion；
* controlled 32-task trace validates theorem-grade controlled launched completion；
* organic recorder verifies strict admission/trace readiness but keeps organic launched completion false until thresholds are met；
* global prototype is not live default。

这个口径是对的。

但论文正文里你仍然有一段写：

```text
strict scheduler-history gate reports strict_history_large_scale_completion_ready=true:
4,020 strict theorem-facing organic launches, 3,992 completions, 35 admitted workload domains, 13 nodes, zero strict unadmitted launched rows.
```

同时又说 live oracle-traced large-scale completion false。

这本身可以成立，但必须保证 artifact 路径明确。正文现在引用了：

```text
md/organic_history_completion_gate_20260614.md
```

这很好，但建议你在 final artifact dashboard 里放一行：

```text
organic_history_completion_gate:
  strict_history_large_scale_completion_ready = true
  live_oracle_traced_large_scale_completion_ready = false
  raw_history_closed = false
```

否则 reviewer 可能混淆 history completion 和 live oracle-traced completion。

---

## 6. Reproduction script：有，但还不是全量 broad gate runner

`reproduce_or_submission.sh` 已经很有价值。它能重跑 Gavel service-unit certificate、Gavel paired holdout、declared finite cover、future admission、production launch/completion、controlled launched completion、organic canary、online/ablation、selected-profile LCB、global dispatcher prototype、environment manifest、dashboard、Lean audit/build、targeted tests、paper PDF 和 latex log scan。

但它目前看起来仍然不是所有 20260613/20260614 broad SOTA/production gates 的全量重跑器。例如 dashboard 里出现了很多 gate：

* `sota_universe_registry_gate`
* `registered_sota_runtime_gate`
* `registered_sota_adapter_closure_gate`
* `sota_admitted_universe_closure_gate`
* `multinode_history_completion_gate`
* `decima_same_domain_benchmark_gate`
* `production_wide_organic_trace_gate`

这些如果在论文或 dashboard 里作为 evidence，就需要么：

1. 加进 reproduction script；
2. 或者明确标成 `precomputed_artifact_not_safe_rerun_by_default`。

我建议 reproduction manifest 里给每个 artifact 一个字段：

```json
"regeneration_mode": "safe_rerun|safe_readonly|precomputed_live_run|requires_external_stack|not_rerun"
```

否则 reviewer 会以为 `make reproduce` 能重跑一切。

---

## 7. Proof supplement packaging 仍是一个独立 checklist

我这轮主要看 repo。关于证明包，之前的问题仍然要确认：你论文说的是完整 Lean supplement，包括 `ScheduleurmUpload.lean`、`lakefile.toml`、`lean-toolchain`、crosswalk、build log、no-sorry audit。正文也继续这么写。

如果最终提交包里这些都在，那没问题；如果只有模块 zip，还不够。
这不是论文主文问题，而是 submission artifact 问题。

---

## 8. 当前版本评分

我会这样打分：

| 项目                                      |        当前分 | 说明                                                               |
| --------------------------------------- | ---------: | ---------------------------------------------------------------- |
| 理论证明主文                                  |       9/10 | 已有 roadmap、lemmas、theorem proof、statewise corollary。             |
| Appendix proof                          |       9/10 | EC proof chain 和 Lean crosswalk 已补齐。                             |
| Lean artifact packaging                 |       8/10 | 取决于最终 supplement 是否包含 lake/build/crosswalk。                      |
| Declared finite cover                   |       9/10 | 225 bucket、classification/positive/boundary split，边界清楚。          |
| Selected LCB / stochastic lower-service |     8.5/10 | 口径正确：LCB lower-service capacity，不 claim mean-service eta。        |
| Production evidence                     |       8/10 | controlled/history/live trace 分开了，但需要 artifact manifest 强绑定。     |
| SOTA/external systems                   | 6.5–7.5/10 | 20260612 旧 snapshot 已修；20260613 仍有 direct fullstack true 字段，需收掉。 |
| OR 投稿准备度                                |     8.2/10 | 理论达标；最大剩余风险是 SOTA claim surface。                                 |

---

## 9. 我建议你下一步只做这 5 件事

### 1. 把 20260613 SOTA gate 也改成 runtime-probe gate

不要让任何 machine-readable field 出现：

```text
direct_fullstack_sota_superiority_ready = true
direct_fullstack_named_sota_superiority_ready = true
```

除非你真的想把论文变成 systems benchmark paper。对 OR 来说不值得。

改成：

```text
named_same_host_runtime_probe_ready = true
named_same_host_runtime_probe_native_better_ready = true
direct_fullstack_named_sota_superiority_ready = false
direct_fullstack_sota_superiority_ready = false
```

### 2. Dashboard 同步改名

把 `SOTA Full-Stack Superiority Gate` 改成：

```text
Named External Runtime-Probe Gate
```

把 strong claim 统一 false。

### 3. Reproduction manifest 标记 regeneration mode

尤其对 SOTA/production live artifacts：

```text
safe_rerun
safe_readonly
precomputed_live_run
requires_external_stack
not_rerun
```

### 4. 主文只保留短 dashboard

长 gate ladder 太长，而且有太多 SOTA/system rows。建议主文只保留核心 OR rows：

```text
stability theorem
declared finite cover
selected LCB
online/ablation
controlled launched completion
strict history completion
external policy-semantics / runtime probes
Lean supplement
```

长表放 EC 或 artifact。

### 5. 最终 proof supplement 打包验证

保证提交包里真的有：

```text
lakefile.toml
lean-toolchain
ScheduleurmUpload.lean
Scheduleurm/*.lean
build.log
theorem_crosswalk.md
manifest.json
```

---

## 最终判断

这次 push **确实改善了旧 SOTA snapshot 的口径**，而且理论证明部分已经达到 OR 稿应有的阅读标准。现在唯一还显著危险的是：

> 你一边在 scope matrix 里说 named external runtime-probe 不是 direct fullstack SOTA superiority，一边 20260613 SOTA gate 仍然写 `direct_fullstack_sota_superiority_ready=true`。

这会让 reviewer 抓住不放。把这一个口径统一后，整篇会稳很多。

我的建议：**现在不要再加新实验、新 gate、新强 claim。做 claim freeze。**
目标是让所有文件都回答同一句话：

> 本文主贡献是 robust candidate MaxWeight stability certificate 和 Scheduleurm measured-service certification framework；外部系统结果只是 bounded runtime-probe / policy-semantics diagnostics，不是 arbitrary direct SOTA superiority。
