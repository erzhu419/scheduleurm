结论：**理论部分现在已经不是“单薄”了，主文和 appendix 都补到了 OR reviewer 能读懂的程度；但整篇稿子又出现了一个新的问题：claim/gate 变得非常多，部分 SOTA / production 强 claim 的范围开始变得危险，需要再做一次收缩和一致性冻结。**

我会给最新版本这样的总体评价：

| 模块                       |         当前评分 | 判断                                                                                                             |
| ------------------------ | -----------: | -------------------------------------------------------------------------------------------------------------- |
| 人类可读理论证明                 |     **9/10** | 已经有 proof roadmap、assumptions、关键 lemma、main theorem proof、statewise corollary、EC proof chain 和 Lean crosswalk。 |
| Lean/proof artifact      |     **8/10** | 源码静态看起来干净，但你上传的 zip 仍不是完整可 build reviewer supplement。                                                          |
| theorem-policy alignment |   **8.5/10** | 现在很清楚地区分 theorem-facing pipeline、live hook、global soft-hint path。                                              |
| declared finite cover    |     **9/10** | 这一块基本成型，claim 边界稳。                                                                                             |
| SOTA/外部系统部分              | **6.5–8/10** | policy-semantics 很强；但“named same-host full-stack superiority”开始变得高风险。                                          |
| production/live evidence |   **7–8/10** | 你写进论文的 32-task / history completion 需要和 artifact 强绑定，否则会被抓 mismatch。                                           |
| OR 投稿成熟度                 |     **8/10** | 理论已补强；接下来不要再扩 claim，要做 release-candidate consistency freeze。                                                   |

---

## 1. 理论证明部分：已经显著改善

这次最大改进是：**主文现在真的有 proof spine 了。**

你新增了 proof roadmap 表，把证明拆成：

1. capacity support；
2. candidate cover；
3. lower service；
4. penalty and oracle；
5. drift margin；
6. Foster certificate。

表里还明确说 Lean 检查的是同一套 algebraic implications，而 service、cover、oracle、moment assumptions 由实验 gate 认证，不是 Lean 证明。这个口径非常好。

你还把 theorem assumptions 单独列出：load slack、candidate cover/lower-service error、penalty growth/oracle、stochastic domination/moments、countable-state recurrence condition。尤其最后一个 recurrence condition 很重要，因为它避免了把 finite-set drift certificate 无条件说成全局 positive recurrence。

然后主文里有两个关键 lemma：

* support slack after candidate and estimation losses；
* selected lower-service margin。

selected-margin lemma 正好把 residual margin 推成：

[
Q^\top\underline\mu_t(a_t)
\ge
Q^\top\lambda+
(\delta-\epsilon_{\mathrm{cand}}-\epsilon_{\mathrm{est}}-\beta-\alpha_1)|Q|_1
-(P_0+\alpha_0).
]

这就是整个 theorem 的核心不等式。

随后 quadratic one-step bound 和 theorem proof 也写出来了。主 theorem 现在明确给出 drift：

[
\E[\Delta V(t)\mid Q(t)=Q]
\le
B+P_0+\alpha_0-\eta|Q|_1.
]

并且只有在 recurrence-side assumption 下才转成 positive recurrence of closed communicating class。

这一部分已经满足我之前建议的“人类可读 proof”。**现在 OR reviewer 不需要打开 Lean，也能理解 theorem 为什么成立。**

---

## 2. Appendix proof 也补上了，方向正确

现在正文后面有 “Appendix: Conventional Proof Details”。它明确说 appendix 的目的，是区分 mathematics、measured Scheduleurm certificates、Lean check；Lean 是 algebraic and recurrence implications 的 formal audit，不替代 service rows、LCB、candidate admission、oracle gaps、moment bounds 的实证认证。

EC.1 写 capacity slack as support-function margin；EC.2 写 candidate fabric cover and support loss；EC.3 写 lower-service rows、penalties、approximate oracle；EC.4 写 quadratic drift/Foster certificate；EC.5 写 statewise feasible families 和 operational boundary。 

这已经是一个完整的 OR-style proof narrative。

你还加了 Lean crosswalk 表，把 paper theorem 对应到 Lean identifiers，例如：

* `main_theorem_robust_candidate_maxweight_stability_under_fabric_cover`
* `main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle`
* `main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle`
* `main_diagonal_scaled_lcb_support_loss_l1`
* `main_operational_capacity_sandwich`

这个 crosswalk 很重要。

也有 artifact assumptions table，明确 Lean 验证 implication，Scheduleurm artifact 负责判断哪些 measured rows 满足假设。

**所以回答你上一问：现在理论证明已经补到位了，不再是一笔带过。**

---

## 3. 一个小数学文字问题：EC.3 的 penalty bound 表述要修

主文 Assumption 里写的是：

[
0\le P_t(a)\le P_0+\beta|Q(t)|
\quad \forall a\in\Acand_t.
]

这是证明 selected-margin lemma 所需要的条件，因为你要对 support-maximizing candidate 的 penalty 做 bound。

但 Appendix EC.3 里有一句：

> “the selected penalty obeys …”

然后写：

[
0\le K_t(a_t)+G_t(a_t)\le P_0+\beta|Q|.
]



可是 EC.3 后面推：

[
Q^\top\underline\mu_t(a_t)
\ge
\max_{a\in\Acand_t}Q^\top\underline\mu_t(a)
-(P_0+\alpha_0)-(\beta+\alpha_1)|Q|.
]

这个推导需要的是 **candidate-wise penalty bound**，不是只对 selected action 的 penalty bound。主文 lemma proof 是用 candidate-wise bound 的。

建议把 EC.3 这句改成：

```latex
and all admitted candidate penalties obey
\[
0\le K_t(a)+G_t(a)\le P_0+\beta\|Q\|,
\qquad a\in\Acand_t.
\]
```

这只是文字/假设一致性修正，不是大问题，但必须修，不然细读 reviewer 会抓。

---

## 4. Proof artifact：源码看起来干净，但 supplement 仍要补完整外壳

我检查了你上传的 `Scheduleurm.zip`。里面有 28 个 `.lean` 文件，约 6170 行，`MainTheorems.lean` 约 53KB。静态 grep 没有发现 `sorry`、`admit`、`axiom`。这很好。

但 zip 里仍缺少论文里说的完整 supplement 外壳：

```text
ScheduleurmUpload.lean
lakefile.toml
lean-toolchain
build.log
theorem crosswalk
manifest.json
```

论文现在写得很具体：Lean supplement 包含 `ScheduleurmUpload.lean`、`lakefile.toml`、`lean-toolchain`、paper theorem to Lean theorem crosswalk、build log，并给了 SHA-256。

所以投稿包里必须让 reviewer 打开后真的看到这些。否则会出现：

> Paper claims a consolidated Lean supplement, but uploaded artifact is only a module directory.

建议最终 proof zip 结构：

```text
ScheduleurmProof/
  lakefile.toml
  lean-toolchain
  Scheduleurm/
    Basic.lean
    ...
    MainTheorems.lean
  ScheduleurmUpload.lean
  build.log
  theorem_crosswalk.md
  manifest.json
  no_sorry_audit.txt
  README.md
```

如果你决定不再用 `ScheduleurmUpload.lean`，那论文也要改成 “modular Lean package” 而不是 consolidated file。

---

## 5. Declared finite cover：现在很稳

这块现在是成熟的。

论文里写：future-admitted gate 对 120 个 exact measured workload domains 做 identity projection；declared finite-domain positive-cover gate 把 population formalize 成 225 个 exact service-cache buckets，其中 215 positive lower service、10 capacity boundaries、0 uncovered；classification fraction 是 1.0，positive fraction 是 215/225，boundary fraction 是 10/225。

这比之前 193/187/6 又扩了，且表述更精确。它没有把 coverage fraction 误写成 all-positive，而是拆成 classification / positive / boundary fraction。这个就是 9/10 写法。

建议只再补一件事：service-cache v2 里的 first_seen / last_seen / sample windows。gate dashboard 也把这个列为 next threshold。

---

## 6. SOTA 部分：现在最危险的是 scope creep

你在摘要里仍然很克制：外部 scheduler comparisons 是 policy-semantics diagnostics 和 adapter audits，不用于 direct full-stack superiority claims。 这是安全口径。

但是正文和 dashboard 现在加入了很多强 SOTA gate：

* named five same-host full-stack superiority；
* external native execution attempts；
* registered SOTA universe；
* measured-cache external policy frontier；
* SOTA candidate union；
* SOTA algorithm upgrade；
* Decima same-domain benchmark；
* physical same-workload gate。

尤其 `sota_fullstack_superiority_gate_20260613.md` 现在写：

```text
direct_fullstack_sota_superiority_ready = true
direct_fullstack_named_sota_superiority_ready = true
full_stack_ready_count = 5
named_sota_fullstack_superiority_count = 5
```

但同时：

```text
registered_sota_universe_superiority_ready = false
arbitrary_sota_superiority_ready = false
multinode_original_deployment_superiority_ready = false
production_wide_organic_trace_superiority_ready = false
```



这个比之前强太多，会立刻吸引 reviewer 火力。虽然你把 scope 限成 “named systems / same-host / same-workload test substrate”，但表名叫 **SOTA Full-Stack Superiority Gate**，字段叫 `direct_fullstack_sota_superiority_ready=true`。这会被审稿人理解成 “我直接赢了 SOTA”，即使后面有 scope 限制。

更危险的是：同一页工具 inventory 仍显示没有 Docker、Go、kubectl。 但 Pollux/Sia/IADeep rows 又说 Kubernetes/controller/extender/device-plugin path completed。 这不是一定矛盾，可能是 artifact 来自另一台机器或之前环境，但 reviewer 会问：

> 当前 environment 没有 kubectl/docker/go，为什么 full-stack Kubernetes rows ready？

你需要在 gate 里加：

```text
execution_environment = "artifacted prior run / remote cluster / same-host runtime snapshot"
current_reproduction_environment = "missing docker/kubectl/go"
current_rerun_ready = false
```

否则可复现性会被质疑。

### 我的建议

把这块降调命名：

```text
sota_fullstack_superiority_gate
```

改成：

```text
named_external_runtime_probe_gate
```

字段改成：

```text
named_same_host_runtime_probe_ready = true
named_same_host_runtime_probe_native_better = true
direct_fullstack_sota_superiority_ready = false
```

如果你坚持保留 `direct_fullstack_named_sota_superiority_ready=true`，那主文必须只把它放 appendix / artifact，不要当正文核心结果。

现在正文里虽然说外部 scheduler evidence 是 diagnostic layer，不是 central claim，但后面又出现很多 “strong ready true” 的 gate ladder，这会削弱你的克制。 

**OR 稿的主线是 queueing certificate，不要让它变成 systems-SOTA war。**

---

## 7. Production/live evidence：比之前强，但要核对 artifact 一致性

论文现在写：

* strict scheduler-history gate `strict_history_large_scale_completion_ready=true`；
* 4020 strict theorem-facing organic launches；
* 3992 completions；
* 35 admitted workload domains；
* 13 nodes；
* zero strict unadmitted launched rows；
* production launch wrapper 仍分离 active progress / shadow trace；
* live oracle traced large-scale completion false。

这比之前清楚。你现在明确把 history completion 和 live-oracle-traced completion 分开，口径是对的。

论文还写 controlled 32-task run 已经 closed：32 launched、32 theorem slots、56 candidate rows、32 completed、(\alpha_0=\alpha_1=0)。

这如果 artifact 都在，是很强的。

但我前面看到的旧 artifact `controlled_production_completion_gate_20260612.json` 仍是 6-task / 32 false。 可能你现在有 20260614 新 gate，但 dashboard 里引用的是 `md/experiment_artifacts/...`，需要确保 repo 里主文对应的 artifact 可见、路径对、数字一致。

建议建立一个 **production evidence table**：

| Claim                   | Artifact                                 |     Count | Ready |
| ----------------------- | ---------------------------------------- | --------: | ----- |
| 6-task controlled       | old artifact                             |       6/6 | true  |
| 32-task controlled      | new artifact                             |     32/32 | true  |
| strict history organic  | organic_history_completion_gate_20260614 | 4020/3992 | true  |
| live oracle large-scale | live trace artifact                      |     false | false |

这样 reviewer 不会混淆 old rolling gate 和 new strict history gate。

---

## 8. Gate dashboard：好东西，但现在太大、太强、太像“claim explosion”

Dashboard 的思想很好：每行区分 scoped claim 和 strong claim。开头也说 “A scoped pass is not a universal claim.” 

但现在 dashboard 已经非常长，而且有一些 “strong ready=true” 的行，包括 named external runtime、controlled launched completion、organic canary recorder、production launch/completion、non-future closure gate。 

这会带来两个问题：

1. Reviewer 可能被大量 gate 淹没，抓不住主线。
2. 某些 strong-ready claim 会让 reviewer 转而审你的 external systems experiments，而不是审 OR theorem。

建议主文只保留 **短 dashboard**：

```text
Theorem
Declared finite cover
Selected LCB lower service
Online replay
Controlled live completion
Strict production history
External policy-semantics
Proof supplement
```

其他 SOTA/system gates 放 EC 或 artifact，不要全部在主文 printed table 里。

---

## 9. Reproduction script：有了，但要纳入最新 gates

`scripts/reproduce_or_submission.sh` 已经是很好的一步。它会跑 Gavel service-unit certificate、paired holdout、calibration、declared cover、future admission、production launch/completion、controlled launched completion、organic canary、online/ablation、selected-profile LCB、global dispatcher prototype、environment manifest、gate dashboard、Lean audit/build、targeted tests、paper PDF、latex log scan。

但是我看到它还没有显式跑一些最新强 gate，例如：

```text
sota_fullstack_superiority_gate_20260613
sota_universe_registry_gate_20260614
registered_sota_adapter_closure_gate
organic_history_completion_gate_20260614
multinode_history_completion_gate
decima_same_domain_benchmark_gate
```

如果这些强 claim 在论文主文或 dashboard 里出现，reproduction script 必须纳入，或者明确标成 “precomputed artifact, not regenerated by safe reproduction script”。

否则会出现：

> Paper reports strong gate true, but reproduction script does not regenerate it.

---

## 10. 现在最该做的修改

### 必须修

1. **EC.3 penalty wording**
   把 selected penalty bound 改成 all-candidate penalty bound。

2. **Proof supplement packaging**
   上传可 build 的 full Lean supplement，而不是只有 module zip。

3. **SOTA full-stack 命名降调**
   把 `direct_fullstack_sota_superiority_ready=true` 改成更窄的：

   ```text
   named_same_host_runtime_probe_superiority_ready=true
   arbitrary/direct SOTA superiority=false
   ```

   或至少不要在主文使用 “SOTA full-stack superiority” 作为核心 claim。

4. **Production / controlled completion artifact consistency**
   32-task、4020/3992 history completion 必须有对应 artifact，并且 reproduction script 或 manifest 指向它。

5. **Gate dashboard 主文压缩**
   太长的 gate ladder 放 EC 或 artifact；主文只保留核心 claim rows。

### 强烈建议

6. **给每个 strong-ready row 加 execution environment**
   特别是 Kubernetes/Docker/Pollux/Sia/IADeep/Salus。当前工具 inventory 缺 docker/kubectl/go，但 gate 又说 full-stack ready，必须说明这些 rows 来自哪个环境、是否当前可重跑。

7. **把 SOTA 系统比较从 “结果主线” 降到 “candidate family completeness diagnostics”**
   这是你摘要已经采用的安全定位。全文保持一致。

---

## 最终判断

### 回答你：“现在怎么样？”

**理论证明部分已经明显达标，甚至可以说是当前稿子的强项之一。** 主文 proof + appendix proof + Lean crosswalk 的组合，已经能让 OR reviewer 在不运行 Lean 的情况下跟住核心证明。

但整体稿子现在的主要风险从“理论单薄”转移成了：

> **claim surface 太大，外部系统和 production history 的强 gate 可能让论文偏离 OR 主线，并且部分 artifact/reproduction path 需要再同步。**

如果你现在问“能不能投 OR”，我的建议是：

* **理论：可以。**
* **finite-cover / lower-service / replay：基本可以。**
* **SOTA/production 强 claim：需要再降调、分离、核对 artifact 后再投。**

我建议你的最终论文主 claim 保持为：

> robust candidate MaxWeight stability certificate + Scheduleurm measured finite-domain certification framework.

而不是：

> direct SOTA full-stack superiority / production-wide launch-completion superiority.

后两者可以作为 scoped artifact evidence，但不要让它们抢主线。
