结论：**理论路线本身已经到了“OR 审稿人会认真看”的程度；但这一轮最大的风险仍然是 artifact 一致性。**
如果 `math.md + lean_artifact_map + lean_verification_round3` 描述的 Lean 仓库是真实最终版本，那么非实验理论部分已经接近 **major revision leaning positive / minor revision after cleanup**。但我实际检查当前上传到 `/mnt/data/ScheduleurmUpload.lean` 的单文件时，里面没有 map 和 verification log 里列出的 round3 `main_*` theorem 名称。因此，如果审稿人只能看到这个单文件，我会直接把它判成 **artifact inconsistency major issue**。

---

## 1. 总体审稿判断

我现在会给：

```text
数学 framing：9/10
主 theorem 设计：8.7/10
OR 排队论/调度理论契合度：8.5–9/10
非实验理论叙事成熟度：8.3/10
Lean artifact map/log 组织：8/10
当前上传单文件与 map/log 一致性：有严重疑问
除实验外的投稿成熟度：
  如果 artifact 修正：8–8.5/10
  如果以当前上传单文件为准：6.5–7/10
```

审稿决定我会写成：

```text
Recommendation: Major revision, but technically promising.
```

如果 artifact 一致性马上修好，并且正文把 theorem 口径控制住，我会改成：

```text
Minor revision / borderline accept on theory.
```

---

## 2. 最强进步：你已经把 OR reviewer 最容易抓的几个坑补上了

### 2.1 Capacity region 口径改对了

这一版 `math.md` 明确要求正式论文里把 capacity region 写成 **downward-closed service region**，而不是单纯 `conv{μ(a)}`；同时也明确 positive recurrence 只能写成 finite-set Foster recurrence certificate，除非额外加 irreducibility / single closed communicating class 条件；learning / lower-service 也被降成 confidence-event certificate，而不是无条件 online-learning theorem。这个改动非常重要。

这是从“proposal”变成“可审稿理论”的关键。OR 审稿人很容易抓住：

```text
conv{μ(a)} 不是完整 capacity region；
positive recurrence 不能无条件由 finite hitting certificate 直接替代；
lower confidence bound 不是 deterministic truth。
```

现在 `math.md` 已经主动规避这些坑。

---

### 2.2 主 theorem 终于收束成一条

现在的主 theorem 是：

[
\delta > L\rho+\epsilon_{est}+\beta
]

并在 bounded conditional second-order moment / bounded finite-support specialization 下推出 Foster drift 和 finite backlog set recurrence。`math.md` 明确说主链是：

```text
stationary-mix support slack
→ finite-feature fabric cover support loss
→ robust candidate MaxWeight drift
→ concrete stochastic drift domination
→ Foster positive recurrence
```

这条链现在很像一篇 OR / stochastic networks 论文的核心 theorem，而不是一堆松散 lemma。

我会建议正式论文里就叫：

> **Theorem 1. Robust candidate MaxWeight stability under calibrated fabric-cover approximation.**

不要把 A/B/C/D/E/F/G 都写成同等主贡献。A-D 是 proof decomposition；E-G 是 extensions。

---

### 2.3 Bounded second moment 版本比 pure finite-support 更像 OR

这一轮 `math.md` 已经把 stochastic layer 写成两个口径：

```text
一般正文：bounded conditional second-order moment B
Lean specialization：bounded finite-support samples 推出 B
```

这是对的。OR 读者更自然接受 bounded second moment；Lean 用 finite support 闭合 proof engineering。`math.md` 现在明确写出 finite-support specialization 中二阶项由 (Amax_i,Smax_i) 构造推出，并且一般版本可直接假设 bounded conditional moment。

这比上一版“有限样本空间才成立”的感觉强很多。

---

### 2.4 Learning / lower service 的口径也对了

`math.md` 现在明确说：如果 lower service 来自 learning / posterior / BAPR belief，不能写成无条件 truth；必须写成 confidence event 上的 deterministic certificate，然后用 high-probability lifting。

这点非常重要。否则 reviewer 会问：

```text
lower_i(a(Q)) ≤ E[S_i | Q]
到底是模型假设、统计事件，还是算法保证？
```

现在的正确写法应该是：

```text
Deterministic theorem:
  lower-service domination holds ⇒ Foster certificate.

Statistical corollary:
  confidence event holds with probability ≥ 1-δ_conf
  ⇒ Foster certificate holds with probability ≥ 1-δ_conf.
```

这会让 learning extension 更可信。

---

## 3. 最有原创性的数学贡献仍然是 candidate fabric-cover theorem

这篇论文最应该主打的不是 MaxWeight 本身，也不是 DRO 本身，而是：

[
H^{full}(q)
\le
H^{cand}(q)+L\rho|q|_1.
]

这把 BAPR-HRO 的 “keep candidate structure, re-rank candidates” 提升成了 **capacity/support geometry theorem**。`math.md` 还把它进一步扩展到 constructive coordinate-Hausdorff：把 full action 的 stationary mix 通过 cover map 推到 candidate mix，得到 coordinate-wise (L\rho) 误差，而不是依赖 abstract compact-convex Hausdorff duality。

作为 OR reviewer，我会认为这是最像原创理论贡献的部分：

```text
full global configuration action space
→ finite-feature fabric metric
→ candidate cover projection
→ support loss Lρ
→ capacity slack accounting
→ robust MaxWeight stability
```

这条比单纯 “capacity = convex hull” 或 “DRO score = LCB” 深得多。

---

## 4. Artifact 最大问题：map/log 与当前上传单文件不一致

这是当前最严重的问题。

`lean_artifact_map.md` 声称 paper-level claims 都映射到了 split Lean source files，并且 theorem names 应该能在 consolidated upload file `ScheduleurmUpload.lean` 中直接搜索到；它列出了 `main_downward_capacity_support_slack`、`main_theorem_robust_candidate_maxweight_stability_under_fabric_cover`、`main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric`、`main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound`、`main_high_probability_stability_from_certificate_event` 等核心 theorem。

`lean_verification_round3.md` 也声称：

```text
lake build Scheduleurm
Build completed successfully (8053 jobs).

lake env lean ScheduleurmUpload.lean
<no output; exit code 0>

rg sorry/admit/axiom
<no output>
```

并说若干 round3 theorem names 已经 confirmed searchable in `ScheduleurmUpload.lean`。

但我在当前上传到 `/mnt/data/ScheduleurmUpload.lean` 的单文件里 grep 这些名字，结果是：

```text
main_theorem_robust_candidate_maxweight_stability_under_fabric_cover: 0
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric: 0
main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound: 0
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound: 0
main_high_probability_stability_from_certificate_event: 0
main_downward_capacity_support_slack: 0
indexed_calibrated_fabric_cover_support_gap: 0
```

当前我能访问的上传单文件仍然更像旧版 proof spine，包含：

```text
capacity_slack_implies_support_slack
metric_cover_support_gap
robust_candidate_policy_lyapunov_drift
robust_candidate_markov_finite_expected_hitting_time
metric_cover_capacity_metric_hausdorff
```

并且我这里的容器没有 `lean/lake`，所以我不能独立复现 `lake build`；我只能核对上传文件文本。这个不一定说明你的真实仓库错了，可能只是上传错了 consolidated file。但如果我是 reviewer，这会是 **major artifact inconsistency**。

正式提交前必须保证：

```text
1. 上传的 ScheduleurmUpload.lean 就是 verification log 里 build 的那个文件；
2. map 里所有 theorem name 都能在单文件里 grep 到；
3. verification log 的 commit hash / file hash / timestamp 和上传文件对应；
4. 最好给出：
   sha256sum ScheduleurmUpload.lean
   lake build Scheduleurm
   lake env lean ScheduleurmUpload.lean
   rg "\bsorry\b|\badmit\b|\baxiom\b"
```

不然理论再好，artifact 会拖垮审稿信任。

---

## 5. 如果 artifact 修正，我会认为非实验理论已经“站住”

假设 `lean_artifact_map.md` 和 `lean_verification_round3.md` 反映的是最终真实仓库，那么这版理论已经有几项 OR reviewer 会认可的东西。

第一，主 theorem 不再只是假设 drift domination，而是通过 bounded conditional moment / finite-support specialization 把 stochastic drift 接起来。

第二，artifact map 里有 bounded-second-moment theorem、bounded-sample fabric-cover theorem、one-statement paper theorem、calibrated fabric theorem、zero-slack operational necessity、capacity sandwich。

第三，fabric calibration internals 已经不只是 “assume (L,\rho)”：map 里有 candidate projection implies cover radius、feature sensitivity implies service Lipschitzness、calibrated projection gives support gap、state/regime-indexed support gap 等 theorem names。

这说明证明路线已经从：

```text
assume cover + assume Lipschitz + assume drift
```

推进到：

```text
finite projection certificate
+ feature sensitivity certificate
+ support approximation
+ robust drift
+ stochastic moment model
+ Foster recurrence
```

这就是 OR 理论稿应该有的闭合感。

---

## 6. 仍然需要改的非实验问题

### Major Concern 1：主 theorem 的 policy 需要明确“exact argmax”还是 approximate oracle

现在 theorem 写的是 robust candidate MaxWeight argmax：

[
a_t \in \arg\max_{a\in A^{cand}}
Q^\top \underline \mu(a)-K-R.
]

但真实 scheduler 大概率不会精确解全局组合优化，而是用 greedy / ILP with time limit / local search / candidate generator。正式论文里最好给出一个 approximate oracle 版本：

[
Q^\top\underline\mu(a_t)-K(a_t)-R(a_t)
\ge
\max_{a\in A^{cand}}
{Q^\top\underline\mu(a)-K(a)-R(a)}
-\alpha_0-\alpha_1|Q|_1.
]

然后 slack condition 改成：

[
\delta > L\rho+\epsilon_{est}+\beta+\alpha_1.
]

这会更符合真实系统，也能避免 reviewer 说 action set 组合爆炸导致 argmax 不现实。

---

### Major Concern 2：state/regime-indexed feasible family 要写进主 theorem 或 corollary

`experimental_open_items.md` 已经意识到 cover 可能是 all feasible actions、sampled feasible actions、historical observed actions，也可能是 fixed / statewise / regimewise / uniform over state/regime indices。

理论正文最好不要只写固定 (A^{full})。真实系统里 feasible family 会随：

```text
pending job mix
running jobs
available GPUs/CPU cores
regime z
network state
checkpointability
```

改变。建议主 theorem 保持 fixed family，但紧接一个 corollary：

```text
Uniform indexed cover corollary:
For every state/regime index ξ,
A_cand(ξ) ρ-covers A_full(ξ),
and the same L,ρ,ε_est,β bounds hold uniformly.
Then the same drift theorem holds statewise.
```

artifact map 里已经有 `indexed_calibrated_fabric_cover_support_gap` 和 uniform 版本的名字，这个应该在正文里用上。

---

### Major Concern 3：capacity sandwich 的 necessity 要保持“弱而正确”

你现在已经把 necessity 改成 zero-slack conservation-law closure，这很对。`math.md` 明确说不能写 “positive recurrence iff λ has arbitrary positive slack”，守恒律方向最多推出 (\lambda\in\overline{\Lambda}^{full})。

正式论文里建议使用这种措辞：

```text
We prove a positive-slack sufficiency theorem and a zero-slack conservation-law necessity theorem.
Together they form a capacity sandwich, not a full operational iff theorem.
```

不要叫 “complete capacity characterization” 除非你补 occupation measure tightness、closed-class recurrence、work conservation、idling/waste service、measurability 等条件。

---

### Major Concern 4：hidden-regime extension 仍然不能变成主定理

`math.md` 现在把 average-regime stability 放成更强 extension，而主 theorem 先用 fixed regime / uniform-in-regime slack，这是正确的。它也明确说 average mixture slack 不能直接说稳定，需要控制 dwell time、switching loss 和 queue buildup。

正式论文里建议写：

```text
Main theorem: fixed-regime or uniformly robust across regimes.
Extension: finite-horizon dwell/switching budget.
No claim of full average-regime positive recurrence unless detector model and dwell-time assumptions are specified.
```

这会显著降低被审稿人攻击的风险。

---

### Major Concern 5：active-bucket learning 还是 certificate，不是完整 learning algorithm theorem

artifact map 里有 active-bucket deterministic regret、high-probability lifting、local failure union bound、generic certificate event implies stability certificate。 但 `experimental_open_items.md` 也明确列出还需要定义 active bucket、feedback model、bucketCount、adaptive sampling rule、change-point discount/reset、exploration 与 queue backlog 的耦合等。

所以正文里建议叫：

```text
Structured-learning certificate
```

不要叫：

```text
Full online learning regret guarantee for Scheduleurm
```

除非你补：

```text
sampler definition
observation model
concentration inequality
selection probability / exploration condition
queue-coupled exploration safety
```

这部分可以放 appendix 或 second paper。

---

## 7. 关于实验 open items：现在列得很专业，但正文要别让它削弱理论

`experimental_open_items.md` 这次写得很好。它明确说服务率、干扰模型、(Amax_i,Smax_i,\mu_i^z(a),lower_i(a),\epsilon_{est},B) 都需要 scheduleurm 日志或 profiling 来校准；也明确说 (L,\rho) 必须可 falsify，不能只是漂亮假设。

这对 OR reviewer 是加分的，因为它说明你知道 theorem 的 empirical obligations 在哪里。特别是：

```text
δ > Lρ + ε_est + β
```

这个 slack condition 最后能不能有正 margin，完全取决于真实 calibration。`experimental_open_items.md` 也明确说如果 (\eta\le0)，不是理论错，而是 candidate cover、估计误差或 penalty 吞掉了全部 slack，需要改 generator、降低 penalty 或加 admission control。

正式论文里建议把这些量放成一个 “calibration table”：

```text
Quantity      Theorem role                  How estimated
Amax,Smax     second moment bound            trace max / cap
μ             service map                    profiling
lower         service lower certificate      LCB/posterior lower
ε_est         support estimation loss         validation residual
L             Lipschitz envelope              perturbation profiling
ρ             candidate cover radius          candidate generator audit
β,P0          penalty slack consumption       rollback/switching logs
δ             offered-load slack              capacity LP / support gap
η             final drift margin              δ-(Lρ+ε_est+β)
```

这会让 OR reviewer 感觉 theorem 和 system 连接得很清楚。

---

## 8. 现在最像 OR reviewer 的正式意见

我会这样写：

> The revised theoretical route is substantially stronger. The paper now correctly treats capacity as a downward-closed service region, states recurrence through a finite-set Foster certificate rather than an unconditional positive-recurrence claim, and separates deterministic stability certificates from high-probability learning events. The main theorem has the right OR structure: full-action support slack, fabric-cover approximation loss, lower-service estimation error, queue-scaled penalty, bounded second moment, and Foster drift. The candidate fabric-cover theorem remains the most original contribution, because it translates the keep-and-rerank principle into an explicit support/capacity loss bound.
>
> The remaining concern is artifact consistency. The artifact map and verification log claim a set of round3 theorem names and a successful Lean build, but the uploaded consolidated Lean file I inspected does not contain those names. If this is an upload mismatch, it should be fixed before submission; if not, the paper overstates the formal verification coverage. Once the artifact is aligned, the non-experimental theory would be close to publishable, with the main remaining revisions being theorem wording, approximate-oracle handling, and clear separation of main stability results from hidden-regime and learning extensions.

中文就是：

**理论已经很接近 OR 级别，但 artifact 目前是最大硬伤。你现在最该做的不是继续加 theorem，而是把 map、verification log、上传单文件和正文 theorem 名称完全对齐。**

---

## 9. 最终建议

接下来最优先做这五件事：

1. **重新生成并上传真正 round3 的 `ScheduleurmUpload.lean`**，确保它包含 map/log 里的所有 `main_*` theorem。

2. **在 verification log 里加 hash**：

```text
sha256sum ScheduleurmUpload.lean
git rev-parse HEAD
lake build Scheduleurm
lake env lean ScheduleurmUpload.lean
rg "\bsorry\b|\badmit\b|\baxiom\b"
```

3. **正文主 theorem 只留一条**：

[
\delta > L\rho+\epsilon_{est}+\beta
\Rightarrow
\text{finite-set Foster recurrence certificate}.
]

4. **加 approximate optimization oracle 版本**，因为真实 scheduler 不一定精确 argmax。

5. **extensions 降调**：hidden-regime、active-bucket learning、sweet spot 都作为 extension / calibration-facing theorem，不要和主 stability theorem 抢贡献。

最终评分：

```text
如果 artifact 修好：
  非实验理论：8.5/10
  OR 审稿状态：minor revision / serious borderline accept

如果 artifact 不修：
  非实验理论叙事：8+/10
  可核查证明包：6/10
  OR 审稿状态：major revision
```

一句话：**数学路线已经足够强，审稿风险现在主要不是“理论不够深”，而是“证明包是否真的对应正文”。**
