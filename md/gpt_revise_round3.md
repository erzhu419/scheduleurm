结论：**这版除实验外已经很接近一篇可认真送审的 OR/queueing-control 理论稿了。**
我作为 OR 审稿人，现在不会再说“只是 proposal”或“proof skeleton”。更准确的评价是：

```text
理论主线：8.3–8.7/10
Lean artifact 组织：7.5–8/10，前提是最终上传文件和 map 完全一致
OR 可读性/叙事：7.5/10
当前非实验部分审稿建议：major revision leaning positive
补完表述与 artifact 后：可到 minor revision / borderline accept 的理论状态
```

这版最大的进步是：你已经把主张收束成一条 paper theorem，而不是 A/B/C/D/E/F/G 全部抢主线。现在的核心链条是：

```text
stationary-mix support slack
→ finite-feature fabric cover support loss
→ robust candidate MaxWeight drift
→ concrete stochastic drift
→ Foster positive recurrence
```

这比“DRO candidate score”深很多，也比上一版的 “if assumptions then drift” 更像 OR 论文。`math.md` 明确把主 theorem 写成：full-action support slack + fabric-cover candidate loss + lower-service estimation loss + queue-scaled penalty + bounded finite-support stochastic model 推出 positive recurrence；同时把 hidden regime、active-bucket learning、sweet spot 放到 extension，而不是主 claim。这个结构是对的。

---

## 1. 我作为审稿人会认可的核心贡献

### 1.1 主 theorem 终于像一篇 OR 理论论文了

新版主 theorem 的形式很清楚：

[
\delta > L\rho+\epsilon_{est}+\beta
]

并且 backlog set (N) 足够大时，integer queue Markov chain positive recurrent。这里 (L\rho) 是 candidate cover 损失，(\epsilon_{est}) 是 lower-service estimation 损失，(\beta) 是 queue-scaled penalty 对 drift margin 的消耗。这个 slack accounting 是 OR reviewer 会喜欢的，因为它把所有工程近似都显式记账，而不是藏在 heuristic 里。

这条定理现在可以作为论文的主句：

> **A candidate-restricted robust MaxWeight policy remains stable if the full-action capacity slack exceeds fabric-cover loss, service-estimation loss, and queue-scaled penalty.**

这已经不是 “BAPR-HRO + scheduler” 的拼接，而是一个真正的 queueing-control theorem。

---

### 1.2 Candidate-set approximation 是最有原创性的数学点

我仍然认为最值得主打的是：

[
H^{full}(q)\le H^{cand}(q)+L\rho|q|_1.
]

它把 BAPR-HRO 的 “keep candidate structure, re-rank candidates” 变成了 capacity/support geometry theorem。更好的是，`math.md` 现在没有硬写普通 Hausdorff，而是区分了 support-function bound、constructive coordinate-Hausdorff bound，以及需要 compact-convex support duality 才能写的 ambient (d_H)。这比上一版严谨很多。

作为 OR 审稿人，我会认为这是文章的理论新意所在。MaxWeight drift 和 Foster recurrence 都是标准工具；真正有你们自己味道的是：

```text
full configuration action space
→ fabric/interference feature metric
→ candidate projection / cover
→ support-capacity loss Lρ
→ robust MaxWeight slack accounting
```

---

### 1.3 Concrete stochastic model 的位置更合理

上一轮我最担心的是 stochastic layer 只是 “drift dominated” 的外部假设。新版 `math.md` 明确说 bounded finite-support arrival/service model 可以推出二阶漂移常数，并把 coordinate arrival/service moment bounds 接到 expected drift；随后 Foster-Lyapunov 推 finite backlog set recurrence。这个方向是对的。

如果 Lean 里的 `main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples'` 确实如 `lean_artifact_map.md` 所列那样存在并通过 build，那这已经解决了我上一轮最大的 theoretical closure concern。

---

### 1.4 你正确处理了 operational necessity

新版没有再写“positive recurrence iff positive slack”。这是非常重要的。`math.md` 现在说 necessity 方向只能推出 zero-slack capacity closure：

[
\lambda \in \overline{\Lambda}^{full}.
]

这在 OR 审稿中会显著降低被打掉的风险。`lean_artifact_map.md` 也把它对应到 `main_operational_conservation_law_necessity` 和 `main_operational_capacity_sandwich`。

我会建议正文里直接叫：

```text
positive-slack sufficiency + zero-slack necessity sandwich
```

不要叫 “capacity characterization iff”，除非你真的补了闭包、不可约性、可稳定策略族、occupation measure tightness 等完整条件。

---

## 2. 非实验部分仍然会被 OR 审稿人追问的地方

### Concern 1：capacity region 应该写成 downward-closed capacity set，而不只是 `conv{μ(a)}`

正文里可以说底座来自 (\operatorname{conv}{\mu(a)})，但严格的 queueing capacity region 更应该写成：

[
\Lambda
=======

{\lambda\ge 0:\exists v\in \operatorname{conv}{\mu(a):a\in\mathcal A},\ \lambda\le v}.
]

也就是 **convex hull 的 downward closure**。原因是服务能力可以浪费，(\lambda) 不需要等于某个 service vector，只要被某个 average service vector coordinate-wise dominated。你们现在 theorem 里实际用的是 coordinate slack：

[
\lambda_i+\delta \le \sum_a x_a\mu_i(a),
]

所以数学是对的；但论文文字里如果只写 (\Lambda=\operatorname{conv}{\mu(a)})，会被挑。`math.md` 已经强调 support slack 而不是 operational iff，这很好，但最终正式定义建议改成 downward-closed set。

---

### Concern 2：positive recurrence 的表述要非常精确

现在主 theorem 说 “integer queue Markov chain positive recurrent via finite backlog sublevel set”。这个可以接受，但 OR reviewer 会问：

```text
是整个 countable-state chain positive recurrent？
还是存在 finite-set recurrent certificate？
是否需要 irreducibility / petite set / closed communicating class？
hitting time 是从所有状态到 finite set，还是 return time from finite set？
```

如果 Lean theorem 证明的是 `PositiveRecurrentViaFiniteSet`，正文就应该定义这个对象，并说明它和标准 positive recurrence 的关系。建议写成：

```text
Theorem proves a finite-set Foster recurrence certificate.
Under the standard irreducibility / single closed communicating class condition, this implies positive recurrence of the queue Markov chain.
```

不要把 Lean 里的 certificate 直接翻译成过强的 “the Markov chain is positive recurrent” 除非你同时给了不可约性或 closed-class 条件。

---

### Concern 3：finite-support stochastic model 适合作为 Lean 闭合模型，但正文最好给 bounded-moment 版本

Lean 里用 finite-support (\Omega) 很自然，因为 expected drift 可以完全形式化。但实际 queueing literature 通常接受更一般的 bounded second moment 或 uniformly bounded arrivals/services。建议正文主定理写成：

```text
bounded conditional second moment + coordinate conditional mean bounds
```

然后附录 / Lean artifact 说明：

```text
The Lean artifact formalizes the finite-support specialization, from which the second-order drift bound is derived constructively.
```

这样不会让 reviewer 觉得模型过窄。你们的 experimental open items 也确实把 (Amax_i,Smax_i,B) 这些 bounded finite-support constants 列成需要校准的量。

---

### Concern 4：`lower_i(a(Q)) ≤ E[S_i|Q]` 是关键假设，必须和 confidence event 绑定

主 theorem 依赖 selected action 的 lower service coordinate-wise dominated by expected service：

```text
lower_i(a(Q)) ≤ E[S_i | Q].
```

这本身没问题，但如果 lower service 来自 learning / posterior / BAPR belief，它不是 deterministic truth，而是高概率事件上的 truth。正文最好把主 theorem 分成两个版本：

```text
Deterministic certificate theorem:
  如果 lower-service domination holds，则稳定。

Learning corollary:
  如果 confidence event E_T 以概率 ≥ 1-δ_conf 成立，
  则在 E_T 上 drift certificate holds。
```

不要把 “lower confidence bound” 和 “true lower bound” 混成一个无条件对象。`experimental_open_items.md` 也已经指出，如果实际系统只有被调度过的配置才有反馈，需要记录 selection probability 或 exploration policy，否则 high-probability concentration 只能是 conditional theorem。

---

### Concern 5：candidate cover 要区分 “full feasible action” 和 “observed/sample action”

你们已经把 (L,\rho) 放到实验 open items 里，这是对的。非实验上仍要在模型定义中明确：

```text
A_full 是什么？
A_cand 覆盖的是所有 feasible actions，还是覆盖 sampled full actions？
cover map π(a) 是否对每个 regime z 都存在？
如果 action feasibility 随 queue state / available jobs / fabric regime 改变，cover 是 uniform 还是 statewise？
```

理论上最好写成：

[
\forall z,\forall x,\forall a\in\mathcal A^{full}(x,z),
\exists \pi_z^x(a)\in\mathcal A^{cand}(x,z)
\quad
d_\Phi(a,\pi_z^x(a))\le \rho.
]

如果正文只写固定 (\mathcal A^{full})，reviewer 可能会问实际 cluster state 改变时 theorem 是否仍适用。你可以先选固定-regime、fixed feasible family 作为主 theorem；但要诚实说 dynamic feasibility 版本需要 uniform cover certificate。

---

### Concern 6：hidden regime extension 现在是合适的，但不要再往主 theorem 里塞

新版把 hidden regime 分成 uniform-in-regime 和 average-regime，这点是正确的。`math.md` 也明确说 average-regime stability 不能偷偷并进主 theorem，需要额外控制 dwell time、switching loss、queue buildup 和 detection delay。

我的建议是：主 paper 只证明 fixed-regime 或 uniform-in-regime corollary。Average-regime 只写成：

```text
Extension / finite-horizon drift budget / future theorem.
```

否则 OR reviewer 会觉得你又把一个难问题压进 assumption 了。

---

### Concern 7：active-bucket learning 是好 extension，但不要让它看起来像完整 learning theorem

`lean_artifact_map.md` 里 active-bucket regret、高概率 lifting、local failure union bound 都已经列出来了。这个很好。

但从 OR 审稿角度，它目前仍然是：

```text
input confidence event ⇒ regret bound event
```

而不是：

```text
specific adaptive sampler ⇒ confidence event ⇒ regret bound.
```

`experimental_open_items.md` 也承认仍需定义 active bucket、feedback model、bucket count、adaptive sampling rule、change-point discount/reset、exploration 与 queue backlog 的耦合。

所以正文里建议叫：

```text
Structured-learning certificate
```

不要叫：

```text
Full online-learning regret theorem for Scheduleurm
```

除非你把 sampler 和 concentration proof 也写完。

---

### Concern 8：artifact map 很有用，但最终审稿包不能只靠 map

`lean_artifact_map.md` 的设计是对的：它列出每个 paper claim 对应的 theorem 名，包括主 theorem、calibration-facing theorem、operational sandwich、hidden-regime extension、active-bucket extension，以及 verification command。

但作为 reviewer，我仍会要求最终 artifact 里有三样东西：

```text
1. lake build Scheduleurm 的完整日志；
2. lake env lean ScheduleurmUpload.lean 的完整日志；
3. rg "\bsorry\b|\badmit\b|\baxiom\b" 的输出。
```

`lean_artifact_map.md` 已经列了这些命令。 但 map 本身不是证明；reviewer 会 grep consolidated file。最终提交时一定要确保 `ScheduleurmUpload.lean` 不是旧版、不是 partial consolidated file，并且 theorem names 与 map 完全一致。

---

## 3. 现在最应该如何写论文贡献

我建议最终 paper contribution 写成三条，不要超过三条主贡献。

第一条：

```text
Configuration-action stochastic processing network for heterogeneous compute fabrics.
```

这里讲全局 action，不是单 job config。它统一 malleable jobs、shareable jobs、gang scheduling、GPU co-location、NUMA/PCIe/NVLink/network placement。`math.md` 对这个定义已经很清楚：服务量依赖 action、hidden regime、共驻、拓扑和随机扰动，而 action 是整个系统的 global allocation pattern。

第二条：

```text
Fabric-cover capacity approximation.
```

主公式：

[
H^{full}(q)\le H^{cand}(q)+L\rho|q|_1.
]

强调这是从 BAPR-HRO 的 keep-and-rerank philosophy 提升到 capacity geometry，而不是简单减少 search space。

第三条：

```text
Robust candidate MaxWeight stability with explicit slack accounting.
```

主公式：

[
\delta>L\rho+\epsilon_{est}+\beta
\Rightarrow
\Delta V(Q)\le B+P_0-\eta|Q|_1.
]

再接 bounded finite-support stochastic model 与 Foster recurrence。

这三条已经足够。`math.md` 也已经提醒不要把 A/B/C/D/E/F/G 都写成同等主 theorem，这个判断应该保留在论文结构里。

---

## 4. 我会给的审稿式评价

如果我是 OR reviewer，我现在会写：

> The revised manuscript has substantially improved its theoretical positioning. The paper no longer presents a local candidate-scoring rule as the main contribution; instead, it develops a configuration-action stochastic processing network and connects full-action support slack, fabric-cover approximation, robust candidate MaxWeight drift, and a concrete bounded finite-support stochastic model into a single stability theorem. The candidate-cover support approximation is the most original mathematical contribution, since it turns a keep-and-rerank principle into an explicit capacity-loss bound. The operational capacity sandwich is also much more credible than an unconditional stability iff claim.
>
> The remaining issues are mostly about precision rather than direction. The capacity region should be defined as a downward-closed service region, not merely a convex hull. The positive-recurrence claim should be stated as a finite-set Foster certificate unless irreducibility or closed-class assumptions are added. The lower-service domination and active-bucket regret results should be clearly presented as conditional certificates unless the learning/sampling mechanism is fully specified. Finally, the Lean artifact map is helpful, but the final submission must include a buildable consolidated Lean file and verification logs.

中文总结就是：

**除实验外，这版理论已经“站住了”，但还没到可以不加修改直接投 OR。剩下的问题不是方向错，而是要把 capacity 定义、positive recurrence 口径、learning confidence event、candidate cover 的 state/regime 依赖、Lean artifact verification 写得更像正式论文。**

---

## 5. 评分更新

我现在会这样打分：

```text
数学问题 framing：9/10
主 theorem 结构：8.5/10
candidate-cover 近似贡献：8.5–9/10
robust MaxWeight drift：8/10
concrete stochastic stability closure：8/10
hidden-regime / learning extensions：7–7.5/10
Lean artifact organization：7.5–8/10，取决于最终 buildable file
OR flagship readiness excluding experiments：7.5–8/10
完整论文潜力：8.5/10+
```

最终建议：

```text
Decision if submitted now without cleanup: major revision.
Decision after fixing theorem wording + artifact verification + notation: minor revision / serious borderline accept on theory.
```

如果目标是 **Stochastic Systems / Queueing Systems / INFORMS JOC**，理论路线已经相当合适。若目标是 **Operations Research flagship**，除了实验，还需要把 operational insight 写得更普适：也就是明确说明这个 theorem 给管理异构 compute fabric 的决策者带来什么结构性结论，而不仅仅是证明某个 scheduler 稳定。
