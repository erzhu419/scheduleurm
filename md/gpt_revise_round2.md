结论：**新版路线明显更像 OR 论文了，但我作为审稿人现在会给 “major revision / promising but not yet accept”。**

最重要的进步是：新版 `math.md` 已经把问题从“BAPR-HRO 风格的候选打分”提升成了真正的 **configuration-action stochastic processing network**，并且明确说这不是无条件 throughput-optimality claim，而是一条 proof spine：先做 stationary-mix / support-function capacity base，再接 candidate-restricted robust MaxWeight drift，再在具体 stochastic arrival/service model 里证明 conditional drift，最后用 Foster-Lyapunov 推 finite-small-set recurrence。这个方向比上版稳很多。

但我也发现一个关键问题：**`math.md` 里声称 Lean 已经对应了一些新版 `main_*` theorem，但我本地检查上传的 `ScheduleurmUpload.lean` 没有这些 `main_*` 声明。** 上传的 Lean 文件里没有 `main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples'`、`main_candidate_restricted_capacity_coordinate_hausdorff`、`main_operational_conservation_law_necessity`、`main_hidden_regime_dwell_switching_drift`、`main_active_bucket_lcb_learning_regret_high_probability` 这些名字。它可能是上传了旧版 Lean，或者 `math.md` 写的是计划中的证明名。如果我是审稿人，这会是 **major consistency issue**。

---

## 1. 总体评分

我会这样打分：

| 维度                      |       评分 | 评语                                                                             |
| ----------------------- | -------: | ------------------------------------------------------------------------------ |
| 研究问题重要性                 |   8.5/10 | 异构 compute fabric + shareable/malleable jobs + hidden regime 是真实且有 OR 味的问题。    |
| 数学路线                    |   8.5/10 | 新版路线已经从“score identity”升级到 support capacity、MaxWeight drift、Foster recurrence。 |
| 当前 `math.md` 完成度        |     8/10 | 比上版强很多，特别是避免了不严谨的 operational iff。                                             |
| 上传 Lean 与 `math.md` 匹配度 |     6/10 | Lean 证明骨架认真，但没有覆盖 `math.md` 声称的新版 main theorem 名称。                             |
| 当前 OR 投稿成熟度             | 6.5–7/10 | 已经不是普通系统 proposal，但还不是可直接接收的 OR paper。                                         |
| 修正后潜力                   |   8.5/10 | 若补齐 Lean/正文一致性、具体 stochastic theorem 和实验校准，潜力很高。                               |

我的审稿意见会是：

```text
Decision: Major revision.
The direction is substantially stronger than the previous version, but the proof claims, Lean artifact, and operational stochastic model must be aligned before the paper can be considered mature.
```

---

## 2. 新版最强的地方

### 2.1 你已经修正了上版最大的问题：不再乱说 capacity iff stability

新版 `math.md` 很关键的一步是说：capacity base 先只做 support-function 形式，不直接宣称 “λ 在 capacity region 内 iff 队列稳定”。它明确写出：

[
q^\top\lambda+\delta|q|_1 \le H^{full}(q)
]

并说明 operational stability iff 还需要 concrete stochastic arrival/service model、conditional expected Lyapunov drift domination、以及 conservation-law necessity。

这是 OR 审稿人会喜欢的，因为它避免了上版最容易被打掉的过度宣称。

### 2.2 Candidate-set approximation 现在是主贡献，而不是配角

新版把 BAPR-HRO 的 “keep structure, re-rank candidates” 数学化为：

[
H^{full}(q) \le H^{cand}(q)+L\rho|q|_1
]

并且进一步解释 candidate set 不是把 action space 偷偷缩小，而是 full action space 的可计算 cover，cover loss 显式进入 slack accounting。这个方向很强。

这也是我认为最像原创 OR 贡献的部分。标准 MaxWeight drift 是经典套路；**真正有新意的是把 configuration hypergraph 的候选近似误差转成 capacity/support-function loss。**

### 2.3 Robust MaxWeight drift 的 slack accounting 写对了

新版 Theorem C 的形式是：

[
\delta>\epsilon_{cand}+\epsilon_{est}+\beta
]

则

[
\Delta V(Q)
\le
B+P_0-
(\delta-\epsilon_{cand}-\epsilon_{est}-\beta)|Q|_1.
]

这比“我们用了 robust score 所以稳定”严谨很多，因为 candidate approximation、lower-service estimation error、switching/risk penalty 都显式吃掉 capacity slack。

作为审稿人，我会认为这是主 theorem 应该长的样子。

### 2.4 Concrete finite-support stochastic model 是正确方向

新版 Theorem D 试图把 stochastic model 具体化：有限样本空间、bounded arrival/service samples、conditional arrival mean、conditional selected service mean、由 bounded samples 推出二阶漂移常数，再接 Foster-Lyapunov。

这比上版 “假设 drift dominated” 强。OR 审稿人会要求你把这个从文字变成正式 theorem；但路线是对的。

---

## 3. 现在最大的硬伤：`math.md` 与 Lean artifact 不一致

这是我最想强调的。

新版 `math.md` 说当前非实验部分已经进一步闭合到：

```text
concrete finite-support stochastic model
zero-slack conservation-law necessity interface
constructive coordinate-Hausdorff candidate capacity set bound
dwell/switching budget
active-bucket high-probability lifting
```

并且列了一批 Lean 对应 theorem 名。

但是上传的 `ScheduleurmUpload.lean` 里我看到的是旧版结构：

```text
candidate_cover_support_gap
metric_cover_support_gap
capacity_slack_implies_support_slack
candidate_capacity_slack_loss
approximate_maxWeight_lyapunov_drift
robust_candidate_policy_lyapunov_drift
robust_candidate_markov_finite_expected_hitting_time
lcb_learning_regret_from_concentration_events
metric_cover_capacity_metric_hausdorff
```

这些本身有价值，但不是 `math.md` 里声称的最新版闭合 theorem。

尤其下面几个不一致很严重：

1. **`math.md` 说 Lean 有 concrete bounded finite-support stochastic stability theorem；上传 Lean 仍然主要通过 `hdominated` 这种外部假设接 stochastic model。**
   这说明 “bounded samples 推出 conditional drift” 还没有在上传 Lean 中真正闭合。

2. **`math.md` 说 necessity 只能是 zero-slack capacity closure；上传 Lean 里 operational necessity 的旧接口仍然像是任意 `δ` 的 positive slack。**
   这在数学上太强，通常是假的。新版 `math.md` 的说法是对的：operational stability 最多推出 closure / zero-slack capacity membership，不能推出任意正 slack。

3. **`math.md` 说有 constructive coordinate-Hausdorff candidate capacity set bound；上传 Lean 里更像是 support-distance theorem + metric Hausdorff theorem under a duality assumption。**
   support-function 版本是够用的，但如果正文说“constructive coordinate-Hausdorff push-forward of stationary mixes”，Lean 里需要真有这个 theorem。

4. **`math.md` 说 active-bucket regret 用 `active.card`；上传 Lean 里我看到的是 generic `Bkt` cardinality envelope，不是 active subset high-probability theorem。**
   这不是致命，但叙事不能说已经解决了 active-bucket union-bound 问题。

所以，我作为审稿人会要求：

```text
Either upload the actual latest Lean file,
or downgrade the text claims to match the uploaded Lean.
```

否则这会被看成 artifact inconsistency。

---

## 4. 我对 Lean 当前证明的评价

### 好的部分

上传 Lean 没有明显 `sorry` / `admit` / `axiom`。这点很好。

它形式化了几个真正有用的东西：

```text
Candidate cover → support-function loss
Capacity slack → support slack
Approximate MaxWeight → Lyapunov drift
Robust candidate score → approximate full support
Foster drift → finite expected hitting time
Integer queue finite small set
LCB confidence event → deterministic regret envelope
Support-distance Hausdorff boundary
```

这已经不是“装饰性 Lean”。它确实证明了 proof spine 里的一大段 algebra。

### 不足部分

但它现在更像：

```text
deterministic algebra + abstract Markov/Foster interface
```

还不是：

```text
full concrete stochastic queueing theorem
```

尤其是 stochastic layer 仍然需要外部给：

```text
conditional drift domination
lower service ≤ true conditional service
confidence event
finite-small-set return / irreducibility condition
support/Hausdorff duality bridge
```

这些作为 proof interface 是合理的，但不能包装成已经完整证明了真实 scheduler 的 positive recurrence。

---

## 5. OR 审稿人会抓的 major concerns

### Major Concern 1：主定理必须合并成一个可读的 paper theorem

现在 `math.md` 有 A/B/C/D/E/F/G，很好，但论文里不能让主贡献分散。建议把主 theorem 写成一个完整版本：

```text
Theorem 1: Robust candidate MaxWeight stability under fabric-cover approximation.

Assume:
1. finite job classes and finite full configuration action family;
2. A_cand ⊂ A_full is a ρ-cover under d_Φ;
3. μ is L-Lipschitz under d_Φ;
4. λ has full-action support slack δ;
5. lower-service model has support error ε_est;
6. switching/risk penalty ≤ P0 + β||Q||1;
7. finite-support stochastic arrival/service samples satisfy boundedness and coordinate mean dominance.

If δ > Lρ + ε_est + β,
then the queue process has negative Foster drift and finite expected hitting time to a finite backlog set.
```

这条 theorem 必须一口气把 `Lρ`、`ε_est`、`β`、`P0`、`B` 都接起来。现在 `math.md` 里已经基本有这个结构。

### Major Concern 2：capacity-region theorem 不能成为“标准 SPN 复述”

`conv{μ(a)}` 本身是标准 stochastic processing network 的 capacity geometry。新意不在这里，而在：

```text
global configuration action
fabric/interference metric
candidate cover approximation
robust lower-service support error
bounded/queue-scaled penalty
hidden-regime extension
```

所以正文里要避免把 “capacity region is convex hull” 说成主要创新。它是底座。主创新应该是：

```text
candidate-restricted robust MaxWeight retains stability with explicit capacity loss Lρ + ε_est + β
```

### Major Concern 3：`d_Φ`、`L`、`ρ` 不能只是假设

这个理论最漂亮的地方也是最脆的地方。你说：

[
d_\Phi(a,a')=\sum_r w_r|\Phi_r(a)-\Phi_r(a')|
]

其中 features 可以是 placement、GPU co-location profile、NUMA/PCIe/NVLink path、memory pressure、network path class、rollback/preemption attribute。

审稿人会问：

```text
Φ 是谁定义的？
为什么 μ 对 d_Φ Lipschitz？
L 怎么估？
ρ 怎么保证？
如果 co-location interference 非连续、非平滑，Lρ 是否巨大？
```

这部分必须靠 scheduleurm profiling / perturbation experiments 支撑。理论可以假设 Lipschitz，但论文必须说明这是可测、可校准、可 falsify 的工程假设，而不是 free assumption。

### Major Concern 4：hidden regime 现在应该放 extension，不要塞进主 theorem

新版区分 uniform-in-regime stability 和 average-regime stability 是正确的。它也明确说 average-regime 不能直接说稳定，需要额外控制 dwell time、switching loss、queue buildup 和 detection delay。

我建议主 paper 只证明固定 regime 或 uniform-in-regime 版本。Average-regime stability 放成 extension 或 conjecture-level theorem。否则文章会过满。

### Major Concern 5：structured learning 还没有到 OR 主 theorem 级别

新版说不要写：

[
\tilde O(\sqrt{|\mathcal A||\mathcal Z|T})
]

而应写 active buckets / semi-bandit factors：

[
\tilde O(\sqrt{|\mathcal B_{active}|T}+N_{cp}\log T+\text{switching})
]

这是对的。

但要成为 OR 主定理，还需要清楚说明：

```text
feedback 是 full-information、semi-bandit、bandit 还是 censored feedback？
同一 action 运行后能观测哪些 job-class service components？
bucket 是固定的还是 adaptive generated？
change point 后旧样本如何 discount？
queue backlog 与 exploration 如何耦合？
```

目前这更适合作为第二篇或 extension。主论文先把 stability + candidate approximation 做扎实。

---

## 6. 我会建议你这样改论文结构

### 主标题方向

不要叫 universal scheduler。更像：

```text
Configuration-Based Queueing Control for Malleable and Shareable Jobs on Heterogeneous Compute Fabrics
```

### 主贡献只保留三条

第一条：

```text
Global configuration-action SPN model for heterogeneous compute fabrics.
```

强调一个 action 可以同时编码：

```text
多 CPU 跑一个任务
一个 CPU 多任务共享
多 GPU gang scheduling
多任务共享一张 GPU
跨 NUMA / PCIe / NVLink / Ethernet / WAN placement
```

第二条：

```text
Candidate fabric-cover approximation theorem.
```

核心是：

[
H^{full}(q)\le H^{cand}(q)+L\rho|q|_1.
]

第三条：

```text
Robust candidate MaxWeight stability theorem.
```

核心是：

[
\delta>L\rho+\epsilon_{est}+\beta
\Rightarrow
\Delta V(Q)\le B+P_0-\eta|Q|_1.
]

然后 stochastic finite-support Foster recurrence 接在同一个 theorem 里。

### 扩展放后面

```text
hidden-regime dwell/switching
structured active-bucket learning
sweet-spot admission threshold
```

这些都可以放，但不要让它们抢主线。Sweet spot 很有 operational insight，但它依赖 profiling 证明 unimodality / submodularity；新版也已经把它定位成 extension + experiment-driven theorem，这是对的。

---

## 7. 对 “能不能达到 OR 标准 9/10 数学深度” 的判断

我的判断变成：

```text
路线：接近 9/10
当前可审稿数学：7.5–8/10
当前 Lean artifact 支撑：6–6.5/10
修正 Lean/正文一致性后：8–8.5/10
若再补实验校准 Lρ、ε_est、β：有机会到 9/10
```

这已经比上一版强很多。上一版主要是“好 proposal + proof skeleton”；这一版 `math.md` 已经知道 OR 审稿人会抓什么，并主动避开了几个大坑：

```text
不再无条件宣称 operational iff；
不再把 DRO score 当主贡献；
不再枚举 |A| regret；
不再把 hidden regime 混成一个 theorem；
不再让 candidate set 缩小 full action space 而不记账。
```

这些都是明显进步。

---

## 8. 最终审稿式评价

我会写成：

> The revised manuscript makes a substantial conceptual improvement. The paper now frames the scheduler as a configuration-action stochastic processing network rather than a candidate-scoring problem. The support-function capacity base, fabric-cover candidate approximation, and explicit slack accounting in robust candidate MaxWeight are the right mathematical objects for an OR contribution. The move from operational capacity “iff” claims to a support/diff/Foster proof chain is especially important.
>
> However, the current proof artifact and the mathematical route are not yet aligned. The manuscript claims several closed Lean theorems for concrete finite-support stochastic stability, zero-slack operational necessity, coordinate-Hausdorff candidate capacity approximation, dwell/switching hidden-regime control, and active-bucket high-probability learning. The uploaded Lean file I inspected does not contain these `main_*` results and still appears to rely on abstract drift-domination, local-return, concentration-event, and duality interfaces. This is not fatal, but the claims must be downgraded or the actual latest Lean development must be supplied.
>
> My recommendation would be major revision. The authors should collapse the theory into one main theorem: full-action support slack plus fabric-cover loss plus lower-service estimation error plus bounded/queue-scaled penalties imply negative Foster drift and positive recurrence under a concrete bounded finite-support arrival/service model. The candidate-cover theorem is the most original mathematical piece and should be emphasized as the BAPR-HRO “keep-and-rerank” principle lifted to capacity geometry.

中文一句话：

**新版已经从“能写成论文的想法”进化到“像一篇 OR 数学论文的骨架”；但要真正到 OR 9/10，必须解决 Lean/正文不一致，并把 concrete stochastic stability、zero-slack necessity、coordinate capacity approximation 这些从文字路线变成可核查的正式 theorem。**
