下面我按 **OR/排队论/调度理论审稿人** 的口径评。结论先说：

**这个方向有论文潜力，但现在不是“已达到 OR 数学 9/10”，而是“有 8.5–9/10 数学路线的 proposal”。**
`math.md` 的建模方向是对的，尤其是把问题提升为 **hidden-regime, configuration-action stochastic processing network**，并把 BAPR/BAPR-HRO 分别放进 belief filtering 与 robust candidate ranking 层；但它目前更像一套强数学论文的蓝图，而不是已经闭合的 OR 级数学结果。`ScheduleurmUpload.lean` 的形式化比普通“伪证明”严肃很多，但它形式化的是 **有限 action、support function、candidate approximation、drift algebra、Foster certificate skeleton**，还没有真正证明完整的随机调度系统在真实 arrival/service process 下 throughput-optimal 或 positive recurrent。

---

## 1. 对现有 scheduleurm 的问题定位

从当前 GitHub README 看，scheduleurm 是一个面向 ML training 的多资源调度器，支持 CPU/RAM/VRAM、异构本地/远程节点、resume、外部进程自动 adopt、历史资源估计等；README 还明确说当前 `dispatch` 是 greedily fills capacity，并且 RL nodes 可以忽略 GPU util、按 “1/3+grace VRAM rule” 打包，而 strict nodes 可以保留 GPU util saturation guard。([GitHub][1])

这和你在 `GPT.md` 里的问题意识一致：它现在最像一个 **工程上可用的 greedy multi-resource scheduler**，而不是一个已经有 queueing-control 理论的 scheduler。讨论里也已经抓住关键：你真正的问题不是“谁先谁后”，而是多任务共驻导致服务率本身变；最需要的是 admission control、GPU sweet spot、rollback 和 effective throughput feedback。`GPT.md` 里把它归到 “state-dependent limited processor sharing queue with dynamic admission/concurrency control” 是合理的。

作为 OR reviewer，我会认为：**工程系统是好起点，但论文贡献不能是“我写了一个调度器”。** 必须抽象出一个新的 stochastic processing network / queueing control 模型，再证明结构性质或稳定性。你们前面的讨论也已经明确：如果只是服务器 benchmark，更像 MLSys/SC/HPDC/SoCC；如果能提出 configuration-based SPN、证明 Goodput-MaxWeight throughput optimality、sweet-spot/admission threshold、unknown-service learning regret/stability，才有 OR/Stochastic Systems/INFORMS JOC 潜力。

---

## 2. `GPT.md` 的评价：问题 framing 很好，但不是数学贡献本身

`GPT.md` 最强的地方有三点。

第一，它没有误判成“没人做”。它明确承认 AlloX、Gavel、Pollux、Sia、IADeep、Salus、MaxWeight/UCB 等都覆盖了局部，但统一地把 **malleable jobs、shareable jobs、CPU/GPU interchangeable resources、GPU co-location interference、拓扑网络、hidden regime、online learning** 放到一个 OR queueing-control 框架里仍有空间。

第二，它正确地把 BAPR/BAPR-HRO 降级为模块，而不是完整解法：BAPR 负责 regime belief / nonstationary adaptation，BAPR-HRO 负责固定候选集的 robust re-ranking；真正完整的 configuration scheduler 还需要一个 MaxWeight / ILP / matching / MPC 之类的 OR 层。

第三，它对 BAPR-HRO 的自我批判比较成熟：BAPR-HRO 的关键 insight 是 “keep structure, re-rank candidates”，但 Wasserstein DRO 和 Lean verification 只是 posterior-risk core 的 supporting layer；完整算法收益很大一部分来自 structural reliability terms，而不是 DRO core 本身。

我的评价是：

```text
GPT.md 作为研究 framing：8/10
GPT.md 作为 OR 数学贡献：5/10
```

它能帮助写 introduction、related work、problem motivation，但还不是 theorem-level contribution。

---

## 3. `math.md` 的核心建模：方向对，而且是 OR 标准路线

`math.md` 的主张是把问题建模为：

> hidden-regime, configuration-action stochastic processing network

这比“多场景模式识别”强很多。文件里明确说：BAPR 是 hidden regime posterior/filtering 层，BAPR-HRO 是 candidate configuration robust ranking 层；真正的数学问题是带未知状态相关服务率、共驻干扰、拓扑约束和切换成本的鲁棒排队控制。

我认为这是正确方向。尤其是下面这三个建模选择很重要。

**第一，action 是全局 configuration pattern，而不是单个 job 的局部配置。**
`math.md` 里明确说不要把动作写成 `c_j = job j 的配置`，而是写成 `a ∈ A`，其中 `a` 可以同时表示 GPU0 上 4 个 RL 任务共驻、GPU2-5 上一个 4-GPU job、CPU socket0 上 rollout workers、NIC 上跨主机 all-reduce 等；这样四个象限都统一进一个 action set。

**第二，资源不是简单向量，而是 resource fabric graph。**
`math.md` 里把 CPU socket、NUMA memory、GPU、GPU memory、unified memory、NIC、host、cluster 作为节点，把 PCIe、NVLink、CXL、10GbE、InfiniBand、WAN 等作为边；并且强调 GPU 共驻不能只用线性 capacity constraint，因为共驻会改变服务率，应该进入 `μ_i^z(a)`。

**第三，核心 theorem 不是 DRO score，而是 capacity/stability/approximation。**
`math.md` 明确把 9/10 数学贡献定位为 capacity region characterization、belief-robust MaxWeight stability、candidate-set approximation，而不是只证明一个 robust score 等价。

所以我会给 `math.md` 这个数学设想的潜力：

```text
数学路线潜力：8.5–9/10
当前写法完成度：7–7.5/10
```

原因是：它提出了 OR reviewer 会认真看的对象，但还没有把 theorem 的 assumptions、proof obligations、novelty boundary 全部收紧。

---

## 4. `math.md` 是否已经达到 OR 标准型 9/10 数学深度？

**还没有。**

它现在是一个很好的 **9/10 数学论文设计图**，但不是已经完成的 9/10 数学。OR 9/10 需要的是类似下面这种闭合链条：

```text
定义 model
→ 定义 capacity region
→ 证明 capacity necessary/sufficient
→ 设计 policy
→ 证明 throughput optimality / stability
→ 处理 approximation / learning / hidden regime
→ 给出 operational insight
```

`math.md` 里这些都提到了，但很多还是 “should prove / 可以证明 / theorem should look like” 的层面。比如 robust capacity region 被定义为 posterior ambiguity 下的可保证区域，Theorem B 说如果 λ 在 robust capacity region 有 slack，则 positive recurrent；但要成为 OR 论文，必须清楚说明：

```text
arrival process 是 i.i.d.、stationary ergodic、还是 adversarial?
service random variables 是否 bounded?
service vector 是否 independent of Q?
action set finite 还是 compact?
switching cost K 是否 bounded?
risk penalty R 是否 bounded，还是会随 Q 增长?
hidden regime 是 exogenous，还是受调度动作影响?
BOCD detection delay 是 assumption，还是 theorem?
posterior lower bound 是 high-probability event，还是 almost-sure eventually valid?
```

如果这些不定，reviewer 会说：**这是一个正确但过宽的 SPN template。**

特别要注意：`math.md` 里最容易被抓的是 “capacity region = conv{μ(a)}” 这一点。这个在有限 action、stationary service、可 randomized scheduling 的 SPN 里是标准事实；新意不在它本身，而在：

```text
1. configuration action 如何统一 shareable + malleable + topology；
2. candidate subset approximation 如何保留 capacity；
3. hidden-regime / learning / robust lower service 如何不破坏 stability；
4. GPU co-location sweet spot 如何产生 threshold 或 submodular admission structure。
```

如果论文只证明 `Λ = conv{μ(a)}`，数学深度不会高。要把它推到 OR 9/10，必须把 `conv{μ(a)}` 作为底座，主打 **candidate-restricted robust MaxWeight under hidden unknown state-dependent service**。

---

## 5. Lean 文件总体评价

我检查了 `/mnt/data/ScheduleurmUpload.lean`，共 2815 行。容器里没有 `lean` 或 `lake` 可执行文件，所以我不能实际编译验证；以下评价基于逐行审查、grep declaration/suspicious keywords 和 theorem statement 检查。

好消息是：文件里没有明显的 `sorry`、`admit`、`axiom`。它确实是认真写的 Lean 文件，不是空壳。它导入 Mathlib，定义了 `ServiceVec`、`dot`、`l1`、`ActionFamily`，并且把全局 action 明确解释为可以编码四种计算调度象限。Lean 文件开头第 29–36 行就写明：一个 `a : A` 可以表示 one job many CPUs、many jobs sharing CPU、one job many GPUs、many jobs co-located on one GPU。

但作为 OR 证明，我会把它分成三类。

### 5.1 真正有价值的形式化部分

Lean 文件对 **support-function algebra** 做得比较扎实。它定义：

```lean
support F μ q = max_{a ∈ F} qᵀ μ(a)
```

并证明了 candidate support gap、metric cover + Lipschitz service → support loss：

```lean
support full μ q ≤ support cand μ q + ε * l1 q
```

这对应 `math.md` 里的 candidate-set approximation theorem。相关核心定理在 Lean 文件第 278–291 行、第 324–365 行。这个部分是有价值的，因为它把 BAPR-HRO “keep candidate structure, re-rank” 的思想转成了 support-function loss。

Lean 文件还形式化了容量区域的 support consequence：如果 `λ + δ·1` 被某个 stationary mix 支持，则对所有 nonnegative queue vector `q` 有：

```lean
qᵀ λ + δ ||q||₁ ≤ H_A(q)
```

这在第 405–435 行定义 `StationaryMix`、`InCapacityWithSlack`，第 500–513 行证明 `capacity_slack_implies_support_slack`。这部分是 MaxWeight drift 证明的正确基础。

第三，它形式化了 queue drift algebra。第 596–627 行给出标准 queue-step square bound 和 Lyapunov drift inequality；第 652–680 行把 approximate MaxWeight loss 接到 negative drift：

```lean
V(Q⁺) - V(Q) ≤ B + cost - (δ - ε) ||Q||₁
```

这是排队论 proof skeleton 的核心。

### 5.2 有价值但仍是“证明框架”的部分

Lean 文件第 1063–1148 行定义了 robust-score maximizer，并证明：

```lean
candidate gap + estimation gap + bounded penalty
→ approximate full MaxWeight
→ Lyapunov drift bound
```

这个和 `math.md` 的 BAPR-HRO-to-MaxWeight 拼接是对应的。问题是它仍然是 deterministic algebra：`hlower_gap`、`hgap`、`hSecond`、`hcap`、`hmax` 都作为假设输入。它证明的是“如果这些假设成立，则 drift bound 成立”，不是证明这些假设在 scheduleurm 的真实 stochastic system 中成立。

第 1599–1828 行形式化了一个抽象 `TransitionExpectation`，然后证明 Foster negative drift implies finite expected hitting time / positive recurrence via finite set。这个比普通 proof sketch 认真很多，尤其它承认 positive recurrence 还需要 finite-set local return assumption。Lean 文件在注释里也说明 local return 是必须由具体 Markov-chain model 提供的额外条件。

第 2189–2250 行定义 `RealQueueTransitionModel` / `NatQueueTransitionModel`，把真实随机模型的关键边界放在 `drift_dominated` 字段里。这很诚实，但也说明真正的随机服务/到达模型还没被 formalized。

### 5.3 最弱的部分：capacity characterization 和 stochastic theorem 还没有真正闭合

Lean 文件里所谓：

```lean
configuration_capacity_region_characterization
```

在第 432–437 行只是：

```lean
InCapacityWithSlack F μ lam δ ↔ StabilizableByStationaryMix F μ lam δ := by
  rfl
```

也就是说它是定义同义反复，而不是“capacity region iff stabilizable queue”的非平凡 theorem。

后面第 2337–2362 行定义了：

```lean
CapacityRegionOperationalSound
CapacityRegionOperationalNecessary
configuration_capacity_region_operational_characterization
```

但 theorem 的内容是：如果你假设 soundness 和 necessity，那么得到 iff。这是逻辑包装，不是证明 necessity/sufficiency。它在形式上完全正确，但 OR reviewer 不会把它当作 capacity-region theorem 的证明。

第 2790–2811 行的 metric Hausdorff theorem 也类似。它把 `CapacitySupportMetricHausdorffDuality` 定义成 assumption，然后在有这个 duality assumption 时推出 metric Hausdorff bound。这是合理的边界标注，但它说明完整的 convex geometry bridge 没有在 Lean 中证明。

所以 Lean 的审稿评价是：

```text
作为 proof engineering：7.5/10
作为 math.md 完整 theorem 的形式化证明：5.5–6.5/10
作为 OR reviewer 会认可的最终数学证明：还不够
```

---

## 6. Lean 和 `math.md` 的覆盖关系

`math.md` 说 9/10 核心是三个 theorem：

1. configuration capacity region theorem
2. belief-robust MaxWeight stability theorem
3. candidate-set approximation theorem


Lean 对这三个的覆盖程度如下：

| math.md 目标                      | Lean 覆盖 | 审稿意见                                                                                                                                                                                |
| ------------------------------- | ------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Configuration capacity region   |      部分 | Lean 形式化了 stationary mix 与 support consequence；但 “capacity iff operational stabilizable” 没有证明，只是假设 soundness/necessity 后包装。                                                         |
| Robust MaxWeight stability      |      中等 | Drift algebra、approximate MaxWeight、Foster certificate 都有；但 stochastic arrival/service model、conditional expectation domination、positive recurrence 的 concrete verification 仍是外部假设。 |
| Candidate-set approximation     |      较强 | support-function 版本比较完整；metric Hausdorff 版本依赖 duality assumption；这是目前最像真正 contribution 的部分。                                                                                         |
| Hidden-regime BAPR layer        |      较弱 | Lean 只证明 joint belief marginalization 和 lower service ≤ belief-weighted service；没有 BOCD detection theorem、piecewise-stationary filtering consistency 或 detection delay bound。       |
| Unknown-service learning regret |   较弱到中等 | Lean 给了 LCB confidence shell 和 sqrt-envelope algebra；但 concentration/adaptive sampling/event probability 是假设，不是从数据过程推导。                                                             |
| Sweet spot threshold            |      较弱 | 单维 unimodal threshold 证明太简单；更有价值的 multiclass/submodular admission 只到边际单调性层面。                                                                                                        |

---

## 7. 作为 OR reviewer，我会给出的主要批评

### Major Concern 1：现在的 theorem 很多是 “if assumptions, then conclusion”

这不是坏事，但要避免把它包装成已经证明了完整调度系统稳定性。Lean 文件很诚实地把关键概率边界写成 `drift_dominated`、`hreturn`、`CapacityRegionOperationalSound`、`CapacityRegionOperationalNecessary`、`CapacitySupportMetricHausdorffDuality` 等 assumption。审稿人会问：

> 这些 assumption 在 scheduleurm 的实际系统中是否成立？
> 是 theorem，还是 modeling assumption？
> 如果是 assumption，有没有 empirical validation 或可检验条件？

### Major Concern 2：`K(a_prev,a)` 和 `R(a)` 可能破坏 throughput optimality

`math.md` 的 policy 是：

```text
argmax_a Q(t)^T μ_lower(a) - K(a_prev,a) - R_t(a)
```

如果 `K` 和 `R` 是 uniformly bounded，那么 MaxWeight stability 通常还能保住，因为大 backlog 时 `Q^T μ` 主导。但如果 `R_t(a)` 会随 queue、network congestion、rollback risk 或 uncertainty 放大，稳定性条件必须重写。Lean 里目前把 penalty 用 `Pmax` bounded 处理，这是对的，但论文正文必须明确这一点。

### Major Concern 3：hidden regime 不能只作为 posterior 加权

`math.md` 里 hidden regime 是 piecewise stationary，并使用 BAPR-style belief。
但如果 regime 变化会改变 capacity region，稳定性有两种完全不同的版本：

```text
uniform-in-regime stability: λ 在每个 regime 的 capacity region 内都有 slack
average-regime stability: λ 只在长期混合 capacity region 内有 slack
```

前者容易但保守；后者需要对 regime dwell time、switching process、queue buildup 做更强分析。现在 `math.md` 两种都提到了，但没有选定主 theorem。

### Major Concern 4：candidate-set approximation 是最有新意的部分，但还要更 operational

`math.md` 的 candidate-set theorem 很漂亮：如果 candidate action set 是 full action set 的 ε-cover，且 service Lipschitz，则 capacity regions 的 Hausdorff gap ≤ Lε。
Lean 也较好地证明了 support-function 版本。

但 OR reviewer 会继续问：

```text
这个 topology/interference metric d(a,a') 怎么定义？
为什么 μ 对它 Lipschitz？
candidate set 是怎么生成的？
工程中如何保证 cover radius ρ？
Lρ 小到有实际意义吗？
```

如果这些只是假设，这个 theorem 会被看成“漂亮但空”。要变强，需要把 `d` 设计成可测的 fabric distance + co-location profile distance，并用 profiling / perturbation experiment 验证 Lipschitz envelope。

### Major Concern 5：learning regret 的 |A| 可能巨大到 vacuous

`math.md` 里提到 regret 形如：

```text
O~(sqrt(|A||Z|T) + N_cp log T + switching)
```

这在全局 configuration action set 上可能完全不可用，因为 `|A|` 是组合爆炸的。Lean 里有 bucket concentration shell，但没有解决组合 action 的统计结构。

若要 OR 级，你需要一种结构化学习假设，例如：

```text
service model factorizes by resource graph；
co-location interference is low-rank / pairwise / submodular；
semi-bandit feedback observes per-job throughput；
only local neighborhood actions are explored；
candidate hypergraph has bounded degree。
```

否则 reviewer 会说 regret bound is formally true but practically vacuous。

---

## 8. 我会建议把论文主线收窄

不要把论文写成：

> universal scheduler for all CS workloads

而写成：

> **Configuration-Based Queueing Control for Malleable and Shareable Jobs on Heterogeneous Compute Fabrics**

主 theorem 不要贪多，建议只保留三条主线：

### Theorem 1：Candidate-restricted capacity approximation

这是最像你们自己的贡献。形式：

```text
If A_cand is a ρ-cover of A_full under topology/interference metric d,
and μ is L-Lipschitz in d,
then support/capacity loss ≤ Lρ.
```

这把 BAPR-HRO 的 “keep candidate structure, re-rank” 变成 OR theorem。

### Theorem 2：Robust candidate MaxWeight stability

形式：

```text
If λ lies in full capacity region with slack δ,
candidate approximation loss + estimation loss + bounded penalty < δ,
then robust candidate MaxWeight has negative Lyapunov drift.
```

Lean 里已经很接近这个 theorem。

### Theorem 3：Operational stochastic stability under verified drift

形式：

```text
For a concrete bounded-arrival/service Markov model,
if conditional drift is dominated by the deterministic drift bound
and finite-small-set return holds,
then positive recurrence / finite expected hitting time follows.
```

Lean 里已经有 certificate 层，但论文要补 concrete stochastic model，而不是只把它作为 `drift_dominated` assumption。

Hidden regime 和 learning 可以先作为 extension，不要都塞进主 theorem。否则 paper 会显得过满。

---

## 9. 对“OR 标准型 9/10 数学深度”的具体评分

我的评分如下：

```text
问题重要性：8.5/10
OR/queueing fit：8/10
数学建模方向：8.5/10
当前 math.md 完成度：7–7.5/10
Lean proof engineering：7.5/10
Lean 对完整 OR theorem 的覆盖：5.5–6.5/10
当前可投 OR 主刊成熟度：5.5–6/10
强修后 OR/Stochastic Systems/JOC 潜力：7.5–8.5/10
```

能不能到 9/10？可以，但前提是把现在的 proposal 变成以下闭合结果：

```text
1. 明确定义一个具体 stochastic queueing model；
2. 证明 candidate-restricted robust MaxWeight 的 throughput stability；
3. 证明 candidate set approximation 有非平凡、可解释的 capacity loss；
4. 证明或验证 hidden regime / learning 的 error 不吞掉 capacity slack；
5. 用 scheduleurm 实测证明 μ(a) 的 state-dependent/co-location 结构是真实存在的。
```

---

## 10. 最像审稿意见的总结

如果我是 OR 审稿人，我会写：

> The paper identifies a practically important scheduling problem arising in heterogeneous ML/RL compute fabrics, where jobs are both malleable and shareable and where service rates depend on co-location, topology, and hidden regimes. The configuration-action abstraction is promising and the candidate-set support approximation theorem is a potentially publishable mathematical contribution. The Lean formalization is unusually careful and covers several useful algebraic layers, including support functions, approximate MaxWeight drift, candidate-set loss, and Foster-style recurrence certificates.
>
> However, the current theory does not yet prove the full queueing-control claim. The capacity characterization is largely definitional unless connected to a concrete stochastic service/arrival model. The robust stability theorem depends on externally supplied support gaps, confidence events, bounded penalties, and drift domination assumptions. The hidden-regime and learning results are currently deterministic shells rather than probabilistic theorems. I would encourage a major revision that narrows the paper around candidate-restricted robust MaxWeight for configuration actions, proves a concrete stochastic stability theorem, and uses the real scheduler only as calibration and validation.

中文一句话：

**这是一个很有希望的 OR/排队论调度论文方向；`math.md` 的方向对，Lean 也不是水证明，但目前还没有真正完成 OR 9/10 的数学闭环。最值得主打的是 candidate-set approximation + robust MaxWeight stability，而不是把 BAPR/BAPR-HRO 或 Lean 形式化本身当主贡献。**

[1]: https://github.com/erzhu419/scheduleurm/tree/codex/scheduler-hetero-dispatch "GitHub - erzhu419/scheduleurm at codex/scheduler-hetero-dispatch · GitHub"
