注：我实际审的是 `paper/main.tex` 对应的 manuscript 内容、`skill/scheduler.py`、`/algorithm` 和实验 artifacts；二进制 `main.pdf` 的版式我没有单独渲染检查。

## 结论

作为 OR reviewer，我**不建议现在直接投 Operations Research 主刊**。如果现在送审，我倾向给 **Reject / Major Revision**，不是因为方向不行，而是因为现在的稿子有一个明显的审稿风险：

> 论文主线是“robust candidate MaxWeight + stability certificate”，但当前 live scheduler 里真正运行的算法更像“规则制候选枚举 + per-task scalar scoring hook”，实验也主要是 measured finite-slice / replay / oracle-bridge artifacts，而不是完整在线调度系统的强实证闭环。

这个方向有 OR 潜力，尤其是“full action space 不缩小、candidate loss 显式从 slack 里扣”的叙事是对的；但现在更像一篇**很有野心的 working paper / systems-OR bridge draft**，还没到 OR 主刊 reviewer 会放心接收的状态。

## 我会怎么打分

| 维度       |    当前分 |     潜力 | 评价                                                                |
| -------- | -----: | -----: | ----------------------------------------------------------------- |
| 问题重要性    |   8/10 |   9/10 | 异构 GPU/CPU fabric、co-location 干扰、RL/ML workload scheduling 是真问题。  |
| OR 建模角度  |   7/10 | 8.5/10 | configuration-action SPN + candidate-cover slack accounting 是好框架。 |
| 理论主线     | 6.5/10 |   8/10 | Theorem 口径比普通系统论文强，但 reviewer 会追问每个假设如何实证闭合。                      |
| 算法实现匹配   |   5/10 |   7/10 | 现在 live hook 还不像论文里的 global robust MaxWeight。                     |
| 实验说服力    | 4.5/10 | 7.5/10 | finite slice/replay 有价值，但 OR 主刊需要更强外部有效性。                         |
| 可复现/审稿友好 |   5/10 |   8/10 | artifact 很多，但 reviewer 需要一键复现实验、Lean artifact、数据字典和 claim matrix。 |

## 我认可的强点

第一，论文的核心抽象是有价值的。稿子没有简单说“我把 action space 缩小了所以能算”，而是明确区分 full action family、candidate family 和 conservative lower-service map，并把 (L\rho)、(\epsilon_{\mathrm{est}})、(\beta)、(\alpha_1) 都作为 slack 消耗项写进稳定性条件。摘要里也把这一点作为主贡献：full-action support slack (\delta) 减去 candidate-cover loss、lower-service loss、penalty 和 oracle error 后，若剩余 margin (\eta>0)，就得到 finite-set Foster recurrence certificate。

第二，主 theorem 的形态比很多“系统调度 + 经验曲线”的文章严谨。Theorem 直接列出 support slack、candidate support loss、lower-service support error、penalty bound、approximate oracle condition、conditional moment bound，并给出
[
\eta=\delta-(\epsilon_{\mathrm{cand}}+\epsilon_{\mathrm{est}}+\beta+\alpha_1)>0
]
下的 negative drift certificate。 这条线如果写实，会比“我们找到了 sweet spot”更像 OR。

第三，工程层的分层是清楚的。`algorithm/README.md` 说明 algorithm layer 是 optional placement policy，默认 `legacy` 保留旧行为，非 legacy policy 可以不改 `scheduler.py` 选择；同一个 README 还把 calibration utilities 分成 fabric metric、service model、penalty fit、oracle audit、capacity LP 等模块。  这对 reviewer 是加分项，因为 proof objects 至少有工程对应物。

第四，论文已经主动限制 claim scope，这点很重要。稿子在 discussion 里承认 strongest empirical claim 是 exact measured finite-slice claim，broader fabric-cover calibration 还需要 (\Phi,L,\rho,\pi) 等 perturbation profiling table；同时承认 SOTA comparison 不是直接运行 Gavel/Pollux/Sia/IADeep/Salus。 这种克制是对的，千万不要删。

## 最大拒稿风险

### 1. 论文里的算法和 live scheduler 里的算法还不是同一个东西

`skill/scheduler.py` 的 dispatch 逻辑仍然是一次给一个 task 选 placement；`pick_placement(task, nodes)` 的候选是 node/GPU 级别，排序后取第一个。  算法层通过 `policy.gpu_fit_block_reason` 和 `policy.gpu_score` 插入额外 gate / score。

但论文的 policy 是
[
\arg\max_{a\in\mathcal A_t^{cand}}{Q^\top \underline{\mu}_t(a)-K_t(a)-G_t(a)},
]
这是全局 action / queue-weighted robust MaxWeight。 当前 `SweetSpotPlacementPolicy` 自己也写明是 “experiment hook, not a theorem by itself”。 它的 score components 主要是 VRAM pressure、util pressure、co-location、sweet-gap、runtime、priority、queue-age 等 scalar 项，而不是明确的 (Q^\top\underline{\mu}(a)) global action objective。

OR reviewer 很可能会问：你证明的是 robust candidate MaxWeight；你实现的到底是不是它？如果不是，论文必须把 live scheduler 部分降级成“engineering hook / policy prototype”，把 theorem-facing evaluation 放在 replay/oracle artifacts 上，而不是暗示 production dispatcher 已经实现了 theorem policy。

### 2. 实验证据太像 finite-slice replay，不像 OR 主刊意义上的系统性计算实验

论文的主实验表确实有不错数字：candidate policy 对 legacy 在 q00/q01/q10/q11 上分别给出 (11.825\times)、(1.079\times)、(1.510\times)、(1.154\times) makespan ratio。 但 artifact 自己也写得很清楚：这是 measured finite service-action slice，不是 production-arrival 或 unmeasured-global-action certificate。

更关键的是，slack certificate 里 (L\rho=0)、(\epsilon_{\mathrm{est}}=0)、(\beta=0)、(\alpha_1=0) 的原因，是 candidate action family equals measured full slice、measured service map used as lower service、exact oracle by construction。 这在数学上干净，但 reviewer 会觉得“你验证的是一个你已经测完并枚举完的小世界”。这可以作为 theorem instantiation，但还不够支撑大标题里的 heterogeneous compute fabrics。

### 3. “SOTA-style baseline” 很容易被 reviewer 攻击

论文明确说没有直接运行外部 scheduler，而是用同一个 measured service cache 上的 policy semantics 来代表 Gavel/Pollux/Sia、SRPT/Gittins/SERPT、IADeep/Salus-like policies。 这比乱跑 baseline 诚实，但 OR reviewer 会要求非常清楚地改名：这不是 “SOTA baseline”，而是 “SOTA-inspired policy-semantics replay”。

现在的表格显示 candidate 并不 Pareto-dominated，但 throughput-table 在 sum makespan 上略优，delay oracle / interference guard 在 mean flow 上略优。 这个结果可以讲成“candidate 在 makespan/flow tradeoff 上稳健”，但不要讲成“beat SOTA”。

### 4. 生产数据 bridge 还不是 live theorem-grade proof

production oracle bridge artifact 明确写了：它是 measured service map + completed-active production population，不是 live scheduler dispatch trace。 它有 3437/3437 mapped records 和 (\alpha_0=\alpha_1=0)，但 `usable_for_live_scheduler_oracle_trace` 是 false，并且剩余要求是捕获真实 scheduler candidate slots，再给每个 candidate enrich `queue_vector`, `lower_service`, `penalty_units`, `score_semantics=robust_maxweight_lower_service`。

另一个 live closure artifact 只有 2 个 trace slots、2 个 candidates，scope 明确说是 one emitted live scheduler candidate-family trace，不是 universal claim over future dispatches。 这在 rebuttal 里会很危险：reviewer 会说“live audit 太小，不能支撑 production claim”。

### 5. 全局 production stability 还没有闭合

`module49_production_load_strict.md` 里 30-day window 有 6957 条记录，但 mapped 只有 4825，unmapped 2132，mapped fraction 约 0.694。 该 artifact 也明确写了 `global_coverage_usable_for_theorem=false`、`usable_for_global_theorem=false`，unmapped reasons 包括 `unmapped_cpu=1034` 和 `unmapped_gpu=943`。 所以论文不能暗示“生产负载整体稳定性已证明”。最多说“mapped measured buckets 上有 load certificate”。

## 我建议怎么改到能投 OR

### A. 把论文 claim matrix 写成第一页就能看懂

建议在 Introduction 后加一张表：

| Claim                      | Scope                                                     | Evidence                                            | Not claimed                            |
| -------------------------- | --------------------------------------------------------- | --------------------------------------------------- | -------------------------------------- |
| Theorem 1 stability        | finite/statewise candidate family under slack assumptions | human proof + Lean artifact                         | not automatic production stability     |
| Finite-slice instantiation | q00/q01/q10/q11 measured action slices                    | measured service cache + replay + slack certificate | not all workloads                      |
| Live scheduler hook        | optional placement scorer                                 | code + small trace audit                            | not theorem-grade global MaxWeight yet |
| Production bridge          | completed-active mapped population                        | service-map oracle bridge                           | not raw 30-day global stability        |

这张表能显著降低 reviewer 误解，也能阻止你自己过度 claim。

### B. 二选一：要么实现 theorem policy，要么改论文定位

现在最核心的选择是：

**路线 1：真的实现 robust candidate MaxWeight dispatch。**
把 live dispatch 的 action 从 “one task -> one node/GPU” 升级为 “current queue vector -> candidate global configurations”，每个 candidate 有 `lower_service`, `penalty`, `oracle_gap`，dispatch 选最大 (Q^\top\underline{\mu}-K-G)。这样论文和系统就对齐。

**路线 2：承认 live scheduler 只是 policy hook，论文主贡献是 theorem + finite-slice replay framework。**
这条更容易短期完成。标题可以弱一点，例如：

> Robust Candidate MaxWeight Certificates for Heterogeneous Compute Scheduling: A Scheduleurm Case Study

不要让 reviewer 以为你已经把 full theorem policy 部署成 production scheduler。

### C. 实验必须补强三类

第一，补 **真实在线 arrival experiment**。现在 static replay 很干净，但 OR reviewer 会想看 Poisson / bursty / adversarial-ish arrivals 下 backlog、flow time、GPU memory failure、rollback frequency、utilization 的曲线。论文自己说 Poisson and load-sweep traces implemented but not primary claim；如果投 OR，建议把它们变成主实验之一。

第二，补 **holdout calibration**。不要只在 measured slice 上令 (\epsilon_{\mathrm{est}}=0)。用 train profiles 建 lower-service map，用 holdout profiles / later time windows 估 (\epsilon_{\mathrm{est}})。`service_model.py` 已经有 empirical Bernstein LCB 和 `min_samples` 机制。 但论文要报告每个 bucket 的 sample count、LCB radius、holdout residual，而不是只给最终 slack。

第三，补 **ablation**。至少要有：

1. legacy rules only；
2. sweetspot scalar scorer；
3. robust lower-service scorer；
4. no candidate cover / exact finite family；
5. no penalty；
6. direct guard vs mean-flow tie-break。

这样 reviewer 才能分清收益来自 OR 算法、来自 hard rules、来自 service profiling，还是来自 legacy cap 被放松。

### D. SOTA 比较要降调

把 “SOTA-style policy baselines” 改成 “policy-semantics replay baselines”。保留 Gavel/Pollux/Sia/IADeep/Salus 作为 related work 和 semantic inspiration，但不要写得像直接公平比较了外部系统。现在稿子已经在文字里承认不是 direct binary execution，这点要继续强化。

### E. Lean artifact 必须可审

论文说 Lean proof artifact 是 `ScheduleurmUpload.lean`，并给了 SHA-256、`lake build`、`lake env lean` 以及 no `sorry/admit/axiom` 的声明。 但 OR reviewer 不会只信 hash。需要 supplementary zip 里有：

`ScheduleurmUpload.lean`、`lakefile.lean`、`lean-toolchain`、一键 build script、theorem-name map、paper theorem 到 Lean theorem 的 crosswalk、以及一个不依赖私有路径的 CI log。

## 我会给你的投稿建议

不要马上投 OR 主刊。先把这篇当成 **R&R 级别 draft** 打磨 1–2 轮。最优顺序是：

1. **先做 claim 收缩**：明确“theory / finite-slice / replay / live hook / production bridge”五者边界。
2. **补 theorem-policy alignment**：至少让 theorem-facing replay 和 live trace 使用同一个 `robust_maxweight_lower_service` score semantics。
3. **补在线负载实验**：Poisson、burst、load sweep、holdout service error、longer live trace。
4. **把 artifact 做成 reviewer 可复现**：一键生成 Tables 1–4、slack certificate、oracle audit、Lean check。
5. **再考虑 OR**。

更现实的投稿路径：如果你想快一点，可以先投 **INFORMS Journal on Computing / MSOM / Stochastic Systems / Queueing Systems / ACM SIGMETRICS-style systems-performance venue** 这类更能接受“理论 + 系统 artifact + replay”的地方。若目标坚持 OR，建议把理论部分打到非常干净，把实验 claim 降到可完全审计的范围，然后把“production scheduler”作为 motivating case study，而不是主证明对象。

一句话：**这个项目有潜力，但现在 OR reviewer 最大的不安是“证明的是 robust global MaxWeight，代码跑的是规则制上挂 scalar score，实验证明的是 measured finite slice”。把这三者对齐或明确切开，才有机会。**
