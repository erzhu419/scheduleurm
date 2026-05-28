我：
那就把刚刚说的scheduler的问题，在BAPR-HRO和BAPR的基础上，从原来的数学是6/10，你看如何建模成9/10的数学问题，先只说数学问题
GPT：
我会把它建模成一个 **“隐藏分段平稳的 configuration-based stochastic processing network”**，再把 BAPR 和 BAPR-HRO 分别放在两个数学层里：

[
\text{BAPR} \Rightarrow \text{隐藏 regime 的 posterior / filtering 层}
]

[
\text{BAPR-HRO} \Rightarrow \text{候选 configuration 的 robust re-ranking / scoring 层}
]

但真正把数学从 6/10 拉到 9/10 的，不是再做一个 scoring rule，而是把它升格成：

> **带未知状态相关服务率、共驻干扰、拓扑约束和切换成本的鲁棒排队控制问题。**

BAPR 已经有 piecewise-stationary regime、BOCD belief、frozen-belief contraction 这类结构；BAPR-HRO 已经有“保留候选集、在线重排序”和 per-candidate Wasserstein DRO score 的思想。你的 scheduler 问题要做成 9/10 数学，需要把这两个东西嵌入一个完整的 queueing-control 模型，而不是只做单任务候选排序。 

先把口径收准：下面是一条 **8.5–9/10 数学路线 proposal + 当前 Lean proof spine**，不是无条件闭合的 throughput-optimality claim。主线不能把完整 operational iff 直接写死；应该先证明 stationary-mix / support-function capacity base、candidate-restricted robust MaxWeight drift，再在具体 stochastic arrival/service model 里证明 conditional drift domination，并由 Foster-Lyapunov drift certificate 推出 finite-small-set return。按 round2 revise，主 paper theorem 已经收束成一条：full-action support slack + fabric-cover candidate loss + lower-service estimation loss + queue-scaled penalty + bounded finite-support stochastic model \(\Rightarrow\) positive recurrence。仍必须靠 scheduleurm profiling / 实验校准的部分单独列在 `md/experimental_open_items.md`。

按 round3 revise，正式论文里 capacity region 必须写成 **downward-closed service region**；positive recurrence 必须写成 **finite-set Foster recurrence certificate**，除非额外加 irreducibility / single closed communicating class 条件；learning / lower-service 部分必须写成 confidence-event certificate，而不是无条件 online-learning theorem。

为了避免 round2 指出的 Lean artifact / 正文不一致，当前 theorem 名称和上传单文件里的可搜索名字单独列在：

```text
md/lean_artifact_map.md
```

其中最核心的 paper theorem 是：

```text
main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
```

它还有一个更强的 calibration-facing 版本：

```text
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric
```

这个版本不直接把 \(L\)-Lipschitz 和 \(\rho\)-cover 当黑箱，而是从 candidate projection certificate、feature-level service sensitivity envelope、feature coefficient domination 推出。上传 `ScheduleurmUpload.lean` 时必须确认这些名字都在单文件里可直接搜索。

round3 后还新增了 bounded-second-moment 主 theorem 版本：

```text
main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound
```

---

# 1. 核心数学对象：configuration stochastic processing network

先定义 job class：

[
i \in \mathcal I
]

例如：

```text
CPU-only 任务
多 CPU 并行任务
单 GPU 任务
单 GPU 多任务共驻任务
多 GPU LLM 训练任务
跨主机多 GPU 任务
ARM 统一内存任务
```

每类任务有队列：

[
Q_i(t)
]

每个时刻有到达：

[
A_i(t)
]

队列演化是：

[
Q_i(t+1)
========

\left[Q_i(t)-S_i(t)\right]^+
+
A_i(t)
]

其中 (S_i(t)) 是在时刻 (t) 被实际完成/推进的 workload。

关键是，服务量 (S_i(t)) 不是固定的，而取决于：

```text
选了什么 configuration
当前处于什么硬件/网络/负载 regime
与哪些任务共驻
拓扑位置
当前系统状态
```

所以写成：

[
S_i(t)
======

S_i\bigl(a_t, Z_t, \omega_t\bigr)
]

其中：

* (a_t)：当前全局 configuration action；
* (Z_t)：隐藏 regime；
* (\omega_t)：随机扰动；
* (S_i)：服务量随机变量。

其期望服务率为：

[
\mu_i^z(a)
==========

\mathbb E[S_i(a,z,\omega)]
]

这里的 (a) 不是“给一个 job 选一个配置”，而是**整个系统的全局配置模式**。

---

# 2. 全局 configuration action

这是最重要的提升。

不要把动作写成：

[
c_j = \text{job } j \text{ 的配置}
]

而要写成：

[
a \in \mathcal A
]

其中 (a) 是一个全局 allocation pattern，例如：

```text
GPU 0: 4 个 RL 任务共驻
GPU 1: 1 个 LLM shard
GPU 2-5: 一个 4-GPU job
CPU socket 0: 64 个 rollout workers
CPU socket 1: 数据预处理任务
NIC 0: 一个跨主机 all-reduce job
```

也就是说：

[
a
=

{(j,c_j,p_j)}_{j \in R_t}
]

其中：

* (j)：正在服务的任务；
* (c_j)：任务选择的 configuration；
* (p_j)：placement，比如 GPU、CPU socket、NUMA node、NIC、跨主机路径。

动作集合 (\mathcal A) 自动包含四个象限：

| 情形            | 数学里怎么表示                              |
| ------------- | ------------------------------------ |
| 多 CPU 跑 1 个任务 | (a) 中某 job 占多个 CPU resource          |
| 1 个 CPU 跑多个任务 | 多个 job 共享同一个 CPU resource            |
| 多 GPU 跑 1 个任务 | 一个 job 占多个 GPU resource，gang service |
| 多任务共享 1 个 GPU | 多个 job 的 placement 指向同一 GPU，服务率非加性   |

这样，所有“CS 任务类型”不需要逐个特殊处理，而是统一成：

[
a \in \mathcal A
]

和

[
\mu^z(a)
========

(\mu_1^z(a),\ldots,\mu_{|\mathcal I|}^z(a))
]

---

# 3. 资源拓扑：不是资源向量，而是资源图

如果只写：

[
\text{CPU}=1000,\quad \text{GPU}=8
]

数学深度不够。

应该定义一个 resource fabric graph：

[
G=(V,E)
]

节点 (V) 包括：

```text
CPU socket
NUMA memory node
GPU card
GPU memory
unified memory pool
NIC
host
cluster
```

边 (E) 包括：

```text
PCIe
NVLink
CXL
10GbE
InfiniBand
普通以太网
WAN link
```

每条边或节点有容量：

[
\kappa_e^z
]

它可以随 regime (z) 变化。例如：

```text
同机 NVLink 正常
跨主机 10GbE 拥塞
ARM unified memory 带宽变成瓶颈
WAN link 不稳定
```

一个 configuration (c) 可以被定义为资源图上的一个子图或超边：

[
c = (H_c, d_c, \chi_c)
]

其中：

* (H_c \subseteq G)：需要占用的资源子图；
* (d_c)：对节点/边资源的需求；
* (\chi_c)：共享、独占、gang、可抢占、可回滚等属性。

可行性约束是：

[
\sum_{(j,c)\in a} d_{c,e}
\leq
\kappa_e^z
\quad
\forall e \in V\cup E
]

但注意，GPU 共驻不能只用线性容量约束，因为共驻会改变服务率。所以共驻干扰放进：

[
\mu_i^z(a)
]

而不是只放进 feasibility constraint。

---

# 4. 隐藏 regime：BAPR 层

让系统 regime 是：

[
Z_t \in \mathcal Z
]

例如：

```text
z1: GPU compute-bound
z2: GPU memory-bandwidth-bound
z3: CPU rollout-bound
z4: PCIe-bound
z5: NVLink-normal
z6: 10GbE-congested
z7: WAN-bottleneck
z8: ARM unified-memory contention
```

(Z_t) 是隐藏的，且 piecewise stationary：

[
Z_t = z_k
\quad
\text{for }
t \in [\tau_k,\tau_{k+1})
]

change points (\tau_k) 未知。

BAPR 的数学思想可以放这里：维护 joint belief：

[
b_t(h,z)
========

\mathbb P(
\text{run-length}=h,,
Z_t=z
\mid
\mathcal F_t
)
]

其中 (\mathcal F_t) 是当前监控历史。

然后 scheduler 不是知道 (Z_t)，而是用 belief-weighted service model：

[
\bar{\mu}_i^t(a)
================

\sum_z b_t(z),\mu_i^z(a)
]

如果想做 robust version，则用 pessimistic lower service：

[
\underline{\mu}_i^t(a)
======================

\inf_{\nu \in \mathcal U_t(i,a)}
\mathbb E_{\nu}[S_i(a,Z_t)]
]

这就是 BAPR 的 regime belief 与 BAPR-HRO 的 pessimistic scoring 的结合。

---

# 5. BAPR-HRO 层：从 candidate ranking 到 robust configuration ranking

BAPR-HRO 里，关键思想是：

> 候选集通常还在，错的是排序。

在 scheduler 里，对应：

> feasible configuration hypergraph 通常不需要每轮重建，真正要更新的是每个 configuration pattern 的 ranking / score。

定义每个 job class 的候选 configuration set：

[
\mathcal C_i
============

{c_{i1},c_{i2},\ldots,c_{im_i}}
]

全局候选 action set 是：

[
\mathcal A
==========

\text{feasible packings of }
{(i,c): c\in \mathcal C_i}
]

然后把 BAPR-HRO 的 scalar robust score 推广成 queue-weighted robust service score：

[
\Psi_t(a)
=========

Q(t)^\top \underline{\mu}_t(a)
-
K(a_{t-1},a)
-
R_t(a)
]

其中：

* (Q(t)^\top \underline{\mu}_t(a))：当前队列压力 × 鲁棒服务率；
* (K(a_{t-1},a))：切换、迁移、回滚、checkpoint 成本；
* (R_t(a))：OOM、通信拥塞、共驻失败、tail-risk 等风险惩罚。

这里 BAPR-HRO 的 Wasserstein-DRO 可以推广为：

[
\underline{\mu}_t(a)
====================

\inf_{\mathbb P:
W_1(\mathbb P,\widehat{\mathbb P}_{t,a})\leq \epsilon_{t,a}}
\mathbb E_{\mathbb P}[\mu(a)]
]

如果服务向量是随机的，则是 vector-valued DRO。

更强的形式是对 queue-weighted service 做 DRO：

[
\inf_{\mathbb P:
W_1(\mathbb P,\widehat{\mathbb P}_{t,a})\leq \epsilon_{t,a}}
\mathbb E_{\mathbb P}
\left[
Q(t)^\top S(a)
\right]
]

这就把 BAPR-HRO 的 per-candidate scalar identity 升级成了 queueing-control 层的 robust action value。

---

# 6. 容量区域：先做 support-function capacity base

如果 regime \(z\) 固定，并且服务率已知，完整动作空间仍然是：

[
\mathcal A_z^{full}
]

不是为了证明方便改成一个很小的动作集合。stationary randomized policy 的服务均值集合是：

[
\mathcal C_z^{full}
=
\operatorname{conv}
\{
\mu^z(a): a\in \mathcal A_z^{full}
\}
]

严格的 queueing capacity region 不是单纯的 convex hull 边界，而是这个 service set 的 downward closure：

[
\Lambda_z^{full}
=
\left\{
\lambda\ge 0:
\exists v\in \mathcal C_z^{full},
\quad
\lambda_i\le v_i,\ \forall i
\right\}.
]

直观上，service 可以浪费；到达率不需要等于某个 service vector，只要被某个 stationary average service vector coordinate-wise 支配即可。Lean 里对应：

```text
InDownwardCapacityWithSlack
downward_capacity_monotone
main_downward_capacity_support_slack
```

如果存在 mixing distribution \(x\in \Delta(\mathcal A_z^{full})\) 和 slack \(\delta>0\)，使得：

[
\lambda_i+\delta
\leq
\sum_{a\in\mathcal A_z^{full}}
x_a\mu_i^z(a),
\quad \forall i\in\mathcal I,
]

那么对任意非负 queue vector \(q\)，有 support-function slack：

[
q^\top \lambda
+
\delta\|q\|_1
\leq
H_z^{full}(q)
:=
\max_{a\in\mathcal A_z^{full}}
q^\top \mu^z(a).
]

这是 capacity/stability 证明的真正底座。它不是 operational stability iff；它只是把 full configuration action space 压成 support function，让 MaxWeight drift 能接上。

operational 的 “队列系统可稳定当且仅当 \(\lambda\in\Lambda_z\)” 要额外补三类条件：

```text
1. concrete stochastic arrival/service model；
2. conditional expected Lyapunov drift dominated by the deterministic drift；
3. necessity / conservation-law：任何 stabilizing policy 的长期平均服务必须落在 capacity set 内。
```

如果 regime 随时间切换，且长期占比为 \(\pi_z\)，则 average capacity base 是：

[
\mathcal C_\pi^{full}
=
\left\{
\sum_z \pi_z
\sum_{a\in\mathcal A_z^{full}}
x_{z,a}\mu^z(a):
x_z\in\Delta(\mathcal A_z^{full})
\right\}.
]

但这里也不能直接说稳定：如果只在长期 mixture 上有 slack，需要额外控制 dwell time、切换损失和 queue buildup。若要得到更稳的主 theorem，应先用 uniform-in-regime slack；average-regime stability 放成更强扩展。

如果服务率未知，还要定义 robust support base：

[
H_t^{rob}(q)
=
\max_{a\in\mathcal A_t^{cand}}
q^\top \underline{\mu}_t(a),
]

或者用 belief-weighted / posterior ambiguity 的 lower service。这里的数学贡献不是简单写出 \(\Lambda^{rob}_t\)，而是证明：

```text
full support slack
→ candidate cover loses at most εcand ||q||1
→ lower-service estimation loses at most εest ||q||1
→ bounded / queue-scaled penalty consumes explicit slack
→ approximate optimization oracle consumes α0 + α1 ||q||1 when exact argmax is unavailable
→ remaining MaxWeight drift is negative.
```

---

# 7. 主 paper theorem 应该收束成一条

一个真正强的数学问题，不应该只证明：

[
\text{score} = \text{DRO upper bound}
]

而应该证明下面这条闭合链。这里的重点是：**full action space 不缩小**；\(\mathcal A^{cand}\) 只是 \(\mathcal A^{full}\) 的可计算 cover，并且 cover loss 要进入 slack accounting。

论文正文最好不要把 A/B/C/D/E/F/G 都写成同等主 theorem。按 round2 的建议，主 paper theorem 应该是一条：

> **Theorem 1: Robust candidate MaxWeight stability under fabric-cover approximation.**

假设：

```text
1. job class 有限，full global configuration action family 有限；
2. A_cand 是 A_full 在 finite-feature fabric metric d_Φ 下的 ρ-cover；
3. 真实服务 μ 对 d_Φ 是 L-Lipschitz；
4. λ 在 full-action downward-closed capacity/support region 内有 slack δ；
5. lower-service model 的 support error 至多 ε_est ||Q||_1；
6. switching / rollback / risk penalty ≤ P0 + β||Q||_1；
7. solver/oracle error ≤ α0 + α1||Q||_1；exact argmax 是 α0=α1=0 的特例；
8. arrival/service 有 bounded conditional second-order moment；Lean artifact 还给出 bounded finite-support specialization；
9. coordinate conditional means 满足 E[A_i|Q]≤λ_i 和 lower_i(a(Q))≤E[S_i|Q]。
```

若：

[
\delta>L\rho+\epsilon_{est}+\beta+\alpha_1,
]

并且 finite backlog set \(N\) 取到足够大，使：

[
\frac12\sum_i(Amax_i^2+Smax_i^2)+P_0+\alpha_0+\alpha
\le
\left(\delta-(L\rho+\epsilon_{est}+\beta+\alpha_1)\right)N,
]

则得到 finite-set Foster recurrence certificate：

```text
PositiveRecurrentViaFiniteSet M.K (natQueueSmallSet N)
```

在标准 irreducibility / single closed communicating class 条件下，这个 finite-small-set certificate 才可翻译成通常意义上的 countable-state Markov chain positive recurrence。正文里不要把这个 certificate 无条件说成完整链的 positive recurrence。

Lean 里这一条已经作为 paper-facing wrapper：

```text
main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
```

真实 scheduler 如果使用 greedy / local search / time-limited ILP，不需要假装精确 argmax；Lean 里有 approximate-oracle 版本：

```text
main_robust_candidate_maxweight_drift_approx_oracle
main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric_with_second_moment_bound_approx_oracle
```

如果要把 \(L\) 和 \(\rho\) 从“assumption”推进成“profiling/candidate-generator certificate”，Lean 里对应：

```text
main_theorem_robust_candidate_maxweight_stability_from_calibrated_fabric
```

如果 feasible family 随 queue state、available jobs 或 regime 改变，cover 不能只写固定 \(\mathcal A^{full}\)。非实验性 theorem 已经支持 indexed/statewise 版本：

```text
indexed_calibrated_fabric_cover_support_gap
indexed_calibrated_fabric_cover_support_gap_uniform
```

主 theorem 可以先用 fixed feasible family；动态 feasibility 版本需要给出对每个 state/regime index 都成立的 uniform cover certificate。

下面的 Theorem A-D 是这条主 theorem 的 proof decomposition，不是四个彼此抢主线的贡献；Theorem E-G 是 extension。

---

## Theorem A：stationary-mix / support-function capacity base

给定完整 global configuration action set \(\mathcal A_z^{full}\)，固定 regime \(z\)。如果存在 stationary mix \(x\in\Delta(\mathcal A_z^{full})\) 和 slack \(\delta>0\)，使得：

[
\lambda_i+\delta
\leq
\sum_{a\in\mathcal A_z^{full}}x_a\mu_i^z(a),
\quad \forall i,
]

那么对任意非负 queue vector \(q\)，有：

[
q^\top\lambda+\delta\|q\|_1
\leq
\max_{a\in\mathcal A_z^{full}}
q^\top\mu^z(a).
]

这个 theorem 才是 capacity-to-MaxWeight 的数学底座。operational stability iff 不是在这里无条件宣称；它要接 concrete stochastic model、drift domination 和 necessity conservation law。Lean 里对应的是 `capacity_slack_implies_support_slack`，而不是把 “stable iff \(\lambda\in\Lambda_z\)” 写成定义同义反复。

---

## Theorem B：finite-feature fabric cover 的 candidate-set capacity approximation

定义 finite-feature fabric/interference metric：

[
d_\Phi(a,a')
=
\sum_{r\in\mathcal D}
w_r|\Phi_r(a)-\Phi_r(a')|.
]

\(\Phi_r(a)\) 可以编码 placement、GPU co-location profile、NUMA / PCIe / NVLink path、memory pressure、network path class、rollback/preemption attribute 等 scheduler 可观察结构。

如果：

```text
A_cand 是 A_full 在 d_Φ 下的 ρ-cover；
真实服务 μ 对 d_Φ 是 L-Lipschitz；
```

则对任意非负 queue vector \(q\)：

[
H^{full}(q)
\leq
H^{cand}(q)+L\rho\|q\|_1.
]

normalized support-function capacity distance 也满足：

[
\sup_{\|q\|_1\leq 1,q\geq0}
\left(H^{full}(q)-H^{cand}(q)\right)
\leq
L\rho.
]

此外，不依赖 compact-convex support/Hausdorff duality，也可以直接证明一个构造性的 capacity-set coordinate-Hausdorff 版本：把 full action 的 stationary mix 通过 cover map \(a\mapsto \pi(a)\in\mathcal A^{cand}\) push forward 到 candidate mix，得到：

[
\forall v\in\Lambda^{full},\exists w\in\Lambda^{cand},
\quad
\|v-w\|_{\infty,\mathrm{coord}}
\leq L\rho,
]

反向因为 \(\mathcal A^{cand}\subseteq \mathcal A^{full}\) 是零误差。Lean 里对应：

```text
main_candidate_restricted_capacity_coordinate_hausdorff
```

如果论文确实要写 ambient metric Hausdorff \(d_H\)，那一版仍然需要 compact convex support-function duality assumption：

[
d_H(\Lambda^{full},\Lambda^{cand})\leq L\rho.
]

这不是缩小动作空间；相反，它证明了 full action space 可以通过可计算候选 cover 近似，而 capacity slack 损失被显式记账。\(L\) 和 \(\rho\) 的数值仍需要由 profiling / perturbation experiment 校准。

round2 里指出 \(d_\Phi,L,\rho\) 不能是 free assumption；Lean 现在有一个 certificate layer：

```text
FabricCandidateProjection.covers
fabric_service_lipschitz_of_feature_sensitivity
calibrated_fabric_cover_support_gap
main_candidate_restricted_capacity_approximation_from_calibration
```

也就是说，实验仍要给出 feature sensitivity coefficients 和 candidate projection，但一旦这些 finite certificates 成立，support loss \(L\rho\) 就不是口头假设，而是 Lean theorem 的结论。

---

## Theorem C：robust candidate MaxWeight drift with slack accounting

定义 robust configuration policy：

[
a_t
\in
\arg\max_{a\in\mathcal A^{cand}}
\{
Q(t)^\top\underline{\mu}_t(a)
-
K(a_{t-1},a)
-
R_t(a)
\}.
]

如果 full action capacity 有 slack \(\delta\)，candidate cover 损失为 \(\epsilon_{cand}\)，lower-service estimation 损失为 \(\epsilon_{est}\)，且 penalty 满足：

[
0\leq K+R_t
\leq
P_0+\beta\|Q(t)\|_1,
]

那么只要：

[
\delta>\epsilon_{cand}+\epsilon_{est}+\beta,
]

就有 deterministic MaxWeight drift：

[
V([Q-\underline{\mu}(a_t)]^+ + \lambda)-V(Q)
\leq
B+P_0
-
\left(\delta-\epsilon_{cand}-\epsilon_{est}-\beta\right)\|Q\|_1.
]

如果用 Theorem B 的 fabric cover，则 \(\epsilon_{cand}=L\rho\)。这条是主 theorem 的核心：切换、风险、candidate approximation、服务估计误差都不能藏起来，必须逐项消耗 slack。

---

## Theorem D：concrete finite-support stochastic stability

选定一个具体 stochastic arrival/service model。令 \(\omega\in\Omega\) 是有限样本空间，每步：

[
Q_i(t+1)
=
\left[Q_i(t)-S_i(Q(t),\omega_t)\right]^+
+
A_i(Q(t),\omega_t).
]

假设：

```text
prob(Q,ω) ≥ 0 且 Σ_ω prob(Q,ω)=1；
bounded conditional second-order moment：E[1/2 Σ_i(A_i^2+S_i^2)|Q]≤B；
可选的 finite-support specialization：A_i(Q,ω)≤Amax_i, S_i(Q,ω)≤Smax_i；
coordinate conditional arrival mean：E[A_i|Q]≤λ_i；
coordinate conditional selected service mean：lower_i(a(Q))≤E[S_i|Q]；
full action capacity slack + fabric cover + robust candidate MaxWeight 条件成立。
```

在一般 queueing 表述里，二阶项可以直接作为 bounded conditional moment \(B\)。在 Lean 的 finite-support specialization 里，二阶漂移常数还可以由 bounded samples 构造性推出：

[
\mathbb E\left[
\frac12\sum_i(A_i^2+S_i^2)
\mid Q
\right]
\leq
\frac12\sum_i(Amax_i^2+Smax_i^2).
]

再结合 Theorem C，得到：

[
\mathbb E[
V(Q(t+1))-V(Q(t))
\mid Q(t)=Q
]
\leq
B+P_0
-
\eta\|Q\|_1,
]

其中：

[
\eta
=
\delta-(L\rho+\epsilon_{est}+\beta)>0.
]

Foster-Lyapunov 推出 hitting finite backlog set；local return 也可以由同一个 drift certificate 在一步之后的 hitting-time bound 推出。因此不需要把 `drift_dominated` 或 finite-small-set return 永久留成黑箱。Lean 里对应的 paper-facing 主 theorem 是：

```text
main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
```

它有两个口径：

```text
main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound
main_concrete_fabric_cover_robust_candidate_stochastic_stability_from_bounded_samples'
```

如果 lower service 来自 learning / posterior / BAPR belief，它不能在正文里被写成无条件 truth；必须写成 confidence event 上的 deterministic certificate。Lean 里现在有 generic lifting：

```text
main_high_probability_stability_from_certificate_event
```

也就是说：sampler-specific 部分要证明 lower-service domination event；一旦该 event 成立，稳定性 certificate 的概率提升是 Lean theorem。

operational necessity 不能写成“positive recurrence iff \(\lambda\) 有任意正 slack”。守恒律方向只能推出 zero-slack capacity closure：若 operationally stable 的模型给出长期 action occupation measure，并且平均服务支配平均到达，则

[
\lambda \in \overline{\Lambda}^{full}.
]

Lean 里对应：

```text
main_operational_conservation_law_necessity
main_operational_capacity_sandwich
```

---

## Theorem E：hidden regime 必须分成两版

hidden regime 不能混着说。应该分成：

```text
uniform-in-regime stability：
  每个 regime z 都有 slack δ_z，并且 inf_z δ_z 足够大。

average-regime stability：
  只有长期 mixture Σ_z π_z C_z 有 slack。
```

uniform-in-regime 版本可以直接接 robust drift，只要 detection / estimation error 不吞掉统一 slack。average-regime 版本更强，但需要额外证明 dwell time、switching process、queue buildup 和 change-point delay 的累计影响：

[
\sum_k
\tau_{\text{detect},k}\|Q(\tau_k)\|
+
\text{estimation loss}
+
\text{switching loss}.
]

这部分不能作为主 theorem 偷偷并进 Theorem D；它应作为 hidden-regime extension。现在已经证明一个有限时域 dwell/switching 版本：若每个 segment 内 detection/switching 标记窗口的 backlog mass 至多是该 segment 总 backlog mass 的 \(\theta\) 倍，再加固定 residual，则 marked-window overhead 只会把 drift margin 从 \(m\) 降到 \(m-\chi\theta\)。Lean 里对应：

```text
main_hidden_regime_dwell_switching_drift
```

---

## Theorem F：structured learning，不用 \(|\mathcal A|\) regret

全局 action set \(\mathcal A^{full}\) 是组合对象，直接写：

[
\tilde O(\sqrt{|\mathcal A||\mathcal Z|T})
]

容易 vacuous。应该用 structured bucket / semi-bandit / factorized feedback：

[
\mathcal B_{active}
=
\{
\text{fabric neighborhood bucket},
\text{co-location profile},
\text{resource-path factor},
\text{job-class service bucket}
\}.
]

在 confidence event 上，regret 应写成：

[
\tilde O
\left(
\sqrt{
|\mathcal B_{active}|T
}
+
N_{cp}\log T
+
\text{switching cost}
\right).
]

这样仍然保留 full action space；学习复杂度来自实际被观测和更新的结构化因子，而不是枚举所有 global packings。

Lean 里对应的 active-bucket event 已经不是全体 bucket 的 union bound，而是显式依赖 `active.card`，并且有从 high-probability input event 到 high-probability regret event 的提升：

```text
main_active_bucket_lcb_learning_regret
main_active_bucket_lcb_learning_regret_high_probability
```

---

## Theorem G：sweet spot / admission threshold 是 extension + experiment-driven theorem

你强调的 “每张 GPU 放几个任务的 sweet spot” 很重要，但它需要吞吐曲线或结构假设支撑。单 GPU 可以设：

[
g_z(n)=\text{同一 GPU 上 }n\text{ 个任务的总 goodput}.
]

若 profiling 证明 \(g_z(n)\) unimodal，则最优 admission limit 是 threshold 型：

[
\text{allow new job}
\iff
n<n_z^*.
]

多类型任务时，用 set function：

[
g_z(S)
]

若 profiling / model 支持 submodularity 或 diminishing returns，则 admission 由 marginal goodput 决定：

[
Q_j\Delta_j(S) > \text{switching/rollback risk}.
]

这条应放 extension 或 experiment-driven theorem。它不是三大主 theorem 的替代品，但可以提供非常强的 operational insight。

---

# 8. 最核心的 9/10 问题可以压成一句话

我会把论文级数学问题写成：

> **A configuration-action stochastic processing network for heterogeneous compute fabrics: prove support-function capacity approximation for full-action fabric covers, robust candidate MaxWeight drift with explicit slack accounting, and positive recurrence for a concrete bounded finite-support arrival/service model; then extend to hidden regimes, structured learning, and admission sweet spots.**

中文就是：

> **面向异构 compute fabric 的 configuration-action stochastic processing network：在 full action space 上证明 fabric-cover 的 support-capacity approximation、带显式 slack 记账的 robust candidate MaxWeight drift，以及 bounded finite-support 到达/服务模型下的 positive recurrence；hidden regime、structured learning 和 sweet spot 作为扩展层继续证明或实验校准。**

这个比“多场景模式识别”强很多。模式识别只是：

[
\mathcal F_t \rightarrow b_t(z)
]

真正的数学问题是：

[
(Q_t,b_t,\mathcal A^{full},\mathcal A^{cand},\Phi)
\rightarrow
a_t.
]

要证明的主链是：

```text
stationary-mix support slack
→ finite-feature fabric cover support loss
→ robust candidate MaxWeight drift
→ concrete stochastic drift domination
→ Foster positive recurrence
```

---

# 9. 从 6/10 到 9/10 的区别

## 6/10 版本

数学大概是：

[
score(c)
========

\widehat{\mu}(c)
-
\beta\sigma(c)
-
\text{risk}(c)
]

然后证明：

[
score(c)
========

\text{some DRO score}
]

这个是 candidate ranking。

它有用，但只是局部。

---

## 9/10 版本

数学应该是：

[
Q_i(t+1)
========

[Q_i(t)-S_i(a_t,Z_t)]^+
+
A_i(t)
]

[
a_t
\text{approximately maximizes over }a\in\mathcal A^{cand}
\{
Q(t)^\top \underline{\mu}_t(a)
-
K(a_{t-1},a)
-
R_t(a)
\}
]

并证明：

[
\delta > L\rho+\epsilon_{est}+\beta+\alpha_1
\Rightarrow
\text{finite-set Foster recurrence certificate under a concrete bounded stochastic model}
]

同时证明：

[
H^{full}(q)-H^{cand}(q)
\leq
L\rho\|q\|_1
]

以及：

[
\text{Regret}(T)
================

\tilde O(\sqrt{T})
+
O(N_{cp}\log T)
]

其中 regret 的 \(\sqrt{T}\) 不能来自枚举 \(|\mathcal A|\)，而要来自 structured active buckets / semi-bandit factors。这才是 OR/排队论意义上的强数学问题。

---

# 10. 主论文只保留三条贡献

不要贪多。真正可以构成一篇强数学论文的主贡献应该是：

1. **Global configuration-action SPN model for heterogeneous compute fabrics**

一个 action 同时编码 malleable jobs、shareable jobs、gang scheduling、GPU co-location、NUMA / PCIe / NVLink / network placement。这是模型贡献，但不要把 “capacity region 是 convex hull” 当主要创新；那只是 SPN 底座。

2. **Candidate fabric-cover approximation theorem**

[
q^\top\lambda+\delta\|q\|_1
\leq
H^{full}(q),
\quad
H^{full}(q)
\leq
H^{cand}(q)+L\rho\|q\|_1.
]

这条是最像原创 OR 数学贡献的部分：它把 BAPR-HRO 的 “keep candidate structure, re-rank candidates” 提升为 support/capacity geometry theorem。

3. **Robust candidate MaxWeight stability theorem**

[
\delta
>
L\rho+\epsilon_{est}+\beta
\Rightarrow
\Delta V(Q)
\leq
B+P_0-\eta\|Q\|_1.
]

并接上 concrete stochastic positive recurrence：

[
\text{bounded finite-support arrival/service}
+
\text{coordinate moment bounds}
+
\text{Foster drift}
\Rightarrow
\text{positive recurrence via finite backlog set}.
]

这三条如果做扎实，就已经比单纯 BAPR-HRO 的 per-candidate DRO scoring 深很多，而且没有把 full action space 缩小掉。Lean 中对应的单条主 theorem 是：

```text
main_theorem_robust_candidate_maxweight_stability_under_fabric_cover
```

然后再作为附加结果给出：

```text
uniform hidden-regime stability and dwell/switching budget
average-regime stability as a stronger extension, not the main claim
structured active-bucket learning regret as extension
sweet-spot threshold with profiling support
```

---

# 11. 最终建模建议

最强的数学建模不是：

> “这是一个多场景模式识别问题。”

而是：

> **这是一个 hidden-regime, configuration-action stochastic processing network。模式识别只是 belief update，核心是 belief-robust queueing control。**

形式上：

[
\boxed{
Q(t+1)
======

[Q(t)-S(a_t,Z_t)]^+
+
A(t)
}
]

[
\boxed{
Z_t
\text{ is hidden, piecewise stationary}
}
]

[
\boxed{
a_t
\in
\mathcal A
==========

\text{feasible global configuration patterns on a resource fabric graph}
}
]

[
\boxed{
\mu^z(a)
========

\mathbb E[S(a,Z_t)]
\text{ encodes co-location, gang scheduling, topology, sharing, interference}
}
]

[
\boxed{
\Lambda
=======

\left\{
\lambda\ge0:
\exists v\in\operatorname{conv}{\mu^z(a):a\in\mathcal A},
\lambda\le v
\right\}
}
]

[
\boxed{
\text{Goal: stability + near-optimality under unknown } \mu^z(a)
\text{ and hidden }Z_t
}
]

这就是从 BAPR / BAPR-HRO 出发，把数学问题提升到 9/10 的方式。
