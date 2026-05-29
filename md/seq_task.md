> Status: historical GPT planning note. This file is kept for provenance only.
> The current reviewer-facing route is `math.md` + `lean_artifact_map.md` +
> `experimental_open_items.md`; do not use this file as the canonical theorem
> statement because it predates the exact/approx-oracle split, load-certified
> operational semantics, and statewise feasible-family theorem.

我会这样判断：

> **如果目标是 OR / 排队论方向，下一步应该以“证明问题”为主，但要先做一个很小的实验来验证数学假设不是空想。**
> 不是先做完整 scheduler 实验，也不是先做端到端 RL。最合适的是：**proof-driven experiments**。

也就是说：

```text
先用小实验确认现象存在；
然后证明一个核心定理；
再让后续实验围绕定理里的常数和假设展开。
```

BAPR 给你的是 hidden regime / piecewise-stationary belief 层；BAPR-HRO 给你的是“候选集保留、在线重排序”的思想。你的 scheduler 要变成强数学论文，下一步不是再扩展算法，而是把“候选配置重排序”变成一个 queueing-control theorem。BAPR 论文里已经强调 piecewise-stationary regime、BOCD belief、frozen belief 这类结构；BAPR-HRO 论文里最强的洞察是“结构未必错，错的是排序”，也就是 keep structure + re-rank。 

---

# 1. 下一步最该解决的不是实验，而是这个定理

我会把核心证明问题设成：

> **候选配置集不完整时，基于鲁棒重排序的 MaxWeight 调度还能稳定多少容量区域？**

这句话非常关键。

因为它直接连接三个东西：

```text
BAPR-HRO: 保留候选集，只重排序
排队论: MaxWeight / capacity region / stability
你的 scheduler: configuration-based 多资源调度
```

不要先证明“某个 score 等价于 Wasserstein DRO”。那个只是 6/10 数学。

真正强的是证明：

> 即使 scheduler 只在一个候选 configuration set 里选动作，只要这个候选集足够覆盖 full configuration space，那么系统的 capacity region 只损失一个可量化的 ε，并且鲁棒 MaxWeight 策略仍然稳定。

---

# 2. 我建议的核心 theorem 形式

定义完整配置动作集：

[
\mathcal A^{full}
]

定义实际维护的候选配置集：

[
\mathcal A^{cand} \subseteq \mathcal A^{full}
]

每个 action (a) 对应服务率向量：

[
\mu^z(a)
]

其中 (z) 是硬件/网络/负载 regime。

定义 full action set 的 support function：

[
H_z^{full}(q)
=============

\max_{a\in \mathcal A^{full}}
q^\top \mu^z(a)
]

candidate action set 的 support function：

[
H_z^{cand}(q)
=============

\max_{a\in \mathcal A^{cand}}
q^\top \mu^z(a)
]

其中 (q) 可以理解成队列压力向量，也就是：

[
q = Q(t)
]

然后证明一个候选集近似条件：

[
H_z^{full}(q) - H_z^{cand}(q)
\leq
\varepsilon_{\text{cand}}|q|_1
\quad
\forall q\geq 0
]

这句话的意思是：

> 对任意队列压力方向，候选配置集能提供的最大服务价值只比完整配置集差 (\varepsilon_{\text{cand}})。

这就是 BAPR-HRO “keep-and-rerank” 思想的数学化。

---

# 3. 核心 stability theorem

然后定义你的调度策略：

[
a_t
\in
\arg\max_{a\in \mathcal A^{cand}}
\left{
Q(t)^\top \underline{\mu}_t(a)
-
K(a_{t-1},a)

R_t(a)
\right}
]

其中：

* (\underline{\mu}_t(a))：鲁棒下界服务率；
* (K(a_{t-1},a))：切换、checkpoint、rollback、迁移成本；
* (R_t(a))：OOM、带宽拥塞、共驻失败风险。

要证明的定理可以是：

> 如果 arrival rate (\lambda) 位于 full capacity region 内部，并且 slack 大于候选集损失、估计误差和 regime 识别误差，那么 candidate-restricted robust MaxWeight policy 稳定队列。

形式上：

[
\lambda \in \Lambda^{full}(\delta)
]

并且：

[
\delta

>

\varepsilon_{\text{cand}}
+
\varepsilon_{\text{est}}
+
\varepsilon_{\text{regime}}
]

则存在 Lyapunov drift：

[
\mathbb E[
V(Q(t+1))-V(Q(t))
\mid Q(t)
]
\leq
B
-

\eta |Q(t)|_1
]

其中：

[
V(Q)=\frac12 \sum_i Q_i^2
]

这就能推出队列稳定性。

这个定理一旦成立，数学味道就完全不一样了。它不是“我给配置打了个分”，而是：

> 我证明了候选配置重排序在排队系统里保留了多少 capacity，并且在不确定服务率和隐藏 regime 下仍然可稳定。

这就是 9/10 数学问题的骨架。

---

# 4. 为什么这个 theorem 比单纯实验更优先？

因为如果没有这个 theorem，你做出来的系统实验会像：

```text
我的 scheduler 比 FIFO 快 20%
```

这更像系统论文。

但如果你有上面的 theorem，实验就变成：

```text
实测 ε_cand 很小；
实测 ε_est 可校准；
实测 regime detection delay 可控；
因此理论解释了为什么 candidate reranking 足够好。
```

这就是 OR / queueing control 的论文结构。

---

# 5. 但在证明之前，应该先做一个小实验

不是大实验，是假设验证实验。

你需要先确认下面几件事确实成立：

## A. 每张 GPU 的服务率确实是 state-dependent

测：

[
g(n)
====

\text{一张 GPU 同时跑 } n \text{ 个任务时的总有效吞吐}
]

例如：

```text
n=1: 1000
n=2: 1800
n=3: 2400
n=4: 2600
n=5: 2300
```

如果真的存在这种先升后降曲线，那么 “sweet spot / admission control” 就有数学意义。

---

## B. 候选配置集是否真的足够覆盖 full configuration space

你不需要一开始枚举所有配置。可以在小规模系统上做：

```text
full set: 所有可行配置
candidate set: 你实际想维护的配置集
```

然后采样很多队列压力向量 (q)，估计：

[
H^{full}(q)-H^{cand}(q)
]

如果这个 gap 很小，BAPR-HRO 式的 keep-and-rerank 就有依据。

如果 gap 很大，说明候选集不够，需要解决的是 configuration generation，而不是 re-ranking。

---

## C. regime 是否可识别

BAPR 层是否有意义，取决于你能不能从监控信号里区分 regime：

```text
GPU memory bandwidth bottleneck
CPU rollout bottleneck
PCIe bottleneck
NUMA bottleneck
10GbE bottleneck
shared unified memory bottleneck
```

你需要测：

```text
同一个配置，在不同 regime 下的 throughput 分布是否明显不同？
```

如果不同 regime 的吞吐分布重叠严重，那么 BOCD / BAPR 式 regime belief 作用有限。

---

# 6. 所以顺序应该是这样

我建议顺序是：

```text
Step 1: 小实验验证建模假设
Step 2: 证明 candidate-restricted robust MaxWeight stability theorem
Step 3: 证明候选集 approximation / capacity loss theorem
Step 4: 加入 hidden regime / BAPR-style belief error
Step 5: 再做完整 scheduler 实验
```

重点是：

> **完整实验应该在核心 theorem 之后做，不应该在 theorem 之前做。**

但小实验必须在 theorem 之前做，否则你可能证明了一个现实中不成立的模型。

---

# 7. 具体下一步：先解决这个 proof problem

我会把第一个正式问题写成：

## Problem 1: Approximate MaxWeight with candidate configurations

给定：

[
Q_i(t+1)
========

[Q_i(t)-S_i(a_t)]^+
+
A_i(t)
]

其中：

[
a_t \in \mathcal A^{cand}
]

而 oracle 可以选：

[
a_t^* \in \mathcal A^{full}
]

如果 policy 满足近似 MaxWeight 条件：

[
Q(t)^\top \mu(a_t)
\geq
\max_{a\in\mathcal A^{full}}
Q(t)^\top \mu(a)
-
\varepsilon |Q(t)|_1

C
]

证明：

只要：

[
\lambda
\in
\Lambda^{full}(\delta)
\quad
\text{且}
\quad
\delta > \varepsilon
]

队列稳定。

这是一条非常干净的主定理。

然后你再把 (\varepsilon) 分解成：

[
\varepsilon
===========

\varepsilon_{\text{cand}}
+
\varepsilon_{\text{score}}
+
\varepsilon_{\text{regime}}
+
\varepsilon_{\text{switch}}
]

分别对应：

```text
候选集不完整
服务率估计误差
hidden regime 识别误差
切换/rollback 成本
```

这个 decomposition 就是论文的数学主线。

---

# 8. 然后再解决第二个 proof problem

## Problem 2: Candidate-set capacity loss

证明：

如果对所有 (q\geq0)：

[
H^{full}(q)-H^{cand}(q)
\leq
\varepsilon_{\text{cand}}|q|_1
]

那么：

[
\Lambda^{cand}
]

是：

[
\Lambda^{full}
]

的 (\varepsilon_{\text{cand}})-近似容量区域。

直观上：

> candidate set 不需要包含所有配置；只要它在所有队列压力方向上的 support function 足够接近 full set，就足够了。

这就是 BAPR-HRO 的 “结构保留、排序更新” 在 queueing control 中的强版本。

---

# 9. 第三个 proof problem 才是 BAPR 层

## Problem 3: Hidden-regime penalty

引入隐藏 regime：

[
Z_t \in \mathcal Z
]

调度器不知道 (Z_t)，只知道 belief：

[
b_t(z)
======

P(Z_t=z\mid \mathcal F_t)
]

定义 belief-weighted robust service：

[
\underline{\mu}_t(a)
====================

\sum_z b_t(z)\underline{\mu}^z(a)
]

或者更保守：

[
\underline{\mu}_t(a)
====================

\min_{z\in \mathcal Z_t}
\underline{\mu}^z(a)
]

要证明的是：

如果 regime change 后检测延迟为：

[
D_{\text{detect}}
=================

O(\log(1/\delta))
]

且每次错误 regime 造成的 support-function loss 有界：

[
\varepsilon_{\text{regime}}(t)
]

那么总 backlog penalty 可以被界定为：

[
O\left(
\sum_k
D_{\text{detect},k}
\cdot
|Q(\tau_k)|
\right)
]

或者证明在每段 stationary segment 足够长时，系统仍然稳定。

这就是 BAPR 的数学真正接入 scheduler 的地方。

---

# 10. 实验应该服务于这些 theorem

实验不要一开始做成：

```text
我写了一个 scheduler，然后看跑得快不快。
```

而应该围绕定理里的量做。

## 实验 1：测 (\mu(a))

对不同 configuration：

```text
单 GPU 1/2/3/4/5 个任务
多 GPU 1 个任务
多 CPU 1 个任务
CPU+GPU 混合任务
跨主机 10GbE 任务
```

测：

[
\mu(a)
]

也就是 effective throughput。

---

## 实验 2：测 (\varepsilon_{\text{cand}})

采样队列压力向量 (q)，比较：

[
\max_{a\in \mathcal A^{full}} q^\top\mu(a)
]

和：

[
\max_{a\in \mathcal A^{cand}} q^\top\mu(a)
]

得到：

[
\varepsilon_{\text{cand}}
]

这会非常有说服力，因为它直接验证“候选集重排序是否足够”。

---

## 实验 3：测 (\varepsilon_{\text{score}})

比较 predicted robust score 和真实 throughput：

[
|\widehat{\mu}(a)-\mu(a)|
]

检查你的 posterior / LCB / DRO 是否校准。

---

## 实验 4：测 regime detection

人为制造 regime change：

```text
加 CPU 压力
加 PCIe 流量
加 network congestion
改变 GPU 共驻组合
```

看 BAPR-style belief 多久检测出来。

---

## 实验 5：最后才是完整 scheduler benchmark

比较：

```text
FIFO
固定每卡 N 个任务
显存优先 bin packing
最短队列优先
greedy goodput
BAPR-HRO scoring
candidate-robust MaxWeight
oracle MaxWeight
```

指标：

```text
mean completion time
makespan
queue stability under load
goodput
GPU/CPU/network utilization
rollback cost
tail latency
```

---

# 11. 我会明确说：先证明，不要先做 RL

不要一上来做：

```text
state = 整个集群状态
action = 所有任务配置
reward = goodput
然后 PPO/SAC 学 scheduler
```

这个会把问题搞散。

你现在要的是 OR/排队论方向，主线应该是：

```text
候选配置集
support-function approximation
robust MaxWeight
hidden-regime error
capacity region
stability
```

RL 可以后面作为估计模块，但不应该是数学主角。

---

# 12. 最终建议

下一步我会这样排优先级：

## 第一优先级：证明问题

证明：

[
\text{candidate-restricted robust MaxWeight}
\Rightarrow
\text{stability under capacity slack}
]

这是主 theorem。

---

## 第二优先级：小实验验证假设

测：

```text
state-dependent service curve
candidate-set support gap
regime separability
score calibration
```

这不是完整实验，而是为了避免数学模型脱离现实。

---

## 第三优先级：完整实验

等 theorem 和假设验证都成立后，再做大规模 scheduler benchmark。

---

一句话总结：

> **先解决证明问题，但不要盲证；先用小实验确认模型的关键假设，然后证明 candidate-restricted robust MaxWeight 的稳定性和 capacity-loss 定理。完整 scheduler 实验应该放在后面，用来验证 theorem 里的 (\varepsilon_{\text{cand}}, \varepsilon_{\text{score}}, \varepsilon_{\text{regime}}) 是否真的小。**
