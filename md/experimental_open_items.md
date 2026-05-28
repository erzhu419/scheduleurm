# Scheduleurm 理论路线中仍需实验 / profiling 才能闭合的项目

本文档只放目前无法靠 Lean/数学证明单独闭合、必须依赖 scheduleurm 真实运行数据、profiling 或扰动实验校准的部分。其余非实验部分已经放到 proof 里的 Scheduleurm 证明链继续推进。

## 1. 服务率与干扰模型校准

需要从 scheduleurm 运行日志和专门 profiling 得到 regime-dependent service map：

[
\mu_i^z(a)=\mathbb E[S_i(a,z,\omega)].
]

最少要记录：

```text
job class / model type / batch size / workload phase
global configuration action a
GPU/CPU/NUMA/NIC placement
同 GPU co-location profile
VRAM/RAM 占用、GPU util、CPU util、I/O、网络指标
单位时间 goodput / progress / failed step / OOM / rollback
```

要产出的实验量：

```text
Amax_i, Smax_i                 bounded finite-support proof 的常数
μ_i^z(a), lower_i(a)           drift 证明中的 service lower bound
ε_est                          lower-service estimation error
B                              second-order drift constant
```

## 2. Fabric metric 的 \(L\) 和 \(\rho\)

Lean 已经证明：如果 candidate set 是 full action space 在 finite-feature fabric metric 下的 \(\rho\)-cover，且服务率对该 metric 是 \(L\)-Lipschitz，则 support loss 和 coordinate capacity-set loss 都至多是 \(L\rho\)。

实验还需要校准：

```text
Φ(a)                           fabric/interference feature
d_Φ(a,a')                      weighted l1 fabric metric
L                              service Lipschitz envelope
ρ                              candidate generator 对 full-action samples 的 cover radius
```

这部分要能被 falsify，不能只写成漂亮假设。建议做 perturbation profiling：固定 job mix，只改变 placement、co-location 数、NUMA/NVLink/PCIe path、网络路径和并发度，测量 \(|\mu(a)-\mu(a')|/d_\Phi(a,a')\) 的上包络。

必须报告：

```text
Φ 的每个 feature 是从 scheduler 哪个 telemetry 字段来的
feature weight w_r 如何设定或拟合
L 的估计分位数 / worst-case envelope
哪些 co-location 或拓扑扰动导致 Lipschitz envelope 变大
ρ 是在全量 feasible action、采样 full action、还是历史 observed action 上估计的
candidate generator 是否存在 bounded-degree / local-neighborhood 结构
cover 是覆盖 all feasible actions、sampled feasible actions、还是 historical observed actions
cover 是 fixed family、statewise family、regimewise family，还是 uniform over all state/regime indices
```

如果某些干扰呈现跳变或非平滑，论文不能硬 claim 小 \(L\rho\)；应把这些区域标成需要 regime split、额外 feature、或 admission guard。

## 3. GPU co-location sweet spot / admission threshold

这部分必须依赖吞吐曲线，不能只靠抽象证明。

需要测：

[
g_z(n)=\text{同一 GPU 上 }n\text{ 个任务的总 goodput}
]

以及多类型任务 set function：

[
g_z(S).
]

要判断：

```text
g_z(n) 是否 unimodal
argmax_n g_z(n) 的 sweet spot 是否稳定
g_z(S) 是否近似 submodular / diminishing returns
不同任务类型组合是否存在明显反协同
admission 阈值是否随 regime z、VRAM、batch size、phase 改变
```

只有这些曲线成立后，sweet spot / admission threshold 才能作为 experiment-driven theorem 写进论文扩展部分。

## 4. Active bucket 的具体采样模型

Lean 已经证明 active-bucket event 下 regret 依赖 \(|B_{active}|\)，并证明了 high-probability input event 可以提升为 high-probability regret bound。

仍需由 scheduleurm 的具体学习/采样机制给出：

```text
active bucket 定义：fabric neighborhood / co-location profile / semi-bandit factor
feedback model：full-information / semi-bandit / bandit / censored feedback
同一 action 执行后可观测哪些 job-class service components
每个 bucket 的观测次数 bucketCount(b)
bucket loss / confidence radius 的具体计算
adaptive sampling rule
bucket 是固定定义还是 online adaptive generated
change point 后旧样本如何 discount / reset
exploration 与 queue backlog 如何耦合，是否会牺牲 stability slack
noise/tail assumption 是否符合 sub-Gaussian、bounded、或 empirical Bernstein
```

如果实际系统只有被调度过的配置才有反馈，需要额外记录 selection probability 或 exploration policy，否则 high-probability concentration 只能作为 conditional theorem，不能作为完整 learning theorem。

lower-service domination 也必须按 confidence event 处理：

```text
需要证明：Pr[∀Q,i, lower_i(a(Q))≤E[S_i|Q]] ≥ 1-δ_conf
如果 lower 来自 BOCD / posterior / empirical LCB，需要记录其训练窗口、reset/discount 规则和 tail assumption
如果只对 observed actions 有 lower bound，则主 theorem 只能用于这些 actions 或需要 exploration/cover 证明
```

## 5. Hidden regime / BOCD 检测延迟

Lean 已经有 dwell-time / switching-window backlog budget：检测和切换窗口只要占每个 segment backlog mass 的 \(\theta\) 比例，就把 drift margin 从 \(m\) 降到 \(m-\chi\theta\)。

还需要实验或具体 detector model 给出：

```text
change point 发生频率和 dwell time 分布
BOCD / detector delay τ_detect 的经验分布或 tail bound
误报率、漏报率
检测窗口内 backlog mass / segment backlog mass 的 θ
switching / rollback / migration cost χ
```

没有这些量，average-regime stability 和 BOCD delay theorem 只能停在条件式 dwell/switching 证明，不能写成 scheduleurm 的实证闭合定理。

## 6. Penalty 与 slack 消耗

robust drift theorem 需要：

[
\delta > L\rho+\epsilon_{est}+\beta+\alpha_1.
]

这里 \(\beta\) 来自 queue-scaled penalty：

[
0\le K+R_t\le P_0+\beta\|Q(t)\|_1.
]

如果 scheduler 不是 exact argmax，而是 greedy / local search / time-limited ILP，需要校准 approximate oracle error：

[
\text{optimal candidate score}-\text{chosen score}
\le
\alpha_0+\alpha_1\|Q(t)\|_1.
]

需要从 scheduleurm 数据估计：

```text
P0                             固定切换/rollback/风险成本
β                              随 backlog 增长的 penalty rate
α0                             bounded optimization oracle error
α1                             queue-scaled optimization oracle error
δ                              目标负载相对 capacity 的 slack
η = δ-(Lρ+ε_est+β+α1)          最终 drift margin
```

如果估计后 \(\eta\le0\)，理论不是错，而是说明 candidate cover、估计误差、penalty 或 solver error 已经吞掉全部 capacity slack，需要改 candidate generator、降低 penalty、改善 oracle 或加 admission control。
