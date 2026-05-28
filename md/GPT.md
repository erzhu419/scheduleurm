我：
排队论优化问题，只在不同机器完成不同任务的时候耗时不同才会出现对么？
GPT：
不对。
排队论（Queueing Theory）里的“优化问题”并不只会在“不同机器做不同任务耗时不同”时出现。那只是其中一种情形（通常属于“异构服务器/异构资源分配”问题）。

更本质地说：

> 只要系统里存在“等待 + 资源有限 + 决策空间”，就会产生优化问题。

即使所有机器完全一样、所有任务耗时也一样，仍然会有很多优化目标。

---

## 一个最简单的例子：机器完全相同

假设：

* 2 台完全一样的服务器
* 每个任务平均处理 1 秒
* 任务随机到达

你仍然可以优化：

### 1. 如何分流任务

例如：

* 随机分配
* 轮询（Round Robin）
* 最短队列优先（Join Shortest Queue）

它们会导致不同：

* 平均等待时间
* 队列长度
* 系统吞吐量

这已经是经典排队论优化。

---

## 即使任务耗时固定，也有优化

比如：

* 每个任务都严格耗时 1 秒
* 单服务器

仍然可以优化：

### 调度顺序

例如：

* FIFO（先来先服务）
* SJF（短作业优先）
* 优先级队列

不同策略：

* 平均响应时间不同
* 长尾延迟不同
* 公平性不同

---

## 排队论里的优化，本质上在优化什么？

常见目标：

| 优化目标     | 例子         |
| -------- | ---------- |
| 最小平均等待时间 | 减少用户等待     |
| 最小尾延迟    | 避免极慢请求     |
| 最大吞吐量    | 单位时间完成更多任务 |
| 最大资源利用率  | CPU 不空转    |
| 公平性      | 不让某类任务一直饿死 |
| 成本最小化    | 少开机器       |

---

## “不同机器耗时不同”属于哪类问题？

这是：

## 异构服务器（heterogeneous servers）

例如：

| 机器 | CPU 型号 | 处理某任务耗时 |
| -- | ------ | ------- |
| A  | 强      | 1s      |
| B  | 弱      | 5s      |

或者：

| 任务类型 | GPU 更快 | CPU 更快 |
| ---- | ------ | ------ |
| 矩阵计算 | GPU    |        |
| IO任务 |        | CPU    |

这时优化会更复杂，因为：

* 不只是“排队”
* 还涉及“任务-资源匹配”

通常会出现：

* Assignment problem
* Scheduling problem
* Load balancing
* Queue-aware routing

这些是排队论 + 运筹优化 + 调度理论的交叉。

---

## 一个关键区别

很多人会把：

* “排队论”
* “调度理论”
* “负载均衡”
* “资源分配”

混在一起。

实际上：

### 经典排队论更关注

* 到达过程
* 服务过程
* 等待时间统计性质

例如：

* M/M/1
* M/M/c
* Jackson Network

### 而复杂优化更偏向

* 调度
* 在线决策
* 资源匹配
* 多目标优化

---

## 一个直观总结

排队论优化并不依赖：

❌ “机器不同”
❌ “任务耗时不同”

真正需要的是：

✅ 有竞争资源
✅ 有等待
✅ 有策略选择

只要：

[
\text{需求} > \text{瞬时可服务能力}
]

优化问题几乎一定会出现。

我：
好,那假如说我现在的问题其实就像你说的是一个服务器问题,我的服务器上一共有八张卡,然后一千多个CPU核心,但是大概实际的处理器也就是六块还是八块,每一块里面有64个核心。我有一大堆的强化学习任务要去训练,因为现在是我一个人在用,所以我现在的排队的方法也没有什么排队的方法,就是全都扔进去,一个一个顺序的,给每一个卡一个,因为强学习任务会导致这个,或者说计算学习任务会导致这个跟一般的排队问题的区别可能在哪呢?它在于这个带宽限制,因为带宽所以会,你如果全排满了其实反而会更慢。我现在给的额度是30%外加一个上下限的512MB的一个显存的一个限制,那你觉得排队论能够如何优化我现在的这个服务器调度器呢? 
GPT：
我会把它视为多资源、会相互干扰的训练任务调度，而不是 8 张卡对应 8 个独立队列；重点是用观测数据找并发拐点、准入控制和 CPU/GPU/带宽配比。
我：
记住，每张卡里不止一个任务哦，只是有30%+512mb的显存限制，有时候4个有时候5个任务，取决于任务的显存
GPT：
新的判断标准是别把 GPU 利用率当主目标，而看总有效吞吐，如 samples/sec、updates/sec、completed runs/day；显存上限只能防 OOM，不能防带宽拥塞。
我：
我以前一直认为，任务之间耗时方差越大，排队论可能才越有用，如果所有任务都是12-18h完成，那么排队论其实只服务于最后那一批任务，因为所有优化方法其实目标都是让机器空闲时间最小，让所有任务同时在所有机器完成，如果方差很小就不存在这个问题，当然排队论有用的前提是ETA很准。
GPT：
你的场景关键不是任务时长差异，而是多任务共卡、带宽干扰和显存碎片会让调度本身改变服务率。优化目标应从排最后一批，转为找每张卡的最佳共驻点。
我：
而且我之前一直认为，与其去优化排队，不如去寻找每张卡上放多少任务的sweet spot，以及动态的加入/回滚任务。

GPT：
你这个判断基本对：现在最关键不是改队列顺序，而是把调度器做成按每张卡动态决定准入、回退和共置的反馈并发控制器。

我：
对，排队论应该算什么问题？planning问题么？它好像是定规则，定如何排队的规则，但是在有反馈信号的情况下，planning肯定不如dynamic control

GPT：
对，你这个说法基本是对的。更精确地说：

> **排队论本身不是 planning 问题，而是一套描述“等待系统”的建模语言。**
> 真正拿来优化时，它可以变成 planning、scheduling，也可以变成 dynamic control。

放到你的服务器调度器里，它更应该被看成：

> **queueing control / online scheduling / admission control 问题**
> 而不是传统的“排好一个静态队列”。

---

## 1. 排队论不是“排一个固定顺序”

很多人一听“排队论”，会想成：

```text
任务 A
任务 B
任务 C
任务 D
...
```

然后问题变成：

```text
谁先谁后？
怎么减少机器空闲？
怎么让最后一批任务尽量同时结束？
```

这更像 **offline scheduling / planning**。

但经典排队论更一般。它关心的是：

```text
任务怎么到达？
服务资源有多少？
服务时间怎么分布？
系统拥塞时等待时间怎么变化？
采用不同服务规则会有什么结果？
```

其中“服务规则”可以是固定的，也可以是动态的。

比如：

```text
FIFO
最短队列优先
优先级队列
最短剩余时间优先
拥塞时拒绝新任务
队列过长时暂停低优先级任务
```

这些都可以被看成排队系统里的 **policy**。

---

## 2. planning 是 open-loop，dynamic control 是 closed-loop

你可以这样区分：

### Planning

Planning 更像：

```text
我提前知道所有任务、预计耗时、资源需求。
然后一次性算出一个未来执行计划。
```

形式上是：

[
a_0, a_1, a_2, ..., a_T
]

也就是提前决定未来动作。

它依赖：

```text
ETA 准
资源模型准
任务行为稳定
中途不发生太多变化
```

所以你说“ETA 很重要”，对 planning 是对的。

---

### Dynamic control

Dynamic control 更像：

```text
我先做一个动作；
观察系统反馈；
根据反馈再决定下一步。
```

形式上是：

[
a_t = \pi(x_t)
]

其中：

* (x_t)：当前系统状态
* (a_t)：当前动作
* (\pi)：控制策略

在你的场景里，状态 (x_t) 包括：

```text
每张 GPU 上有哪些任务
每个任务当前吞吐
每张卡显存占用
GPU memory bandwidth 压力
PCIe / CPU / NUMA 压力
每个任务是否进入慢阶段
队列里还有哪些任务
```

动作 (a_t) 包括：

```text
启动一个任务
把任务放到某张 GPU
暂时不启动
暂停任务
回滚最新加入的任务
降低某张卡并发度
给某任务更多/更少 CPU worker
```

这明显比一次性排队更适合你现在的问题。

---

## 3. 所以你说“有反馈信号时 planning 不如 dynamic control”，在你的场景里基本成立

我会稍微补一句限制：

> 在系统不确定、服务率会变、反馈便宜、回滚可行的情况下，dynamic control 通常比一次性 planning 更强。

你的服务器正好满足这些条件：

```text
任务实际速度不可预测
同卡共驻会改变服务率
4 个任务和 5 个任务的效果要实测
RL 训练阶段会变化
GPU/CPU/带宽瓶颈会随时间漂移
你可以通过 checkpoint / rollback 做纠偏
```

所以这里不应该追求：

```text
一开始把未来 12–18 小时的所有排队顺序算好
```

而应该追求：

```text
每隔一段时间观察一次系统状态，
决定是否加入、保留、暂停或回滚任务。
```

---

## 4. 你的问题更像“带反馈的排队控制问题”

如果用更正式的名字，我会叫它：

> **state-dependent queueing control problem**

或者更工程一点：

> **online multi-resource scheduling with co-location interference**

拆开看是几个问题叠在一起：

| 层次                 | 你的系统里对应什么       |
| ------------------ | --------------- |
| Queueing           | 有大量任务等待训练       |
| Scheduling         | 决定哪个任务先跑        |
| Placement          | 决定任务放到哪张 GPU    |
| Admission control  | 决定现在该不该启动新任务    |
| Bin packing        | 显存、CPU、RAM 等硬约束 |
| Congestion control | 控制带宽/CPU/GPU 争抢 |
| Dynamic control    | 根据吞吐反馈加入或回滚     |
| Online learning    | 学习哪些任务组合适合共驻    |

所以它不是单纯 planning。

更准确地说：

```text
排队论提供建模框架；
调度理论提供任务选择方法；
控制理论提供反馈闭环；
在线学习帮助你学 sweet spot。
```

---

## 5. 传统排队规则其实也是 policy，不是 plan

比如 FIFO 不是提前排完整个未来计划，而是一个规则：

```text
每当有空闲资源，就取最早到达的任务。
```

它是一个简单 policy。

最短队列优先也是：

```text
每个新任务到来时，放到当前最短的队列。
```

它也是动态的，只不过反馈信号很弱，只看：

```text
队列长度
```

而你的调度器要看更丰富的反馈：

```text
总有效吞吐
单任务 slowdown
GPU memory bandwidth
CPU/NUMA 争抢
任务 reward progress
```

所以你不是从“排队论”跳到了“控制论”，而是从：

```text
简单队列规则
```

升级到了：

```text
带性能反馈的 queueing control policy
```

---

## 6. 你这里最核心的控制变量不是顺序，而是并发度

对你来说，最重要的动作不是：

```text
A 先跑还是 B 先跑？
```

而是：

```text
这张卡现在是 3 个任务、4 个任务，还是 5 个任务？
第 5 个任务加进去以后，总吞吐升了还是降了？
要不要回滚第 5 个？
```

这就是控制问题。

可以写成：

[
R_k(S_k)
]

其中：

* (k)：第 (k) 张 GPU
* (S_k)：这张 GPU 上当前共驻任务集合
* (R_k(S_k))：这张卡当前总有效吞吐

你的控制目标是：

[
\max \sum_{k=1}^{8} R_k(S_k)
]

而不是：

[
\max \sum_{k=1}^{8} |S_k|
]

也就是说，不是让每张卡任务数最多，而是让每张卡的有效训练推进速度最大。

---

## 7. 最适合你的框架其实是 MPC：短视规划 + 动态反馈

纯 planning 不够，纯 reactive control 也可能抖动。

最实用的是中间形态：

> **Model Predictive Control，MPC，滚动规划控制。**

意思是：

```text
观察当前状态
根据当前状态做一个短期计划
只执行第一步
观察反馈
重新规划
```

对你的服务器就是：

```text
每 5–15 分钟观察一次
评估每张 GPU 的总有效吞吐
尝试加入一个任务
观察加入后的吞吐变化
好就保留
不好就回滚
更新这类任务的共驻经验
进入下一轮
```

这比一次性规划强，因为它不断纠偏。

也比完全贪心强，因为它可以加一些约束：

```text
不要频繁回滚
不要刚启动就判断
不要因为 1% 噪声做动作
不要饿死老任务
不要连续试同一个坏组合
```

---

## 8. 你可以把调度器写成一个控制器

核心形式可以是：

```text
state  ->  action  ->  feedback  ->  update policy
```

具体到你这里：

```text
state:
    每张卡上有哪些任务
    每个任务当前吞吐
    每个任务 baseline 吞吐
    当前总归一化吞吐
    显存占用
    CPU/NUMA 压力
    最近是否刚回滚过

action:
    不做事
    启动一个任务
    尝试把任务放到某张 GPU
    暂停最新加入任务
    回滚低收益任务
    降低某类任务并发度

feedback:
    加任务后总吞吐是否上升
    单任务 slowdown 是否过大
    GPU/CPU/带宽是否过载
    reward progress 是否异常变慢
```

奖励函数可以简单定义成：

[
\text{reward}
=============

## \sum_i w_i \cdot \frac{\text{current throughput}_i}{\text{solo throughput}_i}

## \alpha \cdot \text{rollback cost}

\beta \cdot \text{severe slowdown penalty}
]

也就是说：

```text
总有效训练吞吐越高越好；
频繁回滚要扣分；
把某些任务拖得太慢也要扣分。
```

---

## 9. 排队论在这里还有用，但不是以“排序算法”的形式有用

它有用的地方是帮你回答这些问题：

```text
系统什么时候已经过载？
新任务什么时候不该进入？
队列里任务太多时，等待时间会怎么增长？
每张卡的服务率如何随并发度变化？
暂停一个任务会不会让整体完成时间更短？
是否应该故意让任务等待，而不是硬塞进去？
```

你之前说“排队论只服务于最后一批任务”，这个判断对一种场景成立：

```text
每台机器一次只跑一个任务
任务耗时差不多
服务率不受共驻影响
没有动态反馈
目标是最小化 makespan
```

但你的系统不是这个模型。

你的系统里，第 5 个任务可能会让前 4 个任务全部变慢。所以调度决策不只是影响“最后一批”，而是会影响所有正在运行任务的服务率。

这时排队论/控制论的价值就大了很多。

---

## 10. 我会给你的问题一个最终分类

如果要一句话分类：

> 你的问题不是传统排队问题，也不是单纯 planning 问题，而是 **带状态反馈的在线多资源调度控制问题**。

更具体：

```text
Queueing control
+ admission control
+ GPU co-location scheduling
+ multi-resource bin packing
+ online learning of interference
+ rollback-based feedback control
```

所以你现在的直觉可以总结成：

```text
静态排队顺序优化：次要
每卡 sweet spot：主要
动态加入/回滚：核心
ETA 准确性：对 planning 重要，但对 feedback control 没那么关键
实时吞吐反馈：最重要
```

最准确的一句话是：

> 你不应该做一个“把任务排好”的 scheduler；
> 你应该做一个“持续把每张 GPU 控制在最佳共驻点附近”的 controller。


对，我记住了：**每张 GPU 不是只能跑一个任务，而是会共驻多个强化学习任务**，现在主要靠“30% + 512MB 显存限制”来决定能塞几个，所以一张卡可能 4 个任务，也可能 5 个任务。

这会把问题从：

> 8 台服务器，每台一次服务 1 个任务

变成：

> 8 个共享资源池，每个资源池里多个任务一起跑；任务越多，单个任务可能越慢，总吞吐也可能下降。

所以你这个问题非常适合用排队论思想优化，但不能直接套最简单的 M/M/8 模型。更准确地说，它是：

**多资源约束 + 共驻干扰 + 状态相关服务率 + 调度优化问题。**

---

## 1. 你现在的问题核心不是“显存够不够”

你现在的限制是：

[
\text{每任务显存} \leq 30% + 512\text{MB}
]

这个限制只能解决一个问题：

> 不要 OOM。

但它不能解决：

> 多个任务放在同一张卡上之后，整体训练是不是更快。

强化学习训练里面，经常瓶颈不在显存，而在：

* GPU memory bandwidth
* PCIe / NVLink 传输
* CPU 到 GPU 的数据搬运
* 环境模拟 CPU 开销
* replay buffer 读写
* dataloader / rollout worker
* NUMA 跨 socket 访问
* 多进程争抢同一张 GPU 的 kernel 调度

所以调度器不能只问：

> 这张卡还剩多少显存？

而应该问：

> 把这个任务加到这张卡上之后，总有效吞吐会增加还是下降？

这是排队论优化在你这里最有用的地方。

---

## 2. 你的 GPU 应该被建模成“可变并发服务器”

普通排队模型里，一个服务器的服务率通常是固定的：

[
\mu
]

但你这里不是。对每张 GPU 来说，服务率应该写成：

[
\mu_k(S_k)
]

其中：

* (k) 是第 (k) 张 GPU
* (S_k) 是当前放在这张 GPU 上的任务集合
* (\mu_k(S_k)) 是这张 GPU 在当前共驻组合下的总训练吞吐

比如一张卡上跑 1 个任务：

[
\mu({A}) = 1000 \text{ samples/s}
]

跑 2 个任务：

[
\mu({A, B}) = 1700 \text{ samples/s}
]

跑 4 个任务：

[
\mu({A, B, C, D}) = 2600 \text{ samples/s}
]

跑 5 个任务：

[
\mu({A, B, C, D, E}) = 2400 \text{ samples/s}
]

那么第 5 个任务虽然“显存放得下”，但它让总吞吐从 2600 掉到 2400，这时就不该放。

所以你的调度目标应该是：

[
\max \sum_{k=1}^{8} R(S_k)
]

而不是：

[
\max \sum_{k=1}^{8} |S_k|
]

也就是说，不是最大化“跑了多少个进程”，而是最大化“总有效训练推进速度”。

---

## 3. 对强化学习任务，建议不要用 GPU 利用率做主指标

很多人会看：

```text
GPU Utilization: 95%
```

然后觉得很好。

但强化学习任务里，这个指标容易误导。因为你真正关心的是：

* env steps/sec
* learner updates/sec
* samples/sec
* episodes/hour
* completed runs/day
* successful experiments/day
* reward improvement per wall-clock hour

更好的主指标是：

[
\text{effective throughput} = \frac{\text{训练进展}}{\text{真实时间}}
]

例如：

```text
task_A 单独跑：1000 env steps/s
task_A 与 4 个任务共驻：500 env steps/s
```

那么它慢了 50%。

如果 5 个任务共驻后每个都慢 60%，总吞吐不一定更高。

---

## 4. 你的调度器应该做“边际收益判断”

每次有新任务要进来时，不应该直接找一张显存够的卡塞进去。

应该对每张 GPU 计算：

[
\Delta R = R(S_k \cup {j}) - R(S_k)
]

其中：

* (S_k) 是第 (k) 张卡上已有任务
* (j) 是新任务
* (R) 是这张卡的总有效吞吐

如果：

[
\Delta R > 0
]

说明加进去后总训练速度变快，可以放。

如果：

[
\Delta R \leq 0
]

说明虽然显存够，但加进去会拖慢整体，不应该放。

这就是你的调度器最核心的优化规则。

---

## 5. 你现在的“30% + 512MB”可以保留，但它只能作为硬约束

我建议把资源分成两类：

### 硬约束

这些不能违反：

```text
显存
CPU 核心
系统内存
文件句柄
进程数
磁盘空间
```

比如：

[
\sum m_j \leq M_{\text{GPU}}
]

这个是硬约束。

### 软约束

这些可以超，但超了会变慢：

```text
GPU compute
GPU memory bandwidth
PCIe bandwidth
CPU memory bandwidth
disk I/O
network I/O
NUMA 跨 socket 流量
```

这些应该进入调度评分，而不是简单地设一个“能不能跑”。

所以你的调度器可以从：

```text
显存够 → 启动
```

升级成：

```text
显存够 + CPU 够 + 预测边际吞吐为正 → 启动
```

---

## 6. 每张卡跑 4 个还是 5 个，不应该由显存决定

你前面说“有时候 4 个，有时候 5 个任务，取决于任务的显存”。

我建议改成：

> 显存决定最多能放几个；吞吐决定实际应该放几个。

举个例子。

假设某张卡 80GB 显存，每个任务限制大约 16GB，所以最多能放 5 个。

但你实际测出来：

| 同卡任务数 | 单任务平均吞吐 |  总吞吐 |
| ----: | ------: | ---: |
|     1 |    1000 | 1000 |
|     2 |     850 | 1700 |
|     3 |     700 | 2100 |
|     4 |     600 | 2400 |
|     5 |     430 | 2150 |

那么最优并发不是 5，而是 4。

如果另一类任务更轻：

| 同卡任务数 | 单任务平均吞吐 |  总吞吐 |
| ----: | ------: | ---: |
|     1 |    1000 | 1000 |
|     2 |     900 | 1800 |
|     3 |     780 | 2340 |
|     4 |     680 | 2720 |
|     5 |     600 | 3000 |

那么这类任务可以放 5 个。

所以调度器里最好不要有一个固定的“每卡最多 N 个任务”。应该是：

```text
每种任务类型有自己的共驻曲线。
```

---

## 7. 强化学习任务要做“任务画像”

你可以给每个训练任务记录一个 profile：

```yaml
job_type: ppo_atari
vram_limit_mb: 16000
cpu_cores: 16
ram_gb: 32
solo_env_steps_per_sec: 12000
solo_gpu_mem_bw: medium
solo_pcie_bw: low
solo_cpu_usage: high
checkpointable: true
priority: normal
```

另一个可能是：

```yaml
job_type: model_based_rl
vram_limit_mb: 22000
cpu_cores: 32
ram_gb: 96
solo_env_steps_per_sec: 2000
solo_gpu_mem_bw: high
solo_pcie_bw: high
solo_cpu_usage: medium
checkpointable: true
priority: high
```

然后调度器不要把所有任务都当成一样。

更合理的是把任务分成几类：

| 类型        | 特征                          | 调度策略                 |
| --------- | --------------------------- | -------------------- |
| 显存重、计算轻   | 占显存但不怎么跑满 GPU               | 可以和计算重任务混放           |
| 计算重、显存中等  | GPU kernel 密集               | 不要太多同卡               |
| PCIe 重    | CPU/GPU 数据来回搬               | 避免同卡或同 PCIe root 下扎堆 |
| CPU env 重 | rollout worker 多            | 要绑定 CPU/NUMA         |
| I/O 重     | replay buffer / dataset 读写多 | 避免集中到同盘              |

你的问题很可能不是“每卡 4 个还是 5 个”，而是：

> 哪 4 个任务可以放一起，哪 4 个任务不能放一起。

---

## 8. 最实用的调度策略：共驻干扰矩阵

你可以测一个很简单的东西：任务之间的 slowdown。

比如任务 A 单独跑：

```text
A: 1000 steps/s
```

A 和 B 同卡跑：

```text
A: 700 steps/s
B: 800 steps/s
```

A 的 slowdown：

[
\frac{700}{1000} = 0.7
]

B 的 slowdown 也类似。

最后形成一个矩阵：

| 共驻组合  | A 保留吞吐 | B 保留吞吐 | C 保留吞吐 |
| ----- | -----: | -----: | -----: |
| A + A |   0.72 |      — |      — |
| A + B |   0.85 |   0.90 |      — |
| A + C |   0.55 |      — |   0.60 |
| B + C |      — |   0.88 |   0.80 |

然后调度器就知道：

```text
A 和 B 可以放一起
A 和 C 不适合放一起
B 和 C 还可以
```

这比“显存够就放”要强很多。

---

## 9. 一个可行的调度打分函数

对每个等待中的任务 (j)，对每张 GPU (k)，算一个 score：

[
\text{score}(j,k)
=================

## \Delta R(j,k)

## \lambda_1 \cdot \text{bandwidth_pressure}

## \lambda_2 \cdot \text{cpu_pressure}

\lambda_3 \cdot \text{numa_penalty}
+
\lambda_4 \cdot \text{age_bonus}
]

含义是：

* (\Delta R(j,k))：把任务放进去带来的吞吐增量
* bandwidth_pressure：显存带宽、PCIe、NVLink 压力
* cpu_pressure：CPU/环境模拟压力
* numa_penalty：跨 socket 惩罚
* age_bonus：任务等太久以后加一点分，防止饿死

然后选 score 最高的 GPU。

如果所有 GPU 的 score 都小于 0，就不要启动新任务，即使显存还有空间。

这是重点：

> 队列为空不是目标；系统吞吐最大才是目标。

有时候让任务在队列里等，比硬塞进去更快。

---

## 10. 简化版伪代码

你的调度器可以长这样：

```python
while True:
    update_gpu_states()
    update_running_job_throughputs()

    for job in pending_jobs_sorted_by_priority():
        best_gpu = None
        best_score = float("-inf")

        for gpu in gpus:
            if not hard_constraints_ok(job, gpu):
                continue

            current_R = estimate_total_throughput(gpu.running_jobs)
            new_R = estimate_total_throughput(gpu.running_jobs + [job])

            delta_R = new_R - current_R

            score = (
                delta_R
                - bandwidth_penalty(job, gpu)
                - cpu_penalty(job, gpu)
                - numa_penalty(job, gpu)
                + age_bonus(job)
            )

            if score > best_score:
                best_score = score
                best_gpu = gpu

        if best_gpu is not None and best_score > 0:
            launch(job, gpu=best_gpu)
        else:
            keep_waiting(job)

    sleep(30)
```

这个比普通 FIFO 强很多。

普通 FIFO 是：

```text
来一个任务，找个地方塞。
```

这个是：

```text
来一个任务，判断它塞进去是否真的让系统更快。
```

---

## 11. 还应该加入“反向动作”：暂停或降并发

只做“启动任务”还不够。

因为强化学习任务的负载可能会随时间变化：

* 前期 rollout 重
* 中期 replay buffer 变大
* 后期 checkpoint 更频繁
* 有些任务进入 evaluation 阶段
* 有些任务 reward 卡住，继续跑价值很低

所以调度器应该周期性判断：

```text
当前共驻组合是不是还合理？
```

如果发现某张卡：

```text
4 个任务时总吞吐 2400
5 个任务时总吞吐 2100
```

就应该暂停一个低优先级任务，回到 4 个。

这个动作很重要，尤其是 RL 任务很长，不像普通 batch job 很快结束。

前提是你的训练任务最好支持：

```text
checkpoint
resume
graceful stop
```

例如调度器发 SIGTERM 后，任务先保存 checkpoint，再退出。

---

## 12. 对你这种“一人使用”的场景，目标可以更激进

如果是多人集群，需要公平性。

但你说现在是你一个人在用，那么调度目标可以很简单：

> 最大化单位时间内完成的有效实验量。

所以你可以牺牲一点公平性，比如：

* 慢任务可以先不跑
* 低优先级任务可以暂停
* 明显没希望的实验可以 early stop
* 高信息量实验优先
* 不追求所有任务平均等待时间最低

这对 RL/hyperparameter sweep 特别重要。

很多时候最优策略不是“把所有实验都跑完”，而是：

1. 每个配置先跑 10% 预算
2. 淘汰明显差的
3. 把 GPU 资源集中给 promising runs
4. 最后只完整训练少数几个

这已经不只是排队论，而是“排队论 + 多臂老虎机/实验设计”。

---

## 13. NUMA 和 CPU 绑定很可能非常关键

你说服务器有很多 CPU 核心，可能是 6 块或 8 块物理处理器，每块 64 核。

这意味着你很可能有明显的 NUMA 拓扑。

强化学习任务常见模式是：

```text
CPU 环境模拟 → 数据送到 GPU → GPU learner 更新 → 再回到 CPU
```

如果 CPU worker 和 GPU 不在同一个 NUMA domain，可能会出现：

```text
跨 socket 内存访问
PCIe root complex 不匹配
CPU memory bandwidth 被打爆
```

然后你会看到一种现象：

```text
GPU 看起来没满，但训练就是慢。
```

所以调度器最好同时决定：

```text
任务放在哪张 GPU
任务绑定哪些 CPU 核心
任务的内存尽量从哪个 NUMA node 分配
```

启动任务时可以类似：

```bash
CUDA_VISIBLE_DEVICES=3 \
numactl --cpunodebind=2 --membind=2 \
python train.py
```

当然具体 GPU 3 对应哪个 NUMA node，需要你根据机器拓扑查。

---

## 14. 一个实际可落地的版本

你可以先不做复杂预测模型，直接做一个简单有效的版本。

### 第一步：让每个任务上报训练进度

比如每 30 秒写一次：

```json
{
  "job_id": "ppo_seed_17",
  "gpu": 3,
  "env_steps_per_sec": 8421,
  "updates_per_sec": 12.3,
  "reward": 138.2,
  "vram_mb": 14200,
  "cpu_percent": 760,
  "timestamp": 1234567890
}
```

重点是：

```text
env_steps_per_sec 或 samples/sec
```

不要只看 nvidia-smi。

---

### 第二步：测单独运行吞吐

每种任务先单独跑 5 到 10 分钟，得到 baseline：

```text
ppo_mujoco: 50000 steps/s
sac_mujoco: 30000 steps/s
dreamer: 8000 steps/s
offline_rl: 2000 updates/s
```

---

### 第三步：测同卡并发曲线

例如同一种任务跑 1、2、3、4、5 个：

| 任务数 | 总 steps/s | 是否推荐 |
| --: | --------: | ---- |
|   1 |       50k | 可以   |
|   2 |       92k | 可以   |
|   3 |      126k | 可以   |
|   4 |      140k | 可以   |
|   5 |      132k | 不推荐  |

那么这个任务类型的默认同卡上限就是 4，而不是显存允许的 5。

---

### 第四步：允许混放，但避免坏组合

比如：

```text
PPO + PPO + PPO + PPO 可以
Dreamer + Dreamer 不行
PPO + Offline RL 可以
SAC + 大 replay buffer 任务 不太行
```

把这些经验写进调度器。

一开始可以很土：

```yaml
bad_colocation:
  - [dreamer, dreamer]
  - [model_based_rl, model_based_rl]
  - [replay_heavy, replay_heavy]

good_colocation:
  - [env_heavy, gpu_heavy]
  - [memory_heavy, compute_light]
```

后面再自动学习。

---

## 15. 对你的场景，我会这样设计调度器

每张 GPU 维护这些状态：

```python
GPUState:
    gpu_id
    running_jobs
    vram_used_mb
    sm_util
    mem_bw_util
    pcie_rx
    pcie_tx
    estimated_total_throughput
    numa_node
```

每个任务维护：

```python
Job:
    job_id
    job_type
    vram_limit_mb
    cpu_cores
    ram_gb
    baseline_throughput
    current_throughput
    priority
    checkpointable
    submit_time
```

调度规则：

```text
1. 显存不够，不放。
2. CPU/NUMA 不合适，扣分。
3. 同卡任务数超过该类型经验上限，不放或重扣分。
4. 预测总吞吐下降，不放。
5. 如果多个 GPU 都能放，选边际收益最高的。
6. 如果所有选择都让总吞吐下降，让任务继续排队。
7. 定期检查正在跑的任务，如果共驻后总吞吐下降，暂停最低优先级任务。
```

---

## 16. 你现在最应该加的不是“更复杂队列”，而是“准入控制”

“排队”听起来像是决定任务顺序。

但你现在最需要的是 admission control，也就是：

> 当前系统已经够满了，新任务到底该不该进来？

你的问题不是没有把任务排好，而是可能启动太多任务，导致大家互相拖慢。

所以第一版优化可以非常简单：

```text
每张 GPU 默认最多 4 个任务。
只有当最近 5 分钟这张 GPU 的总有效吞吐仍然随任务数增加而提高时，才允许第 5 个任务进入。
```

比如：

```python
if gpu.num_jobs < 4:
    allow = True
elif gpu.num_jobs == 4 and predicted_delta_throughput(job, gpu) > 0:
    allow = True
else:
    allow = False
```

这已经会比“显存够就放”稳定很多。

---

## 17. 最关键的一句话

你的系统不应该追求：

```text
8 张卡全部塞满，每张卡 5 个任务
```

而应该追求：

```text
每张卡处在总有效吞吐最高的共驻点
```

对某些任务，这个点可能是：

```text
每卡 5 个
```

对另一些任务，可能是：

```text
每卡 3 个
```

甚至某些重任务可能是：

```text
每卡 1 到 2 个
```

所以排队论能帮你的地方不是套一个标准公式，而是提供一个框架：

> 把 GPU 看成服务率随拥塞变化的资源池，然后通过准入控制、共驻选择、动态暂停和吞吐反馈来最大化整体训练进度。

我：
这个问题，有最接近的论文么，如果是用RL来解决的更好

GPT：
有，而且你这个问题最接近的关键词不是传统 queueing theory，而是：

**GPU sharing / co-location / interference-aware scheduling / goodput-optimized DL scheduling / RL-based cluster scheduling**。

从“像你的程度”来看，我会这样排。

| 优先级 | 论文 / 系统                                                                                                       | 为什么像你的问题                                                                                                                                                                   | 差距                                                                          |
| --: | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
|   1 | **IADeep: Interference-Aware Multiplexing for Deep Learning in GPU Clusters: A Middleware Approach, SC 2023** | 很像你说的“一张 GPU 上 multiplex 多个训练任务，但共驻会互相干扰”。它的核心就是在 cluster 层选择哪些任务适合 multiplex，在 device 层降低干扰。SC23 摘要明确说它针对 single GPU 上 multiplex tasks 的 interference，并联合优化任务分配和设备上的干扰缓解。 | 不是 RL，更偏系统 + heuristic / middleware。([SC23][1])                             |
|   2 | **Hierarchical Resource Partitioning on Modern GPUs: A Reinforcement Learning Approach, 2024**                | 这是最接近“用 RL 决定同卡共驻 + 资源切分”的论文之一。它用 RL 同时优化 job co-scheduling 和 GPU 资源分区，具体讨论 MPS + MIG，并把 state 设计成队列里的 job profile，把 action 设计成共驻 job set 和资源 allocation。                  | 更偏 MIG/MPS 资源分区；如果你现在只是显存 quota + 进程共驻，不一定能直接复现。([arXiv][2])                |
|   3 | **Interference-aware opportunistic job placement for shared distributed deep learning clusters, JPDC 2024**   | 非常像你的“显存够不等于应该放”的问题。它专门研究 shared GPU 上多个 DL job 共驻、memory oversharing、性能干扰、在线 placement adjustment，并用机器学习预测 interference。                                                  | 不是 RL，但对你“30% + 512MB 显存限制、4/5 个任务共驻”的建模非常贴。([科学直接][3])                     |
|   4 | **Gandiva: Introspective Cluster Scheduling for Deep Learning, OSDI 2018**                                    | 很像你的“动态加入/回滚/kill 任务/看反馈”的思路。Gandiva 针对 DL 的 feedback-driven exploration，支持 time-slicing、packing、grow-shrink、migration，并用训练中的反馈动态调整。                                       | 它主要是 DL/HPO 集群调度，不是 RL，也不是专门为每张卡 4–5 个任务的显存 quota 设计。([Microsoft][4])       |
|   5 | **Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications, MLSys 2020**                     | 很适合看底层 primitive：fast job switching、memory sharing、iteration scheduling。它的重点是让多个 DL 应用细粒度共享 GPU，而不是简单独占。                                                                   | 偏 TensorFlow-era 系统实现；不是 RL。([MLSys Proceedings][5])                        |
|   6 | **Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning, OSDI 2021**                     | 对你最有价值的是 **goodput** 这个目标函数：不只是 GPU 利用率，而是“训练实际有效推进速度”。Pollux 会监控 job 状态，建模加/减资源后 goodput 如何变化，并动态重分配资源。                                                                   | 它更关注 elastic DL job 的资源数量、batch size、learning rate，不是单卡多个任务共驻。([USENIX][6]) |

---

## 如果你特别想找“用 RL 做调度”的论文

最经典起点是 **DeepRM**。它把多资源 packing/scheduling 问题变成强化学习问题，强调真实系统复杂、服务时间受干扰、输入有噪声，很符合你前面说的“planning 不如 dynamic control”的直觉。它不是 GPU-specific，但思想上很基础。

然后是 **Decima: Learning Scheduling Algorithms for Data Processing Clusters, SIGCOMM 2019**。它用 RL + 图神经网络学习集群调度策略，目标可以是最小化平均 job completion time。它不是 GPU 共驻问题，但它是“RL 学 scheduler policy”的代表作。([arXiv][7])

更贴近 DL cluster 的是 **DL2: A Deep Learning-driven Scheduler for Deep Learning Clusters**。它先用已有调度器的 trace 做 supervised warm-up，然后接到 live DL cluster 里用 RL fine-tune，在线决定资源分配；这和你这种“先用 heuristic/经验跑起来，再从反馈学策略”的路线很像。([arXiv][8])

还有 **RIFLING: A reinforcement learning-based GPU scheduler for deep learning research and development platforms**，这个名字和你的“一人/研究平台/大量 DL/RL 实验调度”比较贴。摘要里说它把调度建模成 online decision-making，并用 Q-learning 做调度决策。([Wiley Online Library][9])

最近一些 GPU job scheduling with DRL 的方向会更关注 topology / communication cost，例如 2025 的 **Topology-aware GPU job scheduling with deep reinforcement learning and heuristics**，以及 **AttentiveSched / dual-agent DRL** 这类工作，它们更像“分布式训练 job 选哪些 GPU”，而不是“单张卡上塞 4–5 个任务”。([科学直接][10])

---

## 我觉得最像你问题的一句话总结

你的问题最像：

> **interference-aware GPU co-location scheduling with online feedback control**。

如果强行加 RL，就是：

> **RL-based admission control and co-location policy for GPU-sharing workloads**。

它和普通 DL cluster scheduling 的区别是，你的 action 不是简单的：

```text
job -> which GPU
```

而是：

```text
是否把新任务加入某张 GPU
加入哪张 GPU
加入后是否保留
是否回滚
是否降低同卡并发度
是否改变资源 quota
```

所以如果你要自己做，我会优先读：

1. **IADeep**：学怎么建模同卡 multiplex 和 interference。
2. **Hierarchical Resource Partitioning on Modern GPUs: A Reinforcement Learning Approach**：学怎么把“共驻 + 资源切分”写成 RL。
3. **Pollux**：学 reward / objective，尤其是 goodput。
4. **Gandiva / Salus**：学系统 primitive，比如 time-slicing、packing、migration、memory sharing。
5. **DeepRM / Decima / DL2**：学 RL scheduler 的 state/action/reward 设计。

---

## 对你自己的调度器，最可借鉴的 RL 形式

我不会一开始就做完整 RL scheduler。更稳的是：

```text
heuristic scheduler + logging
        ↓
offline simulator / replay
        ↓
contextual bandit or small RL policy
        ↓
online fine-tuning
```

你的 state 可以是：

```text
每张 GPU 当前任务数
每个任务类型
每任务显存占用
每任务 baseline throughput
每任务当前 throughput
GPU memory bandwidth
PCIe / CPU / NUMA 压力
最近一次加入任务后的吞吐变化
```

action 可以是：

```text
do nothing
把某个 pending job 放到 GPU k
回滚 GPU k 上最新加入的任务
暂停低收益任务
允许第 5 个任务进入
禁止第 5 个任务进入
```

reward 可以借鉴 Pollux 的 goodput 思路：

[
reward =
\Delta \text{effective throughput}
----------------------------------

## \alpha \cdot \text{slowdown penalty}

## \beta \cdot \text{rollback cost}

\gamma \cdot \text{OOM / instability penalty}
]

这里最重要的不是 ETA，而是：

```text
加入/移除一个任务之后，总有效训练推进速度是否提高。
```

所以最接近你问题的论文组合不是某一篇，而是：

> **IADeep + Hierarchical Resource Partitioning RL + Pollux + Gandiva/Salus + DeepRM/Decima/DL2**。

其中如果只能读一篇“最像你的”，我会先读 **IADeep**；如果必须是 RL，我会先读 **Hierarchical Resource Partitioning on Modern GPUs: A Reinforcement Learning Approach**，再读 **DL2**。

[1]: https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap290.html "Interference-Aware Multiplexing for Deep Learning in GPU Clusters: A Middleware Approach"
[2]: https://arxiv.org/html/2405.08754v1 "Hierarchical Resource Partitioning on Modern GPUs: A Reinforcement Learning Approach"
[3]: https://www.sciencedirect.com/science/article/abs/pii/S0743731523001466 "Interference-aware opportunistic job placement for shared distributed deep learning clusters - ScienceDirect"
[4]: https://www.microsoft.com/en-us/research/publication/gandiva-introspective-cluster-scheduling-for-deep-learning/ "Gandiva: Introspective Cluster Scheduling for Deep Learning - Microsoft Research"
[5]: https://proceedings.mlsys.org/paper_files/paper/2020/hash/d9cd83bc91b8c36a0c7c0fcca59228f2-Abstract.html "Fine-Grained GPU Sharing Primitives for Deep Learning Applications"
[6]: https://www.usenix.org/conference/osdi21/presentation/qiao "Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning | USENIX"
[7]: https://arxiv.org/abs/1810.01963 "[1810.01963] Learning Scheduling Algorithms for Data Processing Clusters"
[8]: https://arxiv.org/abs/1909.06040 "[1909.06040] DL2: A Deep Learning-driven Scheduler for Deep Learning Clusters"
[9]: https://onlinelibrary.wiley.com/doi/full/10.1002/spe.3066?utm_source=chatgpt.com "RIFLING: A reinforcement learning‐based GPU scheduler ..."
[10]: https://www.sciencedirect.com/science/article/abs/pii/S0743731525001054 "Topology-aware GPU job scheduling with deep reinforcement learning and heuristics - ScienceDirect"

我：排队论方向论文呢？有什么能参考的么？
GPT：
有。你这个问题在排队论里最接近的方向不是经典 `M/M/8`，而是：

> **State-dependent limited processor sharing queues**
> 或者叫：**resource-sharing systems with admission / concurrency control**

直译就是：

> 一个共享资源服务器里，多个任务可以同时服务；但服务器总服务率会随并发数变化；所以要控制同时进入服务区的任务数。

这和你的“一张 GPU 同时跑 4/5 个任务，但带宽争抢会让总吞吐下降”非常像。

---

## 最贴近你问题的一组论文

### 1. **Self-Adaptive Admission Control Policies for Resource-Sharing Systems**

这是我觉得最应该先读的一篇。它把资源共享系统抽象成 **Processor Sharing server with state-dependent service rate + FCFS waiting queue**，核心变量叫 **MPL，Multi-Programming Limit**，也就是“允许同时进入服务的最大任务数”。论文开头就说：随着并发任务增加，吞吐先上升，但超过某点后会因为资源竞争和 thrashing 下降；而它研究的就是怎么选择静态或动态 MPL。这个和你说的“与其优化排队，不如找每张卡放多少任务的 sweet spot，以及动态加入/回滚任务”几乎是同一个抽象。

它还有一个很重要的结论：**让系统运行在吞吐峰值点不一定最小化平均响应时间**；在任务大小高方差时，MPL 选择会变复杂，因为更多并发可能让短任务更快进入服务，但也可能牺牲总体服务率。

映射到你的系统：

```text
GPU = resource-sharing server
同卡并发数 = MPL
GPU 总有效训练吞吐 = state-dependent service rate μ(k)
等待中的训练任务 = FCFS / priority waiting queue
动态加任务/回滚 = dynamic MPL control
```

---

### 2. **Approximations and Optimal Control for State-Dependent Limited Processor Sharing Queues**

这是上面那条线的更理论化版本。它研究的是 **processor sharing server whose service rate depends on the number of jobs occupying the server**，控制方式正是“限制能同时共享服务器的任务数，剩下任务在 FIFO 队列等待”。论文还明确说，好的控制策略需要在 **efficiency** 和 **parallelism** 之间取平衡：一边要让服务器跑在高服务率区间，另一边要避免小任务被大任务挡住。([PubsOnLine][1])

这篇更接近你想做的“调度器核心模型”：

[
\mu(k)
]

表示一张 GPU 上同时跑 (k) 个任务时的总服务率。你的经验曲线可能是：

```text
k=1: 1000
k=2: 1800
k=3: 2400
k=4: 2600
k=5: 2300
```

那最优并发可能就是 4，而不是显存允许的 5。

这篇还研究了 **动态调整 concurrency limit**，这就对应你说的动态加入/回滚任务。([PubsOnLine][1])

---

### 3. **Optimal Admission Control in Queues with Workload-Dependent Service Rates**

这篇的模型非常像“过载后反而变慢”。它假设服务率随着 workload 先增加后减少，然后研究 admission control：系统当前状态下到底接不接新任务。论文在一定条件下证明了 **threshold policy** 是最优的，也就是到达某个负载阈值后就不再接收新任务。([Cambridge University Press & Assessment][2])

对应到你这里，就是：

```text
如果某张 GPU 当前有效吞吐已经在下降区间，
即使显存还够，也不要再往上放任务。
```

所以你可以把它理解成你调度器里的第一版理论依据：

```text
显存约束决定“最多能不能放”；
threshold/admission policy 决定“实际上该不该放”。
```

---

## 如果任务类型不同：读 Processor Sharing / DPS

### 4. **A Survey on Discriminatory Processor Sharing**

普通 processor sharing 假设任务平等共享资源；但你的任务可能不一样：有的 GPU memory bandwidth 重，有的 CPU env 重，有的 replay buffer 重，有的显存重但算力轻。这时候更接近 **Discriminatory Processor Sharing, DPS**。DPS 是多类别 processor sharing：所有 job 同时被服务，但不同类别可以有不同权重 (g_k)，每类得到的服务率由权重和当前任务数共同决定。([Springer][3])

它对你有用的地方不是公式本身，而是建模方式：

```text
不是每个任务平分 GPU；
而是不同任务类型对共享资源的“吃法”不同。
```

你可以把任务粗分为：

```text
env-heavy
gpu-compute-heavy
gpu-memory-bandwidth-heavy
cpu-memory-heavy
io-heavy
```

然后学习不同组合的 slowdown。

---

## 如果你想要“动态控制 / 稳定性”理论：读 MaxWeight 系列

### 5. **Tassiulas & Ephremides: Stability Properties of Constrained Queueing Systems and Scheduling Policies for Maximum Throughput**

这是 MaxWeight / backpressure scheduling 的经典源头。它研究的是在有约束的队列网络里，根据队列长度动态选择服务动作，从而达到最大稳定区域。它本来是通信网络方向，但抽象上很适合“多个队列 + 多个服务器 + 服务动作受约束”的系统。([ENS][4])

对你而言，MaxWeight 的思想可以翻译成：

```text
不要只看哪张卡最空；
要看 pending job 的压力 × 放到某张卡后的边际服务率。
```

比如：

[
\text{score}(j,k)
=================

## Q_j \cdot \hat{\mu}_{j,k}(S_k \cup {j})

Q_j \cdot \hat{\mu}_{j,k}(S_k)
]

其中 (Q_j) 可以是某类任务等待量、优先级或 age。

---

### 6. **MaxWeight with Discounted UCB: Learning While Scheduling in Multi-Server Systems With Unknown Statistics**

这篇非常适合你，因为你其实不知道每种任务在每张 GPU、每种共驻组合下的真实服务率。它研究 multi-server queueing systems，其中不同 job type 在不同 server 上处理时间不同，而且服务统计未知；算法把 **MaxWeight** 和 **discounted UCB** 结合起来，一边调度，一边学习服务率，并且支持非平稳服务率。([Proceedings of Machine Learning Research][5])

这和你的情况很贴：

```text
任务 ETA 不准
同卡共驻后吞吐会变
RL 任务不同阶段服务率会漂移
不同 GPU / NUMA / CPU 绑定会有差异
```

这篇比“纯 RL”更稳，因为它保留了 queueing stability 的结构，又加了 learning。

---

## 如果你想用 RL 解“排队论控制”

### 7. **RL-QN: A Reinforcement Learning Framework for Optimal Control of Queueing Systems**

这是比较正宗的“RL + queueing control”论文。它的问题是：queueing network 的状态空间通常无界，普通 RL 很难直接套；它的做法是在有限状态子集上用 model-based RL，在大 backlog 区域用已知的 stabilizing policy 保证稳定。它测试了 dynamic server allocation、routing、switching 等问题。([arXiv][6])

这个思路对你很有价值：不要让 RL 完全自由控制整台服务器，而是：

```text
正常区域：RL / bandit 学习 sweet spot
危险区域：fallback 到保守策略，比如每卡最多 4 个任务
OOM/吞吐崩溃区：硬规则立即禁止
```

---

### 8. **Queueing Network Controls via Deep Reinforcement Learning**

这篇是用 PPO 做 queueing network control。论文说它在 parallel-server system 和 large-size multiclass queueing networks 上测试，生成的控制策略在多种负载条件下超过已有启发式方法，并在可计算最优策略的场景下接近最优。([PubsOnLine][7])

它适合你参考 state/action/reward 设计，但我不建议你一开始就照着做端到端 PPO。你的实际系统里 rollback、checkpoint、OOM、RL 训练噪声都很麻烦，先做 logging + heuristic + bandit 更稳。

---

## 如果你关心 ETA / 任务时长方差：读 SRPT / Gittins / SERPT

你之前说“如果任务都是 12–18h，排队论似乎只影响最后一批任务；ETA 准才有用”。这对应的是 queueing scheduling 里的 **size-aware scheduling**。

### 9. **Simple Near-Optimal Scheduling for the M/G/1**

这篇总结得很清楚：如果知道每个 job 的大小，**SRPT，Shortest Remaining Processing Time** 最小化平均响应时间；如果不知道每个 job 的大小、只知道 job size distribution，那么 **Gittins policy** 已知可以最小化平均响应时间；但 Gittins 复杂，所以它提出了更简单的 M-SERPT。([arXiv][8])

对你而言，这一支适合解决：

```text
哪个实验先跑？
短任务是否插队？
ETA 有噪声时怎么排？
长任务是否应该降优先级？
```

但它不是你最核心的问题。你最核心的是：

```text
同一张 GPU 上到底该同时跑几个任务，以及该不该接入第 5 个任务。
```

所以 SRPT/Gittins 是第二层，MPL/admission control 是第一层。

---

## 我会建议你的阅读顺序

第一优先级：

1. **Self-Adaptive Admission Control Policies for Resource-Sharing Systems**
2. **Approximations and Optimal Control for State-Dependent Limited Processor Sharing Queues**
3. **Optimal Admission Control in Queues with Workload-Dependent Service Rates**

这三篇最贴近你的“每张卡的 sweet spot + 动态加入/回滚”。

第二优先级：

4. **Discriminatory Processor Sharing survey**

用于建模不同 RL 任务类型共驻时的非均匀干扰。

第三优先级：

5. **MaxWeight with Discounted UCB**
6. **RL-QN**
7. **Queueing Network Controls via Deep RL**

用于把你的调度器从 rule-based 升级到 learning-based。

第四优先级：

8. **Simple Near-Optimal Scheduling for the M/G/1 / Gittins / SRPT**

用于处理 ETA、任务时长分布、短任务优先级。

---

## 对你这个系统，最准确的论文关键词

你可以按这些关键词继续搜：

```text
state-dependent limited processor sharing queues
resource-sharing systems admission control MPL
processor sharing with state-dependent service rate
dynamic concurrency control queueing
workload-dependent service rates admission control
MaxWeight scheduling unknown service rates
queueing control reinforcement learning
learning while scheduling multi-server queues
discriminatory processor sharing
Gittins scheduling M/G/1 unknown job sizes
```

最贴切的一句话是：

> 你的服务器调度问题，在排队论里最像 **state-dependent limited processor sharing queue with dynamic admission/concurrency control**；如果加上任务类型和反馈学习，就是 **multiclass state-dependent processor sharing + learning-based admission control**。

[1]: https://pubsonline.informs.org/doi/pdf/10.1287/stsy.2021.0087 "Approximations and Optimal Control for State-Dependent Limited Processor Sharing Queues"
[2]: https://www.cambridge.org/core/journals/probability-in-the-engineering-and-informational-sciences/article/optimal-admission-control-in-queues-with-workloaddependent-service-rates/215C41699880E5D73BD385FE3ED0DD80 "OPTIMAL ADMISSION CONTROL IN QUEUES WITH WORKLOAD-DEPENDENT SERVICE RATES | Probability in the Engineering and Informational Sciences | Cambridge Core"
[3]: https://link.springer.com/article/10.1007/s11134-006-7586-8 "A survey on discriminatory processor sharing | Queueing Systems | Springer Nature Link"
[4]: https://www.di.ens.fr/~busic/mar/projets/TE92.pdf?utm_source=chatgpt.com "Stability properties of constrained queueing systems and ..."
[5]: https://proceedings.mlr.press/v206/yang23d.html "Learning While Scheduling in Multi-Server Systems With Unknown Statistics: MaxWeight with Discounted UCB"
[6]: https://arxiv.org/abs/2011.07401 "[2011.07401] RL-QN: A Reinforcement Learning Framework for Optimal Control of Queueing Systems"
[7]: https://pubsonline.informs.org/doi/10.1287/stsy.2021.0081 "Queueing Network Controls via Deep Reinforcement Learning | Stochastic Systems"
[8]: https://arxiv.org/abs/1907.10792 "[1907.10792] Simple Near-Optimal Scheduling for the M/G/1"


我：
这个问题，外加考虑多卡1个任务的（LLM训练例如），多cpu并行跑1个任务的，反正就是所有CS类的任务（四个象限：多cpu跑1个任务，1个cpu多个任务，多显卡1个任务，多任务共享一个显卡），中间可能还涉及跨不同的集群，有些是ARM架构统一内存，cpu/gpu共用内存显存，有些是同一个主机上n块芯片n块卡，有些是2台主机用万兆网线连起来，有些则是同一个机房/网络下用普通网络连起来的集群，有些甚至需要跨网。
目前确定没人做这个问题？以及你觉得有意义么？能发OR这个期刊么？

GPT：
我的判断是：

> **不能说“没人做”。**
> 但也不能说“已经被完整解决”。
> 你描述的是一个更大的统一问题：**异构计算资源上的在线队列控制与任务配置选择**。已有工作覆盖了很多局部，但“把多 CPU、多 GPU、共享 GPU、CPU/GPU 可替代、拓扑网络、统一内存、跨集群调度、反馈学习”放进一个统一 OR/排队论模型里，仍然有研究空间。

---

## 1. “没人做”这个说法不成立

你这个问题的很多子块已经被做过，而且不少是很强的系统论文。

比如多资源分配这条线，**DRF** 已经把 CPU、内存、I/O 等多种资源的公平分配形式化为 dominant share 的 max-min fairness 问题，这是多资源调度的经典基础之一。

工业集群调度这条线也很成熟。Google 的 **Borg/Omega/Kubernetes** 经验论文明确讨论了如何把长期服务和 batch jobs 放在统一容器管理系统里，通过共享机器提高资源利用率、降低成本。

CPU/GPU 可替代资源也有人做。**AlloX** 研究的是 hybrid CPU-GPU clusters 中 job 可以在 CPU、GPU 或其他 accelerator 上运行但速度不同的问题；它把调度转成 min-cost bipartite matching，并且在小规模 CPU-GPU 集群和大规模模拟上做了在线调度实验。([Mosharaf Chowdhury][1])

深度学习/GPU 集群调度这条线更拥挤。一个 GPU datacenter scheduling survey 已经把 training 和 inference workload 的调度目标、资源消耗特征、已有 scheduler 做了分类，并指出传统 big data/HPC 调度器不能很好支持 DL workload 对 GPU 的利用。([arXiv][2])

多 GPU 一个任务的分布式训练也有很多。**Themis** 明确说 ML training workload 有两个特殊点：任务需要 gang-scheduled，而且性能对 placement 很敏感。([USENIX][3]) **Tiresias** 也专门处理分布式 DL job 的 scheduling 和 placement，并强调 DL 训练时长常常不可预测，SJF/SRTF 这类策略需要 ETA，而这在 DL 训练里经常不可靠。

异构 GPU 调度也有人做。**Gavel** 用 effective throughput 把现有调度策略推广到 heterogeneous accelerator clusters，而且显式考虑了 space sharing 和 placement sensitivity。([USENIX][4]) **Pollux** 更进一步提出 goodput，把系统吞吐和统计效率结合起来，并联合调整 GPU allocation、batch size、learning rate 等参数。 **Sia** 则是 heterogeneity-aware + goodput-optimized 的 ML cluster scheduler，专门针对 heterogeneous GPU types、GPU count、batch size、resource-adaptive jobs 之间的组合问题。([并行数据实验室][5])

跨地域/跨集群也有人做。**MAST** 处理的是 hyperscale geo-distributed datacenters 里的 ML training global scheduling，目标是同时放置数据和训练任务，避免区域之间 GPU demand 和 supply 失衡。 另一个方向如 **ResLake**，面向 geo-distributed data jobs，把 compute、storage、network 都纳入考虑，因为单看 WAN 或单一资源不够。([VLDB][6])

所以，“完全没人做”这个判断肯定不安全。

---

## 2. 但你说的“统一版本”确实还有空间

现有工作一般只覆盖下面某几项：

| 问题块                         | 已有工作很多吗 | 你的扩展                         |
| --------------------------- | ------: | ---------------------------- |
| 多资源公平分配                     |      很多 | 加入异构性能、干扰、拓扑                 |
| CPU/GPU 可替代调度               |       有 | 加入多任务共享单卡、多卡 gang scheduling |
| 多 GPU 一个任务                  |      很多 | 加入一张 GPU 上多任务共驻和动态回滚         |
| 多任务共享一个 GPU                 |       有 | 加入在线学习 sweet spot 和状态相关服务率   |
| 跨集群/跨地域                     |       有 | 加入 GPU/CPU/统一内存/网络拓扑统一建模     |
| queueing control / learning |       有 | 加入真实 CS workload 的配置空间和共驻干扰  |

你的真正新意不是“调度任务”本身，而是：

> **把所有计算任务都表示成一组可选配置 configurations，然后在异构资源图上做在线 queueing control。**

这个抽象比较有潜力。

一个 job 不再只是：

```text
需要 1 张 GPU
```

而是：

```text
job j 有多个可行配置 C_j：

配置 1：1 个 CPU core，慢
配置 2：16 个 CPU cores，中等
配置 3：1 张 GPU，快
配置 4：4 张 GPU，同机 NVLink，很快
配置 5：4 张 GPU，跨两台机器，受 10GbE 限制
配置 6：和别的任务共享一张 GPU，吞吐取决于共驻组合
```

然后调度器每个时刻决定：

```text
哪个任务进入服务？
用哪个配置？
放在哪些节点/卡/CPU/内存域/网络链路上？
是否允许共驻？
是否暂停/回滚/迁移？
```

这就把你说的四个象限统一了：

| 象限            | 在统一模型里的解释                                       |
| ------------- | ----------------------------------------------- |
| 多 CPU 跑 1 个任务 | 一个 job 选择多 CPU configuration                    |
| 1 个 CPU 跑多个任务 | CPU 上的 processor sharing / time sharing         |
| 多 GPU 跑 1 个任务 | gang-scheduled distributed configuration        |
| 多任务共享 1 个 GPU | state-dependent processor sharing / co-location |

再加上拓扑后，资源不只是：

```text
CPU 数量
GPU 数量
内存数量
```

而是一个图：

```text
CPU socket
NUMA memory
PCIe root complex
GPU
NVLink
NIC
10GbE
rack network
WAN
```

这样，任务配置的服务率可以写成：

[
\mu_j(c, x_t)
]

其中：

* (j)：任务；
* (c)：选择的配置；
* (x_t)：当前系统状态，比如共驻任务、网络负载、NUMA 绑定、GPU memory bandwidth；
* (\mu_j)：有效训练/计算推进速度。

这个模型就比普通 Kubernetes bin packing、普通 DRF、普通 GPU cluster scheduler 更抽象。

---

## 3. 它有意义，但不能写成“所有 CS 任务调度”

我觉得有意义，而且方向不小。原因是现在真实计算平台越来越像你描述的样子：

```text
CPU / GPU / NPU / TPU / FPGA 混合
统一内存和独立显存并存
单机多卡和多机多卡并存
任务有的可并行，有的不可并行
任务有的能共享卡，有的共享就崩
通信拓扑差异巨大
训练、推理、仿真、数据预处理、RL rollout 混在一起
```

CPU-GPU workload placement 的近期综述也把这个问题说得很清楚：高效利用异构 CPU-GPU 系统需要把 workload 放到合适的处理器上，但最优 placement 非常复杂，受 processor 特性、workload 特性、data locality、runtime prediction 和 multi-device code 等因素影响。([UPCommons][7])

但是，论文里不能写成：

> 我解决所有 CS 任务调度。

这个太大，reviewer 会觉得没有边界。

更好的论文问题定义是：

> **Configuration-based online scheduling for heterogeneous compute fabrics with shareable and malleable jobs.**

中文就是：

> **面向异构计算 fabric 的配置化在线调度：同时处理可并行任务、可替代资源、资源共享和共驻干扰。**

这里的 compute fabric 可以包括：

```text
CPU cores
GPU cards
shared/unified memory
GPU memory
PCIe/NVLink/inter-node network
cross-cluster links
```

但你要限定 workload 类型，比如：

```text
batch compute jobs
ML/RL training jobs
LLM fine-tuning jobs
data processing jobs
可 checkpoint / 可 preempt 的长任务
```

这样才像一个可发表问题。

---

## 4. 最有可能成为论文贡献的点

我觉得比较有机会的贡献不是“再造一个 scheduler”，而是下面三个东西的组合。

### 第一，统一建模

你可以提出一个 **configuration-based queueing model**：

[
C_j = {c_{j1}, c_{j2}, ..., c_{jm}}
]

每个配置包括：

```text
需要哪些资源
是否需要 gang scheduling
是否允许 co-location
拓扑约束是什么
服务率是多少
切换/回滚成本是多少
```

比如：

```text
LLM 训练：
    配置 A：1 机 8 卡，NVLink，高吞吐
    配置 B：2 机 16 卡，10GbE，通信慢
    配置 C：1 机 4 卡，batch 小，吞吐中等

RL 训练：
    配置 A：1 张 GPU + 32 CPU cores
    配置 B：与其他 3 个 RL job 共享 1 张 GPU
    配置 C：CPU-only rollout + GPU learner

普通 CPU job：
    配置 A：1 core，慢
    配置 B：64 cores，快但非线性加速
```

这个统一模型本身有价值。

---

### 第二，服务率是状态相关的

你真正关心的是：

[
\mu_j(c, x_t)
]

而不是固定服务率。

比如同一个 job：

```text
单独跑在 GPU 0：1000 steps/s
和 3 个轻任务共驻：850 steps/s
和 1 个 memory-bandwidth-heavy 任务共驻：400 steps/s
跨 10GbE 多机：300 steps/s
在统一内存 ARM 机器上：600 steps/s，但无显存拷贝开销
```

这正好连接到排队论里的 **state-dependent service rate** 和 **processor sharing**。你之前说“找每张卡的 sweet spot 比排队顺序更重要”，在这个模型里就是：

[
\max_k \mu(k)
]

其中 (k) 是同一资源上的并发度。

---

### 第三，在线学习 + 控制

你不可能预先知道所有 (\mu_j(c, x_t))，所以要边跑边学。

一个自然策略是：

[
\text{score}(j,c)
=================

## Q_j(t)\hat{\mu}_j(c,x_t)

## \text{interference penalty}

## \text{switching cost}

\text{network cost}
]

然后每轮选 score 最高的可行配置。

这个可以接 MaxWeight / backpressure，也可以接 bandit/RL。比如已有 queueing-learning 方向已经有 **MaxWeight + discounted UCB** 这种方法，用来在未知、甚至非平稳服务率下同时学习和调度。([Proceedings of Machine Learning Research][8])

你可以不一开始就端到端 RL。更稳的论文路线是：

```text
MaxWeight / queueing control 骨架
+ UCB / Thompson / Bayesian learning 学服务率
+ MPC 做短期重规划
+ rollback / checkpoint 成本建模
```

这样比“我用 PPO 训练一个 scheduler”更容易被 OR 或 SIGMETRICS 接受。

---

## 5. 能不能发 Operations Research？

我的判断：

> **可以作为 OR 方向，但必须有强数学贡献。**
> 如果只是一个系统实现或 benchmark，发 Operations Research 很难；如果有新的排队/调度模型、结构性结论、近似/稳定性/ regret 保证，再加真实系统验证，就有可能。

Operations Research 官方说明里说，它作为 OR 领域 flagship journal，关注的是用 analytical methods 改善 decision-making，并要求文章 operations-focused、scientific、broad；它也明确欢迎新问题域、新数学 formulation、新 methodology。 提交指南也要求 introduction 清楚说明问题、结果以及对 OR 社区的重要性，并且现在也期望提供 code/data 说明。([PubsOnLine][9])

所以你这题投 OR 的关键不是“CS 系统很重要”，而是要回答：

```text
这是不是一个新的 OR 问题？
相比已有 stochastic processing network / queueing control / scheduling 有什么新结构？
有没有可证明的策略？
有没有广泛适用的决策洞察？
```

如果你的贡献只是：

```text
我写了一个调度器，在我的服务器上比 FIFO 快 30%
```

那更像：

```text
MLSys / SC / HPDC / SoCC / SIGMETRICS / NSDI / OSDI
```

如果你的贡献是：

```text
提出 configuration-based stochastic processing network；
证明某类 Goodput-MaxWeight 策略 throughput-optimal；
给出 state-dependent sharing 下的 admission threshold / sweet spot 结构；
在未知服务率下给出学习 regret 或稳定性保证；
用真实 GPU/CPU/RL/LLM workload 校准模型；
```

那就更像 OR / Stochastic Systems / INFORMS Journal on Computing / Queueing Systems。

---

## 6. 我会把论文缩成这个核心问题

不要写：

> Universal scheduler for all CS tasks.

可以写：

> **Online Scheduling of Malleable and Shareable Jobs on Heterogeneous Compute Fabrics**

或者：

> **Configuration-Based Queueing Control for Heterogeneous CPU-GPU Clusters**

核心设定：

```text
1. 有很多 job classes。
2. 每个 job class 有多个 feasible configurations。
3. configuration 可能使用一个资源，也可能 gang 使用多个资源。
4. 多个 job 可以共享同一资源，但服务率随共驻状态变化。
5. 资源之间有拓扑距离和网络带宽。
6. 服务率未知，需要在线学习。
7. 目标是最小化平均 JCT / makespan，或最大化 weighted goodput，同时控制公平性和切换成本。
```

这就能覆盖你的四象限，但不会显得无限大。

---

## 7. 一篇 OR 级论文需要的理论骨架

可以考虑三个层次。

### 层次 A：容量区域

定义所有可行配置集合 (\mathcal{A})，每个 action (a \in \mathcal{A}) 对不同 job class 提供服务率向量：

[
\mu(a)
]

容量区域是：

[
\Lambda = \text{conv}{\mu(a): a \in \mathcal{A}}
]

也就是说，只要到达率 (\lambda) 落在这个区域内，理论上系统可稳定。

### 层次 B：调度策略

用类似 MaxWeight 的策略：

[
a_t
===

\arg\max_{a \in \mathcal{A}}
\sum_i Q_i(t)\hat{\mu}_i(a)
---------------------------

\text{cost}(a)
]

其中 cost 包括：

```text
切换成本
跨网络成本
回滚成本
共驻干扰惩罚
公平性惩罚
```

### 层次 C：未知服务率学习

因为 (\mu_i(a)) 不知道，所以你估计：

[
\hat{\mu}_i(a)
]

可以给出：

```text
已知服务率下的 stability proof
未知服务率下的 learning regret / stability bound
非平稳服务率下的 discounted estimator
```

这会比纯工程 scheduler 更像 OR。

---

## 8. 你现在的服务器可以作为最小实验平台

你自己的机器其实很适合作为 paper prototype：

```text
8 张 GPU
大量 CPU cores
每张 GPU 多任务共驻
RL 任务有明显 CPU/GPU/带宽混合瓶颈
任务可 checkpoint / rollback
```

可以构造四类 workload：

```text
CPU-parallel job：多 CPU 跑一个任务
CPU-sharing job：多个轻任务抢 CPU
GPU-gang job：LLM / distributed training，占多卡
GPU-sharing job：多个 RL job 共享一张卡
```

然后测试不同拓扑：

```text
同 GPU
同机不同 GPU
同 NUMA socket
跨 NUMA socket
跨 PCIe root
跨两台机器 10GbE
普通网络跨主机
```

指标不要只看利用率，要看：

```text
mean JCT
p95 JCT
makespan
weighted goodput
GPU effective throughput
CPU/GPU/network bottleneck time
rollback overhead
fairness
energy / cost，如果能测
```

---

## 9. 我对“有意义吗”的最终判断

有意义，但最有意义的部分不是“做一个万能调度器”，而是：

> **提出一个统一抽象，把可替代资源、可并行任务、共享干扰、拓扑成本和在线学习放进同一个 queueing-control 框架。**

这个问题在系统界有现实需求，在 OR 界也有模型价值。

不过，想发 **Operations Research**，你要避免变成“系统工程论文”。最稳的路径是：

```text
理论模型 + 可证明策略 + 真实系统校准 + 大规模模拟
```

一句话总结：

> **不是没人做；但你描述的“统一的、状态相关的、拓扑感知的、可学习的异构计算排队控制”还没有被一个标准模型完全吃掉。**
> 如果你能把它收敛成 configuration-based queueing control，并证明一些稳定性/近似/阈值/学习性质，它是有 OR 潜力的。

[1]: https://www.mosharaf.com/wp-content/uploads/allox-eurosys20.pdf "AlloX: Compute Allocation in Hybrid Clusters"
[2]: https://arxiv.org/pdf/2205.11913 "Deep Learning Workload Scheduling in GPU Datacenters: Taxonomy, Challenges and Vision"
[3]: https://www.usenix.org/conference/nsdi20/presentation/mahajan "Themis: Fair and Efficient GPU Cluster Scheduling  | USENIX"
[4]: https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak "Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads | USENIX"
[5]: https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sia_sosp23-final.pdf "Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling"
[6]: https://www.vldb.org/pvldb/vol17/p3934-kashaf.pdf "ResLake: Towards Minimum Job Latency and Balanced Resource Utilization in Geo-distributed Job Scheduling"
[7]: https://upcommons.upc.edu/bitstreams/85fd168e-8c7b-442f-923e-ea166c79ef5b/download "Workload Placement on Heterogeneous CPU-GPU Systems"
[8]: https://proceedings.mlr.press/v206/yang23d.html?utm_source=chatgpt.com "Learning While Scheduling in Multi-Server Systems With ..."
[9]: https://pubsonline.informs.org/page/opre/submission-guidelines "Submission Guidelines | Operations Research"

我：
你觉得BAPR和BAPR-HRO这两篇论文的算法，能解决这个configuration-based问题么？目前configuration-based是怎么解决的？我以为的是，本质还是个多场景的模式识别问题

GPT：
我的判断是：

> **BAPR 和 BAPR-HRO 不能“直接解决”完整的 configuration-based 调度问题，但它们能提供两个很关键的模块：**
> **BAPR 负责识别/跟踪场景 regime；BAPR-HRO 负责在固定候选配置集合上做鲁棒重排序。**
> 真正完整的 configuration-based scheduler 还需要第三层：**带资源约束的组合优化 / queueing control 层**。

换句话说，你说“本质还是个多场景的模式识别问题”，这个直觉对了一半。它确实有很强的模式识别成分，但完整问题不是“识别出场景以后选一个 label”这么简单，而是：

[
\text{mode recognition}
+
\text{configuration scoring}
+
\text{constrained combinatorial control}
]

---

## 1. BAPR 能解决哪一部分？

BAPR 最有用的是这部分：

> 当前系统处在哪个 regime？
> 是正常模式、带宽瓶颈模式、CPU rollout 瓶颈模式、跨 NUMA 模式、10GbE 通信瓶颈模式，还是统一内存 contention 模式？

BAPR 论文的核心是 **piecewise-stationary control**：系统长时间处在一个稳定 regime，随后突然发生 regime change。它用 BOCD 做在线 change detection，用 context-conditioning/RMDM 做 mode 表示，并用 adaptive conservatism 在 regime 变化后更保守、稳定后再放松。论文里还把 BAPR operator 写成 mode-conditional Bellman operator 的 belief-weighted mixture，并强调 belief/penalty 冻结时才保留 contraction 结构。

这和你的集群场景非常像。比如：

```text
regime 1: 单机 NVLink 多卡，通信很快
regime 2: 单机 PCIe 多卡，通信中等
regime 3: 两台机器 10GbE，多卡训练通信极慢
regime 4: 多 RL 任务共用一张 GPU，显存够但 memory bandwidth 饱和
regime 5: ARM 统一内存，CPU/GPU 拷贝成本低但共享内存带宽紧张
regime 6: WAN 跨网训练，通信成本主导
```

BAPR 很适合做这种 **regime belief**：

[
b_t(z) = P(\text{当前系统处于 regime } z \mid \text{监控信号})
]

监控信号可以是：

```text
GPU effective throughput residual
GPU memory bandwidth
PCIe/NVLink/NIC throughput
CPU context switch / run queue
NUMA remote access
env steps/sec
samples/sec
rollback/OOM events
ETA prediction error
```

所以，**BAPR 可以作为 configuration-based scheduler 的“场景识别 + 非平稳适应”模块**。

但它不够解决完整调度，因为 BAPR 原问题是一个 RL agent 在连续控制环境里选动作，不是同时给几百个任务选择配置、满足 CPU/GPU/内存/网络/拓扑约束。它没有天然处理：

```text
一个任务占 8 张 GPU 时，别的任务就不能用这些 GPU；
一个任务跨两台机器时，会占用网络带宽；
多个任务共用一张 GPU 时，彼此 throughput 会变；
一个任务被放进某个配置后，会改变其他任务配置的收益。
```

这些是组合优化和排队控制，不只是 RL mode adaptation。

---

## 2. BAPR-HRO 反而更接近 configuration-based 的“结构”

BAPR-HRO 这篇对你的问题更有启发。它的核心观点是：

> 不要在 disruption 后重建整个 hyperpath；
> 保留已有候选结构，只在线重新排序候选项。

论文里明确说，disruption 发生后，原始 hyperpath 通常不是“候选集合错了”，而是“候选排序错了”；所以它采用 keep-and-rerank，而不是 recompute。它还给每个候选 connection 一个 pessimistic posterior score，并证明四项 posterior-risk core 在单候选层面等价于 Wasserstein-1 DRO worst-case expected arrival。

这个思想非常适合迁移到 configuration-based scheduling。

把 transit hyperpath 里的候选 fallback：

```text
line 402
line 311
line 102
line 317
```

替换成 job 的 candidate configurations：

```text
config 1: 1 CPU core
config 2: 64 CPU cores
config 3: 1 GPU
config 4: 4 GPUs, same host
config 5: 8 GPUs, same host NVLink
config 6: 8 GPUs, 2 hosts, 10GbE
config 7: shared 1 GPU with 3 other RL jobs
config 8: ARM unified memory CPU/GPU mode
```

那么 BAPR-HRO 的哲学就变成：

> 不要每次从头搜索所有可能配置；
> 先保留一个结构化的 candidate-configuration hyperpath，
> 然后根据当前 posterior、uncertainty、failure risk、topology risk、interference risk 重新排序。

这非常像你想要的东西。

---

## 3. 但 BAPR-HRO 也不能直接解决完整问题

BAPR-HRO 的 re-ranking 是 per-candidate 的：

[
score(c)
========

\text{nominal cost}
+
\beta \cdot \text{uncertainty}
+
\gamma \cdot \text{failure risk}
+
\text{reliability terms}
]

它适合：

> 一个乘客在一个站点，从若干候选车里选一个。

迁移到计算调度就是：

> 一个任务在若干候选配置里选一个。

但你的集群调度不是单个任务选配置，而是很多任务同时选配置。选择之间有强耦合：

```text
job A 选 8 GPU，会影响 job B 能否启动；
job A 和 job B 共用一张 GPU，会改变双方 throughput；
job C 跨两台机器，会占用网络，影响 job D 的 all-reduce；
job E 多 CPU rollout，会抢 job F 的 CPU memory bandwidth；
job G 进来以后，原来 job H 的最佳配置可能变了。
```

所以不能只做：

[
c_j^* = \arg\min_c score_j(c)
]

还要做：

[
A_t^*
=====

\arg\max_{A \in \mathcal{F}(x_t)}
\sum_{(j,c)\in A}
\text{value}_{j,c}(x_t)
-----------------------

## \text{interference}(A)

\text{switching cost}(A)
]

其中 (\mathcal{F}(x_t)) 是当前资源约束下可行的配置集合。

所以 **BAPR-HRO 可以给每个 configuration 打分，但最终还需要一个 OR/scheduling 层来选一个可行的 configuration set**。

---

## 4. 目前 configuration-based scheduling 通常怎么做？

现有系统界一般不是叫 “configuration-based”，而是叫这些名字：

```text
multi-configuration scheduling
resource-adaptive scheduling
malleable job scheduling
heterogeneity-aware scheduling
interchangeable-resource scheduling
goodput-optimized scheduling
co-location/interference-aware scheduling
```

主流做法大概是四步。

### 第一步：枚举候选配置

给每个 job 生成一组 configuration：

[
C_j = {c_{j1}, c_{j2}, ..., c_{jm}}
]

例如：

```text
CPU-only: 1, 8, 64, 256 cores
GPU single: 1 GPU
GPU gang: 2, 4, 8 GPUs
same-host multi-GPU
multi-host multi-GPU
shared-GPU mode
ARM unified-memory mode
```

AlloX 就很典型：它研究同一个 job 可以有 CPU configuration 或 GPU configuration，CPU/GPU 是 interchangeable resources，且每种资源上的 processing rate 不同；它把调度问题转成 min-cost bipartite matching，并在 Kubernetes 上做在线 profiling 来估计 processing time。([Mosharaf Chowdhury][1])

---

### 第二步：估计每个配置的速度 / goodput

对于每个配置，估计：

[
\mu_j(c, x_t)
]

也就是 job (j) 在配置 (c)、当前系统状态 (x_t) 下的有效服务率。

对于 ML/DL 训练，很多系统不用“GPU 利用率”，而用 **goodput / effective throughput**。Pollux 就是代表：它监控训练中的 job，建模增加或减少资源后 goodput 如何变化，并动态重分配资源，同时考虑 fairness。([arXiv][2])

Sia 更接近“配置空间很大”的版本。它同时考虑 GPU type、GPU count、batch size、job resource-adaptivity，并用 ILP + online throughput model 去做 heterogeneity-aware goodput optimization。Sia 论文明确指出，配置空间很大，穷举 profiling 太贵，所以它只用少量初始 profiling 来 bootstrap throughput model，然后运行中继续 refine。([并行数据实验室][3])

---

### 第三步：解一个优化问题

有了配置集合和收益估计以后，scheduler 每个周期解：

```text
哪些 job 应该跑？
每个 job 用哪个 configuration？
放在哪些机器 / GPU / CPU / NUMA / 网络路径上？
```

常见求解方式包括：

```text
greedy heuristic
min-cost matching
linear programming
integer linear programming
knapsack / generalized assignment
round-based allocation
MaxWeight / backpressure-style online control
```

Gavel 是 heterogeneity-aware DL scheduling 的代表，它把不同 scheduling policy 表达成 optimization problem，并考虑不同模型在不同 accelerator 上的 performance heterogeneity，以及 space sharing。([arXiv][4])

Sia 则是在更大的配置空间里用 ILP 和剪枝来处理 GPU type、GPU count、batch size 等组合。([并行数据实验室][3])

---

### 第四步：用反馈更新模型

因为 (\mu_j(c,x_t)) 不准，所以现有方法通常会：

```text
profile
observe actual throughput
update model
periodically reschedule
checkpoint / preempt / migrate
```

这和你说的 dynamic control 是一致的。对于未知服务率的 queueing scheduling，MaxWeight + UCB 这类方法就是典型思路：一边学习 job type/server pair 的服务率，一边保持队列稳定。([arXiv][5])

---

## 5. 你的“四象限”在现有工作里是被分开处理的

你说的四个象限：

```text
多 CPU 跑 1 个任务
1 个 CPU 跑多个任务
多 GPU 跑 1 个任务
多任务共享 1 个 GPU
```

目前通常不是一个系统统一解决，而是分散在不同文献里：

| 象限            | 现有处理方式                                                                   |
| ------------- | ------------------------------------------------------------------------ |
| 多 CPU 跑 1 个任务 | malleable / parallel job scheduling，speedup curve，HPC/Slurm 风格           |
| 1 个 CPU 跑多个任务 | processor sharing，time sharing，multi-resource fairness                   |
| 多 GPU 跑 1 个任务 | gang scheduling，elastic DL scheduling，Pollux/Sia/Gavel                   |
| 多任务共享 1 个 GPU | GPU sharing / co-location / interference-aware scheduling，如 Salus、IADeep |

Salus 关注细粒度 GPU sharing primitives，比如 fast job switching、memory sharing、iteration scheduling，用这些 primitive 支持 fairness、priority、packing 等策略。([arXiv][6]) IADeep 则明确针对 single GPU 上 multiplex 多个 DL tasks 的 interference，核心是同时优化 cluster-level task assignment 和 device-level interference mitigation。([SC23][7])

所以我会说：

> 目前有很多局部解，但没有一个标准框架把你的四象限、跨 NUMA、跨机器、跨网络、统一内存、GPU co-location、CPU/GPU 可替代、多卡 gang scheduling 全部放进一个统一 configuration-based queueing-control 模型里。

---

## 6. 你说“本质是多场景模式识别”，怎么更准确？

我会改成：

> 它的预测层是多场景模式识别；
> 它的决策层是带约束的组合控制。

模式识别解决的是：

[
x_t \rightarrow z_t
]

也就是：

```text
当前是哪种场景？
这个 job 和这个 configuration 的组合像哪种历史情况？
这个配置在当前 regime 下会不会慢？
```

但 scheduler 还要解决：

[
z_t, Q_t, C_j \rightarrow A_t
]

也就是：

```text
识别出场景以后，到底应该同时启动哪些任务？
哪些任务等待？
哪些任务共享卡？
哪些任务独占卡？
哪些任务用 CPU？
哪些任务用多卡？
哪些任务跨机器？
哪些任务回滚？
```

这是优化 / 控制问题，不只是分类问题。

一个简单例子：

```text
模式识别结果：当前 10GbE 是瓶颈。
```

这只能告诉你：

```text
跨机器 8-GPU LLM 训练很差。
```

但它不能自动回答：

```text
那这 8 张 GPU 应该给一个 8-GPU job，
还是拆成两个 4-GPU job，
还是给 6 个 RL job 共驻，
还是让某些 CPU-heavy rollout 先跑？
```

后者需要资源约束、队列长度、任务优先级、切换成本、fairness、吞吐曲线一起进入优化。

所以更完整的说法是：

> **这是一个 mode-conditioned combinatorial queueing control problem。**
> 多场景模式识别是它的 state estimation layer。

---

## 7. BAPR + BAPR-HRO 可以怎么组合成你的 scheduler？

我会这样设计：

### 第一层：configuration hyperpath

给每个任务预生成候选配置：

```text
job j:
    c1 = 1 GPU, exclusive
    c2 = 1 GPU, shared with up to 4 RL jobs
    c3 = 2 GPUs same host
    c4 = 4 GPUs same host
    c5 = 8 GPUs same host
    c6 = 8 GPUs across 2 hosts over 10GbE
    c7 = CPU-only 64 cores
    c8 = ARM unified-memory CPU/GPU
```

这对应 BAPR-HRO 的“保留候选结构”。

---

### 第二层：BAPR-style regime belief

维护：

[
b_t(z)
]

例如：

```text
z1: GPU compute-bound
z2: GPU memory-bandwidth-bound
z3: CPU rollout-bound
z4: PCIe-bound
z5: NVLink-normal
z6: 10GbE-congested
z7: WAN-unusable
z8: unified-memory contention
```

这对应 BAPR 的 BOCD + context-conditioning。

---

### 第三层：BAPR-HRO-style configuration score

给每个 ((j,c)) 打分：

[
score(j,c)
==========

\widehat{T}*{j,c}
+
\beta \sigma*{j,c}
+
\gamma p_{\text{fail},j,c}
+
\lambda_{\text{topo}} \cdot \text{topology_cost}
+
\lambda_{\text{int}} \cdot \text{interference_risk}
+
\lambda_{\text{switch}} \cdot \text{switching_cost}
]

如果目标是最大化 goodput，也可以写成：

[
value(j,c)
==========

## \widehat{\mu}_{j,c}

## \beta \sigma_{j,c}

## \gamma p_{\text{fail},j,c}

\text{topology/interference/switching penalties}
]

这就是把 BAPR-HRO 的 LCB/DRO scoring 迁移到配置选择上。

---

### 第四层：OR / queueing control 选可行集合

最终不是对每个 job 独立选最高分配置，而是解：

[
A_t^*
=====

\arg\max_{A \in \mathcal{F}(x_t)}
\sum_{(j,c)\in A}
Q_j(t)
\cdot value(j,c)
----------------

\text{global coupling cost}(A)
]

其中：

* (Q_j(t))：任务等待压力 / age / priority；
* (\mathcal{F}(x_t))：资源可行集合；
* (A)：这一轮实际启动/保留/回滚的 job-configuration 集合。

这层才是真正的 scheduler。

---

## 8. 所以两篇论文能不能解决？

我的评价是：

| 组件                            | BAPR | BAPR-HRO | 对 configuration-based scheduler 的作用 |
| ----------------------------- | ---: | -------: | ----------------------------------- |
| 场景变化检测                        |    强 |   弱/不是重点 | 用 BAPR                              |
| regime / context 表示           |    强 |        弱 | 用 BAPR                              |
| 候选集合固定、在线重排序                  |    中 |        强 | 用 BAPR-HRO                          |
| posterior uncertainty penalty |    强 |        强 | 两者都能借                               |
| DRO/LCB scoring               |    中 |        强 | 用 BAPR-HRO                          |
| 多任务资源约束                       |    弱 |        弱 | 需要额外 OR 层                           |
| 多 job 组合优化                    |    弱 |        弱 | 需要 ILP / matching / MaxWeight       |
| queue stability               |    弱 |        弱 | 需要 queueing-control 理论              |
| co-location interference      | 需要扩展 |     需要扩展 | 可以作为 score/interference model       |

所以：

> **BAPR-HRO 更像“configuration candidate ranking”的框架；
> BAPR 更像“non-stationary regime recognition/control”的框架；
> 两者合起来仍然缺一个 constrained scheduler。**

---

## 9. 我觉得你的问题最好的抽象不是“直接用 RL 解决”

如果你一上来做端到端 RL：

```text
state = 整个集群状态
action = 给所有任务选配置
reward = 总 goodput / JCT
```

会非常难，因为 action space 爆炸，而且很多错误动作代价很高，比如 OOM、通信拥塞、训练回滚。

更稳的抽象是：

```text
BAPR-style mode belief
        ↓
BAPR-HRO-style robust scoring
        ↓
MaxWeight / ILP / matching / MPC scheduling
        ↓
feedback update
```

也就是：

> 用 learning 解决“识别和估计”，
> 用 OR 解决“可行组合选择”，
> 用 control 解决“滚动调整”。

这比“把整个 scheduler 交给 PPO/SAC”更像一个能发论文、能落地、也更容易证明性质的方案。

---

## 10. 最终一句话

你说“本质是多场景模式识别”很有洞察，但我会把它升级成：

> **configuration-based 调度的核心是：在非平稳场景下，对每个 job-configuration 的服务率/风险做模式识别与鲁棒估计，然后在资源约束和队列压力下做组合控制。**

BAPR 能做 **场景识别与非平稳适应**；BAPR-HRO 能做 **候选配置的鲁棒重排序**；但完整问题还需要一个 **configuration-level queueing/optimization controller**。这三个拼起来，才是你要的统一框架。

[1]: https://www.mosharaf.com/wp-content/uploads/allox-eurosys20.pdf "AlloX: Compute Allocation in Hybrid Clusters"
[2]: https://arxiv.org/abs/2008.12260 "[2008.12260] Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning"
[3]: https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sia_sosp23-final.pdf "Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling"
[4]: https://arxiv.org/abs/2008.09213 "[2008.09213] Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads"
[5]: https://arxiv.org/abs/2209.01126?utm_source=chatgpt.com "Learning While Scheduling in Multi-Server Systems with Unknown Statistics: MaxWeight with Discounted UCB"
[6]: https://arxiv.org/abs/1902.04610?utm_source=chatgpt.com "Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications"
[7]: https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap290.html "Interference-Aware Multiplexing for Deep Learning in GPU Clusters: A Middleware Approach"

我：评估下BAPR-HRO这篇论文

GPT：
我的总体评价：

> **BAPR-HRO 是一篇“想法很清楚、工程问题真实、实证结果有价值，但理论主贡献和标题叙事有点过重”的论文。**
> 作为 OR 论文，它现在最强的是 **operational insight + real-data validation**；最弱的是 **DRO 理论核心其实只覆盖了完整算法的一小部分，而且这个理论本身偏基础**。

我会把它评成：

| 维度      | 评价                                                       |
| ------- | -------------------------------------------------------- |
| 问题重要性   | 高                                                        |
| 操作洞察    | 强                                                        |
| 算法原创性   | 中等偏强                                                     |
| OR 理论深度 | 中等偏弱                                                     |
| 实证质量    | 中等偏强                                                     |
| OR 接收概率 | 目前偏 borderline，需要强化                                      |
| 更合适期刊   | Transportation Science / INFORMS JOC / MSOM / OR 取决于重写角度 |

---

## 1. 这篇论文真正的核心贡献是什么？

我觉得核心不是 “Wasserstein DRO”，而是这句话：

> **disruption 改变的是候选项的排序，不一定改变候选项集合。**

也就是论文里反复讲的：

```text
keep the hyperpath structure, re-rank the candidates online
```

这个 insight 很好。它和你之前讲 configuration-based scheduler 的关系也很直接：

```text
不要每次重新生成所有 configuration；
先保留一个结构化候选集；
然后根据实时状态重新打分 / 排序。
```

论文在 transit 场景里的说法是：

```text
stochastic hyperpath under disruption is structurally robust but ranking-fragile
```

这个说法是非常好的。论文也做了一个关键负结果：BOCD + recompute 并没有比 static hyperpath 好，原因是 recomputation 可能过度避开 disrupted corridor，从而丢掉仍然可用的 fallback；keep-and-rerank 反而保留这些 fallback。这个是整篇最有价值的 operational finding。

---

## 2. 算法本身其实很简单

算法可以压缩成：

```text
每个 stop 有一组 hyperpath candidates；
实时更新 route-level delay posterior 和 cancellation posterior；
给每个 candidate 一个 pessimistic score；
选 score 最低的 candidate。
```

核心 score 是：

[
\text{score}(c)
===============

\mathbb{E}[T_{\text{dest}}(c)]
+
(\hat{\mu}_r - \mu_0)
+
\beta \hat{\sigma}*r
+
\gamma \hat{p}*{cancel,r}
]

然后 Swiss 实验里用的完整 score 又加了两个结构可靠性项：

[
\lambda_{feas}(1-\phi(c))
+
\lambda_{time}(1-P_{ontime}(c))
]

也就是说，完整算法不是纯 posterior-risk scoring，而是：

```text
posterior delay/cancel risk
+
hyperpath structural reliability
+
online ranking
```

这点很关键。论文自己也承认，Lean 形式化和 Wasserstein DRO 只认证了前四项 posterior-risk core，完整 Swiss ranking objective 的两个 structural reliability terms 不属于 DRO identity，只作为 bounded perturbation 和实证部分处理。

---

## 3. 最大优点：问题定义非常“OR”

这篇论文的 OR 味道主要来自三个地方。

第一，它不是单纯 ML benchmark，而是一个真实运营问题：

```text
公共交通 disruption 发生时，乘客/系统应该如何在线选择 fallback connection？
```

第二，它提出了一个很清楚的 operational policy：

```text
heavy offline planning：预先算 hyperpath；
light online decision：实时重排序。
```

第三，它强调单次旅程是 **single-shot decision**，所以 Thompson Sampling、BAMCP、EXP3 这类探索型方法在 transit 里会有结构性不适配。论文说得很对：一次旅程里探索错了一个 cancelled route，损失无法在同一旅程里补回来；所以 pessimism 比 exploration 更合理。这个 framing 是有价值的。

这点对你的 configuration-based scheduler 也有启发：

```text
如果错误配置代价不可逆，比如 OOM、长时间阻塞、checkpoint rollback 成本高，
那么探索式 RL scheduler 可能不如 pessimistic ranking / safe admission control。
```

---

## 4. 实证部分有亮点

论文的实证不是只有 synthetic。它用了 Zürich 的 GTFS / GTFS-RT 数据，35 天，687 stops，2,000 connections，31 routes，约 358k delay observations；主实验是 35 天 × 18 ODs × 45 trials = 每个方法 28,350 journeys。这个规模在交通 routing 论文里算不错。

主结果也挺清楚：

```text
Static:    E[total] = 49.34 min, reach = 78.2%
V1-LCB:    E[total] = 46.76 min, reach = 81.0%
V2-LCB:    E[total] = 46.47 min, reach = 81.2%
Adaptiveβ: E[total] = 46.76 min, reach = 81.0%
```

也就是大约：

```text
5–6% expected total time reduction
+2.8 to +3.0 percentage points reach rate
```

paired analysis 也比较干净：V1、V2、Adaptive-β 在 35/35 天都优于 Static；V2 的 paired ΔE[total] 是 -2.87 min，95% CI 是 [-3.23, -2.53]。

另一个优点是它做了 leave-one-day-out prior audit，说明 historical prior 不是简单 data leakage：LOO 后各方法 cell-mean E[total] 变化不超过 0.02 min。这个是很加分的。

---

## 5. 但最大问题：DRO 理论不是驱动主结果的核心

这是我作为 reviewer 最会抓的问题。

论文标题叫：

> **A Wasserstein-Robust Scoring Approach**

但 Table 10 的 ablation 显示：

```text
posterior-risk score only 反而比 Static 差；
加 structural reliability objective 之后才变好。
```

具体说：

```text
Static baseline: 50.59

V1 calibrated priors + posterior-risk only: 52.34 (+3.5%)
V2 posterior-risk only: 53.61 (+6.0%)

V1 + structural reliability: 49.09 (-3.0%)
V2 + structural reliability: 46.99 (-7.1%)

Full keep-and-rerank:
V1: 47.44 (-6.2%)
V2: 47.14 (-6.8%)
```

这说明实证收益主要不是来自 Wasserstein posterior-risk core，而是来自：

```text
hyperpath reliability terms
cold-start floor
typed cancellation counters
adaptive candidate screening
historical priors
cross-journey tuning
```

论文自己也很诚实地承认：four-term posterior-risk core alone slightly worsens E[total] relative to static hyperpath，两个 structural reliability terms 是 observed gain 必需的。

这会让 OR reviewer 问：

> 那 Wasserstein DRO 到底是不是主贡献？
> 如果 DRO-certified core 单独不 work，而真正有用的是 hyperpath label reliability terms，那标题和理论叙事是不是偏了？

我的建议是：**不要把论文包装成 DRO 主导，而应该包装成 “risk-augmented hyperpath re-ranking”，其中 DRO 是 posterior-risk submodule 的解释。**

---

## 6. 理论部分：形式化很强，但数学新意一般

Lean 4 formalization 很认真，论文声称有 5,322 lines Lean code，覆盖 W1 pseudometric、Kantorovich-Rubinstein inequality、LCB=DRO identity、Bellman bounds、LCB gap、posterior contraction、Hedge regret 等。这个工程量很大，也确实能减少证明漏洞。

但从 OR 数学贡献角度看，核心等价：

[
\sup_{P: W_1(P,\hat{P}) \leq \epsilon} \mathbb{E}_P[f]
======================================================

\mathbb{E}_{\hat{P}}[f] + \epsilon
]

在 1-Lipschitz / affine cost 下，本质上是 Kantorovich-Rubinstein duality + translation witness。它是一个漂亮、干净、可验证的 identity，但不是特别深的 OR 理论结果。

更重要的是，论文自己也说：

```text
这是 per-candidate scalar identity；
不是 full BA-SSP-MDP 的 global-policy DRO optimality claim。
```

这个限定非常关键。

所以理论部分的问题是：

```text
Lean 形式化很强；
但形式化的是一个相对局部、相对基础的数学核心；
而完整算法的主收益来自理论未覆盖的 reliability terms。
```

如果投 OR，这会是一个 major concern。

---

## 7. Theorem 4 的 bound 对完整算法帮助有限

论文的 LCB suboptimality bound 是：

[
E[cost(\pi_{LCB})] - E[cost(\pi^*)]
\leq
2D(1+\beta)\sigma_{max}
]

这个对 four-term posterior-risk core 有意义。

但完整 objective 加了两个 structural reliability terms。论文承认，这会引入：

[
D(\lambda_{feas}+\lambda_{time})
]

这样的 additive constant，而且在它们的设置里这个 bound 对 single journey 是 vacuous。论文写得很诚实：formal theorem layer 对应 four-term core，full ranking objective 的价值靠 Table 10 实证说明。

这不是致命问题，但会降低 OR 理论贡献强度。

Reviewer 可能会说：

> 你证明了一个简单 score 的性质；
> 但真正 deployed policy 是另一个加了 reliability terms 的 heuristic；
> 所以理论和实证之间有 gap。

---

## 8. baseline fairness 有改善，但还不够

论文已经做了一个好的 fairness audit：它把 reliability terms retrofitted 到 Static 和 SW-LCB，发现 Static + reliability 已经从 50.59 降到 48.74，贡献了 LCB-V1 大约 60% 的 cell-mean gain 和几乎全部 reach-rate improvement。这个 ablation 很有价值。

但它也承认：

```text
PS-SSP、BAMCP、EXP3 没有 retrofitted reliability terms；
因为需要 loss-shaping 或 rollout adapters；
future work 再做。
```

这会削弱 “LCB beats theoretically stronger methods” 的结论。现在更稳妥的说法应该是：

```text
在当前实现和信息接口下，exploration-based methods 表现差；
但还不能完全排除 reliability-aware PS-SSP / BAMCP / EXP3 会追回一部分差距。
```

否则 reviewer 会觉得比较不公平。

---

## 9. 实证外推范围有限

论文自己也承认，Swiss benchmark 是：

```text
Paradeplatz-origin disrupted-route OD panel
not a uniformly sampled city-wide demand panel
```

OD 是刻意筛选的：每个 OD 的 hyperpath 里都同时有 disrupted route 和 safe route。这个设置很适合证明 adaptive ranking 有用，但不能证明 city-wide average superiority。论文在 conclusion 里也明确说，这不是 city-wide average superiority claim，需要 random-origin、city-wide OD panels 和 multi-city tests。

这是诚实的，但对 OR 来说会带来问题：

> 如果只有单城、单起点、筛选过的 disrupted-route stress-test panel，那么贡献更像一个 promising case study，而不是稳健的 generalizable operational method。

要冲 OR，最好至少加：

```text
1. random-origin / random-OD panel；
2. multi-city replication；
3. demand-weighted OD sample；
4. out-of-sample disrupted days；
5. live replay rather than empirical resampling；
6. different network structures：dense mesh vs sparse radial network。
```

---

## 10. 论文叙事有些过满

这篇论文想同时讲很多东西：

```text
hyperpath reranking
Wasserstein DRO
Bayesian posterior
LCB
single-shot irrecoverability
Lean formalization
Swiss GTFS-RT real data
bandit / Bayesian-RL mismatch
cross-domain scope study
adaptive beta
neural surrogate
```

问题是主线会被稀释。

我会建议把主线收紧成：

> **Disrupted stochastic hyperpaths are structurally robust but ranking-fragile.
> A reliability-aware pessimistic reranking rule improves reach and travel time;
> the posterior-risk core admits a Wasserstein-DRO certificate.**

这样会比现在标题里的 “Wasserstein-Robust Scoring” 更准确。

现在最容易被 reviewer 攻击的是：

```text
你说 DRO 是核心，但 ablation 说明 DRO core alone 不行。
```

所以应该主动降调：

```text
DRO is a certificate for one layer, not the whole story.
```

论文其实已经这么写了，但标题、abstract、contribution 排序还可以继续调整。

---

## 11. 对 OR 的可能性

我的判断：

> **有 OR 潜力，但目前更像 major revision / borderline。**

原因是：

OR 喜欢：

```text
新的运营问题抽象
清晰的决策模型
结构性理论
有说服力的实证
可推广洞察
```

BAPR-HRO 有：

```text
新的 operational insight：keep structure + re-rank
真实数据
清楚的负结果：recompute 不好
漂亮的局部 DRO certificate
```

但 OR reviewer 可能会认为：

```text
核心理论太局部；
完整算法偏 heuristic；
实证 stress-test panel 有选择性；
收益主要来自 reliability terms 而不是 DRO；
baseline 还没有完全公平 retrofit。
```

所以我会给现在版本的 OR 命中率一个保守判断：

```text
当前版本：弱 reject / major revision 概率较高
强修后：有机会
```

如果目标是 Transportation Science 或 INFORMS Journal on Computing，可能更自然，因为这篇的问题、数据和算法都是 transportation / routing / computational decision support 味道更重。

---

## 12. 如果我是 reviewer，我会怎么写 recommendation？

我大概会写：

> The paper identifies an interesting and practically relevant failure mode of disruption-aware hyperpath recomputation and proposes a computationally lightweight re-ranking policy. The real-data Zürich evaluation is promising, and the component ablations are unusually transparent. However, the theoretical DRO contribution certifies only a per-candidate posterior-risk core, while the main empirical gains rely on additional structural reliability terms outside the DRO identity. The evaluation is also limited to a selected single-city stress-test panel. I would recommend major revision before publication in a broad OR journal.

翻成中文就是：

```text
想法值得发；
现在证据和叙事还不够稳；
需要重新定位贡献，并补强泛化实验和 baseline。
```

---

## 13. 我会建议作者怎么改

### 第一，改标题

现在标题：

```text
Re-Ranking Stochastic Transit Hyperpaths Under Disruption:
A Wasserstein-Robust Scoring Approach
```

建议改成：

```text
Reliability-Aware Re-Ranking of Stochastic Transit Hyperpaths Under Disruption
```

副标题再说：

```text
A Wasserstein-Certified Posterior-Risk Scoring Layer
```

这样不会被 Table 10 反杀。

---

### 第二，把 Table 10 提前

Table 10 是整篇最重要的诚实证据之一。它应该更早出现，因为它告诉 reviewer：

```text
我们知道收益来自哪里；
不是在假装 DRO core alone 解决了一切。
```

---

### 第三，补 baseline retrofit

至少要做：

```text
PS-SSP + reliability terms
BAMCP + reliability-shaped rollout cost
EXP3 over reliability-augmented arms
Static + same posterior-risk features where applicable
```

哪怕这些方法仍然输，也会让结论更扎实。

---

### 第四，补 city-wide / multi-origin 实验

当前 stress-test panel 是合理的第一步，但 OR 需要更强外推。

最低限度：

```text
random-origin random-OD panel
demand-weighted panel
another Swiss city
sparse network vs dense tram mesh synthetic comparison
```

---

### 第五，强化理论和完整算法之间的桥

现在理论证的是 posterior-risk core，实证赢的是 full score。

可以考虑补一个：

```text
risk-augmented score 的 decision consistency 条件
```

例如在什么条件下：

[
\text{posterior-risk} + \text{reliability penalty}
]

会比 static ranking 有更低 expected timeout risk。

或者证明：

```text
structural reliability terms are sufficient statistics of hyperpath fallback feasibility under certain assumptions
```

这会让 full objective 不只是 heuristic。

---

## 14. 和你关心的 configuration-based 问题的关系

BAPR-HRO 对你的 configuration-based scheduler 最有用的不是 transit 公式，而是这个方法论：

```text
预先维护候选配置集合；
不要每次全局重算；
实时更新 posterior / regime / reliability；
对每个 configuration 做 pessimistic score；
再交给 OR 层做可行组合选择。
```

也就是说：

```text
BAPR-HRO = configuration ranking layer
不是 full configuration scheduler
```

对于你的服务器调度问题，它可以变成：

[
score(j,c)
==========

-\widehat{goodput}(j,c)
+
\beta \cdot uncertainty(j,c)
+
\gamma \cdot failure_risk(j,c)
+
\lambda \cdot reliability_penalty(j,c)
+
switching_cost
]

然后再由 MaxWeight / ILP / greedy controller 选一组不冲突的 ((job, configuration))。

这正好对应你之前的判断：

> configuration-based 本质有很强的模式识别成分，但最终不是分类，而是 scoring + constrained control。

---

## 15. 最终评价

我会这样总结 BAPR-HRO：

> **这是一篇好想法论文，不是一篇很强理论论文。**
> 它最强的贡献是发现并验证了 “hyperpath structure should be kept, ranking should adapt” 这个 operational principle。
> Wasserstein DRO 和 Lean verification 是漂亮的 supporting layer，但不是主结果的主要驱动力。
> 要投 OR，需要把叙事从 “DRO scoring solves disruption routing” 改成 “reliability-aware reranking solves a ranking-fragile hyperpath problem, with a DRO-certified posterior-risk core”，并补强泛化实验与 baseline fairness。

如果按当前版本，我会给：

```text
Idea:        8/10
Empirics:    7/10
Theory:      5.5/10
OR fit:      6/10
After revision OR potential: 7/10
```

作为你这个 configuration-based 调度研究的组件，它很有参考价值；作为单独 OR paper，它需要更谨慎地定位贡献。

