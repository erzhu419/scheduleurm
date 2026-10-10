# 当前主结论：BG（2026-10-04 更新）

项目主算法采用已冻结并通过独立确认的 BG（`BackgroundCompetitionContinuousLaneRollout`）。在四个实测服务目录构建的固定率 Poisson 到达模拟中，相对精确枚举的单通道 MaxWeight，BG 改善整体平均流时间与尾部流时间。主要证据是新种子的确认实验。

96组配对轨迹覆盖4个目录、负载0.8/0.95与种子100001–100012；统计单位为12个种子块。192条主策略轨迹加96条独立MW引擎参考，共288条执行。出生cohort为[300,900]，到达与观察截至1200；策略使用8个共享规划样本，每条路径的未来事件上限为128。

| 确认指标 | MW | BG | 变化 |
|---|---:|---:|---:|
| 1200秒截尾平均流时间 | 31.6886秒 | 29.0502秒 | −8.33% |
| 1200秒逐轨迹共同完成集P90的等权均值 | 65.7411秒 | 54.5442秒 | −17.03% |
| 900秒cohort未完成数 | 1682 | 1554 | −128 |
| 1200秒cohort未完成数 | 2 | 2 | 持平 |

结论为`BACKGROUND_COMPETITION_CONTINUOUS_CONFIRMATION_PASS`；全部12个种子块整体mean改善，原16个节点×负载×截点组合门槛通过。P90为配对轨迹的共同完成ID分位数的等权均值，不是合并任务后的分位数。

复用全部确认缓存，按12个完整到达种子块做20,000次配对bootstrap，1200秒mean相对变化的点态95%区间为[−9.62%,−7.11%]，共同完成集P90为[−20.37%,−13.52%]。这项分析条件于固定规划随机流，沿用原确认结果，不构成新独立确认。[统计设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_confirmation_block_intervals_20261003_design.json)；[区间结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_confirmation_block_intervals_20261003_summary.json)。

BG让窗口后的背景到达持续参与FIFO/MW竞争，评分对象保留可见任务与一个τmax内的采样任务；在原聚合S2≤MW范围内选择严格改善S1的动作，溢出回退MW。[算法实现](/home/researcher/mine_code/scheduleurm/algorithm/background_competition_continuous_lane_rollout.py)；[确认入口](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_background_competition_confirmation.py)。

补充对照已完成192个新单元，原BG/MW轨迹全部复用；方法和预算提前冻结，同一96个到达tape分别运行批SJF及去背景竞争消融。

| 策略 | 1200秒截尾平均流时间 | 1200秒cohort未完成总数 |
|---|---:|---:|
| BG | 29.0502秒 | 2 |
| 去背景竞争 | 29.1669秒 | 2 |
| 批SJF | 222.0747秒 | 9357 |

去背景相对BG的mean差为+0.1167秒，种子块点态95%区间[−0.1521,+0.3660]秒，不能确认背景竞争提供额外mean改善；共同完成集P90差为+1.1927秒，区间[+0.3068,+2.0947]秒。该尾部增量主要来自gpu_a/.95，其P90差+12.4430秒；其余七个组的平均差为−0.4145秒，仅为描述性分组分析。去背景在54/96个tape上mean更低，类别组中有13/24的mean更低、13/24的共同P90更低，故不声称BG逐组支配。主体mean收益来自共享的基础rollout规则，背景分支目前支持有限、场景相关的额外尾部改善。

批SJF在全部96个tape上mean更差，严重积压；node007/.95共同完成集P90反而更低，伴随1200秒未完成数增加1526，不能脱离未完成数解释该分位数。该实验补充普通基线；预算匹配的S1-only rollout开发对照及原96个确认tape上的补充比较均已完成，结果见下文。[对照设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_strong_controls_20261003_design.json)；[完整对照结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_strong_controls_20261003_summary.json)。

预算匹配的S1-only开发对照已完成24条新轨迹，复用原BG、MW、去背景共72个策略轨迹。沿用原四目录×两负载×三个开发种子，仅撤去去背景规则中的S2≤MW候选约束；FIFO、因果count/now、τmax窗口、8个共享样本、128事件上限、MW续排及回退均保持一致。S2仍计算并计费，仅作为诊断量。实现的3项针对性测试与全部24条执行检查通过。

| 开发对照指标 | 去背景S1/S2 | S1-only | 移除约束后的变化 |
|---|---:|---:|---:|
| 1200秒截尾平均流时间 | 31.1700秒 | 31.8337秒 | +2.13% |
| 1200秒共同完成集P90的等权均值 | 59.0253秒 | 66.9263秒 | +13.39% |
| 900秒cohort未完成数 | 431 | 421 | −10 |
| 1200秒cohort未完成数 | 0 | 0 | 持平 |

三个种子块的整体mean和P90均在移除约束后变差，但14/24个tape的mean更低。损失主要集中在node007/.95：mean增加4.9521秒，P90增加57.2636秒，三个种子均退化；其余七组的等权平均差为+0.0511/+0.8492秒，仅作描述。S1-only相对MW仍有5.83%的整体mean改善，但gpu_a/.8的900秒未完成数和node007/.95两个截点的P90未通过原组门槛，结论为`ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_HOLD`。这支持在当前开发协议中保留S2约束，尚不构成独立确认或逐场景支配；S2评分本身也不是实际P90的保证。[对照设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_S1_only_development_20261003_design.json)；[完整结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_S1_only_development_20261003_summary.json)；[S1-only实现](/home/researcher/mine_code/scheduleurm/algorithm/continuous_arrival_s1_only_lane_rollout.py)。

finite/current-policy加深续排分支结束，保留其原full24 HOLD、全部实验代码和负结果。已冻结但未提交的[后续单动作对照](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_later_action_cross_20261003_design.json)作为未执行方案留存，不再推进。其局部条件效果不并入BG确认结果。论文源码暂不修改。

冻结目录已满足投影支配条件，BG的队列包络给出统一加性MW支持损失；在严格容量内部、独立Poisson到达及固定确定性服务等假设下，固定物理窗口漂移可推出异步lane的队列强稳定性。[条件性稳定性推导](/home/researcher/mine_code/scheduleurm/md/BG_async_stability_bridge_20261003.md)记录了常数与完整窗口论据。

局限：原确认的24个类别组有7个mean退化、8个P90退化；Linux/.95/LLM mean/P90增加10.46%/16.08%，11/12个种子mean退化。下述三个新规划流平均后的稳态类别mean/P90退化数为8/24、9/24，突发为9/24、10/24；原突发流及新稳态第二个流仍有组门槛HOLD。新汇总中突发Linux/.8/CNN的900秒未完成差值含`5.55e−17`浮点残差，实际三流平均23→23，按零解释，不计作积压退化。证据覆盖冻结的确定性测量服务模型、固定率及下述周期调制Poisson到达；原确认条件于固定规划流，新敏感性分析条件于三个预设流，尚不估计任意规划随机流的期望。计算费用独立记账且不占服务时钟，live placement集成尚未完成。条件性稳定性推导不提供P90改善或BG相对MW的性能增益定理。本结论不声称逐类支配、一般突发过程的稳健性或真实部署收益。

同一S1-only对手在原96个确认tape上的补充比较已全部完成（epoch：`2026-10-03T16:26:08.359966+00:00`）。复用BG、MW及去背景共288个策略轨迹，参数、全部组及全部种子保持不变；按12个完整种子块、20,000次bootstrap估计配对区间。三个对手的共同完成集分别重算，所以下表P90不是此前BG/去背景配对的P90。

| 补充对照指标 | 去背景S1/S2 | S1-only | 差值及点态95%种子块区间 |
|---|---:|---:|---:|
| 1200秒截尾平均流时间 | 29.1669秒 | 29.9977秒 | +0.8308秒 [+0.3739,+1.2540] |
| 1200秒共同完成集P90的等权均值 | 55.5667秒 | 62.4857秒 | +6.9190秒 [+3.3981,+11.0699] |
| 900秒cohort未完成数 | 1556 | 1608 | +52 [−19,+118] |
| 1200秒cohort未完成数 | 2 | 16 | +14 [+1,+31] |

移除S2后mean增加2.85%（区间[+1.29%,+4.26%]），P90增加12.45%（区间[+6.50%,+17.72%]）；两项均有9/12个种子块退化。node007/.95贡献mean/P90增量的52.9%/69.0%，其余七组的描述性均值差仍为+0.4470/+2.4483秒。该组新增14个最终未完成任务全部来自JAX matmul，该类mean/P90增加62.9926/93.1896秒，CNN与LLM反而改善；全24个类别组中分别有13个mean、15个P90退化。这支持保留S2筛选，同时明确其类别取舍。

S1-only相对BG的mean/P90增加3.26%/15.05%，仅作描述性对照。相对MW的整体mean仍改善5.34%，但node007/.95最终P90和未完成数、以及整体最终未完成数未通过原门槛，保留`ARRIVAL_AWARE_LANE_ROLLOUT_PERFORMANCE_HOLD`。实际option probes为S1-only 115,942,370、去背景114,833,363、BG 118,846,139；共同预算指每次规划的样本和事件上限一致，轨迹演化后的总费用分别记录。本轮属于观察过原确认与开发结果后的补充比较，区间条件于固定规划随机流，不构成新独立BG确认或实际P90保证。完整轨迹及逐tape明细留在服务器，本地仅保留143KB统计汇总。[补充设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_S1_only_supplemental_20261004_design.json)；[完整统计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_S1_only_supplemental_20261004_summary.json)；[完成凭据](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_S1_only_supplemental_20261004_completion_receipt.json)。

到达突发性实验已全部完成（epoch：`2026-10-03T17:08:54.918852+00:00`）：新种子200001–200012、两种到达模式、原八个目录/负载组合和四个策略，共768条新轨迹，统计单位为12个完整到达种子块。突发模式为100秒周期内50秒1.5倍/50秒0.5倍强度，独立随机相位，同一基准Poisson tape经时间变换生成；保持总任务数、300/900/1200边界数量和出生cohort任务ID一致。原服务模型、算法、预算与平均负载保持不变。下表为BG相对MW的1200秒流时间统计及点态95%配对块bootstrap区间；P90按各模式、各策略对分别重算共同完成集。

| 到达模式 | 截尾mean，MW→BG（秒） | mean相对变化 [95%区间] | 共同P90相对变化 [95%区间] | cohort未完成数，MW→BG（900 / 1200秒） | 原门槛 |
|---|---:|---:|---:|---:|---|
| 稳态 | 35.7577→33.2586 | −6.99% [−7.79%,−6.27%] | −13.01% [−14.90%,−11.39%] | 1743→1593 / 1→0 | PASS |
| 周期突发 | 36.6507→35.0571 | −4.35% [−5.32%,−3.44%] | −9.17% [−11.54%,−7.18%] | 2121→2081 / 0→0 | HOLD |

两种模式的全部12个种子块整体mean/P90均改善。突发模式在node007/.8的900秒未完成数200→213，增加6.5%，超过原5%容忍线；其中queued为90→93、inflight为110→120，CNN未完成数63→84，JAX/LLM分别减少7/1。node007/.95的最终mean/P90也小幅退化约0.10%/2.46%。因此保留`BG_EXTERNAL_PERFORMANCE_HOLD`，不调整门槛。突发减弱BG相对MW的绝对收益：配对效应差（突发−稳态）mean为+0.9055秒，区间[+0.4693,+1.2744]；共同P90为+2.8267秒，区间[+1.4992,+4.0720]。

机制对照继续支持S2筛选：撤去S2后，稳态/突发的最终mean增加3.46%/2.59%，P90增加16.68%/12.73%，最终cohort未完成数分别0→10、0→6。背景分支的额外最终mean改善仍未建立，去背景−BG的稳态/突发差为−0.1388/−0.1339秒，两区间均跨零；最终P90差的两区间也跨零。突发模式900秒P90差为+1.4672秒，区间[+0.6465,+2.1873]，仅支持该截点的有限尾部增量，保留原负证据。

首次合并t133870因遗漏`snapshots`失败；[独立分析修复](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_bg_bursty_merge_repair.py)仅补回该字段，t133878成功，未重跑仿真或更改冻结源码、协议及统计参数。本地仅拉取618KB完整统计汇总，轨迹留在服务器。[冻结设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_bursty_arrivals_20261004_design.json)；[完整统计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_bursty_arrivals_20261004_summary.json)；[完成凭据](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_bursty_arrivals_20261004_completion_receipt.json)。该周期模型的条件稳定性取完整周期物理窗口沿用原漂移论据，见[理论记录](/home/researcher/mine_code/scheduleurm/md/BG_async_stability_bridge_20261003.md)。论文源码未改。

规划随机流敏感性实验已完成（epoch：`2026-10-03T23:30:29.309041+00:00`）。三个预设新流复用上述192个到达tape和MW结果，新增576条唯一BG轨迹；样本数、事件预算、算法及原门槛保持不变。先在同一到达单元内平均三个流，再按12个完整到达种子块做20,000次配对bootstrap；同时报告全部三个流的独立分组结果，不选择最佳流。下表区间条件于这三个固定规划流，不将576条轨迹作为独立观测，也不构成新的独立到达确认。

| 到达模式 | 1200秒mean，MW→三流均值（秒） | mean相对变化 [点态95%区间] | 共同P90相对变化 [点态95%区间] | 900 / 1200秒未完成数，MW→三流均值 | 均值原门槛 |
|---|---:|---:|---:|---:|---|
| 稳态 | 35.7577→33.2738 | −6.95% [−7.99%,−5.94%] | −12.97% [−15.17%,−11.04%] | 1743→1605.33 / 1→0 | PASS |
| 周期突发 | 36.6507→35.0054 | −4.49% [−5.47%,−3.53%] | −9.74% [−11.58%,−8.14%] | 2121→2050 / 0→0 | PASS |

全部三个流在两种模式下的整体mean/P90区间均排除零，且各流的12个到达种子块均改善。稳态第二个流仍为HOLD：node007/.8的900秒未完成数138→148（+7.25%），queued为52→58、inflight为86→90；另外两个稳态流及三个突发流均PASS。原突发流的200→213 HOLD保持不变，三流均值PASS不覆盖它。node007/.8突发模式的CNN未完成数三流均值63→76，三个流分别77/75/76，类别取舍也仍存在。突发减弱相对MW收益的配对效应差仍成立：mean为+0.8386秒 [+0.5271,+1.1207]，共同P90为+2.3433秒 [+1.2079,+3.3904]。这些结果支持整体收益对本轮规划随机流变化稳定，不支持每个流、每个组都通过积压门槛。

执行中修复了SSH重试重复启动同一任务的问题；另以独立精确分数证明恢复了unit462被浮点序列化误判的严格S1改善（八条规划样本的S1总和差值为`1/140737488355328`），全部其余冻结检查不变。验证t134573、合并t134574均真实exit0，恢复未新增模拟。保留6个原始失败和18个重试失败记录；576是唯一科学轨迹数，已知实际模拟次数至少578，额外重复费用未知。完整轨迹留在服务器，本地只拉取816KB统计及小型凭据。[冻结设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_planning_streams_20261004_design.json)；[完整统计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_planning_streams_20261004_summary.json)；[完成凭据](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_planning_streams_20261004_completion_receipt.json)。论文源码未改。

服务波动补充的数据准备已完成：沿原组合规则，将四目录13轮来源映射到原35个动作（27 dedicated、8 loaded），保留455条动作×轮次测量记录及原排除规则。训练分布仅使用gate接受的1–12轮，第13轮单独留存；形成1,476个duration/offset联合支持点，loaded的12×12来自两个不同测量campaign的独立复合，不是144次完整动作实跑。其resident overlap仍使用同轮保守速率估计，因此本轮支持的是实测phase派生模型的波动分析，尚非完整动作真实联合时长分布或部署证据。[提取器](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_empirical_service_samples.py)；[625KB样本与均值目录](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_empirical_service_samples_20261004.json)。

经验均值与旧中位数/LCB目录不同，例如Linux LLM/p1平均duration为旧点模型的1.1813倍；已按`requirements/E[max(1,D)]`重算八组arrival rates，两个执行条件及两个策略共用新均值目录与出生tape。[随机服务执行器](/home/researcher/mine_code/scheduleurm/algorithm/experiments/empirical_service_lane_replay.py)在动作确定后整条抽取服务样本，只向策略暴露均值释放预测及已发生的完成状态。[冻结开发实验](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_empirical_service_development_20261004_design.json)已全部完成（epoch：`2026-10-04T01:02:43.343470+00:00`）：八组×三个联合arrival/service种子块×均值/随机执行×MW/BG，共96条唯一轨迹，t134607–t134608、t134609–t134702及合并t134703均真实exit0，0失败/重试、0 shadow/旧轨迹复用。原8项提取/执行器测试及新增9项入口/精确评分/统计测试通过；精确记录不改变BG动作或RNG、不重评分。沿用8样本/128事件和原1%/5%/积压门槛。

| 执行条件 | 1200秒mean，MW→BG（秒；相对变化） | 共同P90，MW→BG（秒；相对变化） | 900 / 1200秒cohort未完成数，MW→BG | 原门槛 |
|---|---:|---:|---:|---|
| 均值 | 35.4337→32.3186；−8.79% | 75.7791→63.0768；−16.76% | 400→378 / 0→0 | HOLD |
| 经验随机服务 | 35.8340→32.5337；−9.21% | 76.7359→64.1640；−16.38% | 410→380 / 0→0 | HOLD |

两个条件的三个联合种子块各自整体mean/P90均改善，7,146个cohort任务均在1200秒前完成；但900秒分组未完成数超过原5%容忍线。均值条件为node007/.8的40→47、node007/.95的82→88；随机条件为gpu_a/.8的35→38、node007/.95的84→91。重复失败的node007/.95在两条件下queued分别60→57、58→56，inflight分别22→31、26→35，LLM在途任务均增加10；均值node007/.8另有queued 16→21。因此未完成数退化包含不同的排队与在途构成，不能全归因于服务波动。24个类别中，均值/随机条件分别有7/5类mean退化、8/8类P90退化，保留全部明细。

配对的“随机−均值”BG相对MW效应差为mean −0.1852秒、P90 +0.1304秒、900秒未完成数−8；三个联合块只作描述统计，不给置信区间、不作为96个独立观测。[234KB完整统计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_empirical_service_development_20261004_summary.json)；[完成凭据](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_empirical_service_development_20261004_completion_receipt.json)。完整轨迹留在服务器，旧确认与全部HOLD保持不变，论文源码未改。

900秒诊断已复用全部96条保存轨迹、48个策略对，原480项计数和48个mean差均对齐；4项针对性测试通过，0新仿真/重评分。node007/.95第三个种子块的10个LLM任务在MW中已完成，在BG中则同处p10批次，均值/随机条件分别于901.373/902.248秒完成。均值条件该帧因规划预算溢出回退MW，随机条件也保留了同状态MW的p10选择，不能归因为该帧BG主动改选大批次。完成代价是真实的：均值条件任务(2,180)虽提前启动14.506秒，执行却延长36.527秒，最终晚完成22.021秒。

同一出生cohort在固定[850,950]窗口内的累计未完成时间，四个HOLD组均下降；但均值node007/.95完整[300,1200]累计量增加37.268 job-s、原tape等权mean增加0.03436秒。均值node007/.8的CNN mean增加1.78367秒，随机gpu_a/.8的LLM mean增加0.98675秒，均保留为类别代价。这是既有三块的事后分解，不是反事实干预或新确认；900门槛与HOLD不变。[诊断代码](/home/researcher/mine_code/scheduleurm/algorithm/experiments/bg_empirical_cutoff_diagnostic.py)；[52,769B压缩摘要](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_empirical_cutoff_20261004_compact.json.gz)，完整明细留服务器。已按冻结的BG、模型、8样本/128事件预算与原门槛完成12个全新arrival/service联合块两条件确认（384条新轨迹）；旧三块不并入确认统计。

冻结确认实验已完成（epoch：`2026-10-04T03:35:03.024872+00:00`）：使用未与开发块重叠的arrival seeds `300004–300015`和service seeds `400004–400015`，八组目录/负载、两种执行条件和MW/BG共384条唯一轨迹；注册、全部unit、合并及完成日志认证均真实exit0，0失败/重试，完整轨迹留在服务器。均值执行的1200秒cohort restricted mean为MW 35.4119→BG 32.4706秒（−8.31%），共同P90为76.4584→63.1952秒（−17.35%）；随机服务分别为35.2945→32.5861秒（−7.67%）和75.9062→63.5066秒（−16.33%）。两条件的900秒未完成数分别为1841→1724、随机为1868→1688，1200秒均为0，均通过原1%/5%/pooled-tail门槛，确认决定均为`BG_CONFIRMATION_PERFORMANCE_PASS`。12个联合块的整体mean/P90均改善；24类别中均值条件仍有6个mean、9个P90退化，随机条件有8个mean、8个P90退化；node007/.8和/.95的900秒组未完成数分别增加3和6（均值），仍在原5%组容忍线内。确认统计条件于固定phase-derived服务库和规划流，12块不是384个独立观测；不与开发三块池化，不自动修改BG或论文。[确认统计投影](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_empirical_service_confirmation_20261004_results_projection.json)；[完成凭据](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_BG_empirical_service_confirmation_20261004_completion_receipt.json)。

[确认结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_background_competition_continuous_confirmation_20260930.json)；[冻结协议](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_background_competition_continuous_confirmation_20260930_protocol.json)；[确认分析](/home/researcher/mine_code/scheduleurm/md/background_competition_continuous_20260930.md)；[finite负结果](/home/researcher/mine_code/scheduleurm/md/finite_BG_regression_20261003.md)。
