已完成可见FIFO等待年龄与预测尾部约束算法，开发集和独立到达确认均通过原性能门槛。论文未改。

新规则在同8条独立规划人口上逐作业计算完整流时间S1与流时间平方和S2，只允许S2不高于同状态普通MaxWeight的根，再选择严格改善S1的动作。FIFO最老作业绑定最早完成offset，忙槽保留完整承诺；原窗口、事件上限128、队列范围、规划随机流和1%/5%门槛均保持。年龄加入S1只是各根共同常数，新增的平方成本约束才会阻止部分老作业被延期。

| 证据 | T平均流时间：MW→候选 | T逐带P90平均：MW→候选 | H未完成：MW→候选 |
|---|---|---|---|
| 开发：24对照、3种子 | 30.939790→28.812990秒（−6.8740%） | 62.545455→55.068202秒（−11.9549%） | 410→369 |
| 独立确认：96对照、12新种子 | 28.011822→26.081807秒（−6.8900%） | 54.014127→47.274216秒（−12.4781%） | 1616→1447 |

开发在北京时间08:13冻结并读取结果后，算法原样进入1001–1012种子的确认；确认在08:21:07冻结，参数与门槛未调整。独立确认的16个节点/负载/H,T分组全部通过，组汇总的均值、P90与未完成数均无回退；两负载T均值分别改善6.0660%/7.4578%。12个种子各自跨8场景的汇总均改善，范围1.8150%–9.5563%；84/96条带T均值改善。T时28419个作业全部完成，双方未完成数均为0。

慢类机制得到实际修复：node007 .95开发组P90由MW86.569、旧无约束98.418降至新81.317秒；独立确认该组P90改善3.3894%，JAX作业加权均值改善3.93%。确认中风险约束1346次改变同状态无约束最优根，node007 .95有226次改向JAX，减少了让快类批处理持续挤占慢类的情况。

本轮33项针对性测试通过。开发48条主轨迹；独立确认192条主轨迹＋96条旧精确控制参考，共288次确认执行，参考不进入主指标共同作业群。所有控制动作与作业时间复现、年龄/风险/预算检查通过；独立从实际作业记录重建共同群与分组指标，结论一致。

限制：证据属于已知到达率和H、目录报告2/4槽的确定性缓存独立槽模型，条件于固定规划随机流；有限窗S2不是实际P90保证。独立确认仍有12条带T均值、15条带T P90回退；node007 .95的JAX仅6/12种子P90改善，seed1005的JAX均值/P90分别回退12.28%/16.91%，这些结果保留。候选总目录探查53857046，MW142607（377.66倍），另有11744180次逐作业成本评估；执行钟未加入计算延迟，没有计算成本或物理部署优势结论。

下一步：在保持已确认规则的基础上，用仅依赖已观测到达的在线到达率估计替代已知率，再冻结配对性能验证，检查这6.89%的增量是否依赖生成模型参数。

代码：[选择器](/home/researcher/mine_code/scheduleurm/algorithm/age_tail_constrained_lane_rollout.py)、[年龄执行引擎](/home/researcher/mine_code/scheduleurm/algorithm/experiments/age_contextual_lane_refill_replay.py)、[确认实验](/home/researcher/mine_code/scheduleurm/algorithm/experiments/aligned_hardware_age_tail_constrained_confirmation.py)。

证据：[开发结果](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_age_tail_constrained_lane_rollout_development_20260930.json)、[独立确认](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_age_tail_constrained_confirmation_20260930.json)、[类别与动作分解](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_age_tail_constrained_confirmation_20260930_mechanism.json)。
