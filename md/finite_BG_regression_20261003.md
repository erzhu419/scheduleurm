# 原24条finite/BG回归分解

已完成48条原轨迹的缓存分析，全部总量与逐类900/1200指标复现；t130175 done/exit0，两项手算测试通过。只回传275,581字节JSON，零新增模拟/规划调用。首次入口缺旧BG协议，改用原finite冻结实验信息后完成，指标与对照未变。

finite−BG的1200秒等权轨迹mean为−0.675638秒：等待−0.711998、服务中时间+0.036360。16条改善、8条退化，原联合门槛仍HOLD。Linux/.95是唯一组mean回归：+0.330605秒，等待+0.455886、服务−0.125281；JAX等待对整群体均值贡献+0.578039，LLM反而贡献−0.240574。

900秒尾部需分开处理：GPU/.8未完成35→40，排队17→18、在途18→22；新增来自CNN+2/JAX+3，LLM持平。GPU/.95为68→75，排队40→47、在途28→28；Linux/.95为29→31，排队17→19、在途12→12。不能用统一LLM或排队护栏解释三处失败。

全24首次分歧均有共同物理前缀；4613个完成评分帧中593次偏离自身BG incumbent，最老头机会提前/延后/不变为224/223/146。此描述不是单动作因果标签。旧320状态的最老头硬保护会撤50个有益动作、18个有害动作，净条件mean退化，故不采用。

下一项已[冻结设计](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_finite_BG_first_action_cross_20261003_design.json)：Linux/.95全部三个seed自然首分歧，只恢复BG首动作，后续统一原finite自由续排，复用事实基线。锚点为120.208/916.316/73.324秒；916秒状态也保留，其900指标应不变。已完成三种子因果对照，详见[结果记录](/home/researcher/mine_code/scheduleurm/md/finite_BG_first_action_cross_20261003.md)；结果将是条件动作效果，不是新策略性能结论。论文未改。

[缓存分析程序](/home/researcher/mine_code/scheduleurm/algorithm/experiments/cached_finite_bg_regression.py)；[小摘要](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_cached_finite_bg_regression_20261003_summary.json)；[完成记录](/home/researcher/mine_code/scheduleurm/md/experiment_artifacts/aligned_hardware_cached_finite_bg_regression_20261003_completion_receipt.json)。
