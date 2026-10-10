# 第三动作标签补齐（2026-10-02）

两个冻结状态各跑一次空干预和一次option5首动作干预，后续均由冻结旧current重排。两项测试通过，4条续排的首提案/队列/原生续排检查通过，空干预完整复现作业、动作、时钟及最终RNG/计数器；t120467/t120468均done/exit0。

两个第三动作均有条件增益：gpu_a负载0.95/seed43/frame76平均流时间−0.595632秒、P90−4.568137秒、900未完成−6；gpu_b负载0.95/seed42/frame125分别−2.241586秒、−14.085880秒、−11。补齐全部22个实际选择后，相对原有限两次current评分的选择，条件流时间差等权平均−0.141617秒、P90差平均−0.852087秒、900未完成差平均−0.636364个/状态，1200均无未完成。所有22个P90共同ID人口都覆盖完整出生cohort。

增益集中在这两个第三动作：它们贡献平均流时间净改善的91.1%、P90净改善的99.5%；其他20状态900未完成差合计反而+3。seed42平均流时间仍+0.100329秒、900未完成+0.5个/状态。两个第三动作分别使JAX/LLM类均值恶化1.264155/0.432946秒。数据支持进入一次固定方案的完整闭环性能试验，保留全部24条开发轨迹和原MW/BG联合门槛，并单列相对有限两次current父算法的性能增量；不筛掉seed42或只选这两个获益状态，不扫描深度/窗口。

局限：这是独立状态单动作干预的条件效果，不能将它们相加当作闭环性能增量。原完整24条HOLD保持；确认集未读取，论文未修改。

证据：[汇总](experiment_artifacts/aligned_hardware_current_policy_third_root_labels_20261002_summary.json)、[22状态摘要](experiment_artifacts/aligned_hardware_current_policy_third_root_labels_20261002_records_summary.json)、[完成记录](experiment_artifacts/aligned_hardware_current_policy_third_root_labels_20261002_completion.json)。完整轨迹留服务器，本地仅取回约57KB结果摘要。
