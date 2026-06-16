"""Build a reviewer-facing dashboard for Scheduleurm OR claim gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


GATES: tuple[dict[str, Any], ...] = (
    {
        "gate": "stability_theorem",
        "artifact": "../proof/ScheduleurmUpload.lean",
        "raw": "../proof/Scheduleurm/build.log",
        "scoped_claim": "Lean-backed robust candidate MaxWeight theorem spine and theorem-name crosswalk.",
        "scoped_key": None,
        "strong_claim": "Automatic production-wide stability without verifying model assumptions.",
        "strong_key": None,
        "blocker": "Operational claims still require the matching stochastic/load certificate.",
        "next_threshold": "Keep theorem names, build log, and no-sorry audit synchronized with the final manuscript.",
    },
    {
        "gate": "declared_finite_domain_positive_cover_gate",
        "artifact": "md/experiment_artifacts/declared_finite_domain_positive_cover_gate_20260612.json",
        "raw": "md/declared_finite_domain_positive_cover_gate_20260612.md",
        "scoped_claim": "Declared finite service-cache domain is classified and has positive lower-service rows or measured boundaries.",
        "scoped_key": "declared_finite_positive_cover_ready",
        "strong_claim": "Arbitrary all-state positive-service fabric cover.",
        "strong_key": "positive_service_all_state_cover_ready",
        "blocker": "Unmeasured future states must be probed before entering the positive theorem population.",
        "next_threshold": "Service-cache v2 with timestamps/sample windows plus perturbation profiling for any broader state universe.",
    },
    {
        "gate": "all_state_conservative_cover_gate",
        "artifact": "md/experiment_artifacts/all_state_conservative_cover_gate_20260612.json",
        "raw": "md/all_state_conservative_cover_gate_20260612.md",
        "scoped_claim": "All scheduler-visible states are safely partitioned into measured-admitted or zero-service probe/defer states.",
        "scoped_key": "all_state_safety_cover_ready",
        "strong_claim": "Positive-service stability for unknown future arrivals.",
        "strong_key": "positive_service_all_state_cover_ready",
        "blocker": "Unknown states have zero theorem service until measured.",
        "next_threshold": "Measure and admit future buckets or prove a finite all-state universe.",
    },
    {
        "gate": "future_production_admission_contract",
        "artifact": "md/experiment_artifacts/future_production_admission_contract_20260612.json",
        "raw": "md/future_production_admission_contract_20260612.md",
        "scoped_claim": "Strict telemetry contract routes measured production jobs to theorem trace and unknown jobs to probe.",
        "scoped_key": "future_production_automatic_theorem_closure_ready",
        "strong_claim": "Every future production job is theorem-grade without probe.",
        "strong_key": "future_jobs_all_theorem_grade_without_probe",
        "blocker": "Future unmeasured jobs must remain outside the positive theorem stream.",
        "next_threshold": "Keep strict admission on by default in trace-only production telemetry.",
    },
    {
        "gate": "gavel_service_unit_equivalence_certificate",
        "artifact": "md/experiment_artifacts/gavel_service_unit_equivalence_certificate_20260612.json",
        "raw": "md/gavel_service_unit_equivalence_certificate_20260612.md",
        "scoped_claim": "Bounded same-trace Gavel adapter compatibility and native simulator metric extraction for q01/q11.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Measured service-unit equivalence and direct full-stack same-workload superiority.",
        "strong_key": "gavel_service_unit_equivalence_ready",
        "blocker": "No calibration yet proves a Gavel simulator step equals a Scheduleurm measured lower-service unit.",
        "next_threshold": "Run q01/q11 service-unit calibration with holdout relative-error certificate.",
    },
    {
        "gate": "gavel_service_unit_calibration_gate",
        "artifact": "md/experiment_artifacts/gavel_service_unit_calibration_gate_20260612.json",
        "raw": "md/gavel_service_unit_calibration_gate_20260612.md",
        "scoped_claim": "q01/q11 profile-aware same-workload native Gavel simulator calibration against paired Scheduleurm holdout windows.",
        "scoped_key": "profile_aware_model_calibration_ready",
        "strong_claim": "Scalar service-unit equivalence between Gavel simulator time and Scheduleurm measured service units.",
        "strong_key": "gavel_service_unit_equivalence_ready",
        "blocker": "The profile-aware same-workload model passes, but the single scalar service-unit map fails the q01/q11 p95 relative-error threshold.",
        "next_threshold": "Promote only if a single service-unit scale reaches holdout p95 relative error <= 5% on the paired q01/q11 windows.",
    },
    {
        "gate": "gavel_resident_delay_jct_holdout_gate",
        "artifact": "md/experiment_artifacts/gavel_resident_delay_jct_holdout_20260613.json",
        "raw": "md/gavel_resident_delay_jct_holdout_20260613.md",
        "scoped_claim": "Measured corner-case resident-delay/JCT holdout shows immediate co-location beats defer under a resident-alone challenge rate.",
        "scoped_key": "gavel_style_resident_delay_jct_holdout_ready",
        "strong_claim": "Direct full-stack Gavel/Pollux/Sia/IADeep/Salus superiority.",
        "strong_key": "strong_claim_ready",
        "blocker": "Holdout is a measured-service Scheduleurm slice, not an external full-stack scheduler run.",
        "next_threshold": "Promote only after direct same-workload external full-stack execution with service-unit/JCT equivalence.",
    },
    {
        "gate": "sota_fullstack_superiority_gate",
        "artifact": "md/experiment_artifacts/sota_fullstack_superiority_gate_20260613.json",
        "raw": "md/sota_fullstack_superiority_gate_20260613.md",
        "scoped_claim": "Scoped same-host same-workload full-stack rows for Gavel, Pollux/AdaptDL, Sia, IADeep, and Salus, each with paired native superiority on the measured probe.",
        "scoped_key": "direct_fullstack_named_sota_superiority_ready",
        "strong_claim": "Directly beats arbitrary SOTA systems, arbitrary future workloads, multi-node original deployments, or production-wide organic traces.",
        "strong_key": "arbitrary_sota_superiority_ready",
        "blocker": "The named five rows are closed, but registered/unbounded SOTA systems and adjacent deployment scopes need their own comparable rows.",
        "next_threshold": "Use the registered-universe gate before any language broader than the five named same-host same-workload systems.",
    },
    {
        "gate": "sota_universe_registry_gate",
        "artifact": "md/experiment_artifacts/sota_universe_registry_gate_20260614.json",
        "raw": "md/sota_universe_registry_gate_20260614.md",
        "scoped_claim": "Named five direct full-stack superiority remains closed while additional registered SOTA systems are inventoried as pending.",
        "scoped_key": "named_five_fullstack_superiority_ready",
        "strong_claim": "Registered SOTA-universe or arbitrary SOTA superiority.",
        "strong_key": "registered_sota_universe_superiority_ready",
        "blocker": "Tiresias, Themis, Gandiva, Shockwave, AlloX, Optimus, and any new SOTA system still need same-host same-workload full-stack rows before universe-level language.",
        "next_threshold": "Add comparable native/full-stack rows for each registered system, then rerun the registry gate; arbitrary unbounded SOTA remains a non-finite claim.",
    },
    {
        "gate": "registered_sota_runtime_gate",
        "artifact": "md/experiment_artifacts/registered_sota_runtime_gate_20260614.json",
        "raw": "md/registered_sota_runtime_gate_20260614.md",
        "scoped_claim": "Runtime/paper inventory is complete for the registered SOTA extension set beyond the named five.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Registered extension systems have runnable same-workload full-stack rows.",
        "strong_key": "registered_extension_fullstack_superiority_ready",
        "blocker": "Tiresias and Optimus need Python2 dependencies, Shockwave needs stubs/Gurobi, AlloX needs a rebuildable Java simulator/full arguments, and Themis/Gandiva need artifacted implementations or reimplementation protocols.",
        "next_threshold": "Close entrypoint smoke and same-workload full-stack rows one registered system at a time.",
    },
    {
        "gate": "sota_admitted_universe_closure_gate",
        "artifact": "md/experiment_artifacts/sota_admitted_universe_closure_gate_20260614.json",
        "raw": "md/sota_admitted_universe_closure_gate_20260614.md",
        "scoped_claim": "Every registered non-adjacent SOTA system is represented by an admitted finite policy-family action in the Scheduleurm+SOTA measured-cache union.",
        "scoped_key": "registered_sota_policy_semantics_universe_ready",
        "strong_claim": "External-binary/full-stack superiority over every registered or arbitrary SOTA system.",
        "strong_key": "registered_sota_universe_superiority_ready",
        "blocker": "Policy-semantics admitted-action superiority is closed; external binary/full-stack superiority still needs same-service-unit adapters for registered systems without direct rows.",
        "next_threshold": "Use admitted-action language for the theory-facing SOTA universe; use direct-system language only for systems with same-workload full-stack rows.",
    },
    {
        "gate": "registered_sota_adapter_closure_gate",
        "artifact": "md/experiment_artifacts/registered_sota_adapter_closure_gate_20260614.json",
        "raw": "md/registered_sota_adapter_closure_gate_20260614.md",
        "scoped_claim": "Every registered non-adjacent SOTA family has a policy/service-unit adapter row in the finite theorem-facing action union.",
        "scoped_key": "registered_policy_adapter_universe_ready",
        "strong_claim": "Direct external-binary superiority for every registered SOTA system.",
        "strong_key": "registered_direct_external_binary_superiority_ready",
        "blocker": "Six registered systems still lack same-host same-workload executable direct-binary superiority rows.",
        "next_threshold": "Close direct executable adapter rows one system at a time; until then use policy/service-unit adapter language.",
    },
    {
        "gate": "future_workload_protocol_gate",
        "artifact": "md/experiment_artifacts/future_workload_protocol_gate_20260614.json",
        "raw": "md/future_workload_protocol_gate_20260614.md",
        "scoped_claim": "Future workload admission protocol routes measured profiles to theorem trace and unseen workload classes to probe-required status.",
        "scoped_key": "future_workload_protocol_ready",
        "strong_claim": "Arbitrary future workload positive-service theorem readiness.",
        "strong_key": "arbitrary_future_workload_theorem_ready",
        "blocker": "Unknown future jobs are deliberately excluded from the positive theorem stream until measured and admitted.",
        "next_threshold": "Promote a future workload only after an exact service certificate or declared finite-cover admission row exists.",
    },
    {
        "gate": "multinode_original_deployment_gate",
        "artifact": "md/experiment_artifacts/multinode_original_deployment_gate_20260614.json",
        "raw": "md/multinode_original_deployment_gate_20260614.md",
        "scoped_claim": "Same-host named full-stack evidence and Scheduleurm-native multi-node history are separated from external SOTA original-deployment claims.",
        "scoped_key": "scheduleurm_multinode_history_completion_ready",
        "strong_claim": "External SOTA original multi-node full-stack superiority.",
        "strong_key": "external_sota_original_multinode_superiority_ready",
        "blocker": "Scheduleurm multi-node history is closed, but external SOTA systems have not all been run through their original multi-node control planes.",
        "next_threshold": "Use Scheduleurm-native multi-node language; require per-system original deployment rows before external multi-node superiority language.",
    },
    {
        "gate": "multinode_history_completion_gate",
        "artifact": "md/experiment_artifacts/multinode_history_completion_gate_20260614.json",
        "raw": "md/multinode_history_completion_gate_20260614.md",
        "scoped_claim": "Scheduleurm-native strict production history spans multiple nodes and GPU nodes with launched/completed theorem-admitted rows.",
        "scoped_key": "scheduleurm_multinode_launched_completion_ready",
        "strong_claim": "Scheduleurm-native multi-node launched/completed production history for the theorem-facing population.",
        "strong_key": "scheduleurm_multinode_launched_completion_ready",
        "blocker": "External original-deployment comparisons remain separate.",
        "next_threshold": "Rerun after major scheduler-history changes and keep external SOTA control-plane language separate.",
    },
    {
        "gate": "multinode_theorem_shadow_gate",
        "artifact": "md/experiment_artifacts/multinode_theorem_shadow_gate_20260614.json",
        "raw": "md/multinode_theorem_shadow_gate_20260614.md",
        "scoped_claim": "Read-only theorem candidate-family shadow enumerates certified lower-service actions across visible GPU nodes and selects a cross-node robust-MaxWeight configuration.",
        "scoped_key": "multinode_theorem_shadow_ready",
        "strong_claim": "Original multi-node full-stack launched-completion superiority.",
        "strong_key": "multinode_original_launch_claim_ready",
        "blocker": "The cross-node theorem shadow is closed, but it is not a launched original-deployment comparison.",
        "next_threshold": "Run launched multi-node controlled traces when production safety and workload isolation permit.",
    },
    {
        "gate": "decima_spark_dag_gate",
        "artifact": "md/experiment_artifacts/decima_spark_dag_gate_20260614.json",
        "raw": "md/decima_spark_dag_gate_20260614.md",
        "scoped_claim": "Decima repository and Spark-DAG simulator import/baseline smoke are audited separately from GPU co-location claims.",
        "scoped_key": "spark_dag_simulator_smoke_ready",
        "strong_claim": "Scheduleurm direct full-stack superiority over Decima's Spark-DAG scheduling setting.",
        "strong_key": "direct_fullstack_gpu_sota_claim_ready",
        "blocker": "Decima is a Spark-DAG simulator; it needs a Spark-DAG workload and metric bridge instead of a GPU co-location probe.",
        "next_threshold": "Build a Spark-DAG benchmark bridge before making Decima-specific performance language.",
    },
    {
        "gate": "decima_same_domain_benchmark_gate",
        "artifact": "md/experiment_artifacts/decima_same_domain_benchmark_gate_20260614.json",
        "raw": "md/decima_same_domain_benchmark_gate_20260614.md",
        "scoped_claim": "Decima Spark-DAG same-domain paired simulator benchmark executes under fixed seeds and reports mixed performance rows.",
        "scoped_key": "spark_dag_same_domain_benchmark_ready",
        "strong_claim": "Scheduleurm GPU co-location full-stack superiority over Decima.",
        "strong_key": "direct_fullstack_gpu_sota_claim_ready",
        "blocker": "Decima is an adjacent Spark-DAG simulator and the same-domain rows are not a GPU co-location comparison.",
        "next_threshold": "Use only Decima same-domain execution/audit language unless a real Spark-DAG comparison objective is added to the paper.",
    },
    {
        "gate": "sota_candidate_union_gate",
        "artifact": "md/experiment_artifacts/sota_candidate_union_gate_20260613.json",
        "raw": "md/sota_candidate_union_gate_20260613.md",
        "scoped_claim": "Scheduleurm can evaluate a candidate set that explicitly contains external-policy-family actions; the fixed Pareto-slack online union policy closes both measured-cache external-policy envelopes within tolerance.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "The measured-cache Pareto-slack replay result directly dominates every external full-stack system on every metric.",
        "strong_key": "strong_claim_ready",
        "blocker": "The fixed Pareto-slack union closes the policy-semantics measured-cache envelopes, but this is still not an external binary/full-stack run.",
        "next_threshold": "Use Pareto-slack policy-semantics dominance for the theory-facing claim; promote direct-system language only after full-stack same-workload execution.",
    },
    {
        "gate": "measured_cache_external_policy_frontier",
        "artifact": "md/experiment_artifacts/sota_strict_dominance_frontier_20260613.json",
        "raw": "md/sota_strict_dominance_frontier_20260613.md",
        "scoped_claim": "The measured-cache external-policy frontier is explicitly diagnosed while the fixed Pareto-slack policy remains within the 0.5% replay tolerance.",
        "scoped_key": "within_tolerance_ready",
        "strong_claim": "The fixed Pareto-slack measured-cache policy strictly reaches ratio >= 1.0 in both metrics on every scenario, without implying direct full-stack SOTA binary superiority.",
        "strong_key": "strict_pareto_ready",
        "blocker": "If rows reopen, strict 1.0 measured-cache closure needs a certified finite trajectory/profile bridge, not full-stack SOTA language.",
        "next_threshold": "Promote only as measured-cache external-policy noninferiority; direct-system language requires the full-stack superiority gate.",
    },
    {
        "gate": "sota_bridge_action_gate",
        "artifact": "md/experiment_artifacts/sota_bridge_action_gate_20260613.json",
        "raw": "md/sota_bridge_action_gate_20260613.md",
        "scoped_claim": "Bridge-action search and safe-probe plan for measured-cache external-policy frontier rows.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Real bridge probes have closed the measured-cache external-policy frontier.",
        "strong_key": "strict_bridge_ready",
        "blocker": "Current measured phase-switch actions do not strictly close every measured-cache frontier row unless the bridge gate is rerun and passes.",
        "next_threshold": "Run prepared bridge probes when a target GPU is free, then admit only lower-confidence bridge rows that close the measured-cache external-policy frontier.",
    },
    {
        "gate": "sota_algorithm_upgrade_gate",
        "artifact": "md/experiment_artifacts/sota_algorithm_upgrade_gate_20260613.json",
        "raw": "md/sota_algorithm_upgrade_gate_20260613.md",
        "scoped_claim": "Opt-in SOTA-facing algorithm upgrade closes adaptive scalarized union, state-dependent marginal cache, bounded lookahead dispatch, reusable ETA/LCB, and expanded SOTA action families.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "The production scheduler default directly beats every external full-stack SOTA system.",
        "strong_key": "strong_claim_ready",
        "blocker": "The gate is replay/certificate evidence; launched production traces and direct external full-stack execution remain separate.",
        "next_threshold": "Enable the opt-in policy on controlled launches and collect global-action theorem traces before production-default language.",
    },
    {
        "gate": "sota_native_execution_attempts",
        "artifact": "md/experiment_artifacts/sota_native_execution_attempts_20260613.json",
        "raw": "md/sota_native_execution_attempts_20260613.md",
        "scoped_claim": "Native execution ledger records Gavel native simulator, Pollux policy-layer optimizer tests, Sia official-artifact probe, and Decima simulator entrypoint execution on this host.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Direct full-stack SOTA superiority over Gavel/Pollux/Sia/IADeep/Salus.",
        "strong_key": "direct_fullstack_sota_superiority_ready",
        "blocker": "No adapter row has direct full-stack same-workload execution; Sia still lacks its official solver/simulator environment or AdaptDL/Kubernetes path, Pollux still lacks Kubernetes/AdaptDLJob stack, and Gavel lacks service-unit equivalence/live cluster execution.",
        "next_threshold": "Run at least one external system end-to-end on identical workload, cluster resources, and JCT/service-unit accounting before using direct superiority language.",
    },
    {
        "gate": "online_ablation_summary_ci",
        "artifact": "md/experiment_artifacts/online_ablation_summary_ci_20260612.json",
        "raw": "md/online_ablation_summary_ci_20260612.md",
        "scoped_claim": "Distributional online replay and ablation summary over existing measured-cache scenarios.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Direct external binary execution or stochastic service generalization.",
        "strong_key": "strong_claim_ready",
        "blocker": "Rows are replay-policy semantics on measured cache, not live external system runs.",
        "next_threshold": "Keep distributional replay separate from live external execution and stochastic LCB lower-service certification.",
    },
    {
        "gate": "corner_case_lower_service_gate",
        "artifact": "md/experiment_artifacts/corner_case_lower_service_gate_20260613.json",
        "raw": "md/corner_case_lower_service_gate_20260613.md",
        "scoped_claim": "Controlled node007/node001 corner-case rows are admitted into a standalone service cache with positive row-level lower-service slack.",
        "scoped_key": "corner_case_lower_service_capacity_ready",
        "strong_claim": "Production-wide or all-state stability for every future corner case.",
        "strong_key": "strong_claim_ready",
        "blocker": "The gate is a controlled finite measured slice; future states still require service admission.",
        "next_threshold": "Merge only explicitly admitted corner-case rows into theorem-facing replay populations; keep default replay cache unchanged.",
    },
    {
        "gate": "selected_profile_holdout_lcb_gate",
        "artifact": "md/experiment_artifacts/selected_profile_holdout_lcb_gate_20260612.json",
        "raw": "md/selected_profile_holdout_lcb_gate_20260612.md",
        "scoped_claim": "Selected-profile aggregate-window stochastic LCB lower-service capacity certificate for the high-backlog theorem profiles.",
        "scoped_key": "selected_profile_stochastic_lcb_ready",
        "strong_claim": "Positive absolute or diagonal mean-service eta for the same selected profiles.",
        "strong_key": "diagonal_normalized_eta_ready",
        "blocker": "The LCB lower-service capacity certificate passes, but absolute and diagonal mean-service eta remain negative because heterogeneous workload units have different scales.",
        "next_threshold": "Use the LCB lower-service certificate for the main theorem; promote mean-service eta only after the absolute or diagonal eta gate becomes positive.",
    },
    {
        "gate": "controlled_launched_completion_gate",
        "artifact": "md/experiment_artifacts/controlled_production_completion_gate_20260612.json",
        "raw": "md/controlled_production_completion_gate_20260612.md",
        "scoped_claim": "32-task controlled launched completion with theorem-grade oracle trace and canary recorder contract.",
        "scoped_key": "controlled_32_task_completion_ready",
        "strong_claim": "Controlled 32-task completion plus strict scheduler-history large-scale launched-completion, with live-oracle large-scale completion kept separate.",
        "strong_key": "strict_history_large_scale_completion_ready",
        "blocker": "Controlled completion is closed; live-oracle-traced large-scale production completion remains false unless a separate live trace closes.",
        "next_threshold": "Keep controlled 6-task, controlled 32-task, strict-history, and live-oracle large-scale flags separate in manuscript wording.",
    },
    {
        "gate": "organic_production_canary_recorder_gate",
        "artifact": "md/experiment_artifacts/organic_production_canary_recorder_gate_20260612.json",
        "raw": "md/organic_production_canary_recorder_gate_20260612.md",
        "scoped_claim": "Strict organic production canary recorder and theorem admission/trace contract are ready.",
        "scoped_key": "organic_production_canary_recorder_ready",
        "strong_claim": "Large organic production launched-completion evidence from live trace or strict history certificate.",
        "strong_key": "large_scale_organic_launched_completion_ready",
        "blocker": "Organic thresholds need enough natural launches, completions, domains, nodes, and zero unadmitted launched rows.",
        "next_threshold": "Closed if either live oracle trace thresholds or strict scheduler-history completion thresholds pass.",
    },
    {
        "gate": "organic_history_completion_gate",
        "artifact": "md/experiment_artifacts/organic_history_completion_gate_20260614.json",
        "raw": "md/organic_history_completion_gate_20260614.md",
        "scoped_claim": "Strict scheduler-history organic production population has enough launched/completed theorem-admitted rows.",
        "scoped_key": "large_scale_organic_history_completion_ready",
        "strong_claim": "Production-wide organic launched-completion evidence for the controlled theorem-facing population.",
        "strong_key": "large_scale_organic_history_completion_ready",
        "blocker": "Raw-history rows outside scheduler control remain excluded; future unknown jobs still require admission/probe.",
        "next_threshold": "Keep strict admission zero-unadmitted before submission and rerun after any new production arrivals.",
    },
    {
        "gate": "production_wide_organic_trace_gate",
        "artifact": "md/experiment_artifacts/production_wide_organic_trace_gate_20260614.json",
        "raw": "md/production_wide_organic_trace_gate_20260614.md",
        "scoped_claim": "Production-wide organic recorder/admission gate separates live-trace, strict-history completion, active-progress, and launched-completion evidence.",
        "scoped_key": "organic_production_canary_recorder_ready",
        "strong_claim": "Large-scale production-wide organic launched-completion evidence for the controlled theorem-facing population.",
        "strong_key": "large_scale_organic_launched_completion_ready",
        "blocker": "Strong production-wide evidence requires either live oracle trace thresholds or strict scheduler-history completion thresholds.",
        "next_threshold": "Keep live-trace and history-completion subclaims separate in manuscript wording.",
    },
    {
        "gate": "production_organic_readiness_bridge_gate",
        "artifact": "md/experiment_artifacts/production_organic_readiness_bridge_gate_20260614.json",
        "raw": "md/production_organic_readiness_bridge_gate_20260614.md",
        "scoped_claim": "Production organic readiness bridge closes strict admission, canary recorder, active-production theorem shadow, read-only queued-production theorem trace, and history completion snapshot.",
        "scoped_key": "organic_readiness_bridge_ready",
        "strong_claim": "Large-scale organic launched-completion evidence through live trace or strict history completion.",
        "strong_key": "large_scale_organic_launched_completion_ready",
        "blocker": "Readiness and queued theorem trace are closed; live oracle trace and history certificate must remain separately reported.",
        "next_threshold": "Use history completion for completed production evidence and keep live oracle trace as an additional subclaim.",
    },
    {
        "gate": "production_launch_completion_gate",
        "artifact": "md/experiment_artifacts/production_launch_completion_gate_20260612.json",
        "raw": "md/production_launch_completion_gate_20260612.md",
        "scoped_claim": "Large-scale active-production progress and non-invasive theorem shadow trace.",
        "scoped_key": "large_scale_active_progress_ready",
        "strong_claim": "Large-scale launched production completion trace.",
        "strong_key": "large_scale_launched_completion_ready",
        "blocker": "Rolling snapshots may lack a theorem shadow subset; launched completion also requires queued production plus low GPU utilization.",
        "next_threshold": "Recover a nonempty theorem shadow subset and launch only when the live gate reports launch_safe=true.",
    },
    {
        "gate": "global_theorem_dispatcher_prototype_gate",
        "artifact": "md/experiment_artifacts/global_theorem_dispatcher_prototype_gate_20260612.json",
        "raw": "md/global_theorem_dispatcher_prototype_gate_20260612.md",
        "scoped_claim": "Pure bounded global robust-MaxWeight action selector has an exact oracle-gap certificate over its enumerated family.",
        "scoped_key": "global_action_dispatch_ready",
        "strong_claim": "Live scheduler default is a deployed global batch MaxWeight dispatcher.",
        "strong_key": "live_scheduler_default_global_dispatcher_ready",
        "blocker": "Prototype is wired only as an opt-in scheduler soft-hint A/B hook; it is not the live default and does not by itself certify launched global-action traces.",
        "next_threshold": "Run controlled launches with global_theorem_maxweight_v1 and collect launched global-action theorem traces.",
    },
    {
        "gate": "universal_claim_closure_gate",
        "artifact": "md/experiment_artifacts/universal_claim_closure_gate_20260614.json",
        "raw": "md/universal_claim_closure_gate_20260614.md",
        "scoped_claim": "All five broad-claim directions have executable scoped boundary gates: arbitrary SOTA universe, future workload protocol, multi-node original deployment, Decima Spark-DAG, and production-wide organic trace.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "All five universal strong claims are true.",
        "strong_key": "strong_claim_ready",
        "blocker": "The package closes scoped/reviewer gates, not the universal theorem that every broad claim is already true.",
        "next_threshold": "Each row in the universal gate must have strong_claim_ready=true before universal language is allowed.",
    },
    {
        "gate": "non_future_claim_closure_gate",
        "artifact": "md/experiment_artifacts/non_future_claim_closure_gate_20260614.json",
        "raw": "md/non_future_claim_closure_gate_20260614.md",
        "scoped_claim": "All finite non-future broad directions close: named external SOTA, registered SOTA policy semantics, production history, Scheduleurm-native multi-node history, and Decima bridge.",
        "scoped_key": "non_future_scoped_closure_ready",
        "strong_claim": "Finite non-future scoped closure excluding arbitrary future workloads.",
        "strong_key": "non_future_scoped_closure_ready",
        "blocker": "Arbitrary future workload and stronger unbounded/external-binary extensions remain excluded.",
        "next_threshold": "Use this row for the paper's non-future closure statement and keep forbidden extensions explicit.",
    },
    {
        "gate": "or_algorithm_upgrade_gate",
        "artifact": "md/experiment_artifacts/or_algorithm_upgrade_gate_20260613.json",
        "raw": "md/or_algorithm_upgrade_gate_20260613.md",
        "scoped_claim": "Optional algorithm-layer upgrade validates global batch candidate construction, state-dependent marginal service rows, online LCB/ETA, and backlog-aware guarded replay without regressions.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "The production scheduler default is a launched global batch MaxWeight dispatcher or the system directly beats external full stacks.",
        "strong_key": "strong_claim_ready",
        "blocker": "The gate is replay/certificate evidence for an opt-in algorithm surface; production launched global-action traces and external full-stack execution remain separate.",
        "next_threshold": "Enable the optional hook for controlled live launches and collect global-action theorem traces before promoting production-default language.",
    },
    {
        "gate": "reviewer_environment_manifest_gate",
        "artifact": "md/experiment_artifacts/reviewer_environment_manifest_gate_20260612.json",
        "raw": "md/reviewer_environment_manifest_gate_20260612.md",
        "scoped_claim": "Current-host reviewer environment and one-command reproduction contract are present.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Clean Docker/Nix container proof.",
        "strong_key": "clean_container_ready",
        "blocker": "No Dockerfile.reviewer or Nix flake is supplied.",
        "next_threshold": "Run the full reproduction script inside a clean container or Nix environment.",
    },
)


def build_gate_status_dashboard() -> dict[str, Any]:
    rows = [_dashboard_row(spec) for spec in GATES]
    scoped_ready_count = sum(1 for row in rows if bool(row.get("scoped_claim_ready")))
    strong_ready_count = sum(1 for row in rows if bool(row.get("strong_claim_ready")))
    return {
        "gate": "gate_status_dashboard",
        "status": "CLAIM_LADDER_DASHBOARD_READY",
        "gate_pass": True,
        "scoped_claim_ready": True,
        "strong_claim_ready": False,
        "pass_meaning": (
            "reviewer-facing claim ladder; a scoped pass row must not be read as "
            "the adjacent strong claim unless strong_claim_ready is true"
        ),
        "row_count": len(rows),
        "scoped_ready_count": scoped_ready_count,
        "strong_ready_count": strong_ready_count,
        "scoped_pending_count": len(rows) - scoped_ready_count,
        "rows": rows,
        "pass": True,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gate Status Dashboard",
        "",
        "This dashboard separates scoped claims from adjacent strong claims. A scoped pass is not a universal claim.",
        "",
        "| Gate | Scoped claim | Scoped ready | Strong claim | Strong ready | Status | Blocker | Next threshold | Artifact | Raw evidence |",
        "|---|---|---:|---|---:|---|---|---|---|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{gate}` | {scoped} | {scoped_ready} | {strong} | {strong_ready} | `{status}` | {blocker} | {next_threshold} | `{artifact}` | `{raw}` |".format(
                gate=row.get("gate"),
                scoped=_cell(row.get("scoped_claim")),
                scoped_ready=str(bool(row.get("scoped_claim_ready"))).lower(),
                strong=_cell(row.get("strong_claim")),
                strong_ready=str(bool(row.get("strong_claim_ready"))).lower(),
                status=row.get("status") or "",
                blocker=_cell(row.get("blocker")),
                next_threshold=_cell(row.get("next_threshold")),
                artifact=row.get("artifact_path"),
                raw=row.get("raw_evidence_path"),
            )
        )
    return "\n".join(lines) + "\n"


def _dashboard_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    artifact = str(spec.get("artifact") or "")
    data = _load_artifact(artifact)
    scoped_key = spec.get("scoped_key")
    strong_key = spec.get("strong_key")
    scoped_ready = bool(data.get(scoped_key)) if scoped_key else Path(REPO_ROOT / artifact).exists()
    strong_ready = bool(data.get(strong_key)) if strong_key else False
    status = str(data.get("status") or ("SCOPED_PASS" if scoped_ready else "SCOPED_PENDING"))
    return {
        "gate": spec.get("gate"),
        "scoped_claim": spec.get("scoped_claim"),
        "scoped_claim_ready": scoped_ready,
        "strong_claim": spec.get("strong_claim"),
        "strong_claim_ready": strong_ready,
        "status": status,
        "blocker": "" if strong_ready else spec.get("blocker"),
        "next_threshold": spec.get("next_threshold"),
        "artifact_path": artifact,
        "raw_evidence_path": spec.get("raw"),
        "artifact_exists": _artifact_path(artifact).exists(),
    }


def _load_artifact(path: str) -> dict[str, Any]:
    p = _artifact_path(path)
    if not p.exists() or p.suffix != ".json":
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _artifact_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gate_status_dashboard()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gate_status_dashboard")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build reviewer-facing gate status dashboard")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "gate_status_dashboard_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "gate_status_dashboard.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
