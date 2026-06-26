.PHONY: verify-gates verify-lean paper-tables proof-supplement artifact-manifest reproduce-or-submission

verify-gates:
	python3 -m algorithm.experiments.gavel_service_unit_equivalence_certificate build
	python3 -m algorithm.experiments.gavel_service_unit_calibration_gate build
	python3 -m algorithm.experiments.declared_finite_domain_positive_cover_gate build
	python3 -m algorithm.experiments.future_production_admission_contract build --admission-mode strict
	python3 -m algorithm.experiments.production_launch_completion_gate build
	python3 -m algorithm.experiments.controlled_production_completion_gate build
	python3 -m algorithm.experiments.organic_production_canary_recorder_gate build
	python3 -m algorithm.experiments.online_ablation_summary_ci build
	python3 -m algorithm.experiments.selected_profile_holdout_lcb_gate build
	python3 -m algorithm.experiments.global_theorem_dispatcher_prototype_gate build
	python3 -m algorithm.experiments.sota_fullstack_superiority_gate build
	python3 -m algorithm.experiments.sota_universe_registry_gate build
	python3 -m algorithm.experiments.registered_sota_adapter_closure_gate build
	python3 -m algorithm.experiments.sota_admitted_universe_closure_gate build
	python3 -m algorithm.experiments.organic_history_completion_gate build
	python3 -m algorithm.experiments.production_wide_organic_trace_gate build
	python3 -m algorithm.experiments.production_organic_readiness_bridge_gate build
	python3 -m algorithm.experiments.registered_sota_runtime_gate build
	python3 -m algorithm.experiments.multinode_history_completion_gate build
	python3 -m algorithm.experiments.multinode_original_deployment_gate build
	python3 -m algorithm.experiments.decima_same_domain_benchmark_gate build
	python3 -m algorithm.experiments.non_future_claim_closure_gate build
	python3 -m algorithm.experiments.reviewer_environment_manifest_gate build
	python3 -m algorithm.experiments.gate_status_dashboard build

verify-lean:
	cd ../proof && ! rg -n --glob '*.lean' '(^|[^A-Za-z_])(sorry|admit|axiom)([^A-Za-z_]|$$)' Scheduleurm ScheduleurmUpload.lean
	cd ../proof && lake build

paper-tables:
	python3 -m algorithm.experiments.online_ablation_summary_ci build
	python3 -m algorithm.experiments.selected_profile_holdout_lcb_gate build
	python3 -m algorithm.experiments.global_theorem_dispatcher_prototype_gate build
	python3 -m algorithm.experiments.gavel_service_unit_calibration_gate build
	python3 -m algorithm.experiments.organic_production_canary_recorder_gate build
	python3 -m algorithm.experiments.reviewer_environment_manifest_gate build
	python3 -m algorithm.experiments.gate_status_dashboard build
	$(MAKE) -C paper

proof-supplement:
	python3 -m algorithm.experiments.or_submission_closure supplement \
		--proof-root ../proof \
		--output-dir md/experiment_artifacts/or_reviewer_supplement_20260612 \
		--markdown-output md/or_reviewer_supplement_20260612.md

artifact-manifest:
	./scripts/reproduce_or_submission.sh

reproduce-or-submission:
	./scripts/reproduce_or_submission.sh
