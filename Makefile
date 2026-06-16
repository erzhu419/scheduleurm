.PHONY: verify-gates verify-lean paper-tables artifact-manifest reproduce-or-submission

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

artifact-manifest:
	./scripts/reproduce_or_submission.sh

reproduce-or-submission:
	./scripts/reproduce_or_submission.sh
