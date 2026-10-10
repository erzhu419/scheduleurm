LATEXMK ?= latexmk
PYTHON ?= python3
ART := ../md/experiment_artifacts
REPLAY := $(ART)/unified_hardware_or_replay_20260809.json
REPLAY_INPUTS := $(ART)/service_cache_v2_cpu_gpu_loaded_final_20260809.json \
	$(ART)/critical_gpu_loaded_action_ledger_20260809.json \
	$(ART)/unified_migration_recalibration_certificate_20260809.json
REPLAY_SOURCES := ../algorithm/experiments/unified_hardware_or_replay.py \
	../simulation/service_cache.py ../simulation/sota_baselines.py \
	../simulation/defaults.py ../simulation/fast_forward.py ../simulation/tasksets.py
HARDWARE_OUTPUTS := generated/unified_hardware_results.tex \
	$(ART)/unified_hardware_paper_results_20260810.json \
	../md/unified_hardware_paper_results_20260810.md
GENERALIZATION_INPUTS := $(ART)/or_generalization_certificate_v6_20260810.json \
	$(ART)/fjsp_solver_action_union_holdout_v1.json \
	$(ART)/port_bacasp_s_external_holdout_r83_r84_20260830.json \
	$(ART)/port_action_union_r81_r82_20260830.json
GENERALIZATION_OUTPUTS := generated/or_generalization_v7_results.tex \
	$(ART)/or_generalization_certificate_v7_20260830.json \
	../md/or_generalization_certificate_v7_20260830.md
BG_CONFIRMATION_OUTPUT := generated/bg_confirmation_results.tex
BG_CONFIRMATION_INPUTS := $(ART)/aligned_hardware_BG_empirical_service_confirmation_20261004_results_projection.json \
	$(ART)/aligned_hardware_BG_empirical_service_confirmation_20261004_completion_receipt.json \
	../algorithm/experiments/bg_confirmation_paper_results.py
QUADRANT_FIGURE := ../md/figures/scheduleurm_quadrant_sota_grid
QUADRANT_OUTPUTS := $(addprefix $(QUADRANT_FIGURE),.pdf .png .svg .tiff _data.json)
GENERATED_INPUTS := generated/unified_hardware_results.tex generated/or_generalization_v7_results.tex $(BG_CONFIRMATION_OUTPUT)
FIGURE_INPUTS := ../md/figures/scheduleurm_fig1_nature_python_combo.pdf ../md/figures/scheduleurm_quadrant_sota_grid.pdf
COMMON_INPUTS := main.tex informs4.cls eqndefns-left.sty informs_Logo.pdf $(GENERATED_INPUTS)

.PHONY: all current-results submission-check clean distclean

all: main.pdf electronic_companion.pdf submission-check

# Frozen protocol: same loads, seeds, and population as the registered replay.
$(REPLAY): $(REPLAY_INPUTS) $(REPLAY_SOURCES) Makefile
	cd .. && $(PYTHON) -m algorithm.experiments.unified_hardware_or_replay \
		--cache md/experiment_artifacts/service_cache_v2_cpu_gpu_loaded_final_20260809.json \
		--loaded-ledger md/experiment_artifacts/critical_gpu_loaded_action_ledger_20260809.json \
		--migration-certificate md/experiment_artifacts/unified_migration_recalibration_certificate_20260809.json \
		--loads 0.50,0.70,0.85,0.95 --seeds 41,42,43 \
		--nominal-online-load 0.85 --jobs-per-workload 16 \
		--output md/experiment_artifacts/unified_hardware_or_replay_20260809.json

$(HARDWARE_OUTPUTS) &: $(REPLAY) $(ART)/critical_gpu_all_hardware_statewise_slack_20260810.json $(ART)/critical_gpu_loaded_action_ledger_merge_20260809.json ../algorithm/experiments/unified_hardware_paper_results.py Makefile
	cd .. && $(PYTHON) -m algorithm.experiments.unified_hardware_paper_results \
		--replay md/experiment_artifacts/unified_hardware_or_replay_20260809.json \
		--slack md/experiment_artifacts/critical_gpu_all_hardware_statewise_slack_20260810.json \
		--loaded-merge md/experiment_artifacts/critical_gpu_loaded_action_ledger_merge_20260809.json \
		--output md/experiment_artifacts/unified_hardware_paper_results_20260810.json \
		--markdown md/unified_hardware_paper_results_20260810.md \
		--tex paper/generated/unified_hardware_results.tex

# V7 aggregates the frozen cross-domain result ledgers; it does not rerun solvers.
$(GENERALIZATION_OUTPUTS) &: $(GENERALIZATION_INPUTS) ../algorithm/experiments/or_generalization_certificate_v7.py Makefile
	cd .. && $(PYTHON) -m algorithm.experiments.or_generalization_certificate_v7 \
		--json-output md/experiment_artifacts/or_generalization_certificate_v7_20260830.json \
		--markdown-output md/or_generalization_certificate_v7_20260830.md \
		--tex-output paper/generated/or_generalization_v7_results.tex --force

$(QUADRANT_OUTPUTS) &: $(REPLAY) ../md/figures/make_scheduleurm_quadrant_sota_grid.py ../algorithm/experiments/unified_hardware_paper_results.py Makefile
	cd .. && $(PYTHON) -m md.figures.make_scheduleurm_quadrant_sota_grid \
		--artifact md/experiment_artifacts/unified_hardware_or_replay_20260809.json \
		--output md/figures/scheduleurm_quadrant_sota_grid

current-results: $(HARDWARE_OUTPUTS) $(GENERALIZATION_OUTPUTS) $(QUADRANT_OUTPUTS) $(BG_CONFIRMATION_OUTPUT)

$(BG_CONFIRMATION_OUTPUT): $(BG_CONFIRMATION_INPUTS)
	cd .. && $(PYTHON) -m algorithm.experiments.bg_confirmation_paper_results \
		--projection md/experiment_artifacts/aligned_hardware_BG_empirical_service_confirmation_20261004_results_projection.json \
		--receipt md/experiment_artifacts/aligned_hardware_BG_empirical_service_confirmation_20261004_completion_receipt.json \
		--output paper/generated/bg_confirmation_results.tex

main.pdf: $(COMMON_INPUTS) references.bib informs2014.bst $(FIGURE_INPUTS)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

electronic_companion.pdf: electronic_companion.tex $(COMMON_INPUTS) main.pdf
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error electronic_companion.tex

submission-check: main.pdf electronic_companion.pdf check_submission_package.py
	python3 check_submission_package.py

clean:
	$(LATEXMK) -c main.tex
	$(LATEXMK) -c electronic_companion.tex

distclean:
	$(LATEXMK) -C main.tex
	$(LATEXMK) -C electronic_companion.tex
	rm -f main.bbl main.blg main.ent main.ttt main.vtc
	rm -f electronic_companion.ent electronic_companion.ttt electronic_companion.vtc
