LATEXMK ?= latexmk
GENERATED_INPUTS := generated/unified_hardware_results.tex generated/or_generalization_v7_results.tex
FIGURE_INPUTS := ../md/figures/scheduleurm_fig1_nature_python_combo.pdf ../md/figures/scheduleurm_quadrant_sota_grid.pdf
COMMON_INPUTS := main.tex informs4.cls eqndefns-left.sty informs_Logo.pdf $(GENERATED_INPUTS)

.PHONY: all submission-check clean distclean

all: main.pdf electronic_companion.pdf submission-check

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
