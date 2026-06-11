LATEXMK ?= latexmk

.PHONY: all clean distclean

all: main.pdf

main.pdf: main.tex references.bib informs4.cls eqndefns-left.sty informs2014.bst informs_Logo.pdf
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -c main.tex

distclean:
	$(LATEXMK) -C main.tex
	rm -f main.bbl main.blg main.ent main.ttt main.vtc
