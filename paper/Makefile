# Makefile for MoCoO paper
# Usage: make [pdf|clean]

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex
FIGDIR = ../benchmarks/figures

# All figures referenced by the article
FIGS = $(wildcard $(FIGDIR)/*.png)

.PHONY: pdf clean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(FIGS)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk \
	      $(MAIN).synctex.gz $(MAIN).spl
