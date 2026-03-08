# Makefile for MoCoO paper
# Usage: make [pdf|clean]

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: pdf clean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk \
	      $(MAIN).synctex.gz $(MAIN).spl
