# parsimony.tex: the method paper.  kintora-precision.tex: the KINTORA case-study paper.
# Both share refs.bib.
PAPERS = parsimony kintora-precision

.PHONY: all clean
all: $(PAPERS:%=%.pdf)

%.pdf: %.tex refs.bib
	pdflatex -interaction=nonstopmode -halt-on-error $*
	-bibtex $*
	pdflatex -interaction=nonstopmode -halt-on-error $*
	pdflatex -interaction=nonstopmode -halt-on-error $*

clean:
	rm -f $(foreach p,$(PAPERS),$(p).aux $(p).bbl $(p).blg $(p).log $(p).out $(p).pdf $(p).toc)
