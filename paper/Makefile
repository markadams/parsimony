PAPER = parsimony

.PHONY: all clean watch
all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER)
	-bibtex $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER)

clean:
	rm -f $(PAPER).{aux,bbl,blg,log,out,pdf,toc}
