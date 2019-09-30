DOC = structuredJets

PDF = $(DOC).pdf
TEX = $(DOC).tex
BBL = $(DOC).bbl
BIB = $(DOC)_sources.bib
FIGDIR = figs
CLS = aastex62.cls

ARXDIR = toArxiv
ARX = toArxiv.tar.gz

.PHONY: clean bbl pdf arxiv apj

default: pdf

bbl: $(BBL)

pdf: $(PDF)

arxiv: $(ARX)

$(PDF): $(TEX) $(BBL)
	pdflatex $(TEX)
	pdflatex $(TEX)

$(BBL): $(TEX) $(BIB)
	pdflatex $(TEX)
	bibtex $(DOC)
	rm -f $(PDF)

$(ARX): $(TEX) $(BBL) $(CLS)
	rm -f $(ARX)
	rm -rf $(ARXDIR)
	mkdir $(ARXDIR)
	cp $(TEX) $(ARXDIR)
	cp $(BBL) $(ARXDIR)
	cp $(CLS) $(ARXDIR)
	cp -r $(FIGDIR) $(ARXDIR)
	tar -czvf $(ARX) $(ARXDIR) 

apj:
	grep -E "(?<=figs).*" $(TEX)

clean:
	rm -rf $(ARXDIR)
	rm -f $(PDF) $(BBL) $(ARX)


