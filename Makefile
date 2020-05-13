DOC = structuredJets

PDF = $(DOC).pdf
TEX = $(DOC).tex
BBL = $(DOC).bbl
BIB = $(DOC)_sources.bib
FIGDIR = figs
CLS = aastex62.cls

APJTOOL = python3 tex2apj.py
TAR = COPYFILE_DISABLE=1 tar

ARXDIR = toArxiv
ARX = toArxiv.tar.gz

APJDIR = toApj
APJ = toApj.tar.gz

.PHONY: clean bbl pdf arxiv apj

default: pdf

bbl: $(BBL)

pdf: $(PDF)

arxiv: $(ARX)

apj: $(APJ)

$(PDF): $(TEX) $(BBL)
	pdflatex $(TEX)
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
	$(TAR) -czvf $(ARX) $(ARXDIR) 

$(APJ): $(TEX) $(BBL) $(CLS) $(PDF)
	rm -f $(APJ)
	rm -rf $(APJDIR)
	$(APJTOOL) $(TEX) $(APJDIR)
	cp $(PDF) $(APJDIR)
	cp $(BBL) $(APJDIR)
	cp $(CLS) $(APJDIR)
	$(TAR) -czvf $(APJ) $(APJDIR) 

clean:
	rm -rf $(ARXDIR)
	rm -rf $(APJDIR)
	rm -f $(PDF) $(BBL) $(ARX) $(APJ)


