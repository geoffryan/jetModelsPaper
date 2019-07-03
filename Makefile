DOC = structuredJets

PDF = $(DOC).pdf
TEX = $(DOC).tex
BBL = $(DOC).bbl
BIB = $(DOC)_sources.bib

default: $(PDF)

$(PDF): $(TEX) $(BBL)
	pdflatex $(TEX)
	pdflatex $(TEX)

$(BBL): $(TEX) $(BIB)
	pdflatex $(TEX)
	bibtex $(DOC)

clean:
	rm -f $(PDF) $(BBL)


