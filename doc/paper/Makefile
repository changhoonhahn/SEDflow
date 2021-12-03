LATEX       = pdflatex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
TMP_SUFFS   = aux bbl blg log dvi ps eps out toc lof lot
CHECK_RERUN =

all: main.pdf 

main.pdf: *.tex
	${LATEX} main
	bibtex main 
	${LATEX} main 
	( grep Rerun main.log && ${LATEX} main ) || echo "Done."
	( grep Rerun main.log && ${LATEX} main ) || echo "Done."

clean:
	${RM} $(foreach suff, ${TMP_SUFFS}, main.${suff})
