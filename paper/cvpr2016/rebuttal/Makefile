fnname1=egrebuttal

all: camera-ready clean

camera-ready: *.tex *.bib
	pdflatex ${fnname1}; bibtex ${fnname1}; pdflatex ${fnname1}; pdflatex ${fnname1}; pdflatex ${fnname1};

check: camera-ready
	pdffonts ${fnname1}.pdf

clean:
	rm -f ${fnname1}.{aux,bbl,blg,brf,log}
