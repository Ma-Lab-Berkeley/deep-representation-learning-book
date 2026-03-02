$bibtex_use = 2;            # clean up biber files when -c (and rerun @ compilation)
$do_cd = 1;                 # relative path compilation (needed for icloud)
$pdf_mode = 1;
$clean_ext = "nav snm";
# $pdf_previewer = 'open -a Preview';
$pdf_previewer = 'open -a Skim';
# XeTeX ignores --extra-mem-bot on the command line; use texmf.cnf instead.
# The trailing ":" appends TeX Live's default search path.
$ENV{'TEXMFCNF'} = '.:';
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode --shell-escape'
