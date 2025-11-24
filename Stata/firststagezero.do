clear
use "C:\Users\B375471\Downloads\Acemoglu_osv\colonial_origins\maketable4\maketable4.dta"

quietly summarize logem4 if shortnam=="CHN", meanonly
replace logem4 = r(mean) if shortnam=="JPN"



 drop if baseco == 1
 drop if africa == 1
 drop if shortnam == "FJI"
 drop if shortnam == "BLZ"
 drop if shortnam == "BRB"
 drop if shortnam == "SUR"
 drop if shortnam == "MMR"
 drop if shortnam == "PNG"
 drop if shortnam == "LAO"

 
 
 reg avexpr logem4
 list if e(sample)
 reg logpgp95 logem4
 eststo fzero
 
 esttab fzero using firstzero2.tex
 list if e(sample)