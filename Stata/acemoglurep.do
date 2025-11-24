***********************************************************
*Creates Table 4: IV Regressions of Log GDP per Capita
***********************************************************
clear
capture log close
cd C:\Users\B375471\Downloads\Acemoglu_osv\colonial_origins\maketable4
log using maketable4, replace

/*Data Files Used
	maketable4.dta
	
*Data Files Created as Final Product
	none
	
*Data Files Created as Intermediate Product
	none*/
	

use maketable4, clear
keep if baseco==1

**********************************
*--Panels A and B, IV Regressions
**********************************
*Columns 1 - 2 (Base Sample)

ivreg logpgp95 (avexpr=logem4), first

outreg2 using acemoglutbl.tex