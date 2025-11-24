*-------------------------------------------------------------*
*  Create LaTeX table of variable names and labels           *
*  Output: varlabels.tex in current working directory        *
*-------------------------------------------------------------*

preserve

* 1. Get list of all variables in the current dataset
ds
local allvars `r(varlist)'
local n : word count `allvars'

* 2. Save labels into locals WHILE original data are loaded
local i = 1
foreach v of local allvars {
    local lab`i' : variable label `v'
    if "`lab`i''" == "" local lab`i' "`v'"   // fallback if no label
    local ++i
}

* 3. Make a small dataset with varname + varlabel
clear
set obs `n'

generate str40  varname  = ""
generate str200 varlabel = ""

local i = 1
foreach v of local allvars {
    replace varname  = "`v'"         in `i'
    replace varlabel = "`lab`i''"    in `i'
    local ++i
}

* 4. Write LaTeX table
tempname fh
file open `fh' using "varlabels.tex", write text replace

file write `fh' "\begin{table}[ht]" _n
file write `fh' "\centering" _n
file write `fh' "\begin{tabular}{ll}" _n
file write `fh' "\toprule" _n
file write `fh' "Variable & Label \\\\" _n
file write `fh' "\midrule" _n

forvalues i = 1/`=_N' {
    file write `fh' "`=varname[`i']' & `=varlabel[`i']' \\\\" _n
}

file write `fh' "\bottomrule" _n
file write `fh' "\end{tabular}" _n
file write `fh' "\caption{Variable names and labels}" _n
file write `fh' "\label{tab:varlabels}" _n
file write `fh' "\end{table}" _n

file close `fh'

restore
