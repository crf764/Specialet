clear
use "C:\Users\B375471\Downloads\Acemoglu_osv\colonial_origins\maketable4\maketable4.dta"

keep if baseco == 1

plausexog uci logpgp95 (avexpr=logem4), gmin(-0.589) gmax(0.117) vce(robust) grid(8)

plausexog ltz logpgp95 (avexpr = logem4), mu(-0.236) omega(0.032364) vce(robust) 



plausexog ltz logpgp95 (avexpr = logem4), mu(-0.236) omega(0.032364) vce(robust) graph(avexpr) graphmu(-0.236 -0.236 -0.236 -0.236 -0.236 -0.236) graphomega(0.02 0.03 0.04 0.05 0.06 0.07) xtitle({&sigma}{sub:{&delta}}) ytitle({&beta}) legend(label(1 "β point estimate") label(2 "Upper CI") label(3 "Lower CI"))

plausexog ltz logpgp95 (avexpr = logem4), mu(-0.236) omega(0.032364) vce(robust) graph(avexpr) graphdelta(0, -0.1, -0.2, -0.3, -0.4, -0.5) graphmu(0 -0.05 -0.1 -0.15 -0.2 -0.25) graphomega(0 0.0008333333 0.0033333333 0.0075 0.0133333333 0.0208333333) xtitle({&delta}) ytitle({&beta}) legend(label(1 "β point estimate") label(2 "Upper CI") label(3 "Lower CI"))



* Define the grid of maximum plausible violations (graph x-axis)
local deltas "0 -0.1 -0.2 -0.3 -0.4 -0.5"

* Build matching mu = delta_max  and  omega = (delta_max / 1.96)^2
local mus ""
local sigmas ""
local omegas ""
foreach d of local deltas {
    local mu = 0                      // expected violation
    local sd = abs(`d') / 1.96          // SD so that ±1.96σ ≈ δ
    local om = `sd'^2
    local mus    "`mus' `mu'"
    local sigmas "`sigmas' `sd'"
    local omegas "`omegas' `om'"
}

* LTZ with those priors
plausexog ltz logpgp95 (avexpr = logem4), ///
    mu(0) omega(0.032364) vce(robust) graph(avexpr) ///
    graphdelta(`deltas') ///
    graphmu(`mus') ///
    graphomega(`omegas') ///
    xtitle("Plausible violation δ") ///
    ytitle("Estimated β") ///
    legend(label(1 "β point estimate") label(2 "Upper CI") label(3 "Lower CI"))

	

