"""
ajr_causalpy_iv.py  •  Bayesian IV identical to CausalPy notebook
================================================================
Fits Acemoglu, Johnson & Robinson (2001) Colonial Origins data with
CausalPy's `InstrumentalVariableRegression` (sampling happens when the
object is instantiated).  After sampling it draws **posterior‑predictive**
replicates using *PyMC 5's* current API and saves the complete
`InferenceData`—posterior **plus** PPC—to a NetCDF file so you never need
to rerun MCMC unless you change hyper‑parameters.

Usage
-----
```python
import ajr_causalpy_iv as ajr
iv, idata = ajr.run(draws=3000, baseline_only=True)

# Posterior predictive check
import arviz as az
az.plot_ppc(idata, data_pairs={"logpgp95": "logpgp95"});

# Later:
idata = az.from_netcdf("ajr_iv_posterior.nc")
```
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import arviz as az
import pandas as pd
import causalpy as cp
from causalpy.pymc_models import InstrumentalVariableRegression
import pymc as pm

# ---------------------------------------------------------------------------
# Default .dta location 
# ---------------------------------------------------------------------------
DATA_PATH = Path(
    r"C:\Users\B375471\Downloads\Acemoglu_osv\colonial_origins\maketable4\maketable4.dta"
)

# ---------------------------------------------------------------------------
# Helper: load data and mirror maketable4.do tweaks
# ---------------------------------------------------------------------------

def _load_data(path: Union[str, Path], baseline_only: bool = True) -> pd.DataFrame:
    """Read AJR data and replicate Stata‑side transformations."""
    df = pd.read_stata(path)

    # Extra continent dummy (AUS, MLT, NZL)
    if "other_cont" not in df.columns and "shortnam" in df.columns:
        df["other_cont"] = 0
        df.loc[df["shortnam"].isin(["AUS", "MLT", "NZL"]), "other_cont"] = 1

    # Baseline colonies (baseco == 1) if requested
    if baseline_only and "baseco" in df.columns:
        df = df[df["baseco"] == 1]

    if df.empty:
        raise ValueError("No rows left after filtering—check path or flags.")
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Main function to run Bayesian IV
# ---------------------------------------------------------------------------

def run(
    *,
    data_path: Union[str, Path] = DATA_PATH,
    draws: int = 2_000,
    tune: int = 1_000,
    chains: int = 4,
    cores: int = 4,
    baseline_only: bool = True,
    target_accept: float = 0.9,
    random_seed: int | None = 42,
    ppc_draws: int | None = None,   # None → one PPC draw per posterior draw
    netcdf_path: Union[str, Path] = "ajr_iv_posterior.nc",
    covariates: list[str] | None = None,  # exogenous controls
):
    """Run Bayesian IV and return `(iv_object, inference_data)`.

    Parameters
    ----------
    covariates : list[str] | None
        Exogenous control variables to include in both first-stage and 
        structural equations. E.g., ['africa', 'asia'] for continent dummies.
    ppc_draws : int | None 
        If not None, only the first `ppc_draws` posterior draws are used 
        to generate posterior‑predictive replicates to speed up PPC.
    
    Returns
    -------
    tuple[cp.InstrumentalVariable, az.InferenceData]
        The fitted IV object and complete inference data with posterior + PPC.
    """

    # 1  Load & prepare data --------------------------------------------------
    df = _load_data(data_path, baseline_only)
    
    # Handle covariates
    if covariates is None:
        covariates = []
    
    # Validate that covariates exist in the data
    missing_covs = [cov for cov in covariates if cov not in df.columns]
    if missing_covs:
        raise ValueError(f"Covariates not found in data: {missing_covs}")

    # Build formulas with covariates
    covariate_terms = " + ".join(covariates) if covariates else ""
    
    if covariate_terms:
        instruments_formula = f"avexpr ~ 1 + logem4 + {covariate_terms}"  # first stage
        formula = f"logpgp95 ~ 1 + avexpr + {covariate_terms}"  # structural equation
    else:
        instruments_formula = "avexpr ~ 1 + logem4"  # first stage
        formula = "logpgp95 ~ 1 + avexpr"  # structural equation

    # Prepare data matrices with required variables
    required_vars_instruments = ["avexpr", "logem4"] + covariates
    required_vars_structural = ["logpgp95", "avexpr"] + covariates
    
    instruments_data = df[required_vars_instruments]
    data = df[required_vars_structural]

    sample_kwargs = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
    )

    # 2  Instantiate InstrumentalVariable — sampling happens internally -------
    iv = cp.InstrumentalVariable(
        instruments_data=instruments_data,
        data=data,
        instruments_formula=instruments_formula,
        formula=formula,
        model=InstrumentalVariableRegression(sample_kwargs=sample_kwargs),
    )

    idata = iv.model.idata  # posterior now in memory

    # 3  Posterior‑predictive draws via PyMC v5 API ---------------------------
    #    Docs: https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html
    if ppc_draws is not None and ppc_draws < idata.posterior.sizes["draw"]:
        idata_subset = idata.sel(draw=slice(0, ppc_draws))
    else:
        idata_subset = idata

    ppc_idata = pm.sample_posterior_predictive(
        model=iv.model,  # underlying PyMC graph
        trace=idata,
        random_seed=random_seed,
        extend_inferencedata=True,
    )

    # 4  Save and return ------------------------------------------------------
    idata.to_netcdf(netcdf_path)

    print(az.summary(idata, var_names=["beta_z", "beta_t"], round_to=3))
    print(f"\nPosterior + PPC saved to → {netcdf_path}")

    return iv, idata


# ---------------------------------------------------------------------------
# Additional helper functions for covariates exploration
# ---------------------------------------------------------------------------

def get_available_covariates(data_path: Union[str, Path] = DATA_PATH, 
                           baseline_only: bool = True) -> list[str]:
    """Get list of potential covariate columns in the AJR dataset.
    
    Returns
    -------
    list[str]
        Available column names that could be used as covariates.
    """
    df = _load_data(data_path, baseline_only)
    
    # Exclude the main variables of interest
    excluded = {"logpgp95", "avexpr", "logem4", "baseco", "shortnam"}
    
    # Get potential covariates
    potential_covs = [col for col in df.columns if col not in excluded]
    
    return sorted(potential_covs)


