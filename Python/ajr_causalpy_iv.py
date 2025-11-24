"""
ajr_causalpy_iv.py  •  Bayesian IV CausalPy implementation for Plausibly Exogenous estimation 
================================================================
Fits Acemoglu, Johnson & Robinson (2001) Colonial Origins data with
CausalPy's `InstrumentalVariableRegression`.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Union
import xarray as xr
import arviz as az
import pandas as pd
import causalpy as cp
from bayesianiv import InstrumentalVariableRegression
import pymc as pm



def load_ajr_data(path: str, baseline_only: bool = True, standardize: bool = False):
    """Load AJR data with exact filtering logic, optionally standardize variables."""
    df = pd.read_stata(path)
    print(f"Original dataset size: {len(df)}")

    # Step 1: Create other_cont dummy if needed
    if "other_cont" not in df.columns and "shortnam" in df.columns:
        df["other_cont"] = 0
        df.loc[df["shortnam"].isin(["AUS", "MLT", "NZL"]), "other_cont"] = 1

    # Step 2: Filter to baseline colonies only (baseco == 1)
    if baseline_only and "baseco" in df.columns:
        df = df[df["baseco"] == 1]
        print(f"After baseco == 1 filter: {len(df)}")

    # Step 3: Drop missing values for required variables
    df = df.dropna(subset=["logem4", "logpgp95", "avexpr"])
    print(f"After dropna: {len(df)} observations")

    # Step 4: Optionally standardize
    stats = {}
    if standardize:
        for var in ["logem4", "avexpr", "logpgp95"]:
            mean_ = df[var].mean()
            std_ = df[var].std(ddof=0)
            stats[var] = {"mean": mean_, "std": std_}

        # Standardize predictors
        df["logem4"] = (df["logem4"] - stats["logem4"]["mean"]) / stats["logem4"]["std"]
        df["avexpr"] = (df["avexpr"] - stats["avexpr"]["mean"]) / stats["avexpr"]["std"]

        # Center outcome (not scale)
        df["logpgp95"] = df["logpgp95"] - stats["logpgp95"]["mean"]

        print("Variables standardized (predictors) and centered (outcome).")

    # Step 5: Reset index
    df = df.reset_index(drop=True)

    if standardize:
        return df, stats
    else:
        return df

def run(
    *,
    data: pd.DataFrame,  # Pre-processed data passed from notebook
    draws: int = 2_000,
    tune: int = 1_000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.9,
    loosen_exclusion: bool = False,
    random_seed: int | None = 42,
    ppc_draws: int | None = None,
    netcdf_path: Union[str, Path] = "ajr_iv_posterior.nc",
    covariates: list[str] | None = None,
    priors: dict | None = None,
    standardize: bool = False,
    stats: dict | None = None,
):
    """Run Bayesian IV on pre-processed data.

    Parameters
    ----------
    data : pd.DataFrame
        Pre-processed dataframe with required variables: logem4, avexpr, logpgp95
    covariates : list[str] | None
        Exogenous control variables to include in both equations
    """
    
    # Handle covariates
    if covariates is None:
        covariates = []
    
    # Validate that required variables exist
    required_vars = ["logem4", "avexpr", "logpgp95"] + covariates
    missing_vars = [var for var in required_vars if var not in data.columns]
    if missing_vars:
        raise ValueError(f"Required variables not found in data: {missing_vars}")

    # Build formulas with covariates
    covariate_terms = " + ".join(covariates) if covariates else ""
    
    
    if covariate_terms:
        instruments_formula = f"avexpr ~ 1 + {covariate_terms} + logem4"
    else:
        instruments_formula = "avexpr ~ 1 + logem4"
    structural_formula = "logpgp95 ~ 1 + avexpr"
    if covariate_terms:
        structural_formula += f" + {covariate_terms}"
    if loosen_exclusion:
        structural_formula += " + logem4"  

    print(f"First-stage formula: {instruments_formula}")
    print(f"Structural formula: {structural_formula}")



    # Prepare data matrices
    required_vars_instruments = ["avexpr", "logem4"] + covariates
    required_vars_structural = ["logpgp95", "avexpr"] + covariates
    if loosen_exclusion:
        required_vars_structural.append("logem4")
    
    instruments_data = data[required_vars_instruments]
    structural_data = data[required_vars_structural]

    sample_kwargs = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
    )

    # Instantiate InstrumentalVariable
    iv = cp.InstrumentalVariable(
        instruments_data=instruments_data,
        data=structural_data,
        instruments_formula=instruments_formula,
        formula=structural_formula,
        model=InstrumentalVariableRegression(sample_kwargs=sample_kwargs),
        priors=priors,
    )

    idata = iv.model.idata

    # Posterior predictive draws
    if ppc_draws is not None and ppc_draws < idata.posterior.sizes["draw"]:
        idata_subset = idata.sel(draw=slice(0, ppc_draws))
    else:
        idata_subset = idata

    ppc_idata = pm.sample_posterior_predictive(
        model=iv.model,
        trace=idata,
        random_seed=random_seed,
        extend_inferencedata=True,
    )

    
    # Save results 
   

    if standardize:
        idata = backtransform_posterior(idata, stats)
        print("Posterior coefficients back-transformed to original units.")

    idata.to_netcdf(netcdf_path)
    print(az.summary(idata, var_names=["beta_z", "beta_t"], hdi_prob=0.95, round_to=3))
    if standardize:
        print(az.summary(idata, var_names=["beta_orig", "delta_orig", "intercept_orig"], hdi_prob=0.95, round_to=3))
    print(f"\nPosterior + PPC saved to → {netcdf_path}")

    return iv, idata


# ---------------------------------------------------------------------------
# Additional helper functions 
# ---------------------------------------------------------------------------

def get_available_covariates(data: pd.DataFrame) -> list[str]:
    """Get list of potential covariate columns in the AJR dataset.
    
    Returns
    -------
    list[str]
        Available column names that could be used as covariates.
    """
    df = data
    # Exclude the main variables of interest
    excluded = {"logpgp95", "avexpr", "logem4", "baseco", "shortnam"}
    
    # Get potential covariates
    potential_covs = [col for col in df.columns if col not in excluded]
    
    return sorted(potential_covs)

def prior_sensitivity(
    scenarios: List[Dict],
    *,
    data: pd.DataFrame,
    covariates: List[str] | None = None,
    baseline_only: bool = True,
    draws: int = 1_000,
    tune: int = 1_000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = 42,
    out_dir: Union[str, Path] = "prior_sensitivity_runs",
) -> List[az.InferenceData]:
    """Fit the model under multiple prior dictionaries.

    Each dict **must** contain a key `"__label__"` for naming.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    idatas: List[az.InferenceData] = []

    for scen in scenarios:
        label = scen.pop("__label__", "scenario")
        print(f"\n— Prior scenario: {label} —")
        idata = run(
            data=data,
            priors=scen,
            covariates=covariates,
            baseline_only=baseline_only,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            netcdf_path=Path(out_dir) / f"ajr_iv_{label}.nc",
        )[1]  # second element is idata

        idata.attrs["label"] = label
        idatas.append(idata)

    return idatas





def backtransform_posterior(idata: az.InferenceData, stats: dict) -> az.InferenceData:
    post = idata.posterior

    # scales
    s_t = stats["avexpr"]["std"]
    s_z = stats["logem4"]["std"]
    m_t = stats["avexpr"]["mean"]
    m_y = stats["logpgp95"]["mean"]
    m_z = stats["logem4"]["mean"]

    covar_names = post["beta_z"].coords["covariates"].values
    instr_names = post["beta_t"].coords["instruments"].values
    idx_treat = int(np.where(covar_names == "avexpr")[0][0])
    idx_delta = int(np.where(covar_names == "logem4")[0][0])

    beta_std  = post["beta_z"].isel(covariates=idx_treat)
    delta_std = post["beta_z"].isel(covariates=idx_delta)
    intercept_std = post["beta_z"].sel(covariates="Intercept")

    beta_orig  = beta_std  * (1.0 / s_t)
    delta_orig = delta_std * (1.0 / s_z)
    intercept_orig = m_y - beta_orig * m_t - delta_orig * m_z

    
    idata_bt = idata.copy()
    idata_bt.posterior = idata_bt.posterior.assign(
        beta_orig=beta_orig,
        delta_orig=delta_orig,
        intercept_orig=intercept_orig,
    )
    return idata_bt
