"""
ajr_causalpy_iv.py  •  Bayesian IV identical to CausalPy notebook
================================================================
Fits Acemoglu, Johnson & Robinson (2001) Colonial Origins data with
CausalPy's `InstrumentalVariableRegression`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import arviz as az
import pandas as pd
import causalpy as cp
from causalpy.pymc_models import InstrumentalVariableRegression
import pymc as pm


def load_ajr_data(path: str, baseline_only: bool = True) -> pd.DataFrame:
    """Load AJR data with exact filtering logic."""
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
    df = df.dropna(subset=['logem4', 'logpgp95', 'avexpr'])
    print(f"After dropna: {len(df)} observations")
    
    # Reset index
    df = df.reset_index(drop=True)
    return df

def run(
    *,
    data: pd.DataFrame,  # Pre-processed data passed from notebook
    draws: int = 2_000,
    tune: int = 1_000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.9,
    random_seed: int | None = 42,
    ppc_draws: int | None = None,
    netcdf_path: Union[str, Path] = "ajr_iv_posterior.nc",
    covariates: list[str] | None = None,
    priors: dict | None = None,
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
        instruments_formula = f"avexpr ~ 1 + logem4 + {covariate_terms}"
        formula = f"logpgp95 ~ 1 + avexpr + {covariate_terms}"
    else:
        instruments_formula = "avexpr ~ 1 + logem4"
        formula = "logpgp95 ~ 1 + avexpr"

    # Prepare data matrices
    required_vars_instruments = ["avexpr", "logem4"] + covariates
    required_vars_structural = ["logpgp95", "avexpr"] + covariates
    
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
        formula=formula,
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
    idata.to_netcdf(netcdf_path)
    print(az.summary(idata, var_names=["beta_z", "beta_t"], round_to=3))
    print(f"\nPosterior + PPC saved to → {netcdf_path}")

    return iv, idata


# ---------------------------------------------------------------------------
# Additional helper functions for covariates exploration
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


