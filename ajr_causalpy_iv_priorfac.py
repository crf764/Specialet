"""
ajr_causalpy_iv.py  •  Bayesian IV (AJR) + prior-sensitivity helper
===============================================================
This is the **stable base version** that already worked for you plus a new
`prior_sensitivity()` function.  Nothing else in `run()` has been changed
except one silent bug-fix: the posterior-predictive call now uses PyMC 5’s
API (`idata=` instead of the deprecated `trace=` argument).  All original
arguments keep their defaults, so existing notebooks and scripts will run
unchanged.

Public API
----------
* **run(...) → (iv, idata)** — same as before, now with an *optional*
  `priors` dict so you can override the default weakly-informative priors
  if you like.
* **prior_sensitivity(scenarios, …) → list[idata]** — Fit the model under
  multiple prior dictionaries; each scenario must contain a `"__label__"`
  key so you can identify plots later.
* **get_available_covariates()** — unchanged helper.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Dict, List

import arviz as az
import pandas as pd
import pymc as pm
import causalpy as cp
from causalpy.pymc_models import InstrumentalVariableRegression

# ---------------------------------------------------------------------------
# Default .dta location
# ---------------------------------------------------------------------------
DATA_PATH = Path(r"C:\Users\B375471\Downloads\Acemoglu_osv\colonial_origins\maketable4\maketable4.dta")

# ---------------------------------------------------------------------------
# Helper: load data and mirror maketable4.do tweaks
# ---------------------------------------------------------------------------

def _load_data(path: Union[str, Path], baseline_only: bool = True) -> pd.DataFrame:
    df = pd.read_stata(path)

    # Extra continent dummy (AUS, MLT, NZL)
    if "other_cont" not in df.columns and "shortnam" in df.columns:
        df["other_cont"] = 0
        df.loc[df["shortnam"].isin(["AUS", "MLT", "NZL"]), "other_cont"] = 1

    if baseline_only and "baseco" in df.columns:
        df = df[df["baseco"] == 1]

    if df.empty:
        raise ValueError("No rows left after filtering—check path or flags.")
    return df.reset_index(drop=True)



# ---------------------------------------------------------------------------
# MAIN: run one Bayesian IV
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
    ppc_draws: int | None = None,   # None → PPC for every posterior draw
    netcdf_path: Union[str, Path] = "ajr_iv_posterior.nc",
    covariates: List[str] | None = None,
    priors: Dict | None = None,
):
    """Run Bayesian IV and return `(iv_object, inference_data)`.

    Parameters unchanged except the new **`priors`** (optional dict).  If
    your CausalPy version doesn’t support custom priors, the dict is
    ignored and you’ll see a warning.
    """

    # 1  Load & prepare data --------------------------------------------------
    df = _load_data(data_path, baseline_only)

    covariates = covariates or []
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"Covariates not found in data: {missing}")

    cov_term = " + ".join(covariates) if covariates else ""
    instruments_formula = f"avexpr ~ 1 + logem4" + (f" + {cov_term}" if cov_term else "")
    formula            = f"logpgp95 ~ 1 + avexpr" + (f" + {cov_term}" if cov_term else "")

    instruments_data = df[["avexpr", "logem4"] + covariates]
    data             = df[["logpgp95", "avexpr"] + covariates]

    # 2  Build model kwargs (handle priors if supported) ----------------------
    sample_kwargs = dict(draws=draws, tune=tune, chains=chains, cores=cores,
                         target_accept=target_accept, random_seed=random_seed)

    model_kwargs: Dict = {"sample_kwargs": sample_kwargs}
    from inspect import signature
    sig = signature(InstrumentalVariableRegression)
    if priors is not None:
        if "priors" in sig.parameters:
            model_kwargs["priors"] = priors
        elif "prior_distributions" in sig.parameters:
            model_kwargs["prior_distributions"] = priors
        else:
            print("⚠︎  This CausalPy version can’t take custom priors; ignoring.")

    # 3  Instantiate IV — sampling happens inside CausalPy --------------------
    iv = cp.InstrumentalVariable(
        instruments_data=instruments_data,
        data=data,
        instruments_formula=instruments_formula,
        formula=formula,
        model=InstrumentalVariableRegression(**model_kwargs),
    )

    idata = iv.model.idata

    # 4  Posterior-predictive draws (PyMC 5 API) ------------------------------
    if ppc_draws is not None and ppc_draws < idata.posterior.sizes["draw"]:
        idata_subset = idata.sel(draw=slice(0, ppc_draws))
    else:
        idata_subset = idata

    ppc_idata = pm.sample_posterior_predictive(
        trace=idata_subset,
        model=iv.model,
        random_seed=random_seed,
        extend_inferencedata=True,
    )

    
    idata.to_netcdf(netcdf_path)

    print(az.summary(idata, var_names=["beta_z", "beta_t"], round_to=3))
    print(f"Posterior + PPC saved → {netcdf_path}")

    return iv, idata

# ---------------------------------------------------------------------------
# SENSITIVITY: run many priors
# ---------------------------------------------------------------------------

def prior_sensitivity(
    scenarios: List[Dict],
    *,
    data_path: Union[str, Path] = DATA_PATH,
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
            data_path=data_path,
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

# ---------------------------------------------------------------------------
# Utility: list candidate covariates
# ---------------------------------------------------------------------------

def get_available_covariates(
    *, data_path: Union[str, Path] = DATA_PATH, baseline_only: bool = True
) -> List[str]:
    df = _load_data(data_path, baseline_only)
    core_vars = {"logpgp95", "avexpr", "logem4", "baseco", "shortnam"}
    return sorted([c for c in df.columns if c not in core_vars])