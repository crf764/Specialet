import pymc as pm
import numpy as np
import pytensor.tensor as pt
from causalpy.pymc_models import PyMCModel

class InstrumentalVariableRegression(PyMCModel):
    """Custom PyMC model for instrumental linear regression

    Example
    --------
    >>> import causalpy as cp
    >>> import numpy as np
    >>> from causalpy.pymc_models import InstrumentalVariableRegression
    >>> N = 10
    >>> e1 = np.random.normal(0, 3, N)
    >>> e2 = np.random.normal(0, 1, N)
    >>> Z = np.random.uniform(0, 1, N)
    >>> ## Ensure the endogeneity of the the treatment variable
    >>> X = -1 + 4 * Z + e2 + 2 * e1
    >>> y = 2 + 3 * X + 3 * e1
    >>> t = X.reshape(10, 1)
    >>> y = y.reshape(10, 1)
    >>> Z = np.asarray([[1, Z[i]] for i in range(0, 10)])
    >>> X = np.asarray([[1, X[i]] for i in range(0, 10)])
    >>> COORDS = {"instruments": ["Intercept", "Z"], "covariates": ["Intercept", "X"]}
    >>> sample_kwargs = {
    ...     "tune": 5,
    ...     "draws": 10,
    ...     "chains": 2,
    ...     "cores": 2,
    ...     "target_accept": 0.95,
    ...     "progressbar": False,
    ... }
    >>> iv_reg = InstrumentalVariableRegression(sample_kwargs=sample_kwargs)
    >>> iv_reg.fit(
    ...     X,
    ...     Z,
    ...     y,
    ...     t,
    ...     COORDS,
    ...     {
    ...         "mus": [[-2, 4], [0.5, 3]],
    ...         "sigmas": [1, 1],
    ...         "eta": 2,
    ...         "lkj_sd": 1,
    ...     },
    ...     None,
    ... )
    Inference data...
    """

    def __init__(self, sample_kwargs=None):
        """
        Initialize the InstrumentalVariableRegression model.
        
        Parameters
        ----------
        sample_kwargs : dict, optional
            Dictionary of keyword arguments to pass to pm.sample()
        """
        PyMCModel.__init__(self, sample_kwargs=sample_kwargs)


    def build_model(self, X, Z, y, t, coords, priors):
        with self:
            # ---------- Coords (robust to None/missing) ----------
            if coords is None:
                coords = {}
            if "instruments" not in coords:
                coords["instruments"] = [f"ins_{j}" for j in range(Z.shape[1])]
            if "covariates" not in coords:
                coords["covariates"] = [f"cov_{j}" for j in range(X.shape[1])]
            self.add_coords(coords)

            ins_names = coords["instruments"]
            cov_names = coords["covariates"]

            # ---------- Sanity check priors lengths ----------
            mus_t, mus_z = priors["mus"][0], priors["mus"][1]
            sig_t, sig_z = priors["sigmas"][0], priors["sigmas"][1]
            if len(mus_t) != len(ins_names) or len(sig_t) != len(ins_names):
                raise ValueError(f"beta_t prior lengths mismatch: mus={len(mus_t)}, sds={len(sig_t)}, dims={len(ins_names)}")
            if len(mus_z) != len(cov_names) or len(sig_z) != len(cov_names):
                raise ValueError(f"beta_z prior lengths mismatch: mus={len(mus_z)}, sds={len(sig_z)}, dims={len(cov_names)}")

            # ---------- Priors ----------
            beta_t = pm.Normal("beta_t", mu=mus_t, sigma=sig_t, dims="instruments")

            # base beta_z as in the original model
            if priors.get("use_truncation", False):
                beta_z_raw = pm.TruncatedNormal(
                    "beta_z_raw",
                    mu=mus_z,
                    sigma=sig_z,
                    lower=0.0,
                    dims="covariates",
                )
            else:
                beta_z_raw = pm.Normal(
                    "beta_z_raw",
                    mu=mus_z,
                    sigma=sig_z,
                    dims="covariates",
                )


            # ---------- Optional conditional prior δ|β ----------
            S_val = priors.get("S", None)
            use_S = (
                isinstance(S_val, (int, float))
                and (S_val is not None)
                and (S_val > 0.0)
                and ("avexpr" in cov_names)
                and ("logem4" in cov_names)
            )

            if use_S:
                idx_beta  = cov_names.index("avexpr")
                idx_delta = cov_names.index("logem4")
                beta = beta_z_raw[idx_beta]
                delta_cond = pm.Normal("delta_cond", mu=0.0, sigma=pt.abs(beta) * float(S_val))
                # replace only the δ entry; keep all others identical
                beta_z = pm.Deterministic(
                    "beta_z",
                    pt.set_subtensor(beta_z_raw[idx_delta], delta_cond),
                    dims="covariates",
                )
            else:
                # exact original behavior
                beta_z = pm.Deterministic("beta_z", beta_z_raw, dims="covariates")

            # ---------- Residual covariance (as in your original) ----------
            sd_dist = pm.Exponential.dist(priors["lkj_sd"], shape=2)
            chol, corr, sigmas = pm.LKJCholeskyCov(
                "chol_cov",
                eta=priors["eta"],
                n=2,
                sd_dist=sd_dist,
            )
            pm.Deterministic("cov", pt.dot(chol, chol.T))

            # ---------- Means & likelihood (unchanged) ----------
            mu_y = pm.Deterministic("mu_y", pt.dot(X, beta_z))
            mu_t = pm.Deterministic("mu_t", pt.dot(Z, beta_t))
            mu   = pm.Deterministic("mu", pt.stack((mu_y, mu_t), axis=1))

            pm.MvNormal(
                "likelihood",
                mu=mu,
                chol=chol,
                observed=np.stack((y.flatten(), t.flatten()), axis=1),
                shape=(X.shape[0], 2),
            )





    def sample_predictive_distribution(self, ppc_sampler="jax"):
        """Function to sample the Multivariate Normal posterior predictive
        Likelihood term in the IV class. This can be slow without
        using the JAX sampler compilation method. If using the
        JAX sampler it will sample only the posterior predictive distribution.
        If using the PYMC sampler if will sample both the prior
        and posterior predictive distributions."""
        random_seed = self.sample_kwargs.get("random_seed", None)

        if ppc_sampler == "jax":
            with self:
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata,
                        random_seed=random_seed,
                        compile_kwargs={"mode": "JAX"},
                    )
                )
        elif ppc_sampler == "pymc":
            with self:
                self.idata.extend(pm.sample_prior_predictive(random_seed=random_seed))
                self.idata.extend(
                    pm.sample_posterior_predictive(
                        self.idata,
                        random_seed=random_seed,
                    )
                )





    def fit(self, X, Z, y, t, coords, priors, ppc_sampler=None):
        """Draw samples from posterior distribution and potentially
        from the prior and posterior predictive distributions. The
        fit call can take values for the
        ppc_sampler = ['jax', 'pymc', None]
        We default to None, so the user can determine if they wish
        to spend time sampling the posterior predictive distribution
        independently.
        """

        # Ensure random_seed is used in sample_prior_predictive() and
        # sample_posterior_predictive() if provided in sample_kwargs.
        # Use JAX for ppc sampling of multivariate likelihood

        self.build_model(X, Z, y, t, coords, priors)
        with self:
            self.idata = pm.sample(**self.sample_kwargs)
        self.sample_predictive_distribution(ppc_sampler=ppc_sampler)
        return self.idata