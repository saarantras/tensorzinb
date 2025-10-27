import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest

from tensorzinb.tensorzinb import TensorZINB


def _simulate_zinb(n, p_nb, p_zi, seed=1337):
    """
    Simple ZINB simulator:
      y ~ 0 w.p. pi
      y ~ NB(mu, theta) otherwise; NB via Gamma-Poisson mixture
    Returns: y (n,), X_nb (DataFrame), X_nb_c (DataFrame 1-col ones),
             X_zi (DataFrame), X_zi_c (DataFrame 1-col ones)
    """
    rng = np.random.default_rng(seed)

    # Design matrices
    X_nb_dense = rng.normal(size=(n, p_nb)).astype(np.float32)
    X_zi_dense = rng.normal(size=(n, p_zi)).astype(np.float32)

    # Coefs
    beta_nb = rng.normal(scale=0.3, size=(p_nb, 1)).astype(np.float32)
    beta_zi = rng.normal(scale=0.5, size=(p_zi, 1)).astype(np.float32)
    b0_nb = np.float32(0.2)
    b0_zi = np.float32(-1.0)
    theta = np.float32(1.5)  # dispersion (>0)

    # Mean/zero-inflation
    eta_nb = b0_nb + X_nb_dense @ beta_nb    # shape (n, 1)
    mu = np.exp(eta_nb).reshape(-1)          # positive
    eta_zi = b0_zi + X_zi_dense @ beta_zi
    pi = 1.0 / (1.0 + np.exp(-eta_zi)).reshape(-1)  # in (0,1)

    # ZINB sampling: Gamma-Poisson with extra zeros
    # Poisson rate lambda = Gamma(shape=theta, scale=mu/theta)
    rate = rng.gamma(shape=float(theta), scale=(mu / float(theta)))
    y = rng.poisson(rate)
    mask_zero = rng.uniform(size=n) < pi
    y[mask_zero] = 0
    y = y.astype(np.int64)

    # Build DataFrames
    nb_cols = [f"x{i}" for i in range(p_nb)]
    zi_cols = [f"z{i}" for i in range(p_zi)]
    X_nb_df = pd.DataFrame(X_nb_dense, columns=nb_cols)
    X_nb_c_df = pd.DataFrame({"intercept": np.ones(n, dtype=np.float32)})
    X_zi_df = pd.DataFrame(X_zi_dense, columns=zi_cols)
    X_zi_c_df = pd.DataFrame({"intercept": np.ones(n, dtype=np.float32)})

    # Also make a sparse version of NB regressors (pandas SparseDtype)
    # Here we sparsify by zeroing most entries then storing as SparseDtype
    mask_keep = rng.uniform(size=X_nb_dense.size) < 0.05  # ~5% density
    X_nb_sparse = X_nb_dense.copy()
    X_nb_sparse[~mask_keep.reshape(n, p_nb)] = 0.0
    # Build pandas SparseDataFrame column-wise with fill_value=0 (required for .sparse.to_coo())
    dtype0 = pd.SparseDtype(np.float32, fill_value=0)
    X_nb_sparse_df = pd.DataFrame(
        {nb_cols[j]: pd.arrays.SparseArray(X_nb_sparse[:, j], dtype=dtype0, fill_value=0)
         for j in range(p_nb)}
    )

    return y, X_nb_df, X_nb_sparse_df, X_nb_c_df, X_zi_df, X_zi_c_df


@pytest.mark.parametrize("n,p_nb,p_zi", [(1500, 12, 3)])
def test_sparse_vs_dense_smoke(n, p_nb, p_zi):
    """
    Smoke test:
      - Fit TensorZINB on dense NB regressors
      - Fit TensorZINB on pandas-sparse NB regressors (auto-detected)
      - Ensure both runs complete and basic attributes look sane.
    This is not a numerical equality test; just checks the new sparse path executes.
    """
    y, X_nb_df, X_nb_sparse_df, X_nb_c_df, X_zi_df, X_zi_c_df = _simulate_zinb(n, p_nb, p_zi)

    # Dense path (baseline)
    model_dense = TensorZINB(
        endog=y,
        exog=X_nb_df.values.astype(np.float32),
        exog_c=X_nb_c_df.values.astype(np.float32),
        exog_infl=X_zi_df.values.astype(np.float32),
        exog_infl_c=X_zi_c_df.values.astype(np.float32),
    )
    res_dense = model_dense.fit(
        epochs=50,
        is_early_stop=True,
        is_reduce_lr=False,
        return_history=True,
    )

    # Sparse path (auto-detected)
    model_sparse = TensorZINB(
        endog=y,
        exog=X_nb_sparse_df,  # pandas SparseDtype
        exog_c=X_nb_c_df.values.astype(np.float32),
        exog_infl=X_zi_df.values.astype(np.float32),
        exog_infl_c=X_zi_c_df.values.astype(np.float32),
    )
    res_sparse = model_sparse.fit(
        epochs=50,
        is_early_stop=True,
        is_reduce_lr=False,
        return_history=True,
    )

    # Basic sanity: results are dicts with required keys
    for res in (res_dense, res_sparse):
        assert isinstance(res, dict)
        for key in ("llf_total", "aic_total", "epochs", "weights"):
            assert key in res
        assert np.isfinite(res["llf_total"])
        assert np.isfinite(res["aic_total"])
        # Optional arrays
        if "llfs" in res:
            assert np.all(np.isfinite(res["llfs"]))
        if "aics" in res:
            assert np.all(np.isfinite(res["aics"]))

    # Optional proximity check on final llf_total magnitudes (very loose; models are stochastic)
    ld = res_dense["llf_total"]
    ls = res_sparse["llf_total"]
    ratio = max(ld, ls) / max(1e-6, min(ld, ls))
    assert ratio < 5.0, f"Final llf_total diverged too much: dense={ld}, sparse={ls}, ratio={ratio}"