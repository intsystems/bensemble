"""
Regression benchmark: PBP vs VI (ELBO) vs Laplace (K-FAC) vs MAP baseline on UCI datasets.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ucimlrepo import fetch_ucirepo

from bensemble.core.ensemble import Ensemble
from bensemble.layers import BayesianLinear
from bensemble.losses import VariationalLoss, GaussianLikelihood
from bensemble.utils import get_total_kl
from bensemble.methods.laplace_approximation import LaplaceApproximation
from bensemble.methods.probabilistic_backpropagation import PBPEngine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    f"Running Bensemble Regression Benchmark on {DEVICE} (PBP itself always runs on CPU — see run_pbp)"
)

HIDDEN = 50
N_SPLITS = 5
TEST_FRACTION = 0.1
NUM_EPOCHS = 40
NUM_POSTERIOR_SAMPLES = 20
LAPLACE_PRIOR_PRECISION = 1.0
MAP_VAL_FRACTION = 0.1  # fraction of training data held out for noise estimation

RESULTS_DIR = Path("results_regression")
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "yacht": dict(uci_id=None, target_col=None, direct=True),
    "energy": dict(uci_id=242, target_col="Y1"),
    "concrete": dict(uci_id=165, target_col=None),
    "power_plant": dict(uci_id=294, target_col=None),
}


def load_dataset(uci_id: int, target_col: str | None = None):
    data = fetch_ucirepo(id=uci_id)
    X = data.data.features.to_numpy(dtype="float64")
    y_df = data.data.targets
    if target_col is not None:
        y = y_df[target_col].to_numpy(dtype="float64")
    else:
        y = y_df.iloc[:, 0].to_numpy(dtype="float64")
    return X, y


def load_yacht_direct():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
    df = pd.read_csv(url, sep=r"\s+", header=None)
    X = df.iloc[:, :-1].to_numpy(dtype="float64")
    y = df.iloc[:, -1].to_numpy(dtype="float64")
    return X, y


def make_split(
    X: np.ndarray, y: np.ndarray, seed: int, test_fraction: float = TEST_FRACTION
):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardize(X_train, y_train, X_test, y_test):
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    if y_std == 0:
        y_std = 1.0

    Xtr = (X_train - x_mean) / x_std
    Xte = (X_test - x_mean) / x_std
    ytr = (y_train - y_mean) / y_std
    yte = (y_test - y_mean) / y_std
    return Xtr, ytr, Xte, yte, y_mean, y_std


def to_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int | None = None, dtype=torch.float32
):
    X_t = torch.tensor(X, dtype=dtype)
    y_t = torch.tensor(y, dtype=dtype).view(-1, 1)
    ds = TensorDataset(X_t, y_t)
    bs = batch_size or len(ds)
    return DataLoader(ds, batch_size=bs, shuffle=True)


def regression_metrics(
    mean: torch.Tensor,
    var: torch.Tensor,
    y_true_std: np.ndarray,
    y_mean: float,
    y_std: float,
):
    mean_np = mean.detach().cpu().numpy().reshape(-1)
    var_np = var.detach().cpu().numpy().reshape(-1)

    mean_real = mean_np * y_std + y_mean
    var_real = var_np * (y_std**2)
    y_real = y_true_std * y_std + y_mean

    rmse = float(np.sqrt(np.mean((mean_real - y_real) ** 2)))
    nlpd = float(
        np.mean(
            0.5 * np.log(2.0 * np.pi * var_real)
            + 0.5 * (y_real - mean_real) ** 2 / var_real
        )
    )
    return rmse, nlpd


# ---------------- Method runners ----------------


def run_pbp(Xtr, ytr, Xte, yte, epochs):
    D = Xtr.shape[1]
    train_loader = to_loader(Xtr, ytr, dtype=torch.float64)
    engine = PBPEngine(layer_sizes=[D, HIDDEN, 1], device=torch.device("cpu"))

    t0 = time.time()
    for _ in tqdm(range(epochs), desc="PBP", leave=False):
        engine.fit(train_loader, num_epochs=1, prior_refresh=1)
    elapsed = time.time() - t0

    X_test_t = torch.tensor(Xte, dtype=torch.float64)
    mean, var = engine._predictive_mean_var(X_test_t)
    return mean, var, elapsed


def run_vi(Xtr, ytr, Xte, yte, epochs):
    D = Xtr.shape[1]
    train_loader = to_loader(Xtr, ytr, batch_size=32)

    model = nn.Sequential(
        BayesianLinear(D, HIDDEN, prior_sigma=1.0),
        nn.ReLU(),
        BayesianLinear(HIDDEN, 1, prior_sigma=1.0),
    ).to(DEVICE)
    likelihood = GaussianLikelihood().to(DEVICE)
    criterion = VariationalLoss(likelihood, alpha=1.0, num_batches=len(train_loader))
    opt = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=1e-2
    )

    t0 = time.time()
    model.train()
    for _ in tqdm(range(epochs), desc="VI (ELBO)", leave=False):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            preds = model(x)
            loss = criterion(preds, y, get_total_kl(model))
            loss.backward()
            opt.step()
    elapsed = time.time() - t0

    model.eval()
    ens = Ensemble.from_stochastic(
        model, num_samples=NUM_POSTERIOR_SAMPLES, mode="bayesian"
    )

    X_test_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        samples = ens.predict_members(X_test_t)

    mean = samples.mean(dim=0)
    epistemic_var = samples.var(dim=0, unbiased=False)
    aleatoric_var = likelihood.sigma**2
    total_var = epistemic_var + aleatoric_var
    return mean.cpu(), total_var.cpu(), elapsed


def run_laplace(Xtr, ytr, Xte, yte, epochs):
    """
    Post-hoc Laplace approximation with K-FAC.
    - Point estimate obtained via MLE (no weight decay).
    - Prior precision added only during curvature estimation.
    - Observation noise estimated on a held-out validation subset.
    """
    D = Xtr.shape[1]
    n_train = Xtr.shape[0]
    n_val = max(1, int(n_train * MAP_VAL_FRACTION))
    Xtr_train, ytr_train = Xtr[:-n_val], ytr[:-n_val]
    Xtr_val, ytr_val = Xtr[-n_val:], ytr[-n_val:]

    train_loader = to_loader(Xtr_train, ytr_train, batch_size=32)

    model = nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)).to(
        DEVICE
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)  # no weight decay, pure MLE
    loss_fn = nn.MSELoss()

    t0 = time.time()
    model.train()
    for _ in tqdm(range(epochs), desc="Laplace (K-FAC)", leave=False):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    model.eval()

    # Noise variance on validation set
    with torch.no_grad():
        X_val_t = torch.tensor(Xtr_val, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(ytr_val, dtype=torch.float32).view(-1, 1).to(DEVICE)
        val_preds = model(X_val_t)
        noise_var_hat = torch.mean((val_preds - y_val_t) ** 2).item()

    la = LaplaceApproximation(
        model, likelihood="regression", prior_precision=LAPLACE_PRIOR_PRECISION
    )
    la.compute_curvature(to_loader(Xtr_train, ytr_train, batch_size=32))
    ens = Ensemble.from_posterior(la, n_members=NUM_POSTERIOR_SAMPLES, temperature=1.0)
    ens.eval()
    elapsed = time.time() - t0

    X_test_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        samples = ens.predict_members(X_test_t)

    mean = samples.mean(dim=0)
    epistemic_var = samples.var(dim=0, unbiased=False)
    total_var = epistemic_var + noise_var_hat
    return mean.cpu(), total_var.cpu(), elapsed


def run_map(Xtr, ytr, Xte, yte, epochs):
    """
    Plain MAP baseline (no Bayesian posterior).
    """
    D = Xtr.shape[1]
    n_train = Xtr.shape[0]
    n_val = max(1, int(n_train * MAP_VAL_FRACTION))
    Xtr_train, ytr_train = Xtr[:-n_val], ytr[:-n_val]
    Xtr_val, ytr_val = Xtr[-n_val:], ytr[-n_val:]

    train_loader = to_loader(Xtr_train, ytr_train, batch_size=32)

    model = nn.Sequential(nn.Linear(D, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)).to(
        DEVICE
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)  # no weight decay
    loss_fn = nn.MSELoss()

    t0 = time.time()
    model.train()
    for _ in tqdm(range(epochs), desc="MAP", leave=False):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    model.eval()

    with torch.no_grad():
        X_val_t = torch.tensor(Xtr_val, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(ytr_val, dtype=torch.float32).view(-1, 1).to(DEVICE)
        val_preds = model(X_val_t)
        noise_var_hat = torch.mean((val_preds - y_val_t) ** 2).item()

    elapsed = time.time() - t0

    X_test_t = torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_mean = model(X_test_t)

    total_var = torch.full_like(pred_mean, noise_var_hat)
    return pred_mean.cpu(), total_var.cpu(), elapsed


METHOD_RUNNERS = {
    "PBP": run_pbp,
    "VI (ELBO)": run_vi,
    "Laplace (K-FAC)": run_laplace,
    "MAP (baseline)": run_map,
}


def main():
    all_results = []

    for dataset_name, cfg in DATASETS.items():
        print(f"\n=== Dataset: {dataset_name} ===")
        X, y = (
            load_yacht_direct()
            if cfg.get("direct")
            else load_dataset(cfg["uci_id"], cfg.get("target_col"))
        )
        print(f"  n={X.shape[0]}, d={X.shape[1]}")

        for method_name, runner in METHOD_RUNNERS.items():
            rmses, nlpds, times = [], [], []

            for split_idx in range(N_SPLITS):
                Xtr_raw, ytr_raw, Xte_raw, yte_raw = make_split(X, y, seed=split_idx)
                Xtr, ytr, Xte, yte, y_mean, y_std = standardize(
                    Xtr_raw, ytr_raw, Xte_raw, yte_raw
                )

                mean, var, elapsed = runner(Xtr, ytr, Xte, yte, epochs=NUM_EPOCHS)
                rmse, nlpd = regression_metrics(mean, var, yte, y_mean, y_std)

                rmses.append(rmse)
                nlpds.append(nlpd)
                times.append(elapsed)
                print(
                    f"  [{method_name}] split {split_idx}: "
                    f"RMSE={rmse:.4f} NLPD={nlpd:.4f} time={elapsed:.1f}s"
                )

            all_results.append(
                {
                    "Dataset": dataset_name,
                    "Method": method_name,
                    "N": X.shape[0],
                    "RMSE_mean": np.mean(rmses),
                    "RMSE_std": np.std(rmses),
                    "NLPD_mean": np.mean(nlpds),
                    "NLPD_std": np.std(nlpds),
                    "Time_mean_s": np.mean(times),
                }
            )

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "regression_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
