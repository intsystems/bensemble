import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from bensemble.core.ensemble import Ensemble
from bensemble.layers import BayesianLinear
from bensemble.uncertainty.decomposition import decompose_classification_uncertainty
from bensemble.metrics import expected_calibration_error
from bensemble.losses import VariationalLoss
from bensemble.utils import get_total_kl
from bensemble.methods.laplace_approximation import LaplaceApproximation
from bensemble.search.nes import RandomSearcher, EvolutionarySearcher
from bensemble.search.bayesian import NESBayesianSampler
from bensemble.search.space import SearchSpace

# --- 1. GLOBAL CONFIG & SEED ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 75

MC_DROPOUT_SAMPLES = 30
LAPLACE_SAMPLES = 10
VI_SAMPLES = 10
NES_ENSEMBLE_SIZE = 3
NES_POOL_SIZE = 9

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
print(f"Running Bensemble Benchmark on {DEVICE}")

# --- 2. DATASETS ---
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)


def add_noise(x, std=0.7):
    return x + torch.randn_like(x) * std


transform_noisy = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        transforms.Lambda(lambda x: add_noise(x, 0.7)),
    ]
)

print("Loading Datasets...")
train_set = datasets.CIFAR10(
    "./data", train=True, download=True, transform=train_transform
)
test_set = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
noisy_test_set = datasets.CIFAR10(
    "./data", train=False, download=True, transform=transform_noisy
)
ood_set = datasets.SVHN("./data", split="test", download=True, transform=transform)

val_size = 5000
train_indices = list(range(len(train_set)))
random.Random(42).shuffle(train_indices)
val_indices, fit_indices = train_indices[:val_size], train_indices[val_size:]

train_set_fit = Subset(train_set, fit_indices)
val_set_clean = datasets.CIFAR10(
    "./data", train=True, download=True, transform=transform
)
val_set = Subset(val_set_clean, val_indices)

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": 2,
    "pin_memory": torch.cuda.is_available(),
}
train_loader = DataLoader(train_set_fit, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_set, **loader_kwargs)
test_loader = DataLoader(test_set, **loader_kwargs)
noisy_loader = DataLoader(noisy_test_set, **loader_kwargs)
ood_loader = DataLoader(ood_set, **loader_kwargs)


# --- 3. MODEL ARCHITECTURES ---
def get_resnet(dropout=0.0):
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    if dropout > 0:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 10))
    return model


class BayesianLastLayerResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = get_resnet()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.out = BayesianLinear(in_features, 10)

    def forward(self, x):
        x = self.backbone(x)
        return self.out(x)


class ResNetSpace(SearchSpace):
    def sample(self):
        return {"dropout": 0.0}

    def mutate(self, cfg):
        return {"dropout": 0.1}

    def build(self, cfg):
        return get_resnet(dropout=cfg.get("dropout", 0.0)).to(DEVICE)


# --- 4. TRAINING UTILS & REGISTRY ---
def train(model, loader, epochs=EPOCHS, desc="Training"):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        loop = tqdm(loader, desc=f"{desc} [Epoch {ep + 1}/{epochs}]", leave=False)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        scheduler.step()
    return model


def _nes_candidate_train_fn(m):
    """Cheap training used for candidates during search (2 epochs), shared by
    NESBS, NES-RS, and NES-RE so all three NES variants get the same search
    budget per candidate."""
    return train(m, train_loader, epochs=2, desc="NES Candidate")


def _finalize_nes_ensemble(models, desc="NES Final Retrain"):
    """Fully retrain NES-selected members for the same EPOCHS budget as every
    other method, then wrap + eval()."""
    retrained = [train(m, train_loader, epochs=EPOCHS, desc=desc) for m in models]
    ens = Ensemble.from_models(retrained)
    ens.eval()
    return ens


METHODS = {}


def register(name):
    def deco(fn):
        METHODS[name] = fn
        return fn

    return deco


@register("Single Net")
def single():
    m = train(get_resnet().to(DEVICE), train_loader, desc="Single Net")
    ens = Ensemble.from_models([m])
    ens.eval()
    return ens


@register("Deep Ensemble")
def deep_ensemble():
    models = [
        train(get_resnet().to(DEVICE), train_loader, desc=f"DeepEns {i + 1}")
        for i in range(NES_ENSEMBLE_SIZE)
    ]
    ens = Ensemble.from_models(models)
    ens.eval()
    return ens


@register("MC Dropout")
def mc_dropout():
    m = train(get_resnet(dropout=0.3).to(DEVICE), train_loader, desc="MC Dropout")
    ens = Ensemble.from_stochastic(m, num_samples=MC_DROPOUT_SAMPLES, mode="dropout")
    ens.eval()
    return ens


LAPLACE_TEMPERATURE = 1.0


@register("Laplace (K-FAC)")
def laplace():
    m = train(get_resnet().to(DEVICE), train_loader, desc="Laplace Base")
    m.eval()  # freeze BN running stats before fitting/sampling the posterior
    la = LaplaceApproximation(m, likelihood="classification")
    la.compute_curvature(train_loader)

    ens = Ensemble.from_posterior(
        la, n_members=LAPLACE_SAMPLES, temperature=LAPLACE_TEMPERATURE
    )

    ens.eval()
    return ens


@register("VI (ELBO)")
def vi():
    m = BayesianLastLayerResNet().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = VariationalLoss(
        nn.CrossEntropyLoss(reduction="none"), num_batches=len(train_loader)
    )

    vi_epochs = EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=vi_epochs)

    m.train()
    for ep in range(vi_epochs):
        kl_w = min(1.0, (ep + 1) / (vi_epochs * 0.75))
        loop = tqdm(train_loader, desc=f"VI [Epoch {ep + 1}/{vi_epochs}]", leave=False)
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(m(x), y, get_total_kl(m) * kl_w)
            loss.backward()
            opt.step()
        scheduler.step()

    ens = Ensemble.from_stochastic(m, num_samples=VI_SAMPLES, mode="bayesian")

    ens.eval()
    return ens


@register("NESBS (SVGD)")
def nesbs():
    sampler = NESBayesianSampler(
        space=ResNetSpace(),
        train_fn=_nes_candidate_train_fn,
        pool_size=NES_POOL_SIZE,
        ensemble_size=NES_ENSEMBLE_SIZE,
        svgd_steps=5,
    )
    ens = sampler.sample_svgd(val_loader=val_loader)
    return _finalize_nes_ensemble(ens.member_modules, desc="NESBS Final Retrain")


@register("NES-RS")
def nes_rs():
    searcher = RandomSearcher(
        space=ResNetSpace(),
        pool_size=NES_POOL_SIZE,
        ensemble_size=NES_ENSEMBLE_SIZE,
        train_fn=_nes_candidate_train_fn,
    )
    selected_ens = searcher.search(val_loader=val_loader)
    return _finalize_nes_ensemble(
        selected_ens.member_modules, desc="NES-RS Final Retrain"
    )


@register("NES-RE")
def nes_re():
    searcher = EvolutionarySearcher(
        space=ResNetSpace(),
        pool_size=NES_POOL_SIZE,
        ensemble_size=NES_ENSEMBLE_SIZE,
        population_size=5,
        num_parent_candidates=3,
        train_fn=_nes_candidate_train_fn,
    )
    selected_ens = searcher.search(val_loader=val_loader)
    return _finalize_nes_ensemble(
        selected_ens.member_modules, desc="NES-RE Final Retrain"
    )


# --- 5. EVALUATION & PLOTTING ---
def plot_ood_histogram(id_epis, ood_epis, method_name):
    plt.figure(figsize=(7, 4))
    sns.kdeplot(
        id_epis.numpy(), fill=True, color="blue", alpha=0.5, label="In-Dist (CIFAR-10)"
    )
    sns.kdeplot(ood_epis.numpy(), fill=True, color="red", alpha=0.5, label="OOD (SVHN)")
    plt.title(f"Epistemic Uncertainty: {method_name}")
    plt.xlabel("Epistemic Uncertainty")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(RESULTS_DIR / f"ood_hist_{safe_name}.png", dpi=300)
    plt.close()


def evaluate(ens, loader):
    probs_all, labels_all, epis_all = [], [], []
    for x, y in loader:
        with torch.no_grad():
            preds = ens.predict_members(x.to(DEVICE))
            probs = torch.softmax(preds, dim=-1)
            _, _, epis = decompose_classification_uncertainty(probs)
        probs_all.append(probs.mean(0).cpu())
        labels_all.append(y.cpu())
        epis_all.append(epis.cpu())
    return torch.cat(probs_all), torch.cat(labels_all), torch.cat(epis_all)


results = []

for name, fn in METHODS.items():
    print(f"\n=== Running {name} ===")
    set_seed(42)

    start_time = time.time()
    ens = fn()
    train_time = time.time() - start_time

    id_p, id_y, id_u = evaluate(ens, test_loader)
    sh_p, sh_y, sh_u = evaluate(ens, noisy_loader)
    ood_p, _, ood_u = evaluate(ens, ood_loader)

    acc = (id_p.argmax(1) == id_y).float().mean().item()
    ece = float(expected_calibration_error(id_p, id_y))
    acc_sh = (sh_p.argmax(1) == sh_y).float().mean().item()

    auroc = roc_auc_score(
        np.concatenate([np.zeros(len(id_u)), np.ones(len(ood_u))]),
        np.concatenate([id_u.numpy(), ood_u.numpy()]),
    )

    plot_ood_histogram(id_u, ood_u, name)

    results.append(
        {
            "Method": name,
            "ID Acc": acc,
            "ID ECE": ece,
            "Shift Acc": acc_sh,
            "OOD AUROC": auroc,
            "Time (s)": round(train_time, 1),
        }
    )

# --- 6. SAVE RESULTS AND RADAR CHART ---
df = pd.DataFrame(results)
csv_path = RESULTS_DIR / "classification_benchmark_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")
print(df.to_markdown(index=False))

methods_to_plot = [
    "Single Net",
    "Deep Ensemble",
    "MC Dropout",
    "VI (ELBO)",
    "NESBS (SVGD)",
    "NES-RS",
    "NES-RE",
]
df_plot = df[df["Method"].isin(methods_to_plot)].reset_index(drop=True)

labels = ["ID Acc", "Calibration (1-ECE)", "OOD AUROC", "Shift Acc"]
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors = sns.color_palette("husl", len(methods_to_plot))

for i, (_, row) in enumerate(df_plot.iterrows()):
    values = [row["ID Acc"], 1 - row["ID ECE"], row["OOD AUROC"], row["Shift Acc"]]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=row["Method"], color=colors[i])
    ax.fill(angles, values, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontweight="bold")
ax.set_ylim(0, 1.0)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.title("Bensemble Method Comparison", pad=20, fontsize=14, fontweight="bold")
plt.savefig(RESULTS_DIR / "radar_chart.png", bbox_inches="tight", dpi=300)
print(f"Radar chart saved to {RESULTS_DIR / 'classification_radar_chart.png'}")
