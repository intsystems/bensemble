"""Run the shipped notebook code in fresh processes without a Jupyter server."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = sorted(
    path for path in EXAMPLES.rglob("*.ipynb") if ".ipynb_checkpoints" not in path.parts
)
assert NOTEBOOKS, "No example notebooks found"

RUN_NOTEBOOK = """
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

path = Path(sys.argv[1])
namespace = {"__name__": "__main__"}
figure_count = 0

def render_figures():
    global figure_count
    for number in plt.get_fignums():
        figure = plt.figure(number)
        figure_count += 1
        figure.savefig(Path(sys.argv[2]) / f"figure-{figure_count}.png")
    plt.close("all")

plt.show = render_figures
notebook = json.loads(path.read_text(encoding="utf-8"))
for index, cell in enumerate(notebook["cells"], 1):
    if cell["cell_type"] == "code":
        print(f"Executing {path.name}, cell {index}", flush=True)
        source = "".join(cell["source"])
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)

# Train, tuning, and test observations must be disjoint.
splits = [set(ids.tolist()) for ids in namespace["split_indices"].values()]
assert all(splits)
assert sum(map(len, splits)) == len(set.union(*splits))
assert len(set.union(*splits)) == len(namespace["features"])

# Learn useful structure without requiring exact metric snapshots.
if "test_rmse" in namespace:
    baseline = (namespace["y_test"] - namespace["y_train"].mean()).square().mean().sqrt()
    assert 0 < namespace["test_rmse"] < baseline.item()

probabilities = namespace.get("test_probabilities", {})
if "test_probs" in namespace:
    probabilities = {"mixture": namespace["test_probs"]}
for name, probs in probabilities.items():
    assert torch.isfinite(probs).all(), name
    assert (probs >= 0).all(), name
    assert torch.allclose(probs.sum(dim=-1), torch.ones(len(probs)), atol=1e-5), name
    accuracy = (probs.argmax(dim=-1) == namespace["y_test"]).float().mean()
    assert accuracy > 0.75, (name, accuracy.item())

if "scaler" in namespace:
    assert namespace["scaler"].temperature.item() > 0
if "renyi_loss" in namespace:
    assert namespace["renyi_loss"] < namespace["elbo_loss"] - 1e-4
"""


@pytest.mark.parametrize(
    "notebook", NOTEBOOKS, ids=lambda path: str(path.relative_to(EXAMPLES))
)
def test_example_runs(notebook, tmp_path):
    document = json.loads(notebook.read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in document["cells"])
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None, (
                "Clear execution counts before committing"
            )
            assert not cell["outputs"], "Clear generated outputs before committing"

    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", RUN_NOTEBOOK, str(notebook), str(tmp_path)],
        cwd=notebook.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
