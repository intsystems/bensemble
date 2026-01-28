import torch
import torch.optim as optim
from bensemble.layers import BayesianLinear
from bensemble.losses import VariationalLoss, GaussianLikelihood
from bensemble.utils import get_total_kl


def test_mlp_overfit_one_batch_elbo():
    """
    Tests if we can learn simple relation Y = 2*X.
    """
    torch.manual_seed(42)
    x = torch.randn(16, 10)
    y = x[:, :1] * 2.0

    model = torch.nn.Sequential(
        BayesianLinear(10, 16), torch.nn.ReLU(), BayesianLinear(16, 1)
    )

    likelihood = GaussianLikelihood()
    criterion = VariationalLoss(likelihood, alpha=1.0, num_batches=1)

    optimizer = optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=0.1
    )

    initial_loss = float("inf")
    final_loss = float("inf")

    model.train()
    for epoch in range(3000):
        optimizer.zero_grad()

        preds = model(x)
        kl = get_total_kl(model)
        loss = criterion(preds, y, kl)

        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss

    # NOTE: it's just a random number
    assert final_loss < 100.0
