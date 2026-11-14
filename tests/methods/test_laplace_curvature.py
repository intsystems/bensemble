import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bensemble.methods.laplace_approximation import LaplaceApproximation


def _curvature(model, inputs, likelihood="regression", batch_size=4):
    with torch.no_grad():
        targets = (
            torch.zeros(len(inputs), dtype=torch.long)
            if likelihood == "classification"
            else torch.zeros_like(model(inputs))
        )
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)
    laplace = LaplaceApproximation(model, likelihood=likelihood)
    laplace.compute_curvature(loader)
    return laplace


def _unregularized_h(laplace, name):
    factor = laplace.kronecker_factors[name]["H"]
    identity = torch.eye(len(factor), dtype=factor.dtype, device=factor.device)
    return (
        factor - (laplace.prior_precision + laplace.damping) * identity
    ) / laplace.dataset_size


@pytest.mark.parametrize("inplace", [False, True])
def test_inactive_relu_has_zero_hidden_curvature(inplace):
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False),
        nn.ReLU(inplace=inplace),
        nn.Linear(1, 1, bias=False),
    ).double()
    with torch.no_grad():
        model[0].weight.fill_(1)
        model[2].weight.fill_(2)

    laplace = _curvature(model, torch.tensor([[-1.0]], dtype=torch.float64))

    torch.testing.assert_close(
        _unregularized_h(laplace, "0"), torch.zeros(1, 1).double()
    )
    torch.testing.assert_close(
        _unregularized_h(laplace, "2"), torch.ones(1, 1).double()
    )


@pytest.mark.parametrize("activation", [nn.ReLU, nn.Tanh])
@pytest.mark.parametrize("likelihood", ["regression", "classification"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_hidden_curvature_matches_autograd(activation, likelihood, batch_size):
    model = nn.Sequential(nn.Linear(2, 3), activation(), nn.Linear(3, 2)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -0.2], [-0.5, 1.0], [0.3, 0.7]]))
        model[0].bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        model[2].weight.copy_(torch.tensor([[1.0, -0.7, 0.2], [0.5, 0.4, -1.0]]))
        model[2].bias.copy_(torch.tensor([0.2, -0.1]))
    inputs = torch.tensor(
        [[-1.0, 0.2], [0.3, 1.4], [1.2, -0.8], [-0.5, -1.0]],
        dtype=torch.float64,
    )
    reference = []
    for sample in inputs:
        hidden = model[0](sample).detach()

        def suffix(z):
            return model[2](model[1](z))

        output = suffix(hidden).detach()
        jacobian = torch.autograd.functional.jacobian(suffix, hidden)
        if likelihood == "classification":
            output_hessian = torch.autograd.functional.hessian(
                lambda logits: F.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([0]), reduction="sum"
                ),
                output,
            )
        else:
            output_hessian = torch.eye(len(output), dtype=output.dtype)
        reference.append(jacobian.T @ output_hessian @ jacobian)

    laplace = _curvature(model, inputs, likelihood, batch_size)

    torch.testing.assert_close(
        _unregularized_h(laplace, "0"), torch.stack(reference).mean(0)
    )


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_curvature_preserves_parameter_gradients_in_inference_context(frozen, context):
    model = nn.Sequential(nn.Linear(1, 2), nn.Tanh(), nn.Linear(2, 1)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0], [-0.5]]))
        model[0].bias.zero_()
        model[2].weight.copy_(torch.tensor([[2.0, 1.0]]))
        model[2].bias.zero_()
    parameters = list(model.parameters())
    for index, parameter in enumerate(parameters):
        if index % 2 == 0:
            parameter.grad = torch.full_like(parameter, 0.25)
        parameter.requires_grad_(not frozen)
    gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
    inputs = torch.tensor([[-1.0], [0.5]], dtype=torch.float64)
    hidden = F.linear(inputs, model[0].weight, model[0].bias).detach()
    derivatives = (1 - hidden.tanh().square()) * model[2].weight.detach()
    expected = torch.einsum("bi,bj->ij", derivatives, derivatives) / len(inputs)

    with context():
        laplace = _curvature(model, inputs)

    torch.testing.assert_close(_unregularized_h(laplace, "0"), expected)
    for parameter, before in zip(parameters, gradients):
        assert parameter.requires_grad is not frozen
        if before is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, before)
    assert not laplace.hook_handles
    assert not laplace._pre_activations
    assert all(not module._forward_hooks for module in model.modules())
    assert all(
        not factor.requires_grad
        for factors in laplace.kronecker_factors.values()
        for factor in factors.values()
    )


def test_curvature_follows_functional_residual_graph():
    class ResidualModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.output = nn.Linear(2, 2, bias=False)
            self.hidden = nn.Linear(2, 2, bias=False)

        def forward(self, inputs):
            hidden = self.hidden(inputs)
            return self.output(torch.tanh(hidden)) + hidden

    model = ResidualModel().double()
    with torch.no_grad():
        model.hidden.weight.copy_(torch.tensor([[1.0, 0.5], [-0.5, 1.0]]))
        model.output.weight.copy_(torch.tensor([[0.3, -0.4], [0.5, 0.7]]))
    inputs = torch.tensor([[-1.0, 0.2], [0.5, 1.5]], dtype=torch.float64)
    reference = []
    for sample in inputs:
        hidden = model.hidden(sample).detach()
        jacobian = torch.autograd.functional.jacobian(
            lambda z: model.output(torch.tanh(z)) + z, hidden
        )
        reference.append(jacobian.T @ jacobian)

    laplace = _curvature(model, inputs)

    torch.testing.assert_close(
        _unregularized_h(laplace, "hidden"), torch.stack(reference).mean(0)
    )
    torch.testing.assert_close(
        _unregularized_h(laplace, "output"), torch.eye(2, dtype=inputs.dtype)
    )


def test_failed_refit_cleans_up_reused_linear_hooks():
    class ReusedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.reuse = False

        def forward(self, inputs):
            output = self.linear(inputs)
            return self.linear(output) if self.reuse else output

    model = ReusedLinear().double()
    inputs = torch.tensor([[1.0, -1.0], [0.2, 0.5]], dtype=torch.float64)
    loader = DataLoader(TensorDataset(inputs, torch.zeros_like(inputs)), batch_size=2)
    laplace = LaplaceApproximation(model)
    laplace.compute_curvature(loader)
    assert laplace.is_fitted
    model.reuse = True

    with pytest.raises(ValueError, match="reused Linear"):
        laplace.compute_curvature(loader)

    assert not laplace.is_fitted
    assert not laplace.hook_handles
    assert not laplace._pre_activations
    assert not laplace.activations
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(RuntimeError, match="curvature not computed"):
        laplace.sample_models(1)
