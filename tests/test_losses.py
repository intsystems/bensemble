import torch

from bensemble.losses import GaussianLikelihood, VariationalLoss

def test_gaussian_likelihood_sigma_positive():
    """Initial sigma is positive."""
    assert GaussianLikelihood().sigma > 0


def test_gaussian_likelihood_sigma_responds_to_init():
    """Lower init_log_sigma produces a smaller sigma value."""
    assert GaussianLikelihood(init_log_sigma=-4.0).sigma < GaussianLikelihood(init_log_sigma=2.0).sigma


def test_gaussian_likelihood_output_shape():
    """Output shape matches the prediction tensor shape."""
    gl = GaussianLikelihood()
    preds = torch.randn(8, 3)
    assert gl(preds, torch.randn(8, 3)).shape == preds.shape


def test_gaussian_likelihood_output_finite():
    """All output values are finite."""
    gl = GaussianLikelihood()
    out = gl(torch.randn(16, 4), torch.randn(16, 4))
    assert torch.isfinite(out).all()


def test_gaussian_likelihood_grad_flows():
    """Gradients flow to log_sigma after backward."""
    gl = GaussianLikelihood()
    gl(torch.randn(8, 3), torch.randn(8, 3)).sum().backward()
    assert gl.log_sigma.grad is not None


def test_gaussian_likelihood_on_device(device):
    """Output tensor is on the same device as the input."""
    gl = GaussianLikelihood().to(device)
    preds = torch.randn(8, 3, device=device)
    out = gl(preds, torch.randn(8, 3, device=device))
    assert out.device.type == device.type


def test_variational_loss_returns_scalar():
    """Loss output is a finite scalar."""
    vl = VariationalLoss(GaussianLikelihood())
    loss = vl(torch.randn(4, 8, 1), torch.randn(8, 1), torch.tensor(1.0))
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_variational_loss_higher_kl_raises_loss():
    """Higher KL term increases the total loss."""
    gl = GaussianLikelihood()
    vl = VariationalLoss(gl, alpha=1.0, num_batches=1)
    preds, target = torch.randn(2, 8, 1), torch.randn(8, 1)
    assert vl(preds, target, torch.tensor(100.0)) > vl(preds, target, torch.tensor(0.0))


def test_variational_loss_num_batches_scales_kl():
    """More batches → smaller per-batch KL contribution → lower loss."""
    gl = GaussianLikelihood()
    preds, target, kl = torch.randn(2, 8, 1), torch.randn(8, 1), torch.tensor(10.0)
    loss_1 = VariationalLoss(gl, alpha=1.0, num_batches=1)(preds, target, kl)
    loss_10 = VariationalLoss(gl, alpha=1.0, num_batches=10)(preds, target, kl)
    assert loss_10 < loss_1


def test_variational_loss_renyi_path_finite():
    """Rényi-alpha path yields a finite loss."""
    vl = VariationalLoss(GaussianLikelihood(), alpha=0.5)
    loss = vl(torch.randn(4, 8, 1), torch.randn(8, 1), torch.tensor(0.5))
    assert torch.isfinite(loss)


def test_variational_loss_grad_flows():
    """Gradients flow through predictions during backward."""
    vl = VariationalLoss(GaussianLikelihood())
    preds = torch.randn(2, 8, 1, requires_grad=True)
    vl(preds, torch.randn(8, 1), torch.tensor(1.0)).backward()
    assert preds.grad is not None
