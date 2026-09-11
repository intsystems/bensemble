import torch

from bensemble.losses import GaussianLikelihood, VariationalLoss


def test_gaussian_likelihood_sigma_positive():
    """Initial sigma is positive."""
    assert GaussianLikelihood().sigma > 0


def test_gaussian_likelihood_sigma_responds_to_init():
    """Lower init_log_sigma produces a smaller sigma value."""
    assert (
        GaussianLikelihood(init_log_sigma=-4.0).sigma
        < GaussianLikelihood(init_log_sigma=2.0).sigma
    )


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


def test_variational_loss_flat_target_matches_column_target():
    """A (B,) target scores exactly like the same target shaped (B, 1), with and without a sample axis."""
    torch.manual_seed(0)
    vl = VariationalLoss(GaussianLikelihood())
    target, kl = torch.randn(8), torch.tensor(0.0)

    for preds in (torch.randn(8, 1), torch.randn(4, 8, 1)):
        flat = vl(preds, target, kl)
        column = vl(preds, target.unsqueeze(1), kl)
        torch.testing.assert_close(flat, column)


def test_variational_loss_multi_output_is_scalar_and_differentiable():
    """With out_dim > 1 the loss is still a scalar and backward() works on both paths."""
    torch.manual_seed(0)
    target = torch.randn(8, 3)
    for alpha in (1.0, 0.5):
        vl = VariationalLoss(GaussianLikelihood(), alpha=alpha)
        preds = torch.randn(4, 8, 3, requires_grad=True)
        loss = vl(preds, target, torch.tensor(1.0))
        assert loss.shape == ()
        loss.backward()
        assert preds.grad is not None


def test_variational_loss_elbo_sums_over_outputs():
    """
    ELBO = -(sum of log-likelihood over batch and outputs - KL). The data term
    must not be averaged over the output dimension.
    """
    torch.manual_seed(0)
    gl = GaussianLikelihood()
    preds, target, kl = torch.randn(1, 8, 3), torch.randn(8, 3), torch.tensor(5.0)

    loss = VariationalLoss(gl, alpha=1.0)(preds, target, kl)
    expected = gl(preds[0], target).sum() + kl

    torch.testing.assert_close(loss, expected)


def test_variational_loss_classification_sums_over_batch():
    """
    A class-index target (cross-entropy) must reduce the data term the same way
    regression does: summed over the batch, not averaged, before KL is added.
    """
    torch.manual_seed(0)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    logits, labels, kl = (
        torch.randn(8, 10),
        torch.randint(0, 10, (8,)),
        torch.tensor(3.0),
    )

    loss = VariationalLoss(ce, alpha=1.0)(logits, labels, kl)
    expected = ce(logits, labels).sum() + kl

    torch.testing.assert_close(loss, expected)


def test_variational_loss_classification_accepts_sample_axis():
    """Stacked (K, B, C) logits with (B,) labels work and average over the samples."""
    torch.manual_seed(0)
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    logits, labels = torch.randn(4, 8, 10), torch.randint(0, 10, (8,))

    loss = VariationalLoss(ce, alpha=1.0)(logits, labels, torch.tensor(0.0))
    expected = torch.stack([ce(s, labels).sum() for s in logits]).mean()

    torch.testing.assert_close(loss, expected)
