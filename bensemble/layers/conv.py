import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from .base import BaseBayesianLayer


class BayesianConv2d(BaseBayesianLayer):
    """Bayesian 2D Convolutional layer with Local Reparameterization Trick."""

    _mc_weight: torch.Tensor | None
    _mc_bias: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        prior_sigma: float = 1.0,
        init_sigma: float = 0.1,
    ):
        """Initializes the BayesianConv2d layer.

        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution. Defaults to 1.
            padding: Zero-padding added to both sides of the input. Defaults to 0.
            dilation: Spacing between kernel elements. Defaults to 1.
            groups: Number of blocked connections from input to output channels. Defaults to 1.
            prior_sigma: Standard deviation of the Gaussian prior distribution. Defaults to 1.0.
            init_sigma: Initial standard deviation for posterior parameters. Defaults to 0.1.
        """
        super().__init__(prior_sigma=prior_sigma)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_sigma = init_sigma
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)

        self.w_mu = nn.Parameter(torch.empty(weight_shape))
        self.w_rho = nn.Parameter(torch.empty(weight_shape))

        self.b_mu = nn.Parameter(torch.empty(out_channels))
        self.b_rho = nn.Parameter(torch.empty(out_channels))

        # Temporary draws follow the module across devices and stay out of checkpoints.
        self.register_buffer("_mc_weight", None, persistent=False)
        self.register_buffer("_mc_bias", None, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes layer weights and bias parameters."""
        init.kaiming_normal_(self.w_mu, mode="fan_in", nonlinearity="relu")
        init.zeros_(self.b_mu)

        rho_init = math.log(math.exp(self.init_sigma) - 1.0)
        self.w_rho.data.fill_(rho_init)
        self.b_rho.data.fill_(rho_init)

    def _draw_mc_weights(self) -> None:
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)
        weight = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        bias = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        self.register_buffer("_mc_weight", weight, persistent=False)
        self.register_buffer("_mc_bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Uses posterior means in eval, sampled weights for MC, and local noise in training.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Convolved output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        weight = getattr(self, "_mc_weight", None) if self.training else None
        if weight is not None or not self.training:
            return F.conv2d(
                x,
                self.w_mu if weight is None else weight,
                self.b_mu if weight is None else self._mc_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        conv_mu = F.conv2d(
            x,
            self.w_mu,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        conv_var = F.conv2d(
            x.pow(2),
            w_sigma.pow(2),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        conv_var = conv_var + b_sigma.pow(2).view(1, -1, 1, 1)

        eps = torch.randn_like(conv_mu)
        out = conv_mu + eps * torch.sqrt(conv_var + 1e-8)

        return out + self.b_mu.view(1, -1, 1, 1)
