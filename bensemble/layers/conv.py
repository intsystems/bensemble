import math
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import BaseBayesianLayer


class BayesianConv2d(BaseBayesianLayer):
    """
    Bayesian Convolutional Layer (2D) with Local Reparameterization Trick.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        prior_sigma: float = 1.0,
        init_sigma: float = 0.1,
    ):
        super().__init__()
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
        self.prior_sigma = prior_sigma

        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)

        self.w_mu = nn.Parameter(torch.empty(weight_shape))
        self.w_rho = nn.Parameter(torch.empty(weight_shape))

        self.b_mu = nn.Parameter(torch.empty(out_channels))
        self.b_rho = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.w_mu, mode="fan_in", nonlinearity="relu")
        init.zeros_(self.b_mu)

        rho_init = math.log(math.exp(self.init_sigma) - 1.0)
        self.w_rho.data.fill_(rho_init)
        self.b_rho.data.fill_(rho_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.conv2d(
                x,
                self.w_mu,
                self.b_mu,
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
