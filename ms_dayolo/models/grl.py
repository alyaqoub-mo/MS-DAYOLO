"""
Gradient Reversal Layer (GRL) for Domain Adaptation

This module implements the Gradient Reversal Layer from:
    Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by
    Backpropagation. ICML 2015.

The GRL acts as an identity function during forward pass but reverses and
scales gradients during backward pass, enabling adversarial domain adaptation.
"""

from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer implementation as a custom autograd function.

    Forward pass: Identity (output = input)
    Backward pass: Gradient reversal with scaling (grad_input = -lambda * grad_output)

    This creates an adversarial training dynamic where the feature extractor
    is trained to fool the domain classifier.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass - identity function.

        Args:
            ctx: Context object to save information for backward pass
            x: Input tensor of any shape
            lambda_: Gradient reversal weight/strength

        Returns:
            Output tensor (same as input)
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)  # Identity, but creates a new tensor in computation graph

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass - reverse and scale gradients.

        Args:
            ctx: Context object with saved information from forward pass
            grad_output: Gradient of loss with respect to output

        Returns:
            Tuple of (grad_input, grad_lambda)
            - grad_input: Reversed and scaled gradient
            - grad_lambda: None (lambda is not trainable)
        """
        grad_input = grad_output.neg() * ctx.lambda_  # Reverse sign and scale
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.

    This layer performs identity operation in forward pass but reverses
    gradients during backpropagation. The reversal strength is controlled
    by lambda parameter which can be dynamically adjusted during training.

    Usage:
        >>> grl = GradientReversalLayer(lambda_=1.0)
        >>> x = torch.randn(4, 128, 32, 32, requires_grad=True)
        >>> y = grl(x)
        >>> loss = y.sum()
        >>> loss.backward()
        >>> # x.grad will be negative (reversed)

    Attributes:
        lambda_ (float): Gradient reversal strength. Higher values mean
            stronger adversarial training. Typically scheduled from 0 to 1.
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Initialize Gradient Reversal Layer.

        Args:
            lambda_: Initial gradient reversal weight. Default: 1.0
                - lambda = 0.0: No gradient reversal (normal training)
                - lambda = 1.0: Full gradient reversal
                - Typically scheduled from 0 to 1 during training
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gradient reversal layer.

        Args:
            x: Input tensor of shape [B, C, H, W] or any shape

        Returns:
            Output tensor (same as input in forward pass)
        """
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """
        Update gradient reversal weight.

        This method is typically called by a scheduler during training
        to gradually increase adversarial strength.

        Args:
            lambda_: New gradient reversal weight
        """
        self.lambda_ = lambda_

    def get_lambda(self) -> float:
        """
        Get current gradient reversal weight.

        Returns:
            Current lambda value
        """
        return self.lambda_

    def extra_repr(self) -> str:
        """
        Extra representation string for the layer.

        Returns:
            String with lambda value
        """
        return f'lambda={self.lambda_}'


class GradientReversalLayerHook(nn.Module):
    """
    Alternative implementation of GRL using backward hooks.

    This is simpler but slightly less efficient than the autograd function approach.
    Use this if you need more flexibility or encounter issues with the autograd version.

    Usage:
        >>> grl = GradientReversalLayerHook(lambda_=1.0)
        >>> x = torch.randn(4, 128, 32, 32, requires_grad=True)
        >>> y = grl(x)
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Initialize GRL with hook-based implementation.

        Args:
            lambda_: Gradient reversal weight
        """
        super(GradientReversalLayerHook, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient reversal hook.

        Args:
            x: Input tensor

        Returns:
            Output tensor with backward hook registered
        """
        # Clone to create a new tensor in the computation graph
        y = x.clone()

        # Register hook for gradient reversal
        y.register_hook(lambda grad: -self.lambda_ * grad)

        return y

    def set_lambda(self, lambda_: float) -> None:
        """Update gradient reversal weight."""
        self.lambda_ = lambda_

    def get_lambda(self) -> float:
        """Get current gradient reversal weight."""
        return self.lambda_

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'lambda={self.lambda_}'


# Alias for convenience
GRL = GradientReversalLayer
