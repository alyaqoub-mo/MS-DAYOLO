"""
Unit tests for Gradient Reversal Layer (GRL)

Tests cover:
- Forward pass (identity operation)
- Backward pass (gradient reversal)
- Lambda scaling
- Dynamic lambda updates
- Both autograd and hook implementations
"""

import pytest
import torch
import torch.nn as nn

from ms_dayolo.models.grl import (
    GradientReversalLayer,
    GradientReversalLayerHook,
    GradientReversalFunction,
)


class TestGradientReversalFunction:
    """Test the custom autograd function."""

    def test_forward_is_identity(self):
        """Test that forward pass is identity operation."""
        x = torch.randn(4, 128, 32, 32, requires_grad=True)
        lambda_ = 1.0

        y = GradientReversalFunction.apply(x, lambda_)

        # Output should equal input
        assert torch.allclose(x, y), "Forward pass should be identity"
        assert y.shape == x.shape, "Shape should be preserved"

    def test_backward_reverses_gradients(self):
        """Test that backward pass reverses gradients."""
        x = torch.randn(4, 128, 32, 32, requires_grad=True)
        lambda_ = 1.0

        # Forward pass
        y = GradientReversalFunction.apply(x, lambda_)

        # Create gradient flowing backward
        grad_output = torch.ones_like(y)
        y.backward(grad_output)

        # Gradient should be reversed (negative)
        expected_grad = -grad_output
        assert torch.allclose(x.grad, expected_grad), "Gradients should be reversed"

    def test_lambda_scaling(self):
        """Test that lambda scales gradients correctly."""
        lambda_values = [0.0, 0.5, 1.0, 2.0]

        for lambda_ in lambda_values:
            x = torch.randn(4, 128, 32, 32, requires_grad=True)

            # Forward pass
            y = GradientReversalFunction.apply(x, lambda_)

            # Backward pass
            grad_output = torch.ones_like(y)
            y.backward(grad_output)

            # Expected gradient: -lambda * grad_output
            expected_grad = -lambda_ * grad_output
            assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
                f"Gradients should be scaled by lambda={lambda_}"


class TestGradientReversalLayer:
    """Test the GRL module."""

    def test_initialization(self):
        """Test GRL initialization."""
        # Default lambda
        grl = GradientReversalLayer()
        assert grl.lambda_ == 1.0, "Default lambda should be 1.0"

        # Custom lambda
        grl = GradientReversalLayer(lambda_=0.5)
        assert grl.lambda_ == 0.5, "Lambda should be set correctly"

    def test_forward_shape_preservation(self):
        """Test that forward pass preserves tensor shape."""
        grl = GradientReversalLayer(lambda_=1.0)

        # Test different shapes
        shapes = [
            (4, 128, 32, 32),  # [B, C, H, W]
            (8, 256, 16, 16),
            (1, 512, 8, 8),
            (16, 64),  # [B, D]
        ]

        for shape in shapes:
            x = torch.randn(*shape, requires_grad=True)
            y = grl(x)
            assert y.shape == x.shape, f"Shape {shape} should be preserved"

    def test_forward_is_identity(self):
        """Test that forward pass is identity."""
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 128, 32, 32, requires_grad=True)
        y = grl(x)

        assert torch.allclose(x, y), "Forward pass should be identity"

    def test_backward_gradient_reversal(self):
        """Test gradient reversal in backward pass."""
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 128, 32, 32, requires_grad=True)

        # Forward
        y = grl(x)
        loss = y.sum()

        # Backward
        loss.backward()

        # All gradients should be -1 (reversed from +1)
        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad), "Gradients should be reversed"

    def test_lambda_scaling_in_backward(self):
        """Test that lambda scales gradients correctly."""
        lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]

        for lambda_ in lambda_values:
            grl = GradientReversalLayer(lambda_=lambda_)
            x = torch.randn(4, 128, 32, 32, requires_grad=True)

            # Forward
            y = grl(x)
            loss = y.sum()

            # Backward
            loss.backward()

            # Expected: -lambda * 1
            expected_grad = -lambda_ * torch.ones_like(x)
            assert torch.allclose(x.grad, expected_grad, atol=1e-6), \
                f"Gradients should be scaled by lambda={lambda_}"

    def test_set_lambda(self):
        """Test dynamic lambda update."""
        grl = GradientReversalLayer(lambda_=0.0)
        assert grl.get_lambda() == 0.0

        # Update lambda
        grl.set_lambda(0.5)
        assert grl.get_lambda() == 0.5
        assert grl.lambda_ == 0.5

        # Test that new lambda affects gradients
        x = torch.randn(4, 128, 32, 32, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()

        expected_grad = -0.5 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)

    def test_in_sequential(self):
        """Test GRL in nn.Sequential."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            GradientReversalLayer(lambda_=1.0),
            nn.Conv2d(64, 128, 3, padding=1),
        )

        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        y = model(x)

        # Should work without errors
        assert y.shape == (2, 128, 32, 32)

        # Backward should work
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_extra_repr(self):
        """Test string representation."""
        grl = GradientReversalLayer(lambda_=0.5)
        repr_str = grl.extra_repr()
        assert 'lambda=0.5' in repr_str


class TestGradientReversalLayerHook:
    """Test the hook-based GRL implementation."""

    def test_forward_is_identity(self):
        """Test that forward pass is identity."""
        grl = GradientReversalLayerHook(lambda_=1.0)
        x = torch.randn(4, 128, 32, 32, requires_grad=True)
        y = grl(x)

        assert torch.allclose(x, y), "Forward pass should be identity"

    def test_backward_gradient_reversal(self):
        """Test gradient reversal."""
        grl = GradientReversalLayerHook(lambda_=1.0)
        x = torch.randn(4, 128, 32, 32, requires_grad=True)

        y = grl(x)
        loss = y.sum()
        loss.backward()

        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad), "Gradients should be reversed"

    def test_lambda_scaling(self):
        """Test lambda scaling."""
        for lambda_ in [0.0, 0.5, 1.0, 2.0]:
            grl = GradientReversalLayerHook(lambda_=lambda_)
            x = torch.randn(4, 128, 32, 32, requires_grad=True)

            y = grl(x)
            loss = y.sum()
            loss.backward()

            expected_grad = -lambda_ * torch.ones_like(x)
            assert torch.allclose(x.grad, expected_grad, atol=1e-6)


class TestGRLNumericalStability:
    """Test numerical stability and edge cases."""

    def test_zero_lambda(self):
        """Test with lambda=0 (no gradient reversal)."""
        grl = GradientReversalLayer(lambda_=0.0)
        x = torch.randn(4, 128, 32, 32, requires_grad=True)

        y = grl(x)
        loss = y.sum()
        loss.backward()

        # With lambda=0, gradients should be zero
        expected_grad = torch.zeros_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)

    def test_very_large_lambda(self):
        """Test with very large lambda values."""
        grl = GradientReversalLayer(lambda_=100.0)
        x = torch.randn(4, 128, 32, 32, requires_grad=True)

        y = grl(x)
        loss = y.sum()
        loss.backward()

        # Should still work, just with large gradients
        assert not torch.isnan(x.grad).any(), "Should not produce NaN"
        assert not torch.isinf(x.grad).any(), "Should not produce Inf"

    def test_negative_lambda(self):
        """Test with negative lambda (double reversal = positive)."""
        grl = GradientReversalLayer(lambda_=-1.0)
        x = torch.randn(4, 128, 32, 32, requires_grad=True)

        y = grl(x)
        loss = y.sum()
        loss.backward()

        # Double negative = positive
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)

    def test_with_different_dtypes(self):
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            grl = GradientReversalLayer(lambda_=1.0)
            x = torch.randn(4, 128, 32, 32, dtype=dtype, requires_grad=True)

            y = grl(x)
            loss = y.sum()
            loss.backward()

            assert x.grad.dtype == dtype, f"Gradient dtype should match input dtype"


class TestGRLGradientFlow:
    """Test gradient flow through complex networks."""

    def test_gradient_flow_in_network(self):
        """Test gradient flow through a network with GRL."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.grl = GradientReversalLayer(lambda_=1.0)
                self.conv2 = nn.Conv2d(64, 1, 3, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.grl(x)
                x = self.conv2(x)
                return x

        model = SimpleNet()
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} should have gradient"

    def test_multiple_grls_in_network(self):
        """Test multiple GRL layers in one network."""
        class MultiGRLNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.grl1 = GradientReversalLayer(lambda_=0.5)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.grl2 = GradientReversalLayer(lambda_=1.0)
                self.conv3 = nn.Conv2d(128, 1, 3, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.grl1(x)
                x = self.conv2(x)
                x = self.grl2(x)
                x = self.conv3(x)
                return x

        model = MultiGRLNet()
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # All parameters should have gradients
        for param in model.parameters():
            assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
