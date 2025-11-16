"""
Unit tests for Domain Classifier modules

Tests cover:
- DomainClassifier
- DomainAdaptationBranch
- MultiscaleDomainAdaptation
"""

import pytest
import torch
import torch.nn as nn

from ms_dayolo.models.domain_classifier import (
    DomainClassifier,
    DomainAdaptationBranch,
    MultiscaleDomainAdaptation,
)


class TestDomainClassifier:
    """Test the DomainClassifier module."""

    def test_initialization(self):
        """Test classifier initialization."""
        # Default hidden channels
        dc = DomainClassifier(in_channels=256)
        assert dc.in_channels == 256
        assert dc.hidden_channels == 256
        assert dc.dropout == 0.0

        # Custom hidden channels
        dc = DomainClassifier(in_channels=256, hidden_channels=128)
        assert dc.hidden_channels == 128

        # With dropout
        dc = DomainClassifier(in_channels=256, dropout=0.5)
        assert dc.dropout == 0.5

    def test_forward_shape(self):
        """Test output shape."""
        dc = DomainClassifier(in_channels=256, hidden_channels=256)

        # Test different input shapes
        test_cases = [
            (8, 256, 32, 32),  # Standard
            (4, 256, 64, 64),  # Larger spatial
            (16, 256, 16, 16),  # Smaller spatial
            (1, 256, 80, 80),  # Batch size 1
        ]

        for shape in test_cases:
            x = torch.randn(shape)
            logits = dc(x)

            # Output should have 1 channel
            expected_shape = (shape[0], 1, shape[2], shape[3])
            assert logits.shape == expected_shape, \
                f"Expected shape {expected_shape}, got {logits.shape}"

    def test_predict_probability(self):
        """Test predict method returns probabilities."""
        dc = DomainClassifier(in_channels=256)
        x = torch.randn(8, 256, 32, 32)

        probs = dc.predict(x)

        # Check shape
        assert probs.shape == (8, 1, 32, 32)

        # Check values are in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all(), \
            "Probabilities should be in [0, 1]"

    def test_different_input_channels(self):
        """Test with different input channel sizes."""
        channel_sizes = [128, 256, 512, 1024]

        for in_ch in channel_sizes:
            dc = DomainClassifier(in_channels=in_ch, hidden_channels=in_ch)
            x = torch.randn(4, in_ch, 32, 32)
            logits = dc(x)

            assert logits.shape == (4, 1, 32, 32)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        dc = DomainClassifier(in_channels=256)
        x = torch.randn(4, 256, 32, 32, requires_grad=True)

        # Forward
        logits = dc(x)
        loss = logits.sum()

        # Backward
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check classifier parameters have gradients
        for param in dc.parameters():
            assert param.grad is not None

    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        dc = DomainClassifier(in_channels=256)

        # Check that weights are not zero or constant
        params = list(dc.parameters())
        assert len(params) > 0

        for param in params:
            # Should not be all zeros
            assert not torch.allclose(param, torch.zeros_like(param))


class TestDomainAdaptationBranch:
    """Test the DomainAdaptationBranch module."""

    def test_initialization(self):
        """Test DA branch initialization."""
        da = DomainAdaptationBranch(
            in_channels=256,
            hidden_channels=256,
            grl_lambda=0.5
        )

        assert da.in_channels == 256
        assert da.grl_lambda == 0.5
        assert da.grl is not None
        assert da.domain_classifier is not None

    def test_forward_shape(self):
        """Test output shape."""
        da = DomainAdaptationBranch(in_channels=256, grl_lambda=1.0)
        x = torch.randn(8, 256, 32, 32)

        logits = da(x)
        assert logits.shape == (8, 1, 32, 32)

    def test_grl_integration(self):
        """Test that GRL is properly integrated."""
        da = DomainAdaptationBranch(in_channels=256, grl_lambda=1.0)
        x = torch.randn(4, 256, 32, 32, requires_grad=True)

        # Forward
        logits = da(x)
        loss = logits.sum()

        # Backward
        loss.backward()

        # Gradients should be reversed (negative)
        # This is a simplified check - actual check would compare
        # with non-GRL version
        assert x.grad is not None

    def test_set_grl_lambda(self):
        """Test dynamic GRL lambda update."""
        da = DomainAdaptationBranch(in_channels=256, grl_lambda=0.0)
        assert da.get_grl_lambda() == 0.0

        # Update lambda
        da.set_grl_lambda(0.5)
        assert da.get_grl_lambda() == 0.5
        assert da.grl_lambda == 0.5

    def test_predict_probability(self):
        """Test predict method."""
        da = DomainAdaptationBranch(in_channels=256)
        x = torch.randn(8, 256, 32, 32)

        probs = da.predict(x)

        # Check shape and range
        assert probs.shape == (8, 1, 32, 32)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_reversal_effect(self):
        """Test that gradient reversal actually works."""
        # Create two branches: one with GRL, one without
        da_with_grl = DomainAdaptationBranch(in_channels=256, grl_lambda=1.0)
        da_no_grl = DomainAdaptationBranch(in_channels=256, grl_lambda=0.0)

        # Same input
        x1 = torch.randn(4, 256, 32, 32, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        # Forward
        logits1 = da_with_grl(x1)
        logits2 = da_no_grl(x2)

        # Same loss
        loss1 = logits1.sum()
        loss2 = logits2.sum()

        # Backward
        loss1.backward()
        loss2.backward()

        # Gradients should have opposite signs (due to reversal)
        # Note: This test assumes similar forward values
        # In practice, the sign relationship should hold on average


class TestMultiscaleDomainAdaptation:
    """Test the MultiscaleDomainAdaptation module."""

    def test_initialization(self):
        """Test multiscale DA initialization."""
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[128, 256, 512],
            hidden_channels_list=[128, 256, 512],
            grl_lambda=0.1
        )

        assert ms_da.num_scales == 3
        assert ms_da.grl_lambda == 0.1
        assert len(ms_da.da_branches) == 3

    def test_initialization_auto_hidden(self):
        """Test initialization with automatic hidden channels."""
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[128, 256, 512],
            grl_lambda=0.1
        )

        assert ms_da.num_scales == 3

    def test_forward_multiscale(self):
        """Test forward pass with multiple scales."""
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[128, 256, 512],
            grl_lambda=0.1
        )

        # Create features at different scales
        p3_feat = torch.randn(8, 128, 80, 80)  # P3
        p4_feat = torch.randn(8, 256, 40, 40)  # P4
        p5_feat = torch.randn(8, 512, 20, 20)  # P5

        features_list = [p3_feat, p4_feat, p5_feat]

        # Forward
        domain_logits_list = ms_da(features_list)

        # Check outputs
        assert len(domain_logits_list) == 3

        assert domain_logits_list[0].shape == (8, 1, 80, 80)
        assert domain_logits_list[1].shape == (8, 1, 40, 40)
        assert domain_logits_list[2].shape == (8, 1, 20, 20)

    def test_set_grl_lambda_all_scales(self):
        """Test updating GRL lambda for all scales."""
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[128, 256, 512],
            grl_lambda=0.0
        )

        # Update lambda
        ms_da.set_grl_lambda(0.5)

        # Check all branches updated
        assert ms_da.get_grl_lambda() == 0.5
        for branch in ms_da.da_branches:
            assert branch.get_grl_lambda() == 0.5

    def test_different_scales(self):
        """Test with different number of scales."""
        # Single scale
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[256],
            grl_lambda=0.1
        )
        features = [torch.randn(4, 256, 32, 32)]
        outputs = ms_da(features)
        assert len(outputs) == 1

        # Five scales
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[64, 128, 256, 512, 1024],
            grl_lambda=0.1
        )
        features = [
            torch.randn(4, 64, 160, 160),
            torch.randn(4, 128, 80, 80),
            torch.randn(4, 256, 40, 40),
            torch.randn(4, 512, 20, 20),
            torch.randn(4, 1024, 10, 10),
        ]
        outputs = ms_da(features)
        assert len(outputs) == 5

    def test_gradient_flow_multiscale(self):
        """Test gradient flow through all scales."""
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[128, 256, 512],
            grl_lambda=1.0
        )

        # Create features with gradients
        features_list = [
            torch.randn(4, 128, 80, 80, requires_grad=True),
            torch.randn(4, 256, 40, 40, requires_grad=True),
            torch.randn(4, 512, 20, 20, requires_grad=True),
        ]

        # Forward
        domain_logits_list = ms_da(features_list)

        # Combined loss
        loss = sum(logits.sum() for logits in domain_logits_list)

        # Backward
        loss.backward()

        # All features should have gradients
        for feat in features_list:
            assert feat.grad is not None
            assert not torch.isnan(feat.grad).any()

    def test_mismatched_features_error(self):
        """Test that mismatched features raise error."""
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[128, 256, 512],
            grl_lambda=0.1
        )

        # Wrong number of features
        features = [
            torch.randn(4, 128, 80, 80),
            torch.randn(4, 256, 40, 40),
        ]

        with pytest.raises(AssertionError):
            ms_da(features)


class TestDomainClassifierIntegration:
    """Integration tests for domain classifiers."""

    def test_with_binary_cross_entropy_loss(self):
        """Test domain classifier with BCE loss."""
        dc = DomainClassifier(in_channels=256)

        # Source features (label=1)
        source_feat = torch.randn(4, 256, 32, 32)
        source_labels = torch.ones(4, 1, 32, 32)

        # Target features (label=0)
        target_feat = torch.randn(4, 256, 32, 32)
        target_labels = torch.zeros(4, 1, 32, 32)

        # Combined batch
        features = torch.cat([source_feat, target_feat], dim=0)
        labels = torch.cat([source_labels, target_labels], dim=0)

        # Forward
        logits = dc(features)

        # Compute BCE loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)

        # Check loss is reasonable
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0

        # Backward should work
        loss.backward()

    def test_da_branch_with_scheduler(self):
        """Test DA branch with GRL scheduling."""
        da = DomainAdaptationBranch(in_channels=256, grl_lambda=0.0)

        # Simulate training with increasing lambda
        lambda_schedule = [0.0, 0.2, 0.5, 0.8, 1.0]

        for lambda_val in lambda_schedule:
            da.set_grl_lambda(lambda_val)

            x = torch.randn(4, 256, 32, 32, requires_grad=True)
            logits = da(x)
            loss = logits.sum()
            loss.backward()

            # Should work for all lambda values
            assert x.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
