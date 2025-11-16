"""
Unit tests for GRL Schedulers

Tests cover:
- GRLScheduler (sigmoid scheduling)
- LinearGRLScheduler
- ConstantGRLScheduler
- WarmupGRLScheduler
- CosineAnnealingLRWithWarmup
"""

import pytest
import torch
import math

from ms_dayolo.models.grl import GradientReversalLayer
from ms_dayolo.models.domain_classifier import (
    DomainAdaptationBranch,
    MultiscaleDomainAdaptation,
)
from ms_dayolo.training.scheduler import (
    GRLScheduler,
    LinearGRLScheduler,
    ConstantGRLScheduler,
    WarmupGRLScheduler,
    CosineAnnealingLRWithWarmup,
)


class TestGRLScheduler:
    """Test the sigmoid GRL scheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=10000,
            gamma=10.0
        )

        assert scheduler.max_steps == 10000
        assert scheduler.gamma == 10.0
        assert scheduler.current_step == 0

    def test_lambda_at_start(self):
        """Test lambda value at training start."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=10000,
            gamma=10.0
        )

        lambda_start = scheduler.get_lambda()
        # At p=0, lambda should be close to 0
        assert lambda_start < 0.1, f"Lambda at start should be near 0, got {lambda_start}"

    def test_lambda_at_end(self):
        """Test lambda value at training end."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=1000,
            gamma=10.0
        )

        # Step to end
        for _ in range(1000):
            scheduler.step()

        lambda_end = scheduler.get_lambda()
        # At p=1, lambda should be close to 1
        assert lambda_end > 0.9, f"Lambda at end should be near 1, got {lambda_end}"

    def test_lambda_at_middle(self):
        """Test lambda value at training midpoint."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=1000,
            gamma=10.0
        )

        # Step to middle
        for _ in range(500):
            scheduler.step()

        lambda_mid = scheduler.get_lambda()
        # At p=0.5, lambda should be around 0
        # (sigmoid function crosses 0 at p=0.5 with this formula)
        assert -0.1 < lambda_mid < 0.1, f"Lambda at middle should be near 0, got {lambda_mid}"

    def test_sigmoid_curve(self):
        """Test that lambda follows sigmoid curve."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=100,
            gamma=10.0
        )

        lambdas = []
        for _ in range(101):
            lambdas.append(scheduler.get_lambda())
            scheduler.step()

        # Lambda should be monotonically increasing
        for i in range(len(lambdas) - 1):
            assert lambdas[i] <= lambdas[i + 1], "Lambda should increase monotonically"

        # Should start near 0 and end near 1
        assert lambdas[0] < 0.1
        assert lambdas[-1] > 0.9

    def test_step_updates_grl(self):
        """Test that step() updates GRL lambda."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=100,
            gamma=10.0
        )

        # Initial lambda
        initial_lambda = grl.get_lambda()

        # Step scheduler
        scheduler.step()

        # GRL lambda should be updated
        new_lambda = grl.get_lambda()
        assert new_lambda != initial_lambda

    def test_multiple_grls(self):
        """Test scheduler with multiple GRL layers."""
        grls = [
            GradientReversalLayer(lambda_=0.0),
            GradientReversalLayer(lambda_=0.0),
            GradientReversalLayer(lambda_=0.0),
        ]
        scheduler = GRLScheduler(
            grl_layers=grls,
            max_steps=100,
            gamma=10.0
        )

        # Step scheduler
        scheduler.step()

        # All GRLs should have same lambda
        lambda_values = [grl.get_lambda() for grl in grls]
        assert all(abs(lam - lambda_values[0]) < 1e-6 for lam in lambda_values)

    def test_with_da_branch(self):
        """Test scheduler with DomainAdaptationBranch."""
        da_branch = DomainAdaptationBranch(in_channels=256, grl_lambda=0.0)
        scheduler = GRLScheduler(
            grl_layers=[da_branch],
            max_steps=100,
            gamma=10.0
        )

        scheduler.step()
        # Should update DA branch's GRL
        assert da_branch.get_grl_lambda() > 0

    def test_with_multiscale_da(self):
        """Test scheduler with MultiscaleDomainAdaptation."""
        ms_da = MultiscaleDomainAdaptation(
            in_channels_list=[128, 256, 512],
            grl_lambda=0.0
        )
        scheduler = GRLScheduler(
            grl_layers=ms_da,
            max_steps=100,
            gamma=10.0
        )

        scheduler.step()
        # Should update all branches
        assert ms_da.get_grl_lambda() > 0

    def test_state_dict(self):
        """Test state dict save/load."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=100,
            gamma=10.0
        )

        # Step a few times
        for _ in range(10):
            scheduler.step()

        # Save state
        state = scheduler.state_dict()
        assert 'current_step' in state
        assert state['current_step'] == 10

        # Create new scheduler and load state
        grl2 = GradientReversalLayer(lambda_=0.0)
        scheduler2 = GRLScheduler(
            grl_layers=[grl2],
            max_steps=50,  # Different max_steps
            gamma=5.0      # Different gamma
        )
        scheduler2.load_state_dict(state)

        # State should match
        assert scheduler2.current_step == 10
        assert scheduler2.max_steps == 100
        assert scheduler2.gamma == 10.0


class TestLinearGRLScheduler:
    """Test linear GRL scheduler."""

    def test_linear_increase(self):
        """Test that lambda increases linearly."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = LinearGRLScheduler(
            grl_layers=[grl],
            max_steps=100,
            max_lambda=1.0
        )

        lambdas = []
        for _ in range(101):
            lambdas.append(scheduler.get_lambda())
            scheduler.step()

        # Check linearity
        for i, lam in enumerate(lambdas):
            expected = i / 100.0
            assert abs(lam - expected) < 1e-6, f"Step {i}: expected {expected}, got {lam}"

    def test_custom_max_lambda(self):
        """Test with custom max lambda."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = LinearGRLScheduler(
            grl_layers=[grl],
            max_steps=100,
            max_lambda=0.5
        )

        # Step to end
        for _ in range(100):
            scheduler.step()

        final_lambda = scheduler.get_lambda()
        assert abs(final_lambda - 0.5) < 1e-6


class TestConstantGRLScheduler:
    """Test constant GRL scheduler."""

    def test_constant_lambda(self):
        """Test that lambda remains constant."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = ConstantGRLScheduler(
            grl_layers=[grl],
            max_steps=100,
            lambda_value=0.5
        )

        lambdas = []
        for _ in range(100):
            lambdas.append(scheduler.get_lambda())
            scheduler.step()

        # All lambdas should be equal
        assert all(abs(lam - 0.5) < 1e-6 for lam in lambdas)


class TestWarmupGRLScheduler:
    """Test warmup GRL scheduler."""

    def test_warmup_phase(self):
        """Test that lambda is 0 during warmup."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = WarmupGRLScheduler(
            grl_layers=[grl],
            max_steps=1000,
            warmup_steps=100,
            gamma=10.0
        )

        # During warmup, lambda should be 0
        for _ in range(100):
            lam = scheduler.get_lambda()
            assert lam == 0.0, f"During warmup, lambda should be 0, got {lam}"
            scheduler.step()

    def test_after_warmup(self):
        """Test that lambda increases after warmup."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = WarmupGRLScheduler(
            grl_layers=[grl],
            max_steps=1000,
            warmup_steps=100,
            gamma=10.0
        )

        # Step through warmup
        for _ in range(100):
            scheduler.step()

        # After warmup, lambda should start increasing
        lambda_after_warmup = scheduler.get_lambda()
        scheduler.step()
        lambda_next = scheduler.get_lambda()

        assert lambda_next > lambda_after_warmup

    def test_warmup_to_end(self):
        """Test full warmup to end schedule."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = WarmupGRLScheduler(
            grl_layers=[grl],
            max_steps=200,
            warmup_steps=50,
            gamma=10.0
        )

        lambdas = []
        for _ in range(201):
            lambdas.append(scheduler.get_lambda())
            scheduler.step()

        # First 50 steps: lambda = 0
        assert all(lam == 0.0 for lam in lambdas[:50])

        # After warmup: lambda increases
        for i in range(50, len(lambdas) - 1):
            assert lambdas[i] <= lambdas[i + 1]

        # At end: lambda should be close to 1
        assert lambdas[-1] > 0.9


class TestCosineAnnealingLRWithWarmup:
    """Test cosine annealing LR scheduler with warmup."""

    def test_initialization(self):
        """Test scheduler initialization."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = CosineAnnealingLRWithWarmup(
            optimizer,
            T_max=1000,
            warmup_steps=100,
            eta_min=0.0
        )

        assert scheduler.T_max == 1000
        assert scheduler.warmup_steps == 100
        assert scheduler.eta_min == 0.0

    def test_warmup_phase(self):
        """Test linear warmup phase."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = CosineAnnealingLRWithWarmup(
            optimizer,
            T_max=1000,
            warmup_steps=10,
            eta_min=0.0
        )

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        # LR should increase linearly during warmup
        for i in range(len(lrs) - 1):
            assert lrs[i] < lrs[i + 1], "LR should increase during warmup"

    def test_cosine_phase(self):
        """Test cosine annealing after warmup."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = CosineAnnealingLRWithWarmup(
            optimizer,
            T_max=100,
            warmup_steps=10,
            eta_min=0.0
        )

        # Skip warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        # Record LRs during cosine phase
        lrs = []
        for _ in range(90):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        # LR should decrease during cosine annealing
        assert lrs[0] > lrs[-1]

        # Final LR should be close to eta_min
        assert lrs[-1] < 0.01

    def test_no_warmup(self):
        """Test with warmup_steps=0."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = CosineAnnealingLRWithWarmup(
            optimizer,
            T_max=100,
            warmup_steps=0,
            eta_min=0.0
        )

        # Should start cosine annealing immediately
        lr_start = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        lr_next = optimizer.param_groups[0]['lr']

        assert lr_next < lr_start


class TestSchedulerEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_max_steps(self):
        """Test with max_steps=1 (edge case)."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=1,
            gamma=10.0
        )

        # Should not crash
        scheduler.step()

    def test_resume_from_middle(self):
        """Test resuming scheduler from middle of training."""
        grl = GradientReversalLayer(lambda_=0.0)
        scheduler = GRLScheduler(
            grl_layers=[grl],
            max_steps=100,
            gamma=10.0,
            start_step=50  # Resume from step 50
        )

        # Lambda should reflect step 50
        lambda_50 = scheduler.get_lambda()

        # Create fresh scheduler and step to 50
        grl2 = GradientReversalLayer(lambda_=0.0)
        scheduler2 = GRLScheduler(
            grl_layers=[grl2],
            max_steps=100,
            gamma=10.0
        )
        for _ in range(50):
            scheduler2.step()

        lambda_50_fresh = scheduler2.get_lambda()

        # Should be approximately equal
        assert abs(lambda_50 - lambda_50_fresh) < 1e-6

    def test_empty_grl_list(self):
        """Test with empty GRL list."""
        scheduler = GRLScheduler(
            grl_layers=[],
            max_steps=100,
            gamma=10.0
        )

        # Should not crash
        scheduler.step()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
