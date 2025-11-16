"""
Schedulers for MS-DAYOLO Training

This module implements schedulers for:
1. GRL (Gradient Reversal Layer) weight scheduling
2. Learning rate scheduling (standard PyTorch schedulers)
"""

from typing import List, Optional, Union
import math

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ms_dayolo.models.grl import GradientReversalLayer
from ms_dayolo.models.domain_classifier import DomainAdaptationBranch, MultiscaleDomainAdaptation


class GRLScheduler:
    """
    Scheduler for Gradient Reversal Layer weight (lambda parameter).

    Implements the sigmoid scheduling strategy from the original paper:
        lambda_p = 2 / (1 + exp(-gamma * p)) - 1

    where:
        p = current_step / max_steps (progress from 0 to 1)
        gamma = steepness parameter (default: 10.0)

    This schedule ensures:
    - Early training (p ≈ 0): lambda ≈ 0 (focus on detection task)
    - Mid training (p ≈ 0.5): lambda ≈ 0 (gradual increase)
    - Late training (p ≈ 1): lambda ≈ 1 (strong domain adaptation)

    Usage:
        >>> grl_layer = GradientReversalLayer(lambda_=0.0)
        >>> scheduler = GRLScheduler(
        ...     grl_layers=[grl_layer],
        ...     max_steps=10000,
        ...     gamma=10.0
        ... )
        >>> for step in range(10000):
        ...     # Training code...
        ...     scheduler.step()
    """

    def __init__(
        self,
        grl_layers: Union[
            List[GradientReversalLayer],
            List[DomainAdaptationBranch],
            MultiscaleDomainAdaptation
        ],
        max_steps: int,
        gamma: float = 10.0,
        start_step: int = 0,
    ):
        """
        Initialize GRL Scheduler.

        Args:
            grl_layers: List of GRL layers or DA branches to schedule, or
                a MultiscaleDomainAdaptation module
            max_steps: Total number of training steps
            gamma: Steepness parameter for sigmoid function. Default: 10.0
                - Higher values = steeper transition
                - Lower values = smoother transition
            start_step: Starting step (for resuming training). Default: 0
        """
        # Handle different input types
        if isinstance(grl_layers, MultiscaleDomainAdaptation):
            self.grl_layers = [grl_layers]
            self.use_multiscale = True
        else:
            self.grl_layers = grl_layers
            self.use_multiscale = False

        self.max_steps = max_steps
        self.gamma = gamma
        self.current_step = start_step

        # Compute initial lambda
        self._update_lambda()

    def _compute_lambda(self, progress: float) -> float:
        """
        Compute lambda value for given progress.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Lambda value in [0, 1]
        """
        # Sigmoid schedule: lambda = 2 / (1 + exp(-gamma * p)) - 1
        lambda_p = 2.0 / (1.0 + math.exp(-self.gamma * progress)) - 1.0
        return max(0.0, min(1.0, lambda_p))  # Clamp to [0, 1]

    def _update_lambda(self) -> None:
        """Update lambda for all GRL layers."""
        progress = self.current_step / self.max_steps
        lambda_p = self._compute_lambda(progress)

        for grl_layer in self.grl_layers:
            if hasattr(grl_layer, 'set_grl_lambda'):
                # DomainAdaptationBranch or MultiscaleDomainAdaptation
                grl_layer.set_grl_lambda(lambda_p)
            elif hasattr(grl_layer, 'set_lambda'):
                # GradientReversalLayer
                grl_layer.set_lambda(lambda_p)
            else:
                raise TypeError(f"Unsupported GRL layer type: {type(grl_layer)}")

    def step(self) -> float:
        """
        Update GRL lambda for current training step.

        Returns:
            Current lambda value
        """
        self._update_lambda()
        self.current_step += 1
        return self.get_lambda()

    def get_lambda(self) -> float:
        """
        Get current lambda value without updating.

        Returns:
            Current lambda value
        """
        progress = self.current_step / self.max_steps
        return self._compute_lambda(progress)

    def get_progress(self) -> float:
        """
        Get current training progress.

        Returns:
            Progress in [0, 1]
        """
        return self.current_step / self.max_steps

    def state_dict(self) -> dict:
        """
        Get scheduler state for checkpointing.

        Returns:
            State dictionary
        """
        return {
            'max_steps': self.max_steps,
            'gamma': self.gamma,
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load scheduler state from checkpoint.

        Args:
            state_dict: State dictionary
        """
        self.max_steps = state_dict['max_steps']
        self.gamma = state_dict['gamma']
        self.current_step = state_dict['current_step']
        self._update_lambda()


class LinearGRLScheduler(GRLScheduler):
    """
    Linear GRL scheduler (simpler alternative to sigmoid).

    Lambda increases linearly from 0 to max_lambda:
        lambda = max_lambda * (current_step / max_steps)

    Usage:
        >>> scheduler = LinearGRLScheduler(
        ...     grl_layers=[grl_layer],
        ...     max_steps=10000,
        ...     max_lambda=1.0
        ... )
    """

    def __init__(
        self,
        grl_layers: Union[
            List[GradientReversalLayer],
            List[DomainAdaptationBranch],
            MultiscaleDomainAdaptation
        ],
        max_steps: int,
        max_lambda: float = 1.0,
        start_step: int = 0,
    ):
        """
        Initialize Linear GRL Scheduler.

        Args:
            grl_layers: GRL layers to schedule
            max_steps: Total training steps
            max_lambda: Maximum lambda value. Default: 1.0
            start_step: Starting step. Default: 0
        """
        self.max_lambda = max_lambda
        super().__init__(grl_layers, max_steps, gamma=0.0, start_step=start_step)

    def _compute_lambda(self, progress: float) -> float:
        """Compute lambda using linear schedule."""
        return self.max_lambda * progress


class ConstantGRLScheduler(GRLScheduler):
    """
    Constant GRL scheduler (no scheduling).

    Lambda remains constant throughout training.

    Usage:
        >>> scheduler = ConstantGRLScheduler(
        ...     grl_layers=[grl_layer],
        ...     max_steps=10000,
        ...     lambda_value=0.5
        ... )
    """

    def __init__(
        self,
        grl_layers: Union[
            List[GradientReversalLayer],
            List[DomainAdaptationBranch],
            MultiscaleDomainAdaptation
        ],
        max_steps: int,
        lambda_value: float = 1.0,
        start_step: int = 0,
    ):
        """
        Initialize Constant GRL Scheduler.

        Args:
            grl_layers: GRL layers to schedule
            max_steps: Total training steps
            lambda_value: Constant lambda value. Default: 1.0
            start_step: Starting step. Default: 0
        """
        self.lambda_value = lambda_value
        super().__init__(grl_layers, max_steps, gamma=0.0, start_step=start_step)

    def _compute_lambda(self, progress: float) -> float:
        """Compute lambda using constant value."""
        return self.lambda_value


class WarmupGRLScheduler(GRLScheduler):
    """
    GRL scheduler with warmup period.

    Lambda remains at 0 during warmup, then follows sigmoid schedule.

    Usage:
        >>> scheduler = WarmupGRLScheduler(
        ...     grl_layers=[grl_layer],
        ...     max_steps=10000,
        ...     warmup_steps=1000,
        ...     gamma=10.0
        ... )
    """

    def __init__(
        self,
        grl_layers: Union[
            List[GradientReversalLayer],
            List[DomainAdaptationBranch],
            MultiscaleDomainAdaptation
        ],
        max_steps: int,
        warmup_steps: int,
        gamma: float = 10.0,
        start_step: int = 0,
    ):
        """
        Initialize Warmup GRL Scheduler.

        Args:
            grl_layers: GRL layers to schedule
            max_steps: Total training steps
            warmup_steps: Number of warmup steps (lambda=0)
            gamma: Steepness for sigmoid after warmup
            start_step: Starting step. Default: 0
        """
        self.warmup_steps = warmup_steps
        super().__init__(grl_layers, max_steps, gamma=gamma, start_step=start_step)

    def _compute_lambda(self, progress: float) -> float:
        """Compute lambda with warmup."""
        if self.current_step < self.warmup_steps:
            return 0.0

        # Adjust progress to start from 0 after warmup
        adjusted_progress = (
            (self.current_step - self.warmup_steps) /
            (self.max_steps - self.warmup_steps)
        )
        adjusted_progress = max(0.0, min(1.0, adjusted_progress))

        # Sigmoid schedule after warmup
        lambda_p = 2.0 / (1.0 + math.exp(-self.gamma * adjusted_progress)) - 1.0
        return max(0.0, min(1.0, lambda_p))


class CosineAnnealingLRWithWarmup(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.

    Combines linear warmup with cosine annealing decay.

    Usage:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        >>> scheduler = CosineAnnealingLRWithWarmup(
        ...     optimizer,
        ...     T_max=10000,
        ...     warmup_steps=1000,
        ...     eta_min=0.0001
        ... )
        >>> for epoch in range(epochs):
        ...     for batch in dataloader:
        ...         # Training...
        ...         scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warmup_steps: int = 0,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        """
        Initialize Cosine Annealing LR with Warmup.

        Args:
            optimizer: Wrapped optimizer
            T_max: Maximum number of iterations
            warmup_steps: Number of warmup steps. Default: 0
            eta_min: Minimum learning rate. Default: 0
            last_epoch: The index of last epoch. Default: -1
        """
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / (self.warmup_steps + 1)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (
                self.T_max - self.warmup_steps
            )
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
