"""
Domain Classifier for Domain Adaptation

This module implements domain classifiers used in MS-DAYOLO to distinguish
between source and target domains. Used in conjunction with Gradient Reversal
Layers to learn domain-invariant features.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ms_dayolo.models.grl import GradientReversalLayer


class DomainClassifier(nn.Module):
    """
    Domain classifier for binary domain classification (source vs target).

    Architecture:
        Input Features [B, C, H, W]
            ↓
        Conv(C → hidden_channels, 1x1) + BatchNorm + LeakyReLU
            ↓
        Conv(hidden_channels → 1, 1x1)
            ↓
        Output Logits [B, 1, H, W]

    The classifier outputs logits for each spatial location. During training,
    these are compared against domain labels (1=source, 0=target) using
    binary cross-entropy loss.

    Usage:
        >>> dc = DomainClassifier(in_channels=256, hidden_channels=256)
        >>> features = torch.randn(8, 256, 32, 32)
        >>> logits = dc(features)
        >>> print(logits.shape)  # [8, 1, 32, 32]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize Domain Classifier.

        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden units. If None, uses same as in_channels
            dropout: Dropout probability (0.0 = no dropout). Default: 0.0
        """
        super(DomainClassifier, self).__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Classifier architecture
        layers = []

        # Layer 1: Conv + BN + Activation
        layers.extend([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ])

        # Optional dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Layer 2: Conv to 1 channel (binary classification)
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize layer weights using appropriate strategies."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through domain classifier.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Domain logits [B, 1, H, W]
            - Positive values → predicted as source domain
            - Negative values → predicted as target domain
        """
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get domain predictions (probabilities).

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Domain probabilities [B, 1, H, W] in range [0, 1]
            - Values close to 1 → source domain
            - Values close to 0 → target domain
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def extra_repr(self) -> str:
        """Extra representation string."""
        return (
            f'in_channels={self.in_channels}, '
            f'hidden_channels={self.hidden_channels}, '
            f'dropout={self.dropout}'
        )


class DomainAdaptationBranch(nn.Module):
    """
    Complete domain adaptation branch combining GRL + Domain Classifier.

    This is the complete module used at each feature pyramid level in MS-DAYOLO.

    Architecture:
        Input Features [B, C, H, W]
            ↓
        Gradient Reversal Layer (GRL)
            ↓
        Domain Classifier
            ↓
        Domain Logits [B, 1, H, W]

    Usage:
        >>> da_branch = DomainAdaptationBranch(
        ...     in_channels=256,
        ...     hidden_channels=256,
        ...     grl_lambda=0.1
        ... )
        >>> features = torch.randn(8, 256, 32, 32)
        >>> domain_logits = da_branch(features)
        >>> print(domain_logits.shape)  # [8, 1, 32, 32]

        >>> # Update GRL weight during training
        >>> da_branch.set_grl_lambda(0.5)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        grl_lambda: float = 1.0,
        dropout: float = 0.0,
    ):
        """
        Initialize Domain Adaptation Branch.

        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden units in classifier
            grl_lambda: Initial gradient reversal strength. Default: 1.0
            dropout: Dropout probability in classifier. Default: 0.0
        """
        super(DomainAdaptationBranch, self).__init__()

        self.in_channels = in_channels
        self.grl_lambda = grl_lambda

        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambda_=grl_lambda)

        # Domain Classifier
        self.domain_classifier = DomainClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DA branch.

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Domain logits [B, 1, H, W]
        """
        # Apply gradient reversal
        x = self.grl(x)

        # Classify domain
        domain_logits = self.domain_classifier(x)

        return domain_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get domain predictions (probabilities).

        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Domain probabilities [B, 1, H, W]
        """
        x = self.grl(x)
        return self.domain_classifier.predict(x)

    def set_grl_lambda(self, lambda_: float) -> None:
        """
        Update gradient reversal weight.

        This is typically called by a scheduler during training.

        Args:
            lambda_: New gradient reversal weight
        """
        self.grl.set_lambda(lambda_)
        self.grl_lambda = lambda_

    def get_grl_lambda(self) -> float:
        """
        Get current gradient reversal weight.

        Returns:
            Current lambda value
        """
        return self.grl.get_lambda()

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'in_channels={self.in_channels}, grl_lambda={self.grl_lambda}'


class MultiscaleDomainAdaptation(nn.Module):
    """
    Multiscale domain adaptation module with DA branches at multiple scales.

    This implements the core MS-DAYOLO innovation: domain adaptation at
    multiple feature pyramid levels (P3, P4, P5).

    Usage:
        >>> ms_da = MultiscaleDomainAdaptation(
        ...     in_channels_list=[128, 256, 512],  # P3, P4, P5
        ...     hidden_channels_list=[128, 256, 512],
        ...     grl_lambda=0.1
        ... )
        >>> p3_feat = torch.randn(8, 128, 80, 80)
        >>> p4_feat = torch.randn(8, 256, 40, 40)
        >>> p5_feat = torch.randn(8, 512, 20, 20)
        >>> domain_logits_list = ms_da([p3_feat, p4_feat, p5_feat])
        >>> print(len(domain_logits_list))  # 3
    """

    def __init__(
        self,
        in_channels_list: list,
        hidden_channels_list: Optional[list] = None,
        grl_lambda: float = 1.0,
        dropout: float = 0.0,
    ):
        """
        Initialize Multiscale Domain Adaptation.

        Args:
            in_channels_list: List of input channels for each scale [P3, P4, P5]
            hidden_channels_list: List of hidden channels for each scale.
                If None, uses same as in_channels_list
            grl_lambda: Initial gradient reversal strength
            dropout: Dropout probability
        """
        super(MultiscaleDomainAdaptation, self).__init__()

        if hidden_channels_list is None:
            hidden_channels_list = in_channels_list

        assert len(in_channels_list) == len(hidden_channels_list), \
            "in_channels_list and hidden_channels_list must have same length"

        self.num_scales = len(in_channels_list)
        self.grl_lambda = grl_lambda

        # Create DA branch for each scale
        self.da_branches = nn.ModuleList([
            DomainAdaptationBranch(
                in_channels=in_ch,
                hidden_channels=hidden_ch,
                grl_lambda=grl_lambda,
                dropout=dropout,
            )
            for in_ch, hidden_ch in zip(in_channels_list, hidden_channels_list)
        ])

    def forward(self, features_list: list) -> list:
        """
        Forward pass through multiscale DA.

        Args:
            features_list: List of feature maps at different scales
                [P3_features, P4_features, P5_features]
                where each has shape [B, C_i, H_i, W_i]

        Returns:
            List of domain logits at each scale
                [P3_logits, P4_logits, P5_logits]
                where each has shape [B, 1, H_i, W_i]
        """
        assert len(features_list) == self.num_scales, \
            f"Expected {self.num_scales} feature maps, got {len(features_list)}"

        domain_logits_list = []
        for features, da_branch in zip(features_list, self.da_branches):
            domain_logits = da_branch(features)
            domain_logits_list.append(domain_logits)

        return domain_logits_list

    def set_grl_lambda(self, lambda_: float) -> None:
        """
        Update gradient reversal weight for all scales.

        Args:
            lambda_: New gradient reversal weight
        """
        for da_branch in self.da_branches:
            da_branch.set_grl_lambda(lambda_)
        self.grl_lambda = lambda_

    def get_grl_lambda(self) -> float:
        """Get current gradient reversal weight."""
        return self.grl_lambda

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f'num_scales={self.num_scales}, grl_lambda={self.grl_lambda}'
