"""
CNN Model Architecture for Diabetic Retinopathy Detection
=========================================================
Transfer learning models with ImageNet-pretrained backbones
(EfficientNet-B0, EfficientNet-B3, ResNet50) and custom
classification heads optimized for fundus image classification.

Supports progressive unfreezing for fine-tuning.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


class RetinopathyModel(nn.Module):
    """
    Fundus image classifier using transfer learning.

    Architecture:
        1. Pretrained backbone (EfficientNet or ResNet)
        2. Adaptive average pooling
        3. Custom classification head:
           FC → BatchNorm → ReLU → Dropout → FC

    Args:
        architecture: Backbone name ('efficientnet_b0', 'efficientnet_b3', 'resnet50').
        num_classes: Number of output classes (2 for binary, 5 for multi-class).
        pretrained: Whether to load ImageNet-pretrained weights.
        dropout: Dropout rate in the classification head.
        hidden_dim: Hidden layer size in the classifier.
    """

    SUPPORTED_ARCHITECTURES = [
        "efficientnet_b0",
        "efficientnet_b3",
        "resnet50",
        "resnet101",
    ]

    def __init__(
        self,
        architecture: str = "efficientnet_b3",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.4,
        hidden_dim: int = 512,
    ):
        super().__init__()

        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Choose from: {self.SUPPORTED_ARCHITECTURES}"
            )

        self.architecture = architecture
        self.num_classes = num_classes

        # Create backbone via timm (universal interface)
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            global_pool="avg",
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features_dim = self.backbone(dummy).shape[-1]

        logger.info(
            f"Backbone: {architecture} (features_dim={features_dim}, "
            f"pretrained={pretrained})"
        )

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

        # Track frozen state
        self._backbone_frozen = False

    def _init_classifier(self):
        """Initialize classifier head with Kaiming normal."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """Freeze all backbone parameters (for initial training of the head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Backbone frozen. Trainable params: {trainable:,} / {total:,} "
            f"({trainable / total * 100:.1f}%)"
        )

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Backbone unfrozen. All {trainable:,} params trainable.")

    def unfreeze_layers(self, n: int):
        """
        Progressively unfreeze the last N layers/blocks of the backbone.

        For EfficientNet: unfreezes last N blocks.
        For ResNet: unfreezes last N layer groups (layer1..layer4).

        Args:
            n: Number of layer groups to unfreeze from the end.
        """
        if "efficientnet" in self.architecture:
            # EfficientNet blocks
            blocks = list(self.backbone.blocks.children())
            total_blocks = len(blocks)
            unfreeze_from = max(0, total_blocks - n)
            for i, block in enumerate(blocks):
                if i >= unfreeze_from:
                    for param in block.parameters():
                        param.requires_grad = True
            logger.info(
                f"Unfroze last {n}/{total_blocks} EfficientNet blocks"
            )
        elif "resnet" in self.architecture:
            # ResNet layer groups
            layer_groups = [
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            ]
            unfreeze_from = max(0, len(layer_groups) - n)
            for i, group in enumerate(layer_groups):
                if i >= unfreeze_from:
                    for param in group.parameters():
                        param.requires_grad = True
            logger.info(
                f"Unfroze last {n}/{len(layer_groups)} ResNet layer groups"
            )

    def get_gradcam_target_layer(self):
        """
        Get the target layer for Grad-CAM visualization.

        Returns the last convolutional layer of the backbone.
        """
        if "efficientnet" in self.architecture:
            # Last block of EfficientNet
            return self.backbone.blocks[-1]
        elif "resnet" in self.architecture:
            return self.backbone.layer4[-1]
        else:
            raise ValueError(
                f"Grad-CAM target not defined for {self.architecture}"
            )

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


def get_model(config: Dict[str, Any]) -> RetinopathyModel:
    """
    Factory function to create a RetinopathyModel from config.

    Args:
        config: Configuration dictionary with model settings.

    Returns:
        Configured RetinopathyModel instance.
    """
    model_cfg = config.get("model", {})

    model = RetinopathyModel(
        architecture=model_cfg.get("architecture", "efficientnet_b3"),
        num_classes=model_cfg.get("num_classes", 5),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.4),
        hidden_dim=model_cfg.get("hidden_dim", 512),
    )

    # Freeze backbone if configured
    if model_cfg.get("freeze_backbone", True):
        model.freeze_backbone()

    params = model.count_parameters()
    logger.info(
        f"Model created: {model_cfg.get('architecture', 'efficientnet_b3')} — "
        f"Total: {params['total']:,}, Trainable: {params['trainable']:,}"
    )

    return model
