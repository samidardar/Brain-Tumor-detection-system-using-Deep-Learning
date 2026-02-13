"""
Tests for the CNN model architecture.
"""

import torch
import pytest

from src.models.cnn_model import RetinopathyModel, get_model
from src.models.loss_functions import FocalLoss, WeightedCrossEntropyLoss, get_loss_function


class TestRetinopathyModel:
    def test_forward_pass_efficientnet_b0(self):
        """Model should produce correct output shape with EfficientNet-B0."""
        model = RetinopathyModel(
            architecture="efficientnet_b0", num_classes=5,
            pretrained=False, dropout=0.3
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 5)

    def test_forward_pass_resnet50(self):
        """Model should produce correct output shape with ResNet50."""
        model = RetinopathyModel(
            architecture="resnet50", num_classes=5,
            pretrained=False, dropout=0.3
        )
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 5)

    def test_binary_mode(self):
        """Binary classification should output 2 classes."""
        model = RetinopathyModel(
            architecture="efficientnet_b0", num_classes=2,
            pretrained=False
        )
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 2)

    def test_freeze_backbone(self):
        """Freezing backbone should reduce trainable params."""
        model = RetinopathyModel(
            architecture="efficientnet_b0", num_classes=5, pretrained=False
        )
        params_before = model.count_parameters()
        model.freeze_backbone()
        params_after = model.count_parameters()
        assert params_after["trainable"] < params_before["trainable"]

    def test_unfreeze_backbone(self):
        """Unfreezing should restore all trainable params."""
        model = RetinopathyModel(
            architecture="efficientnet_b0", num_classes=5, pretrained=False
        )
        total = model.count_parameters()["total"]
        model.freeze_backbone()
        model.unfreeze_backbone()
        assert model.count_parameters()["trainable"] == total

    def test_gradcam_target_layer(self):
        """Should return a valid target layer for Grad-CAM."""
        model = RetinopathyModel(
            architecture="efficientnet_b0", num_classes=5, pretrained=False
        )
        layer = model.get_gradcam_target_layer()
        assert layer is not None

    def test_invalid_architecture_raises(self):
        """Invalid architecture name should raise ValueError."""
        with pytest.raises(ValueError):
            RetinopathyModel(architecture="invalid_model", num_classes=5)

    def test_get_model_factory(self):
        """Factory function should create model from config."""
        config = {
            "model": {
                "architecture": "efficientnet_b0",
                "num_classes": 5,
                "pretrained": False,
                "dropout": 0.3,
                "hidden_dim": 256,
                "freeze_backbone": True,
            }
        }
        model = get_model(config)
        assert isinstance(model, RetinopathyModel)
        assert model._backbone_frozen is True


class TestLossFunctions:
    def test_focal_loss_shape(self):
        """Focal loss should return scalar."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # scalar

    def test_focal_loss_with_weights(self):
        """Focal loss should work with class weights."""
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = FocalLoss(alpha=weights, gamma=2.0)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_weighted_ce(self):
        """Weighted CE should return scalar loss."""
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = WeightedCrossEntropyLoss(class_weights=weights)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_get_loss_function_factory(self):
        """Factory should return loss module."""
        config = {"training": {"loss_function": "focal"}, "class_balance": {"auto_compute_weights": True}, "model": {"num_classes": 5}}
        loss_fn = get_loss_function(config, class_counts=[1000, 200, 100, 50, 30])
        assert isinstance(loss_fn, torch.nn.Module)
