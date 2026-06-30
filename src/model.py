import torch
import torch.nn as nn
import timm


class ChestXrayClassifier(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()

        # Load EfficientNet-B3 pretrained on ImageNet
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,      # Remove default classification head
            global_pool="avg"   # Global average pooling
        )

        # Feature dimension of EfficientNet-B3
        feature_dim = self.backbone.num_features  # 1536

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)  # No sigmoid here
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)