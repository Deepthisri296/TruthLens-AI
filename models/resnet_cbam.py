import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from models.cbam import CBAM


class ResNetCBAM(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the original classifier
        self.features = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )

        # Add CBAM
        self.cbam = CBAM(2048)

        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = self.cbam(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x