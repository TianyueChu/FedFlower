from torchvision import models
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights

class CelebAMobileNet(nn.Module):
    """
    A custom MobileNetV2 model for the CelebA dataset with a frozen feature extractor
    and a customizable classifier head.

    Args:
        num_classes (int): The number of output classes for the classifier head.
    """

    def __init__(self, num_classes=2):
        super(CelebAMobileNet, self).__init__()

        # Load the pre-trained MobileNetV2 model
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Freeze the feature extractor
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace the classifier head with a new fully connected layer
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

        # Initialize the new classifier head
        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize the classifier head parameters."""
        nn.init.kaiming_uniform_(self.model.classifier[1].weight)
        nn.init.zeros_(self.model.classifier[1].bias)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes].
        """
        return self.model(x)


# Instantiate the model
# model = CelebAMobileNet(num_classes=4)