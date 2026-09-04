import torch

from preprocessing.dataset import get_dataloaders
from models.resnet_cbam import ResNetCBAM


# Load dataset
train_loader, validation_loader, test_loader = get_dataloaders(
    "data/rvf10k_subset",
    batch_size=4
)

# Create model
model = ResNetCBAM(num_classes=2)

# Get one real batch
images, labels = next(iter(train_loader))

print("Images:", images.shape)
print("Labels:", labels)

# Forward pass
with torch.no_grad():
    outputs = model(images)

print("Outputs:", outputs.shape)
print("Predicted classes:", torch.argmax(outputs, dim=1))
print("Actual classes:", labels)

print("FULL PIPELINE TEST: SUCCESS")
