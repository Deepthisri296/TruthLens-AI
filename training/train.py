import os
import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing.dataset import get_dataloaders
from models.resnet_cbam import ResNetCBAM
from training.train_utils import train_one_epoch, validate


# ==============================
# Device Configuration
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==============================
# Load Dataset
# ==============================

train_loader, validation_loader, test_loader = get_dataloaders(
    "data/rvf10k_subset",
    batch_size=32
)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(validation_loader.dataset))
print("Test images:", len(test_loader.dataset))


# ==============================
# Load Model
# ==============================

model = ResNetCBAM(num_classes=2).to(device)


# ==============================
# Loss Function
# ==============================

criterion = nn.CrossEntropyLoss()


# ==============================
# Optimizer
# ==============================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==============================
# Training Configuration
# ==============================

epochs = 5

best_validation_accuracy = 0.0

os.makedirs("models/checkpoints", exist_ok=True)


# ==============================
# Training Loop
# ==============================

for epoch in range(epochs):

    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 40)

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    validation_loss, validation_acc = validate(
        model,
        validation_loader,
        criterion,
        device
    )

    print(f"Train Loss       : {train_loss:.4f}")
    print(f"Train Accuracy   : {train_acc:.2f}%")
    print(f"Validation Loss  : {validation_loss:.4f}")
    print(f"Validation Acc.  : {validation_acc:.2f}%")

    # Save best model
    if validation_acc > best_validation_accuracy:

        best_validation_accuracy = validation_acc

        torch.save(
            model.state_dict(),
            "models/checkpoints/best_model.pth"
        )

        print("✓ Best model saved!")


# ==============================
# Training Completed
# ==============================

print("\n" + "=" * 50)
print("Training Completed!")
print(f"Best Validation Accuracy: {best_validation_accuracy:.2f}%")
print("Best model saved at:")
print("models/checkpoints/best_model.pth")
print("=" * 50)