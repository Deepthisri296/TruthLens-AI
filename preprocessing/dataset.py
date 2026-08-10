from torchvision import datasets
from torch.utils.data import DataLoader

from preprocessing.transforms import train_transform, test_transform


def get_dataloaders(data_dir, batch_size=32):

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )

    validation_dataset = datasets.ImageFolder(
        root=f"{data_dir}/validation",
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader, test_loader

    return train_loader, valid_loader