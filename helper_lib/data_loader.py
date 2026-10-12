from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(
    data_dir,
    batch_size=32,
    train=True,
    dataset_name="CIFAR10",
):
    """Create a data loader for CIFAR-10 or FashionMNIST."""

    if dataset_name == "CIFAR10":
        # Module 4 models use 32x32 RGB images.
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )

    elif dataset_name == "FashionMNIST":
        # Pad 28x28 grayscale images to 32x32 for Module 5.
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor(),
        ])

        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform,
        )

    else:
        raise ValueError(
            "dataset_name must be 'CIFAR10' or 'FashionMNIST'"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
    )
