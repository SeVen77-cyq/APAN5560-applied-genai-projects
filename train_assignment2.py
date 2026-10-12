import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.assignment2_model import Assignment2CNN
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import get_device, save_model


def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}", flush=True)

    # Use the same preprocessing for training, testing, and API inference.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_data = datasets.CIFAR10(
        root="data/cifar10", train=True,
        download=True, transform=transform
    )
    test_data = datasets.CIFAR10(
        root="data/cifar10", train=False,
        download=True, transform=transform
    )

    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=64, shuffle=False, num_workers=0
    )

    model = Assignment2CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = train_model(
        model, train_loader, criterion, optimizer,
        device=device, epochs=5
    )
    loss, accuracy = evaluate_model(
        model, test_loader, criterion, device=device
    )

    save_model(model, "models/assignment2_cnn.pth")

    results = {
        "dataset": "CIFAR10",
        "input_shape": [3, 64, 64],
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "seed": 42,
        "classes": train_data.classes,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/assignment2_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("Saved: models/assignment2_cnn.pth")
    print("Saved: results/assignment2_metrics.json")
    print("Assignment 2 training completed.")


if __name__ == "__main__":
    main()
