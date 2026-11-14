import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion, train_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("diffusion", "energy"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Train on a random subset of this size.")
    parser.add_argument("--sampling-steps", type=int, default=20)
    default_device = ("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.batch_size <= 2048 or args.sampling_steps < 1:
        parser.error("Use positive epochs and sampling steps and a batch size from 1 to 2048.")
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    if args.limit is not None:
        if not 1 <= args.limit <= len(dataset):
            parser.error("--limit must be within the size of the CIFAR-10 training set.")
        dataset = Subset(dataset, torch.randperm(len(dataset))[:args.limit].tolist())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = get_model(args.model.capitalize()).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4 if args.model == "diffusion" else 1e-4)
    if args.model == "diffusion":
        train_diffusion(model, loader, nn.MSELoss(), optimizer, args.device, args.epochs)
    else:
        train_energy(model, loader, optimizer, args.device, args.epochs, args.sampling_steps)
    path = Path("models") / f"assignment4_{args.model}.pth"
    path.parent.mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(), path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
