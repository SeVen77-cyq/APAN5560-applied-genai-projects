from tqdm.auto import tqdm
import torch
from torch import nn


class Generator(nn.Module):
    """Assignment 3: noise -> 7x7 -> 14x14 -> 28x28."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.layers(self.fc(z).reshape(-1, 128, 7, 7))


class Discriminator(nn.Module):
    """Return a real/fake probability for each image."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

    def forward_logits(self, x):
        # Skip only the final Sigmoid when computing the training loss.
        return self.layers[:-1](x)


class GAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, z):
        return self.generator(z)


def train_gan(model, data_loader, criterion, optimizer,
              device="cpu", epochs=10):
    """optimizer must contain separate 'generator' and 'discriminator' optimizers."""
    model.to(device)
    opt_g = optimizer["generator"]
    opt_d = optimizer["discriminator"]

    for epoch in range(epochs):
        model.train()
        total_g = total_d = 0.0
        count = 0

        for real, _ in tqdm(data_loader, desc="GAN batches"):
            real = real.to(device)
            batch = real.size(0)
            ones = torch.ones(batch, 1, device=device)
            zeros = torch.zeros(batch, 1, device=device)

            # Train D on real images and detached generated images.
            opt_d.zero_grad(set_to_none=True)
            noise = torch.randn(batch, model.latent_dim, device=device)
            fake = model.generator(noise)
            loss_d = (
                criterion(model.discriminator.forward_logits(real), ones)
                + criterion(model.discriminator.forward_logits(fake.detach()), zeros)
            )
            loss_d.backward()
            opt_d.step()

            # Freeze D's parameters while retaining training-mode BatchNorm.
            model.discriminator.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            loss_g = criterion(model.discriminator.forward_logits(fake), ones)
            loss_g.backward()
            opt_g.step()
            model.discriminator.requires_grad_(True)
            model.discriminator.train()

            total_d += loss_d.item() * batch
            total_g += loss_g.item() * batch
            count += batch

        if count == 0:
            raise ValueError("The training data loader is empty.")
        print(
            f"GAN epoch {epoch + 1}/{epochs}: "
            f"D={total_d / count:.4f}, G={total_g / count:.4f}",
            flush=True,
        )
    return model
