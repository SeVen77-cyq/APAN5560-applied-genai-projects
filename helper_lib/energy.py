import torch
from torch import nn

from helper_lib.diffusion import IMAGE_SHAPE


class EnergyModel(nn.Module):
    """Return one scalar energy per RGB image, with lower values preferred."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
        )

    def forward(self, images):
        return self.network(images).squeeze(1)


def langevin_sample(model, images, steps=60, step_size=10.0):
    for _ in range(steps):
        with torch.enable_grad():
            images = images.detach().requires_grad_(True)
            gradient, = torch.autograd.grad(model(images).sum(), images)
        with torch.no_grad():
            images = images - step_size * gradient.clamp(-0.03, 0.03)
            images = (images + 0.005 * torch.randn_like(images)).clamp(-1, 1)
    return images.detach()


def sample_energy(model, device, num_samples=10, steps=60):
    images = torch.empty(num_samples, *IMAGE_SHAPE, device=device).uniform_(-1, 1)
    return langevin_sample(model, images, steps=steps)
