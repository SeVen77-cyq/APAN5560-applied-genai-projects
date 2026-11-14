import math

import torch
from torch import nn

IMAGE_SHAPE = (3, 32, 32)


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
    )


class Diffusion(nn.Module):
    """Timestep-conditioned U-Net predicting the added RGB noise."""

    def __init__(self, steps=100):
        super().__init__()
        self.register_buffer("frequencies", 10000 ** (-torch.arange(16) / 16))
        # Cosine cumulative signal schedule, with a finite final reverse step.
        times = torch.linspace(0, 1, steps + 1)
        signal = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
        betas = (1 - signal[1:] / signal[:-1]).clamp(1e-5, 0.999)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", (1 - betas).cumprod(0))
        self.input = nn.Conv2d(3, 32, 3, padding=1)
        self.down1 = block(32, 32)
        self.down2 = block(32, 64)
        self.middle = block(64, 96)
        self.up2 = block(96 + 64, 64)
        self.up1 = block(64 + 32, 32)
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.time_projection = nn.Linear(32, 32)
        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, images, timesteps):
        phase = timesteps[:, None] * self.frequencies[None, :]
        time_features = self.time_projection(torch.cat((phase.sin(), phase.cos()), dim=1))
        high = self.down1(self.input(images) + time_features[:, :, None, None])
        low = self.down2(self.pool(high))
        middle = self.middle(self.pool(low))
        decoded = self.up2(torch.cat((self.upsample(middle), low), dim=1))
        return self.output(self.up1(torch.cat((self.upsample(decoded), high), dim=1)))


@torch.no_grad()
def sample_diffusion(model, device, num_samples=10, diffusion_steps=100):
    if not 1 <= diffusion_steps <= len(model.betas):
        raise ValueError("diffusion_steps must be between 1 and the trained schedule length.")
    images = torch.randn(num_samples, *IMAGE_SHAPE, device=device)
    # DDIM uses the same trained schedule even when taking fewer reverse steps.
    timeline = torch.linspace(len(model.betas) - 1, 0, diffusion_steps).round().long().tolist()
    for step, previous in zip(timeline, timeline[1:] + [-1]):
        timesteps = torch.full((num_samples,), step, device=device, dtype=torch.long)
        alpha = model.alpha_bar[step]
        noise = model(images, timesteps)
        clean = ((images - (1 - alpha).sqrt() * noise) / alpha.sqrt()).clamp(-1, 1)
        if previous >= 0:
            next_alpha = model.alpha_bar[previous]
            images = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * noise
    return clean
