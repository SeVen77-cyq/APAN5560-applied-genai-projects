import math
import torch
import matplotlib.pyplot as plt

from helper_lib.diffusion import Diffusion, sample_diffusion
from helper_lib.energy import EnergyModel, sample_energy


def generate_samples(model, device, num_samples=10,
                     save_path=None, show=True, *, diffusion_steps=100):
    """Generate and plot samples from a supported image generator."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    model = model.to(device)
    was_training = model.training
    model.eval()
    try:
        if isinstance(model, Diffusion):
            samples = (sample_diffusion(model, device, num_samples, diffusion_steps).cpu() + 1) / 2
        elif isinstance(model, EnergyModel):
            samples = (sample_energy(model, device, num_samples).cpu() + 1) / 2
        else:
            with torch.no_grad():
                z = torch.randn(num_samples, model.latent_dim, device=device)
                if hasattr(model, "generator"):
                    samples = (model.generator(z).cpu() + 1) / 2
                else:
                    samples = model.decoder(z).cpu()
    finally:
        model.train(was_training)

    samples = samples.clamp(0, 1)
    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(columns * 2, rows * 2),
        squeeze=False,
    )
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < num_samples:
            if samples.size(1) == 1:
                ax.imshow(samples[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(samples[index].permute(1, 2, 0).numpy())
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return samples
