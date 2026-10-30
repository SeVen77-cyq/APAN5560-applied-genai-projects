import math
import torch
import matplotlib.pyplot as plt


def generate_samples(model, device, num_samples=10,
                     save_path=None, show=True):
    """Generate and plot samples from either a GAN or a VAE."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    model = model.to(device)
    was_training = model.training
    model.eval()
    try:
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
            ax.imshow(samples[index, 0].numpy(),
                      cmap="gray", vmin=0, vmax=1)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return samples
