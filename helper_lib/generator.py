import math
import torch
import matplotlib.pyplot as plt


def generate_samples(model, device, num_samples=10):
    """Generate images from standard-normal latent samples."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive.")

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn(
            num_samples,
            model.latent_dim,
            device=device,
        )
        samples = model.decoder(z).cpu()

    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2, rows * 2),
        squeeze=False,
    )

    for index, ax in enumerate(axes.flat):
        ax.axis("off")

        if index < num_samples:
            ax.imshow(
                samples[index, 0].numpy(),
                cmap="gray",
                vmin=0,
                vmax=1,
            )

    fig.tight_layout()
    plt.show()

    return samples
