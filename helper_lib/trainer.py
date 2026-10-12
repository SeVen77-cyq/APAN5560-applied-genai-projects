import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def train_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
):
    """Train a classification model."""

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress = tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for inputs, labels in progress:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                accuracy=f"{total_correct / total_samples:.4f}",
            )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={total_loss / total_samples:.4f}, "
            f"accuracy={total_correct / total_samples:.4f}"
        )

    return model


def vae_loss_function(recon_x, x, mu, logvar):
    """Use the loss weighting from the Module 5 notebook."""

    reconstruction_loss = F.binary_cross_entropy(
        recon_x,
        x,
        reduction="sum",
    )

    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return 500 * reconstruction_loss + kl_divergence


def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
):
    """Train a VAE using reconstruction and KL losses."""

    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        progress = tqdm(
            data_loader,
            desc=f"VAE Epoch {epoch + 1}/{epochs}",
        )

        for inputs, _ in progress:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            reconstruction, mu, logvar = model(inputs)
            loss = criterion(
                reconstruction,
                inputs,
                mu,
                logvar,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += inputs.size(0)

            progress.set_postfix(
                loss_per_image=f"{total_loss / total_samples:.4f}"
            )

        print(
            f"VAE Epoch {epoch + 1}: "
            f"loss per image={total_loss / total_samples:.4f}"
        )

    return model
