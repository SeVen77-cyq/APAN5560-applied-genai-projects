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


from helper_lib.gan import train_gan


def train_diffusion(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    model.to(device).train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, _ in data_loader:
            images = images.to(device)
            times = torch.randint(len(model.betas), (images.size(0),), device=device)
            alpha = model.alpha_bar[times, None, None, None]
            noise = torch.randn_like(images)
            noisy = alpha.sqrt() * images + (1 - alpha).sqrt() * noise
            loss = criterion(model(noisy, times), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Diffusion epoch {epoch + 1}: loss={total_loss / len(data_loader):.4f}", flush=True)
    return model


def train_energy(model, data_loader, optimizer, device="cpu", epochs=10, sampling_steps=20):
    from helper_lib.energy import langevin_sample

    model.to(device).train()
    replay = torch.empty(2048, 3, 32, 32, device=device).uniform_(-1, 1)
    for epoch in range(epochs):
        total_loss = 0.0
        for images, _ in data_loader:
            images = images.to(device)
            indices = torch.randperm(len(replay), device=device)[:images.size(0)]
            fresh = torch.rand(images.size(0), 1, 1, 1, device=device) < 0.05
            start = torch.where(fresh, torch.empty_like(images).uniform_(-1, 1), replay[indices])
            negatives = langevin_sample(model, start, steps=sampling_steps)
            replay[indices] = negatives
            real_energy = model(images)
            fake_energy = model(negatives)
            regularization = real_energy.square().mean() + fake_energy.square().mean()
            loss = real_energy.mean() - fake_energy.mean() + 0.1 * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Energy epoch {epoch + 1}: loss={total_loss / len(data_loader):.4f}", flush=True)
    return model
