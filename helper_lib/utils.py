from pathlib import Path
import torch


def get_device():
    """Choose an available GPU, otherwise use CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def save_model(model, path):
    """Save model weights."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device="cpu"):
    """Load weights into a model with the same architecture."""

    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model
