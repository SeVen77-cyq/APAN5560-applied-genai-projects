import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
from helper_lib.lstm import LSTMModel, tokenize


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["gan", "lstm"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default=select_device())
    parser.add_argument("--corpus", default="data/corpus.txt")
    args = parser.parse_args()
    random.seed(77)
    torch.manual_seed(77)
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    print(f"Training {args.model} on {args.device}", flush=True)

    if args.model == "gan":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = datasets.MNIST(
            "data", train=True, download=True, transform=transform,
        )
        loader = DataLoader(
            dataset, batch_size=128, shuffle=True, num_workers=0,
        )
        model = get_model("GAN").to(args.device)
        optimizers = {
            "generator": torch.optim.Adam(
                model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
            "discriminator": torch.optim.Adam(
                model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
        }
        # Save after each epoch so completed training is preserved.
        for epoch in range(args.epochs or 20):
            print(f"Starting GAN epoch {epoch + 1}", flush=True)
            train_gan(model, loader, nn.BCEWithLogitsLoss(), optimizers,
                      device=args.device, epochs=1)
            torch.save({
                "generator_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.generator.state_dict().items()
                },
                "latent_dim": 100,
                "epochs_completed": epoch + 1,
            }, "models/assignment3_gan.pth")
            generate_samples(
                model, args.device, num_samples=25,
                save_path="outputs/gan_samples.png", show=False,
            )
        print("Saved models/assignment3_gan.pth and outputs/gan_samples.png")
    else:
        corpus_path = Path(args.corpus)
        if not corpus_path.is_file():
            raise SystemExit(
                f"Missing {corpus_path}. Add a plain-text training corpus first."
            )
        words = tokenize(corpus_path.read_text())
        sequence_length = 12
        if len(words) < sequence_length + 1:
            raise SystemExit("The corpus must contain at least 13 words.")
        vocabulary = sorted(set(words))
        word_to_id = {word: index for index, word in enumerate(vocabulary)}
        ids = torch.tensor([word_to_id[word] for word in words])
        windows = ids.unfold(0, sequence_length + 1, 1)
        loader = DataLoader(
            TensorDataset(windows[:, :-1], windows[:, 1:]),
            batch_size=64, shuffle=True,
        )
        model = LSTMModel(len(vocabulary)).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs or 40):
            model.train()
            total = count = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(inputs)
                loss = criterion(
                    logits.reshape(-1, len(vocabulary)),
                    targets.reshape(-1),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * targets.numel()
                count += targets.numel()
            print(f"LSTM epoch {epoch + 1}: loss={total / count:.4f}",
                  flush=True)

        torch.save({
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "vocabulary": vocabulary,
            "embedding_dim": 64,
            "hidden_dim": 128,
        }, "models/module7_lstm.pth")
        Path("outputs/lstm_vocabulary.json").write_text(
            json.dumps(vocabulary, indent=2)
        )
        print("Saved models/module7_lstm.pth")


if __name__ == "__main__":
    main()
