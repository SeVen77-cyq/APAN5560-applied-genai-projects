import re
import torch
from torch import nn


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens, hidden=None):
        values, hidden = self.lstm(self.embedding(tokens), hidden)
        return self.output(values), hidden


@torch.inference_mode()
def generate_with_lstm(model, vocabulary, start_word,
                       length=20, temperature=1.0):
    """length includes the initial word."""
    if length < 1 or temperature <= 0:
        raise ValueError("length and temperature must be positive.")
    words = tokenize(start_word)
    if len(words) != 1:
        raise ValueError("Please provide exactly one starting word.")
    word_to_id = {word: index for index, word in enumerate(vocabulary)}
    word = words[0]
    if word not in word_to_id:
        raise ValueError(f"Starting word '{word}' is not in the training vocabulary.")

    model.eval()
    device = next(model.parameters()).device
    current = torch.tensor([[word_to_id[word]]], device=device)
    generated = [word]
    hidden = None
    for _ in range(length - 1):
        logits, hidden = model(current, hidden)
        probabilities = torch.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = int(torch.multinomial(probabilities, 1).item())
        generated.append(vocabulary[next_id])
        current = torch.tensor([[next_id]], device=device)
    return " ".join(generated)
