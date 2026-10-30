import io
import math
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from helper_lib.gan import Generator
from helper_lib.lstm import LSTMModel, generate_with_lstm

router = APIRouter()
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class ImageGenerationRequest(BaseModel):
    num_samples: int = Field(default=10, ge=1, le=100)


class LSTMGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1, max_length=100)
    length: int = Field(default=20, ge=1, le=200)
    temperature: float = Field(default=1.0, gt=0, le=5)


@lru_cache(maxsize=1)
def load_gan():
    path = MODEL_DIR / "assignment3_gan.pth"
    if not path.is_file():
        raise HTTPException(503, "GAN weights are missing. Train the GAN first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Generator(checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["generator_state_dict"])
    return model.eval()


@lru_cache(maxsize=1)
def load_lstm():
    path = MODEL_DIR / "module7_lstm.pth"
    if not path.is_file():
        raise HTTPException(503, "LSTM weights are missing. Train the LSTM first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    vocabulary = checkpoint["vocabulary"]
    model = LSTMModel(
        len(vocabulary), checkpoint["embedding_dim"], checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), vocabulary


@router.post("/generate_digits", responses={200: {"content": {"image/png": {}}}})
def generate_digits(request: ImageGenerationRequest):
    model = load_gan()
    with torch.inference_mode():
        noise = torch.randn(request.num_samples, model.latent_dim)
        pixels = (
            (model(noise)[:, 0] + 1).div(2).clamp(0, 1).mul(255)
            .round().to(torch.uint8).numpy()
        )
    columns = min(5, request.num_samples)
    rows = math.ceil(request.num_samples / columns)
    grid = Image.new("L", (columns * 28, rows * 28))
    for index, array in enumerate(pixels):
        grid.paste(Image.fromarray(array),
                   ((index % columns) * 28, (index // columns) * 28))
    stream = io.BytesIO()
    grid.save(stream, format="PNG")
    return Response(content=stream.getvalue(), media_type="image/png")


@router.post("/generate_with_rnn")
def generate_with_rnn(request: LSTMGenerationRequest):
    model, vocabulary = load_lstm()
    try:
        text = generate_with_lstm(
            model, vocabulary, request.start_word,
            request.length, request.temperature,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"generated_text": text, "model": "LSTM"}
