import io
import math
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from helper_lib.diffusion import sample_diffusion
from helper_lib.energy import sample_energy
from helper_lib.gan import Generator
from helper_lib.model import get_model
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
    return png_grid(pixels)


def png_grid(pixels):
    num_samples, height, width = pixels.shape[:3]
    columns = min(5, num_samples)
    rows = math.ceil(num_samples / columns)
    grid = Image.new("RGB" if pixels.ndim == 4 else "L", (columns * width, rows * height))
    for index, array in enumerate(pixels):
        grid.paste(Image.fromarray(array), ((index % columns) * width, (index // columns) * height))
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


class CIFARGenerationRequest(BaseModel):
    num_samples: int = Field(default=4, ge=1, le=16)


class DiffusionGenerationRequest(CIFARGenerationRequest):
    diffusion_steps: int = Field(default=50, ge=1, le=100)


@lru_cache(maxsize=2)
def load_assignment4_model(name):
    path = MODEL_DIR / f"assignment4_{name.lower()}.pth"
    if not path.is_file():
        raise HTTPException(503, f"{name} weights are missing. See ASSIGNMENT4.md.")
    model = get_model(name)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False)


def rgb_response(samples):
    pixels = (samples.add(1).mul(127.5).clamp(0, 255).round()
              .to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    return png_grid(pixels)


@router.post("/generate_diffusion", response_class=Response,
             responses={200: {"content": {"image/png": {}}}})
def generate_diffusion(request: DiffusionGenerationRequest):
    model = load_assignment4_model("Diffusion")
    return rgb_response(sample_diffusion(model, "cpu", request.num_samples, request.diffusion_steps))


@router.post("/generate_energy", response_class=Response,
             responses={200: {"content": {"image/png": {}}}})
def generate_energy(request: CIFARGenerationRequest):
    model = load_assignment4_model("Energy")
    return rgb_response(sample_energy(model, "cpu", request.num_samples))
