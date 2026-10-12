from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from helper_lib.assignment2_model import Assignment2CNN

router = APIRouter()
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "assignment2_cnn.pth"
)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


@lru_cache(maxsize=1)
def get_classifier():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model weights missing. Run train_assignment2.py first.",
        )

    model = Assignment2CNN()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


@router.post("/classify")
def classify_image(file: UploadFile = File(...)):
    model = get_classifier()

    try:
        with Image.open(BytesIO(file.file.read())) as image:
            inputs = preprocess(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image.",
        )

    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1)[0]

    index = int(probabilities.argmax().item())
    return {
        "predicted_class": CLASSES[index],
        "class_index": index,
        "confidence": float(probabilities[index].item()),
        "probabilities": {
            name: float(probabilities[i].item())
            for i, name in enumerate(CLASSES)
        },
    }
