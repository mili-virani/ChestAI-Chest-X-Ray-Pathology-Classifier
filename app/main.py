import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES
from src.inference import predict_and_explain
from src.model import ChestXrayClassifier

MODEL_PATH = ROOT / "models" / "best_model.pth"


def _load_model_from_checkpoint() -> tuple[ChestXrayClassifier, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXrayClassifier(num_classes=len(CLASSES), pretrained=False)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.device = _load_model_from_checkpoint()
    yield


app = FastAPI(
    title="ChestAI X-Ray Classifier API",
    description="Multi-label chest X-ray disease classification",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Upload a PNG or JPG chest X-ray.",
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    predictions, _ = predict_and_explain(
        image,
        model=app.state.model,
        device=app.state.device,
    )

    top_diagnosis = max(predictions, key=predictions.get)
    confidence = predictions[top_diagnosis]

    return {
        "probabilities": predictions,
        "top_diagnosis": top_diagnosis,
        "confidence": confidence,
    }
