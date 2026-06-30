from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.gradcam import generate_gradcam
from src.model import ChestXrayClassifier


def _preprocess_image(image: Image.Image) -> torch.Tensor:
    """Resize to 224x224 and apply ImageNet normalization."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def _load_model(checkpoint_path: Path, device: torch.device) -> ChestXrayClassifier:
    model = ChestXrayClassifier(num_classes=len(CLASSES), pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_and_explain(
    image: Image.Image,
    checkpoint_path: str | Path | None = None,
    model: ChestXrayClassifier | None = None,
    device: torch.device | None = None,
) -> tuple[dict[str, float], Image.Image]:
    """
    Load a trained model, predict class probabilities, and generate a Grad-CAM overlay.

    Returns:
        predictions_dict: Mapping of NIH ChestX-ray14 class names to sigmoid probabilities.
        gradcam_pil_image: Grad-CAM heatmap overlaid on the input image.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        if checkpoint_path is None:
            raise ValueError("Either checkpoint_path or model must be provided.")
        model = _load_model(Path(checkpoint_path), device)

    input_tensor = _preprocess_image(image).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    predictions_dict = {
        class_name: float(probabilities[idx]) for idx, class_name in enumerate(CLASSES)
    }

    _, gradcam_pil_image = generate_gradcam(model, image, device=device)

    return predictions_dict, gradcam_pil_image
