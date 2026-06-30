import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD
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


def generate_gradcam(
    model: ChestXrayClassifier,
    image: Image.Image,
    device: torch.device | None = None,
) -> tuple[dict[str, float], Image.Image]:
    """
    Run inference and generate a Grad-CAM heatmap for the top predicted class.

    Returns:
        A tuple of (class_probabilities, overlay_image) where class_probabilities
        maps each NIH ChestX-ray14 class name to its predicted probability and
        overlay_image is the heatmap blended onto the original PIL image.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    original_image = image.convert("RGB")
    original_rgb = np.array(original_image, dtype=np.float32) / 255.0

    input_tensor = _preprocess_image(original_image).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    class_probabilities = {
        class_name: float(probabilities[idx]) for idx, class_name in enumerate(CLASSES)
    }
    top_class_idx = int(np.argmax(probabilities))

    target_layers = [model.backbone.conv_head]
    targets = [ClassifierOutputTarget(top_class_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (original_image.width, original_image.height),
        interpolation=cv2.INTER_LINEAR,
    )

    overlay_rgb = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)
    overlay_image = Image.fromarray(overlay_rgb)

    return class_probabilities, overlay_image
