"""Loads the existing trained checkpoint and runs inference + Grad-CAM.

This preserves the exact working logic from the original app.py:
- ResNet18, ImageNet weights=None at load time, fc replaced with Linear(in, 2)
- state_dict + classes loaded from backend/model/best_model.pt
- Grad-CAM computed on layer4[-1] via forward/backward hooks
"""
import io
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models

from config import MODEL_PATH, IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"[model_service] Model loaded. Classes: {CLASSES}. Device: {DEVICE}")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Experimental mitigation for the resolution/sharpness shortcut.
#
# CIFAKE's REAL images are 32x32 CIFAR-10 thumbnails and its FAKE images were
# generated to match that same native resolution. The model was never shown
# genuinely high-resolution, sharp real-world photography, so fine detail and
# natural camera noise are, in effect, out-of-distribution — and empirically
# push predictions toward FAKE regardless of actual origin.
#
# domain_normalize() closes that resolution gap by routing every input
# through the same 32px bottleneck the training data effectively has, before
# upscaling to the network's expected 224x224 input. This does NOT make the
# model a better general AI-detector — it was never trained on high-res
# generator output (Midjourney/DALL-E/SDXL etc.) at any resolution — it only
# removes resolution as a spurious cue, so a real phone photo isn't penalized
# just for being sharp and detailed.
# ---------------------------------------------------------------------------
CIFAKE_NATIVE_SIZE = 32


def domain_normalize(img: Image.Image) -> Image.Image:
    tiny = img.resize((CIFAKE_NATIVE_SIZE, CIFAKE_NATIVE_SIZE), Image.BICUBIC)
    return tiny.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

_activations = {}
_gradients = {}


def _save_activation(name):
    def hook(module, inp, output):
        _activations[name] = output.detach()
    return hook


def _save_gradient(name):
    def hook(module, grad_input, grad_output):
        _gradients[name] = grad_output[0].detach()
    return hook


_target_layer = model.layer4[-1]
_target_layer.register_forward_hook(_save_activation("target"))
_target_layer.register_full_backward_hook(_save_gradient("target"))


def generate_gradcam(input_tensor, class_idx):
    """Returns a normalized (0-1) heatmap the same H,W as IMG_SIZE."""
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()

    acts = _activations["target"][0]
    grads = _gradients["target"][0]

    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(DEVICE)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((IMG_SIZE, IMG_SIZE))
    return np.array(cam_img) / 255.0


def overlay_heatmap(original_img, cam, alpha=0.45):
    """Overlay the Grad-CAM heatmap (red-ish) on top of the original image."""
    resized = original_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    original_np = np.array(resized).astype(np.float32) / 255.0

    heatmap = np.zeros((*cam.shape, 3), dtype=np.float32)
    heatmap[..., 0] = cam
    heatmap[..., 1] = cam * 0.3
    heatmap[..., 2] = 1 - cam

    blended = (1 - alpha) * original_np + alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return Image.fromarray(np.uint8(blended * 255))


def image_to_base64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict(original_img: Image.Image, normalize_domain: bool = False):
    """Runs classification + Grad-CAM on a PIL image (already RGB).

    normalize_domain: experimental mitigation (see domain_normalize above).
    When True, the image is passed through a 32px bottleneck before
    classification to remove resolution/sharpness as a spurious cue.

    Returns a dict: prediction, confidence, probabilities, heatmap_b64, heatmap_img
    """
    model_input_img = domain_normalize(original_img) if normalize_domain else original_img

    input_tensor = preprocess(model_input_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    input_tensor_grad = preprocess(model_input_img).unsqueeze(0).to(DEVICE)
    input_tensor_grad.requires_grad_(False)
    cam = generate_gradcam(input_tensor_grad, pred_idx)
    heatmap_img = overlay_heatmap(original_img, cam)

    return {
        "prediction": CLASSES[pred_idx],
        "confidence": round(confidence * 100, 2),
        "probabilities": {CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASSES))},
        "heatmap_b64": image_to_base64(heatmap_img),
        "heatmap_img": heatmap_img,
        "domain_normalized": normalize_domain,
    }
