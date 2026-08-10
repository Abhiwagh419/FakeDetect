import io
import base64
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model.pt")
IMG_SIZE = 224

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = checkpoint["classes"] 

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print(f"Model loaded. Classes: {CLASSES}. Device: {DEVICE}")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

gradients = {}
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

target_layer = model.layer4[-1]
target_layer.register_forward_hook(save_activation("target"))
target_layer.register_full_backward_hook(save_gradient("target"))

def generate_gradcam(input_tensor, class_idx):
    """Returns a normalized (0-1) heatmap the same H,W as the input image."""
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()

    acts = activations["target"][0]     
    grads = gradients["target"][0]       

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
    original_img = original_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    original_np = np.array(original_img).astype(np.float32) / 255.0

    heatmap = np.zeros((*cam.shape, 3), dtype=np.float32)
    heatmap[..., 0] = cam          
    heatmap[..., 1] = cam * 0.3
    heatmap[..., 2] = 1 - cam      

    blended = (1 - alpha) * original_np + alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return Image.fromarray(np.uint8(blended * 255))


def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    try:
        original_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    input_tensor = preprocess(original_img).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(False)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    input_tensor_grad = preprocess(original_img).unsqueeze(0).to(DEVICE)
    cam = generate_gradcam(input_tensor_grad, pred_idx)
    overlay_img = overlay_heatmap(original_img, cam)

    return jsonify({
        "prediction": CLASSES[pred_idx],
        "confidence": round(confidence * 100, 2),
        "probabilities": {CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASSES))},
        "heatmap": image_to_base64(overlay_img),  
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": CLASSES, "device": str(DEVICE)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
