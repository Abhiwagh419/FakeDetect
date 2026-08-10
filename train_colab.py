# ============================================================
# AI-Generated Image Detector — Training Script
# Run this in Google Colab (Runtime > Change runtime type > GPU)
# ============================================================

# ---- STEP 1: Install/setup ----
# Run in a Colab cell:
# !pip install kaggle torch torchvision scikit-learn matplotlib -q

# ---- STEP 2: Download dataset from Kaggle ----
# 1. Go to kaggle.com/settings -> API -> Create New Token (downloads kaggle.json)
# 2. In Colab, run:
#
# from google.colab import files
# files.upload()  # upload kaggle.json here
#
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
# !unzip -q cifake-real-and-ai-generated-synthetic-images.zip -d /content/cifake

# ---- STEP 3: Training code ----

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import time
import os

DATA_DIR = "/content/cifake"   # expects DATA_DIR/train/{REAL,FAKE}, DATA_DIR/test/{REAL,FAKE}
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Transforms ---
# Standard ImageNet normalization since we're using an ImageNet-pretrained backbone
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Datasets ---
# ImageFolder expects: train/REAL/*.jpg, train/FAKE/*.jpg (and same for test/)
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

print("Classes:", train_dataset.classes)  # should print ['FAKE', 'REAL'] or similar
# IMPORTANT: note the class order — index 0 and 1 map to this list.
# We'll save this mapping alongside the model so the Flask app agrees with it.

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Model: ResNet18 pretrained on ImageNet, fine-tuned for binary classification ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze early layers (keep general low-level features), fine-tune later layers
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# Replace final layer for binary classification (Real vs Fake)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# --- Training loop ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels

best_acc = 0.0
for epoch in range(EPOCHS):
    start = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion)
    scheduler.step()

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f}s")

    # Save best checkpoint
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes,   # save class order — critical for correct inference later
        }, "/content/best_model.pt")
        print(f"  -> New best model saved (val_acc={val_acc:.4f})")

_, _, preds, labels = evaluate(model, test_loader, criterion)
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=train_dataset.classes))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))
