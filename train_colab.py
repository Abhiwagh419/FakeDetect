# ============================================================
# AI-Generated Image Detector — Training Script v3
# Run this in Google Colab (Runtime > Change runtime type > GPU)
#
# WHAT'S NEW IN v3 (vs v2)
# -----------------------------------------------------------
# v2 fixed the "sharp real photo -> FAKE" bug by training with resolution/
# compression augmentation instead of patching it at inference time.
#
# v3 addresses a SEPARATE, bigger gap: CIFAKE is built on CIFAR-10, which
# only has 10 classes — airplane, automobile, bird, cat, deer, dog, frog,
# horse, ship, truck. There are NO people in it at all, real or fake. So a
# model trained only on CIFAKE has never seen a human face in either
# class, and its behavior on portraits/selfies is essentially undefined —
# not because it's bad at telling real people from AI people, but because
# it's never been shown either.
#
# v3 fixes this by merging in a real-vs-AI-generated FACES dataset
# alongside CIFAKE, so the model actually learns what real vs. fake looks
# like for people too, not just objects.
#
# WHAT THIS STILL WON'T DO
# -----------------------------------------------------------
# - It won't perfectly match whatever specific generator YOU use to make
#   your test images. Different tools (Midjourney, SDXL, DALL-E 3, Flux,
#   etc.) leave different fingerprints. This gets you much closer for
#   faces in general, but if your own generated test image comes from a
#   tool very different from what's in the face dataset below, it may
#   still be inconsistent. There's no way around this except training on
#   images from that exact tool, which is a much bigger undertaking.
# - It's still a binary classifier making a probabilistic guess, not a
#   forensic guarantee.
# ============================================================

# ---- STEP 1: Install/setup ----
# Run in a Colab cell:
# !pip install kaggle torch torchvision scikit-learn matplotlib -q

# ---- STEP 2: Download datasets from Kaggle ----
# 1. Go to kaggle.com/settings -> API -> Create New Token (downloads kaggle.json)
# 2. In Colab, run:
#
# from google.colab import files
# files.upload()  # upload kaggle.json here
#
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
#
# # Dataset 1: CIFAKE (objects/animals/vehicles — same as before)
# !kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
# !unzip -q cifake-real-and-ai-generated-synthetic-images.zip -d /content/cifake
#
# # Dataset 2: real vs AI-generated FACES — fills the "no people" gap
# !kaggle datasets download -d philosopher0808/real-vs-ai-generated-faces-dataset
# !unzip -q real-vs-ai-generated-faces-dataset.zip -d /content/faces
#
# ---- STEP 3: INSPECT the faces dataset structure before merging ----
# Every Kaggle dataset organizes its folders a bit differently, and the
# uploader can restructure it at any time. Run this and LOOK at the output
# before continuing — you need real leaf folder paths for the config below:
#
# !find /content/faces -type d | head -50
#
# You're looking for folders that contain real face images and folders
# that contain AI-generated face images. Update FACE_REAL_DIRS and
# FACE_FAKE_DIRS below to match what you actually see. If the dataset
# splits into train/valid/test internally, you can list multiple folders
# per class — this script does its own shuffling and re-splitting, so it
# doesn't matter which original split a file came from.

# ---- STEP 4: Training code ----

import io
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = "/content/cifake"   # CIFAKE lives here; we merge face images INTO this same tree
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# CONFIG — edit these two lists after running the `find` command above.
# Point them at whatever leaf directories actually contain the images.
# ============================================================
FACE_REAL_DIRS = [
    "/content/faces/data_source/data_source/ffhq",
]
FACE_FAKE_DIRS = [
    "/content/faces/data_source/data_source/fake/stable_diffusion",
    "/content/faces/data_source/data_source/fake/thispersondoesnotexist",
    "/content/faces/data_source/data_source/fake/sfhq/pt1",
    "/content/faces/data_source/data_source/fake/sfhq/pt2",
    "/content/faces/data_source/data_source/fake/sfhq/pt3",
    "/content/faces/data_source/data_source/fake/sfhq/pt4",
    "/content/faces/data_source/data_source/fake/faceswap",
]

# Caps keep total training size (and Colab runtime) manageable. CIFAKE
# already contributes 50k/50k train. Adding 20k/20k faces brings the total
# to 70k/70k — a meaningful chunk of "people" without blowing up epoch time.
MAX_FACE_TRAIN_PER_CLASS = 20000
MAX_FACE_TEST_PER_CLASS = 3000


def collect_images(dirs):
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  [skip] not found: {d}")
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(d, f))
    return files


def water_fill_quotas(dir_counts, total_target):
    """Given {dir: available_count} and a total target, allocate quotas so
    small directories get fully included and the remaining target is spread
    evenly across the rest. Without this, pooling every source and sampling
    randomly lets big folders (e.g. 20k images) drown out small ones (e.g.
    ~1k), even though the small one might be the most relevant generator."""
    quotas = {d: 0 for d in dir_counts}
    remaining_dirs = set(dir_counts)
    remaining_target = total_target
    for _ in range(len(dir_counts) + 1):
        if not remaining_dirs or remaining_target <= 0:
            break
        share = remaining_target // len(remaining_dirs)
        if share == 0:
            for d in list(remaining_dirs):
                if remaining_target <= 0:
                    break
                avail = dir_counts[d] - quotas[d]
                if avail > 0:
                    quotas[d] += 1
                    remaining_target -= 1
            break
        finalized = []
        for d in remaining_dirs:
            avail = dir_counts[d] - quotas[d]
            take = min(avail, share)
            quotas[d] += take
            remaining_target -= take
            if dir_counts[d] <= quotas[d]:
                finalized.append(d)
        for d in finalized:
            remaining_dirs.discard(d)
    return quotas


def merge_faces_into_cifake(face_real_dirs, face_fake_dirs, max_train_per_class, max_test_per_class, seed=0):
    """Copies a fairly-sampled set of face images into CIFAKE's own
    train/{REAL,FAKE} and test/{REAL,FAKE} folders. Sampling is done
    per-source-directory (via water_fill_quotas) rather than pooled-then-
    random, so a small but important source (e.g. a specific generator)
    isn't drowned out by a much larger one. Filenames are prefixed with
    'face_' so they can never collide with CIFAKE's own filenames."""
    rng = random.Random(seed)
    for label, dirs, cls_name in [("real", face_real_dirs, "REAL"), ("fake", face_fake_dirs, "FAKE")]:
        per_dir_files = {d: collect_images([d]) for d in dirs}
        dirs_with_files = {d: files for d, files in per_dir_files.items() if files}
        if not dirs_with_files:
            print(f"  [warning] no images found for {cls_name} — check FACE_{cls_name}_DIRS above")
            continue

        total_target = max_train_per_class + max_test_per_class
        dir_counts = {d: len(files) for d, files in dirs_with_files.items()}
        quotas = water_fill_quotas(dir_counts, total_target)

        combined = []
        for d, files in dirs_with_files.items():
            shuffled = files[:]
            rng.shuffle(shuffled)
            taken = shuffled[:quotas[d]]
            combined.extend(taken)
            print(f"    {d}: available={len(files)} taken={len(taken)}")
        rng.shuffle(combined)

        n_train = min(max_train_per_class, len(combined))
        train_files = combined[:n_train]
        test_files = combined[n_train:n_train + max_test_per_class]

        train_dst = os.path.join(DATA_DIR, "train", cls_name)
        test_dst = os.path.join(DATA_DIR, "test", cls_name)
        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(test_dst, exist_ok=True)

        for i, src in enumerate(train_files):
            shutil.copy(src, os.path.join(train_dst, f"face_{label}_{i}{os.path.splitext(src)[1]}"))
        for i, src in enumerate(test_files):
            shutil.copy(src, os.path.join(test_dst, f"face_{label}_{i}{os.path.splitext(src)[1]}"))

        print(f"{cls_name}: merged {len(train_files)} train + {len(test_files)} test face images "
              f"(from {sum(dir_counts.values())} total available across {len(dirs_with_files)} sources)")


print("Merging face dataset into CIFAKE folder structure...")
merge_faces_into_cifake(FACE_REAL_DIRS, FACE_FAKE_DIRS, MAX_FACE_TRAIN_PER_CLASS, MAX_FACE_TEST_PER_CLASS)


# --- Custom augmentations (same as v2 — still needed on top of the merge) ---

class RandomResolutionJitter:
    """Randomly degrade then restore resolution, so the model can't use
    sharpness as a shortcut for REAL vs FAKE."""

    def __init__(self, p=0.6, min_size=24, max_size=200):
        self.p = p
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        target = random.randint(self.min_size, self.max_size)
        down = img.resize((target, target), Image.BICUBIC)
        return down.resize((w, h), Image.BICUBIC)


class RandomJPEGCompression:
    def __init__(self, p=0.4, quality_range=(35, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    def __init__(self, p=0.25, radius_range=(0.3, 1.5)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    RandomResolutionJitter(p=0.6),
    RandomJPEGCompression(p=0.4),
    RandomGaussianBlur(p=0.25),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Datasets ---
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

print("\nClasses:", train_dataset.classes)
print(f"Total train images: {len(train_dataset)} | Total test images: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Model ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for name, param in model.named_parameters():
    if not any(layer in name for layer in ("layer3", "layer4", "fc")):
        param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)


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

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": train_dataset.classes,
        }, "/content/best_model.pt")
        print(f"  -> New best model saved (val_acc={val_acc:.4f})")

_, _, preds, labels = evaluate(model, test_loader, criterion)
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=train_dataset.classes))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))

# ============================================================
# A note on the accuracy number
# ------------------------------------------------------------
# Don't be surprised if val_acc comes out lower than your original 97.63%.
# You're now testing on a harder, more diverse mix (objects AND faces,
# varied sharpness, varied compression) instead of a narrow, uniformly-
# blurry one. A somewhat lower number on a harder, more honest test set is
# a sign of a MORE useful model, not a worse one — worth explaining this
# explicitly in your report if the number moves.
# ============================================================