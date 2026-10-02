import io
import os
import random
import shutil
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = "./cifake-real-and-ai-generated-synthetic-images"   # CIFAKE lives here; everything else merges INTO this same tree
BATCH_SIZE = 32          # reduced from 64 to stay safely within 6GB VRAM
EPOCHS = 14
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 4          # your local CPU is dedicated, not shared like Colab's — safe to raise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"   # mixed precision — big win on your Ampere GPU

CHECKPOINT_DIR = "./checkpoints"
LOCAL_RESUME_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# ============================================================
# CONFIG — face dataset (same sources/counts as the Colab v4 run)
# ============================================================
FACE_REAL_DIRS = [
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/ffhq",
]
FACE_FAKE_DIRS = [
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/fake/stable_diffusion",
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/fake/thispersondoesnotexist",
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/fake/sfhq/pt1",
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/fake/sfhq/pt2",
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/fake/sfhq/pt3",
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/fake/sfhq/pt4",
    "./real-vs-ai-generated-faces-dataset/data_source/data_source/fake/faceswap",
]
MAX_FACE_TRAIN_PER_CLASS = 20000
MAX_FACE_TEST_PER_CLASS = 3000

# ============================================================
# CONFIG — nature/scene dataset (capped moderately, not maxed — see v5 notes)
# ============================================================
NATURE_CSV = "./ai-vs-human-generated-dataset/train.csv"
NATURE_ROOT = "./ai-vs-human-generated-dataset"
NATURE_REAL_LABEL = 0   # per dataset documentation: 0 = authentic/human-captured
NATURE_FAKE_LABEL = 1   # per dataset documentation: 1 = AI-generated
MAX_NATURE_TRAIN_PER_CLASS = 20000   # reduced from v4's 36,000 (full pool) — diminishing returns
MAX_NATURE_TEST_PER_CLASS = 3000

MIX_REAL_DIRS = [
    "./ai-generated-images-vs-real-images/train/real",
    "./ai-generated-images-vs-real-images/test/real",
]
MIX_FAKE_DIRS = [
    "./ai-generated-images-vs-real-images/train/fake",
    "./ai-generated-images-vs-real-images/test/fake",
]
MAX_MIX_TRAIN_PER_CLASS = 15000   # out of 24,000 available in train/ alone (30,000 pooled)
MAX_MIX_TEST_PER_CLASS = 3000


# ============================================================
# Dataset merging helpers
# ============================================================

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
    evenly across the rest, instead of a big folder drowning out a small
    but relevant one under naive pooled-random sampling."""
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


def merge_faces_into_cifake(real_dirs, fake_dirs, max_train_per_class, max_test_per_class, seed=0, file_prefix="face"):
    rng = random.Random(seed)
    for label, dirs, cls_name in [("real", real_dirs, "REAL"), ("fake", fake_dirs, "FAKE")]:
        per_dir_files = {d: collect_images([d]) for d in dirs}
        dirs_with_files = {d: files for d, files in per_dir_files.items() if files}
        if not dirs_with_files:
            print(f"  [warning] no images found for {cls_name} — check the configured dirs above")
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
            shutil.copy(src, os.path.join(train_dst, f"{file_prefix}_{label}_{i}{os.path.splitext(src)[1]}"))
        for i, src in enumerate(test_files):
            shutil.copy(src, os.path.join(test_dst, f"{file_prefix}_{label}_{i}{os.path.splitext(src)[1]}"))

        print(f"{cls_name}: merged {len(train_files)} train + {len(test_files)} test images "
              f"(from {sum(dir_counts.values())} total available across {len(dirs_with_files)} sources)")


def merge_csv_labeled_into_cifake(csv_path, root_dir, fake_label, real_label,
                                   max_train_per_class, max_test_per_class, seed=1,
                                   file_prefix="nature"):
    df = pd.read_csv(csv_path)
    rng = random.Random(seed)
    for label_value, cls_name in [(real_label, "REAL"), (fake_label, "FAKE")]:
        subset = df[df["label"] == label_value]["file_name"].tolist()
        paths = [os.path.join(root_dir, f) for f in subset]
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            print(f"  [warning] no images found for {cls_name} in {csv_path} — check label values")
            continue
        rng.shuffle(paths)
        n_train = min(max_train_per_class, max(0, len(paths) - max_test_per_class))
        n_test = min(max_test_per_class, len(paths) - n_train)
        train_files = paths[:n_train]
        test_files = paths[n_train:n_train + n_test]

        train_dst = os.path.join(DATA_DIR, "train", cls_name)
        test_dst = os.path.join(DATA_DIR, "test", cls_name)
        os.makedirs(train_dst, exist_ok=True)
        os.makedirs(test_dst, exist_ok=True)

        for i, src in enumerate(train_files):
            shutil.copy(src, os.path.join(train_dst, f"{file_prefix}_{cls_name.lower()}_{i}{os.path.splitext(src)[1]}"))
        for i, src in enumerate(test_files):
            shutil.copy(src, os.path.join(test_dst, f"{file_prefix}_{cls_name.lower()}_{i}{os.path.splitext(src)[1]}"))

        print(f"{cls_name}: merged {len(train_files)} train + {len(test_files)} test nature/scene images "
              f"(from {len(paths)} total available)")


# ============================================================
# Custom augmentations
# ============================================================

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

# ============================================================
# Train/eval loops (AMP-aware)
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


def safe_image_loader(path):
    """Wraps the default image loader so a single corrupt/truncated file
    (unavoidable at this scale — 210,000 images pulled from 4 different
    web-scraped datasets) can't crash the entire training run. On failure,
    logs which file it was and substitutes a neutral gray placeholder
    instead — with only a handful of bad files out of 210k, this has no
    measurable effect on training. You can optionally go clean up the
    flagged files later for a fully pristine dataset, but it isn't
    necessary for training to succeed."""
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception as e:
        print(f"  [warning] unreadable image, using placeholder: {path} ({e})")
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type != "cuda":
        print("WARNING: CUDA not detected — check your PyTorch install (see SETUP step 2) "
              "and that `nvidia-smi` works in this terminal. Training on CPU would take days, not hours.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} | Mixed precision (AMP): {USE_AMP}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\nMerging face dataset into CIFAKE folder structure...")
    merge_faces_into_cifake(FACE_REAL_DIRS, FACE_FAKE_DIRS, MAX_FACE_TRAIN_PER_CLASS, MAX_FACE_TEST_PER_CLASS,
                             seed=0, file_prefix="face")

    print("\nMerging nature/scene dataset...")
    merge_csv_labeled_into_cifake(NATURE_CSV, NATURE_ROOT, NATURE_FAKE_LABEL, NATURE_REAL_LABEL,
                                   MAX_NATURE_TRAIN_PER_CLASS, MAX_NATURE_TEST_PER_CLASS)

    print("\nMerging DALL-E/MidJourney/Stable Diffusion mix dataset...")
    merge_faces_into_cifake(MIX_REAL_DIRS, MIX_FAKE_DIRS, MAX_MIX_TRAIN_PER_CLASS, MAX_MIX_TEST_PER_CLASS,
                             seed=2, file_prefix="mix")

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

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform, loader=safe_image_loader)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform, loader=safe_image_loader)

    print("\nClasses:", train_dataset.classes)
    print(f"Total train images: {len(train_dataset)} | Total test images: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if not any(layer in name for layer in ("layer3", "layer4", "fc")):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # ---- Resume support (local safety net against reboots/Ctrl+C) ----
    start_epoch = 0
    best_acc = 0.0
    if os.path.exists(LOCAL_RESUME_PATH):
        print(f"\nFound existing checkpoint at {LOCAL_RESUME_PATH} — resuming.")
        state = torch.load(LOCAL_RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        best_acc = state["best_acc"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming at epoch {start_epoch + 1}/{EPOCHS} (best_acc so far: {best_acc:.4f})")
        if start_epoch >= EPOCHS:
            print("All epochs already completed according to saved state — skipping to evaluation.")
    else:
        print("\nNo existing checkpoint found — starting fresh from epoch 1.")

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE, USE_AMP)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE, USE_AMP)
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
            }, BEST_MODEL_PATH)
            print(f"  -> New best model saved to {BEST_MODEL_PATH} (val_acc={val_acc:.4f})")

        # Save full resume state every epoch (not just on improvement), so
        # an interruption after a non-improving epoch still resumes right.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
        }, LOCAL_RESUME_PATH)

    print("\nAll requested epochs complete (or already were, per saved state).")
    _, _, preds, labels = evaluate(model, test_loader, criterion, DEVICE, USE_AMP)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=train_dataset.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    # REQUIRED on Windows: DataLoader with num_workers > 0 spawns child
    # processes that re-import this script. Without this guard, each child
    # would re-run the whole script from the top, causing a crash or an
    # infinite process-spawning loop. This is a Windows-specific quirk —
    # Colab (Linux) never needed it.
    main()