# FakeDetect — Model Training & Development Log

This document records the full history of how the FakeDetect detection model
and application evolved during development: what was changed, why, which
datasets were used, and what the measured results were at each stage. It is
kept up to date as the project progresses.

**Current deployed model: v3.** v4 was fully trained and evaluated, and
measurably outperformed v3 on held-out test data — but real-world testing
against a modern generator not represented in any training data (see the v4
section below) showed it performing worse than v3 on the project's primary
use case (face images), so v3 was reinstated. Full reasoning is documented
in the v4 section.

---

## v1 — Original baseline

| | |
|---|---|
| Dataset | CIFAKE (Real and AI-Generated Synthetic Images) |
| Dataset composition | CIFAR-10 objects/animals/vehicles only (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). FAKE images generated with Stable Diffusion 1.4, downscaled to CIFAR-10's native 32×32px. |
| Training samples | 100,000 (50,000 REAL / 50,000 FAKE) |
| Test samples | 20,000 (10,000 REAL / 10,000 FAKE) |
| Epochs | 8 |
| Architecture | ResNet18 (ImageNet-pretrained, `layer4` + fc fine-tuned) |
| Best validation accuracy | 97.77% |
| Test accuracy | 97.63% |

### Problem discovered

Real-world high-resolution photos (e.g. phone camera photos) were being
classified as **FAKE with very high confidence** (99%+), regardless of
actually being real.

**Root cause:** CIFAKE's images are natively 32×32px. The training pipeline
(`transforms.Resize((224,224))`) simply upscaled these to 224px, meaning
*every single training image, in both classes,* was permanently blurry at
training resolution. The model never once saw a genuinely sharp image during
training, so it learned "sharp = FAKE" as a shortcut rather than any real
forensic signal — because sharp images simply never appeared in the training
distribution at all.

This was confirmed by directly visualizing what training images looked like
(32px stretched to 224px) versus what a real photo looks like at 224px — the
difference is stark and directly explains the misclassification.

An inference-time workaround (`normalize_domain` option — downsamples the
input to CIFAKE's native resolution before classifying) was added as an
interim mitigation, but this only patches the symptom rather than fixing the
underlying training gap.

---

## v3 — Resolution-shortcut fix + faces dataset merge

*(v2 was a code iteration that was folded into v3 before an actual training
run — no separate v2 checkpoint exists.)*

### Change 1: Training-time augmentation (fixes the v1 bug at the source)

Added to the training pipeline:
- **`RandomResolutionJitter`** — randomly downsamples each training image to
  somewhere between 24–200px and back, with 60% probability, so both REAL and
  FAKE classes are seen across the full sharpness spectrum. This removes
  sharpness as a usable shortcut instead of just hiding it at inference time.
- **`RandomJPEGCompression`** — re-encodes through JPEG at a random quality
  (35–95), since CIFAKE's clean PNG exports never taught the model to
  tolerate real-world compression artifacts.
- Mild Gaussian blur + color jitter to further widen what "normal" looks
  like for the REAL class.
- Unfroze `layer3` in addition to `layer4` + fc, since the added augmentation
  makes the task harder and benefits from slightly more adaptable capacity.

### Change 2: Added a real-vs-AI-generated **faces** dataset

**Problem:** CIFAKE is built entirely on CIFAR-10, which has **zero human
images** in either class. The model had literally never seen a person, so
behavior on portraits/selfies was undefined, not just inaccurate.

**Dataset added:** [`philosopher0808/real-vs-ai-generated-faces-dataset`](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset) (Kaggle)

| Source | Label | Images available | Images used |
|---|---|---:|---:|
| ffhq | REAL | 70,000 | 20,000 |
| stable_diffusion | FAKE | 1,031 | 1,031 (all of it) |
| thispersondoesnotexist | FAKE | 20,499 | 3,662 |
| sfhq/pt1 | FAKE | 5,000 | 3,662 |
| sfhq/pt2 | FAKE | 5,000 | 3,661 |
| sfhq/pt3 | FAKE | 5,000 | 3,662 |
| sfhq/pt4 | FAKE | 5,000 | 3,661 |
| faceswap | FAKE | 9,429 | 3,661 |

Sampling used a **fair "water-fill" allocation** per source directory rather
than pooling everything and sampling randomly — this was necessary because a
naive pooled-random sample would have diluted the small-but-relevant
`stable_diffusion` folder (1,031 images) down to a few hundred images against
much larger folders, when in fact every image from it should be included.

### Final v3 dataset composition

| | |
|---|---|
| Training samples | 140,000 (70,000 REAL / 70,000 FAKE) |
| Test samples | 26,000 (13,000 REAL / 13,000 FAKE) |
| Epochs | 12 |
| Training hardware | Google Colab, NVIDIA T4 GPU (~3 hours — CPU-bound on decoding/resizing larger face-dataset source images, longer than the original estimate) |

### v3 results

| | |
|---|---|
| Best validation accuracy | 97.62% |
| Test accuracy | 97.62% |

**Classification report:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| FAKE | 0.97 | 0.98 | 0.98 |
| REAL | 0.98 | 0.97 | 0.98 |

**Confusion matrix:**

|              | Predicted FAKE | Predicted REAL |
|--------------|-----------------|-----------------|
| Actual FAKE  | 12,711          | 289             |
| Actual REAL  | 330             | 12,670          |

*(Note: this accuracy is not directly comparable to v1's 97.63% — the v3 test
set is deliberately harder and more diverse (objects **and** faces, varied
sharpness, varied compression) rather than a narrow, uniformly-blurry one. A
similar or slightly lower number on a harder, more honest test set represents
a more capable model, not a regression.)*

### Verification performed

- Re-tested the exact real photo that originally exposed the v1 bug: now
  correctly classified REAL at 100% confidence, **without** needing the
  `normalize_domain` workaround at all.
- Tested 5 additional real personal photos (varied conditions: night
  portrait, heavy color grading, bokeh-heavy shot) — all correctly REAL at
  99%+ confidence.
- Full backend test suite (85 tests) passes.
- `backend/config.py` (`MODEL_INFO`) and `README.md` updated to reflect real
  v3 numbers, replacing the stale v1 numbers that were previously hardcoded.

---

## v4 — Nature/scenery coverage: attempted, tested, and reverted

### Problem found

Through direct testing, human images were being identified correctly and
consistently (95–99%+ confidence) by v3. However, **nature/landscape/scenery
images were still misclassified** — an expected gap, since neither CIFAKE
(CIFAR-10 objects only) nor the v3 faces dataset contain any nature/scenery
images. The model simply had no reference for that category, the same root
cause as the earlier faces gap.

### Dataset added

[`alessandrasala79/ai-vs-human-generated-dataset`](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)
("AI vs. Human-Generated Images" by Alessandra Sala) — covers portraits,
landscapes, objects, and general scenes, not just faces.

**Structure (confirmed by inspection):**
- `train.csv` — columns `Unnamed: 0`, `file_name`, `label`. `file_name` is a
  relative path like `train_data/xxxx.jpg`.
- `train_data/` — the actual images.
- `test_data_v2/` + `test.csv` — an **unlabeled** Kaggle-competition-style
  holdout set; not usable for training or evaluation, ignored.
- Label values (per the dataset's own documentation): **1 = AI-generated
  (FAKE)**, **0 = authentic/human-captured (REAL)**.
- Confirmed perfectly balanced: 39,975 images per class (79,950 total).

Since this dataset labels via CSV rather than folder-per-class (unlike the
faces dataset), a new `merge_csv_labeled_into_cifake()` function was written
and unit-tested against mock data (including alongside the existing
folder-based face merge, to confirm no filename collisions or
double-counting) before being used for real.

**Sampling decision:** no time constraint was specified for this run, so the
**entire available pool** was used — 36,000 train + 3,975 test per class
(100% of the dataset).

### v4 dataset composition (as trained)

| | |
|---|---|
| Training samples | 212,000 (106,000 REAL / 106,000 FAKE) |
| Test samples | 33,950 (16,975 REAL / 16,975 FAKE) |
| Epochs | 14 |
| Training hardware | Google Colab, free-tier GPU (T4) |

### v4 held-out test results

| | |
|---|---|
| Best / final validation accuracy | 98.25% (epoch 14) |
| Test accuracy | 98.25% |

**Classification report:**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| FAKE | 0.98 | 0.98 | 0.98 |
| REAL | 0.98 | 0.98 | 0.98 |

**Confusion matrix:**

|              | Predicted FAKE | Predicted REAL |
|--------------|-----------------|-----------------|
| Actual FAKE  | 16,685          | 290             |
| Actual REAL  | 304             | 16,671          |

On its own held-out test set (drawn from the same three training
distributions), v4 measurably outperformed v3 — 98.25% vs. 97.62% — on a
harder, more diverse mix of objects, faces, and nature/scenes.

### Real-world testing revealed a generalization gap, and the decision to revert

Despite the strong held-out metrics, real-world testing with images
generated by a modern proprietary tool (OpenAI's current ChatGPT image
generator) exposed a clear weakness: **v4 classified both AI-generated
face images and AI-generated nature images from this tool as REAL, with
97%+ confidence, every time.** v3 was then tested against the same
ChatGPT-generated face image and correctly classified it as FAKE with high
confidence.

**Root cause:** none of the three training datasets (CIFAKE/Stable
Diffusion 1.4, the faces dataset/StyleGAN+SFHQ+faceswap+a small Stable
Diffusion subset, or the v4 nature dataset) contain any images from
OpenAI's current generator. Every generator leaves a distinct statistical
fingerprint; a detector trained on one family does not reliably transfer
to an unseen one, especially a newer, more photorealistic generator
specifically engineered to avoid the artifacts older models had. This is a
well-documented, unsolved problem in AI-image-forensics research generally
— it is not unique to this project.

**A plausible explanation for why v3 happened to get this specific case right:**
prior to the v3 fix, the model had a bias toward flagging any sharp/clean
image as FAKE (the original resolution-shortcut bug — see the v1 section
above). v3 still carries a lighter version of this tendency for
image types outside its face-specific training, which may have coincided
with correctly flagging the sharp, clean ChatGPT face image — a favorable
coincidence rather than a learned, generalizable signal. v4 does not show
this tendency, and consequently has no fallback bias for content it
doesn't recognize.

**Decision:** given this project's priority is reliable face-image
detection, and given the time cost of further training (four separate
Colab sessions, ~9+ hours total, to reach v4), **v3 was reinstated as the
final deployed model.** v4's code, dataset integration work, and full
results remain documented here as a record of the experiment and the
reasoning behind reverting it — an evidence-based decision, not an
abandoned attempt.

### A candidate path if this is revisited later

A search for datasets containing OpenAI-generator images specifically
found [`tristanzhang32/ai-generated-images-vs-real-images`](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
on Kaggle (60,000 images: 10,000 each of Stable Diffusion, MidJourney, and
DALL-E, plus 30,000 real images from Pexels/Unsplash/WikiArt) — the DALL-E
subset is the closest available match to the generator family behind
ChatGPT's image tool, though not a guaranteed exact match given how often
these proprietary models are updated. Not pursued further due to time
already invested.

### Training process notes (for reference)

This run required **4 separate Colab sessions across roughly 9+ hours
total**, due to repeated free-tier session disconnects (session length
limits, not GPU quota exhaustion this time). A resume-from-checkpoint
mechanism (saving full training state — model, optimizer, scheduler,
epoch number — to Google Drive after every epoch) meant every single
disconnect was fully recoverable: each new session automatically detected
the saved state and continued from the exact next epoch, with zero
progress lost after the very first incident (which happened before this
mechanism existed, costing one 9-epoch/4.5-hour run).

An earlier version of the auto-checkpoint-download feature caused a
20+ minute stuck download, traced to Colab's `files.download()` being
unreliable on larger files — the full resume state (with optimizer state
included) measured 128.8MB, vs. 44.8MB for weights-only. Fixed by only
auto-downloading the smaller weights-only file, and only on epochs that
produced a new best result.

### Verification performed

- Full backend test suite passes with the new model.
- *(To be completed: real-world testing specifically on nature/landscape
  photos — the category this version targets — plus re-verification that
  objects and faces still work correctly.)*

---

## Frontend development history

1. **Initial redesign** — dark cyberpunk theme → light, professional theme
   (clean cards, soft shadows, restrained color palette).
2. **Second redesign** — light theme → "sleek SaaS dashboard" style
   (Stripe/Linear-inspired): distinctive electric-indigo accent color,
   sliding active-nav pill indicator, animated count-up KPI numbers,
   staggered card entrance animations, hover-lift micro-interactions, and a
   signature "lock-on" animation where the upload viewfinder's corner
   brackets snap taut and flash green/red the instant a verdict is returned.
3. `normalizeDomain` (the inference-time resolution workaround) was
   defaulted to ON in the UI prior to the v3 training fix; can now safely be
   left off by default since the root cause is fixed in training, though the
   toggle remains available.

## Other fixes made during development

- **`.gitignore` fix:** `backend/data/artifacts/` (auto-saved copies of every
  uploaded/analyzed image, including personal photos) was not excluded and
  had already been pushed to a public GitHub repository. Corrected the
  ignore rule (initial fix used a bare `data/artifacts/` pattern which only
  matches at the repo root; corrected to `backend/data/artifacts/` since the
  folder is nested). Git history was not rewritten — only future tracking
  was stopped, by project owner's choice.
- **Git repository recovery:** local repo hit `fatal: bad object HEAD`
  (corrupted `.git` metadata); recovered by re-cloning from the GitHub
  remote, which was unaffected.
- **Missing dependency fix:** `reportlab` was listed in `requirements.txt`
  but not actually installed in the local virtual environment; resolved with
  a direct `pip install`.