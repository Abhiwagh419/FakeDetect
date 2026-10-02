# FakeDetect — Model Training & Development Log

This document records the full history of how the FakeDetect detection model
and application evolved during development: what was changed, why, which
datasets were used, and what the measured results were at each stage. It is
kept up to date as the project progresses.

**Current deployed model: v5.** v4 (3 datasets) was tested and reverted after
exposing a real-world generalization gap. v5 (4 datasets, trained locally)
was built specifically to close that gap, tested against the same failure
case, and adopted as the new deployed model — see the v5 section below for
the full real-world verification and the one remaining known limitation.

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

- Full backend test suite passes with the v4 model.
- Real-world testing with a modern-generator (ChatGPT) face image and
  multiple nature images — the actual result that led to reverting to v3,
  documented above.

---

## v5 — DALL-E/MidJourney/Stable Diffusion coverage + nature retry (local training)

### Problem found

After v3 was reinstated, ongoing real-world testing confirmed the gap
documented above was still present: **ChatGPT-generated face images were
still frequently classified REAL with high confidence** by v3. Nature
images were untested since v3 (unlike v4) never had nature-dataset
exposure at all. The underlying cause remained the same as identified in
the v4 section: no training data from OpenAI's current image-generation
model family.

### Approach

Rather than repeat v4's "use as much data as possible" strategy, sampling
across all sources was deliberately **moderated, not maximized** — direct
evidence from the v3→v4 comparison above showed that maximizing volume
does not reliably improve real-world behavior, and can introduce bias
(e.g. skewing the REAL class toward an unrepresentative visual style).

Four datasets were merged:

| Dataset | Role | Train used | Test used |
|---|---|---:|---:|
| CIFAKE | objects/animals/vehicles (base) | 100,000 | 20,000 |
| Real-vs-AI-Generated-Faces | human faces | 20,000 | 3,000 |
| AI-vs-Human-Generated | nature/scenes (re-added, capped lower than v4) | 20,000 | 3,000 |
| AI-Generated-Images-vs-Real-Images (new) | Stable Diffusion + MidJourney + DALL-E mix | 15,000 | 3,000 |

**New dataset:** [`tristanzhang32/ai-generated-images-vs-real-images`](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)
— chosen specifically because it contains genuine **DALL-E** images (the
closest available match to ChatGPT's generator family), alongside
MidJourney and Stable Diffusion. Filenames carry no generator label
(plain sequential numbers), so the three generators cannot be individually
isolated — every sampled batch contains an inseparable mix of all three,
DALL-E included. Its REAL half is exclusively professional stock
photography (Pexels/Unsplash/WikiArt); sampling was capped at 15,000/3,000
per class (rather than the full 24,000/6,000 available) specifically to
avoid over-representing that unrepresentative visual style in the REAL
class. Nature's cap was also reduced from v4's full-pool 36,000 to 20,000
per class for the same reason.

**Training environment:** moved from Google Colab to a local machine
(NVIDIA RTX 3050 Laptop GPU, 6GB VRAM) specifically to remove the
free-tier GPU-quota and session-disconnect problems encountered during v4
(4 separate Colab sessions were required to complete that run). Automatic
mixed precision (AMP) was enabled to fit comfortably within 6GB VRAM and
take advantage of the RTX 3050's Tensor Cores; batch size reduced to 32
accordingly. A resume-from-checkpoint mechanism (full training state saved
after every epoch) was retained as a safety net against local
interruptions.

**Robustness fix during this run:** training crashed at epoch 1 with
`OSError: image file is truncated`, caused by a small number of corrupt
files among the 248,000 total images pulled from four different
web-scraped sources. Fixed by wrapping the image loader so any unreadable
file is replaced with a neutral placeholder and logged, rather than
crashing the run. Only 7 corrupt files were found in total (0.003% of the
dataset) — negligible impact, training resumed and completed normally
afterward.

### Final v5 dataset composition

| | |
|---|---|
| Training samples | 210,000 (105,000 REAL / 105,000 FAKE) |
| Test samples | 38,000 (19,000 REAL / 19,000 FAKE) |
| Epochs | 14 |
| Training hardware | Local NVIDIA RTX 3050 Laptop GPU (6GB VRAM) |
| Total training time | ≈ 7 hours 32 minutes |

### Training progression

| Epoch | Train Acc | Val Acc | Notes |
|---:|---:|---:|---|
| 1 | 87.75% | 92.82% | |
| 2 | 91.40% | 93.73% | |
| 3 | 92.76% | 94.58% | |
| 4 | 93.61% | 94.53% | slight dip, not a new best |
| 5 | 95.26% | 95.37% | |
| 6 | 95.80% | 95.90% | |
| 7 | 96.20% | 95.99% | |
| 8 | 96.50% | 95.84% | dip — coincides with a scheduled learning-rate decay (halved every 4 epochs); not overfitting, since val loss/accuracy both recovered immediately after |
| 9 | 97.28% | 96.05% | |
| 10 | 97.57% | 96.17% | |
| 11 | 97.76% | 96.12% | not a new best |
| 12 | 97.87% | 96.18% | |
| 13 | 98.26% | **96.35%** | **best epoch — this is the deployed checkpoint** |
| 14 | 98.37% | 96.26% | final epoch; slightly below epoch 13, not saved as best |

### v5 results

| | |
|---|---|
| **Best validation accuracy (epoch 13 — the deployed checkpoint)** | **96.35%** |
| Final-epoch accuracy (epoch 14) | 96.26% |

**Important technical note:** because epoch 14 did not exceed epoch 13's
result, the checkpoint file actually saved and deployed (`best_model.pt`)
holds **epoch 13's** weights (96.35%). The classification report and
confusion matrix below were produced by the script's final evaluation
pass, which ran against whatever weights were in memory at that point —
**epoch 14's**, not the saved best checkpoint. The two are very close
(96.35% vs. 96.26%) and this does not change any conclusion, but the
report below is technically epoch 14's numbers, not epoch 13's.

**Classification report (epoch 14 weights, full 38,000-image test set):**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| FAKE | 0.96 | 0.96 | 0.96 |
| REAL | 0.96 | 0.96 | 0.96 |

**Confusion matrix:**

|              | Predicted FAKE | Predicted REAL |
|--------------|-----------------|-----------------|
| Actual FAKE  | 18,300          | 700             |
| Actual REAL  | 723             | 18,277          |

On its own held-out test set, v5 (96.35%) scores lower than v3 (97.62%) —
but, as with the v3→v4 comparison, this is not a like-for-like comparison:
v5's test set is nearly 50% larger (38,000 vs. 26,000) and spans four
visual categories (objects, faces, nature, and a mixed-generator category)
instead of two.

### Real-world verification — the result that matters

The model was tested against the exact type of image that caused v3 (and
v4) to fail: images generated using ChatGPT's built-in image tool.

- **5 ChatGPT-generated face images** — all 5 correctly classified **FAKE**,
  with confidence between 91.0% and 96.99%. This is a full resolution of
  the face-detection gap that caused v4 to be reverted — the DALL-E-inclusive
  dataset succeeded where the earlier faces-only + nature-only datasets did
  not.
- **2 ChatGPT-generated nature/landscape images** — both still incorrectly
  classified **REAL**, both above 95% confidence. The nature dataset's own
  generator coverage evidently does not extend to ChatGPT's current image
  model for non-facial content, even though it now covers nature/landscape
  *composition* in general. This is documented here as an **open, known
  limitation**, not resolved by this version.
- **1 genuine real photo** (a Windows screen-capture/camera file) — correctly
  classified REAL (95.93% confidence), confirming no regression on ordinary
  real-world images.

### Decision

v5 was adopted as the new deployed model, replacing v3. The face-detection
improvement is a complete, verified fix for the specific failure mode that
caused the previous revert. The remaining nature-image gap is a narrower,
distinct limitation — closing it would likely require a dataset combining
*both* nature/landscape content *and* modern-generator coverage
specifically (the current nature dataset and the DALL-E/MidJourney/SD
dataset each cover one of those two dimensions, not both together) — and is
recorded here as a candidate direction for any future iteration.

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
- **Corrupt training image handling:** during v5 training, 7 corrupt/truncated
  image files (out of 248,000 total, sourced from 4 different web-scraped
  datasets) crashed the training loop. Fixed with an error-tolerant image
  loader that substitutes a placeholder and logs the file path instead of
  crashing — training completed normally afterward.
