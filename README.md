# AI-Generated Image Detector — Build Guide

Detects whether an image is a real photograph or AI-generated (diffusion/GAN),
using a self-trained CNN. No external API calls — the trained model does all
the work locally.

---

## Architecture

```
[User uploads image]
        |
        v
   React frontend  --(POST /api/predict, local network)-->  Flask backend
        |                                                          |
        |                                                          v
        |                                          Loads best_model.pt (your
        |                                          trained ResNet18) into
        |                                          PyTorch, runs inference
        |                                          + Grad-CAM locally
        |                                                          |
        <-----------------  JSON: prediction, confidence, heatmap  -
```

Training happens once (on Colab, needs internet). Inference happens every
time the app is used (fully offline, on your machine).

---

## Step 1 — Train the model (Google Colab)

1. Go to https://colab.research.google.com, new notebook.
2. Runtime > Change runtime type > **T4 GPU**.
3. Get a Kaggle API token: kaggle.com/settings > API > "Create New Token"
   (downloads `kaggle.json`).
4. In a Colab cell:
   ```python
   !pip install kaggle -q
   from google.colab import files
   files.upload()  # select kaggle.json
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
   !unzip -q cifake-real-and-ai-generated-synthetic-images.zip -d /content/cifake
   ```
5. Check the folder structure matches `/content/cifake/train/{REAL,FAKE}` and
   `/content/cifake/test/{REAL,FAKE}` — adjust `DATA_DIR` in the script if the
   unzip produces different folder names.
6. Copy the contents of `training/train_colab.py` into cells and run.
   ~8 epochs on a T4 GPU should take well under an hour.
7. Watch the printed validation accuracy per epoch — it should climb into the
   90%+ range on CIFAKE (it's a fairly separable dataset).
8. Download the result:
   ```python
   from google.colab import files
   files.download("/content/best_model.pt")
   ```

## Step 2 — Set up the backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir model
# move your downloaded best_model.pt into backend/model/
python app.py
```

Flask will start on `http://localhost:5000`. Test it's alive:
```bash
curl http://localhost:5000/api/health
```

## Step 3 — Set up the frontend

This is a single React component (`App.jsx`). Fastest path: create a Vite
React app and drop it in.

```bash
cd frontend
npm create vite@latest . -- --template react
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Replace the generated `src/App.jsx` with the provided `App.jsx`, add Tailwind
directives to `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Then:
```bash
npm run dev
```

Opens on `http://localhost:5173`. Upload an image, click Detect — it calls
your local Flask server, gets back prediction + confidence + Grad-CAM heatmap.

---

## What to say in your viva/demo

- "I fine-tuned a ResNet18, pretrained on ImageNet, on the CIFAKE dataset —
  60k real and 60k AI-generated images."
- "I froze the early layers and only fine-tuned layer4 and the final
  classifier — this avoids overfitting on a relatively small dataset and
  trains faster."
- "Inference is fully local — no API calls. The model weights are a file I
  own, loaded directly into PyTorch in my Flask backend."
- "The Grad-CAM heatmap shows which pixels most influenced the model's
  decision — this makes the model's reasoning interpretable, not a black
  box."
- Have your confusion matrix and classification report (printed at the end
  of training) ready to show — that's your accuracy/precision/recall story.

## Possible extensions if you have time left

- Add a second dataset (e.g. real vs Midjourney faces) to test generalization
  beyond CIFAKE's specific generator.
- Try EfficientNet-B0 instead of ResNet18 and compare accuracy.
- Add a "batch upload" mode — upload a folder, get a CSV of results.
