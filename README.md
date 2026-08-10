
# FakeDetect

### AI Image Detection using Deep Learning

FakeDetect is a machine-learning-based web application that analyzes images and predicts whether they are **AI-generated (FAKE)** or **REAL**.

The project uses a **ResNet18 Convolutional Neural Network with Transfer Learning**, trained on the **CIFAKE dataset**.

---

## Current Features

- AI-generated image detection
- REAL / FAKE classification
- Prediction probability
- Image upload and preview
- Grad-CAM visualization
- Flask REST API
- React + Vite frontend
- PyTorch-based inference
- CPU inference support
- Model health-check API

---

## Machine Learning Model

**Architecture:** ResNet18  
**Learning Method:** Transfer Learning  
**Framework:** PyTorch  
**Dataset:** CIFAKE  
**Classes:**

```text
FAKE
REAL
````

The model uses an ImageNet-pretrained ResNet18 and is fine-tuned for binary classification of real and AI-generated images.

---

## Dataset

The model was trained using the CIFAKE dataset.

| Split    |   REAL |   FAKE |   Total |
| -------- | -----: | -----: | ------: |
| Training | 50,000 | 50,000 | 100,000 |
| Testing  | 10,000 | 10,000 |  20,000 |

Total images used:

**120,000**

---

## Model Performance

### Best Validation Accuracy

**97.77%**

### Test Accuracy

**97.63%**

The model was evaluated on 20,000 previously unseen test images.

### Confusion Matrix

                 Predicted
                 FAKE    REAL

Actual FAKE     9828     172
Actual REAL      302    9698


### Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| FAKE  |      0.97 |   0.98 |     0.98 |
| REAL  |      0.98 |   0.97 |     0.98 |

---

## Explainable AI

FakeDetect uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize image regions that contributed to the model's classification.

The system provides:


Original Image
       +
Grad-CAM Heatmap


Grad-CAM is used for model interpretability and should not be considered independent proof that an image is AI-generated.

---

## Technology Stack

### Machine Learning

* Python
* PyTorch
* TorchVision
* NumPy
* ResNet18
* CIFAKE Dataset

### Backend

* Python
* Flask
* Flask-CORS
* Pillow

### Frontend

* React
* Vite
* Tailwind CSS
* Lucide React

---

## Project Structure

```text
FakeDetect/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── model/
│       └── best_model.pt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
├── training/
│   └── train_colab.py
│
├── README.md
└── .gitignore
```
```
The exact structure may evolve during further development.

---

## Running the Backend

```powershell
cd backend
```

Create the virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify the trained model:

```powershell
Test-Path .\model\best_model.pt
```

Start Flask:

```powershell
python app.py
```

Backend:

```text
http://localhost:5000
```

Health endpoint:

```text
http://localhost:5000/api/health
```

---

## Running the Frontend

Open another terminal:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## API

### Health Check

```http
GET /api/health
```

Used to verify that the Flask backend and trained model are available.

### Image Prediction

```http
POST /api/predict
```

Accepts an image using:

```text
multipart/form-data
```

and returns the model's prediction and probability information.

---

## Current Status

### Completed

* [x] CIFAKE dataset preparation
* [x] ResNet18 training
* [x] Transfer learning
* [x] Model evaluation
* [x] Best model checkpoint
* [x] Flask backend
* [x] React frontend
* [x] Image upload
* [x] FAKE / REAL prediction
* [x] Prediction probabilities
* [x] Grad-CAM
* [x] Health-check API

### Planned / Under Development

* [ ] Advanced image analysis
* [ ] EXIF metadata extraction
* [ ] Image statistics
* [ ] Digital forensic indicators
* [ ] SHA-256 file analysis
* [ ] Professional PDF reports
* [ ] Analysis history
* [ ] Dashboard
* [ ] Model information page
* [ ] Downloadable analysis data
* [ ] Advanced UI/UX

---

## Limitations

The reported **97.63% accuracy** applies specifically to the CIFAKE test set.

Model probability for an individual image should not be interpreted as the model's overall accuracy.

The system may perform differently on images from distributions that differ significantly from the CIFAKE dataset.

Grad-CAM provides model interpretability but does not constitute definitive proof of image authenticity.

---

## Project

**FakeDetect — AI Image Detection & Digital Image Analysis**

Developed as an academic Computer Engineering project.

**Author:** Abhishek Wagh
**Government Polytechnic Mumbai**

---
