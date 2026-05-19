# 🫁 CoughSense — AI-Powered Cough Sound Analysis

Early warning and respiratory risk screening using audio signal processing and machine learning.

---

## 🏗️ Project Structure

```
cough-analyzer/
├── backend/
│   ├── main.py          ← FastAPI server
│   ├── model.py         ← Feature extraction + MLP model
│   └── requirements.txt
├── frontend/
│   └── index.html       ← Single-file web UI
├── render.yaml          ← Render.com deployment config
├── start.sh             ← Local startup script
└── README.md
```

---

## ⚡ Local Setup (5 minutes)

### Step 1 — Prerequisites
- Python 3.9+
- pip

### Step 2 — Install & Run Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
> ⏳ On first run, the model trains on synthetic data (~30 seconds). After that it loads instantly.

### Step 3 — Open Frontend
Open `frontend/index.html` in your browser (double-click the file).  
The API URL defaults to `http://localhost:8000`.

### Step 4 — Test
1. Click **Start Recording** and cough (or upload a .wav file)
2. Click **Analyze Cough Sample**
3. View the risk assessment result

---

## 🚀 Deploy to Render.com (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: CoughSense AI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cough-analyzer.git
git push -u origin main
```

### Step 2 — Deploy Backend on Render
1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` — just click **Deploy**
4. Wait ~3 minutes for build. Your backend URL will be:  
   `https://cough-analyzer.onrender.com`

### Step 3 — Deploy Frontend on Render (Static Site)
1. Go to Render → **New** → **Static Site**
2. Connect same GitHub repo
3. Set **Publish directory** to `frontend`
4. Click **Deploy**

### Step 4 — Connect Them
In the frontend UI, update the **API** field to your Render backend URL.  
Or hardcode it in `index.html` (change the default value of `#api-url` input).

---

## 🤖 How the AI Works

| Step | What Happens |
|------|-------------|
| **Audio Input** | User uploads WAV/MP3 or records via microphone |
| **Feature Extraction** | Librosa extracts 40 MFCCs, chroma, spectral contrast, mel spectrogram, ZCR, RMS |
| **Normalization** | Features normalized with pre-fit mean/std scaler |
| **Classification** | 3-layer MLP classifies into: Healthy / Mild Risk / High Risk |
| **Output** | Prediction + confidence + probabilities + audio stats + recommendations |

### Features Used
- **MFCC (40 coefficients)** — captures vocal tract shape
- **Chroma** — pitch class energy
- **Spectral Contrast** — dynamic range across frequency bands  
- **Mel Spectrogram** — perceptual frequency representation
- **Zero Crossing Rate** — signal irregularity
- **RMS Energy** — amplitude envelope

---

## 📊 Model Details

- **Architecture:** 3-layer MLP (NumPy only, no TensorFlow dependency)
- **Training Data:** 800 synthetic audio samples per class
- **Classes:** Healthy | Mild Risk | High Risk
- **Training:** 300 epochs, SGD with backpropagation
- **Saved to:** `backend/cough_model.pkl`

> **Note:** The model is trained on synthetic data for demonstration. For real-world accuracy, replace with a labeled dataset like COUGHVID or Coswara.

---

## 🔬 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API + model status |
| POST | `/analyze` | Upload audio file, get prediction |
| GET | `/docs` | Swagger interactive API docs |

### Example Response
```json
{
  "prediction": "Mild Risk",
  "confidence": 72.4,
  "risk_level": "Mild Risk",
  "description": "Some irregular patterns detected...",
  "recommendations": ["Rest and stay hydrated", "Monitor temperature"],
  "probabilities": {
    "Healthy": 18.3,
    "Mild Risk": 72.4,
    "High Risk": 9.3
  },
  "audio_stats": {
    "duration_seconds": 2.5,
    "db_level": -18.4,
    "rms_energy": 0.0342,
    "zero_crossing_rate": 0.0812
  }
}
```

---

## ⚠️ Disclaimer
This tool is for **educational and screening purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
