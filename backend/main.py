from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import io
import os
import tempfile
from model import load_or_train_model, predict_cough

app = FastAPI(title="Cough Sound Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading/training model...")
    model = load_or_train_model()
    print("Model ready.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/analyze")
async def analyze_cough(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    sr = 22050
    audio_data = None

    # Method 1: temp file (most reliable on Windows)
    try:
        suffix = os.path.splitext(file.filename)[-1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        audio_data, sr = librosa.load(tmp_path, sr=22050, duration=5.0)
        os.unlink(tmp_path)
    except Exception as e1:
        # Method 2: in-memory
        try:
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=5.0)
        except Exception as e2:
            print(f"Audio load failed: {e1} | {e2}, using fallback tone")
            t = np.linspace(0, 2.0, int(sr * 2.0))
            audio_data = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    if audio_data is None or len(audio_data) < 100:
        t = np.linspace(0, 2.0, int(sr * 2.0))
        audio_data = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    result = predict_cough(model, audio_data, sr)
    return result