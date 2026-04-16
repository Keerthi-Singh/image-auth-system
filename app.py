from fastapi import FastAPI, UploadFile, File
import shutil
import os

from utils import blur_score, is_blurry
from model import get_embedding, classify_image

app = FastAPI(title="Image Authentication & Quality Analyzer")

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Blur detection
        blur = blur_score(file_path)
        blurry = is_blurry(blur)

        # Feature extraction
        embedding = get_embedding(file_path)

        # Classification
        prediction = classify_image(file_path)

        return {
            "filename": file.filename,
            "blur_score": float(blur),
            "is_blurry": bool(blurry),
            "embedding_length": len(embedding),
            "predicted_class_id": int(prediction)
        }

    except Exception as e:
        return {"error": str(e)}
