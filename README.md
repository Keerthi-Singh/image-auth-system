# 🧠 Smart Image Authentication & Quality Analyzer

A computer vision system that analyzes image quality and extracts unique feature embeddings (digital fingerprints).

## 🚀 Features
- Blur detection using Laplacian variance
- Feature extraction using ResNet (PyTorch)
- Image classification (pretrained CNN)
- REST API using FastAPI

## 📸 Use Case
- Image quality filtering
- Visual fingerprinting
- Preprocessing for authentication systems

## 🛠 Tech Stack
- Python
- OpenCV
- PyTorch
- FastAPI

## ⚡ Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## 🔗 API Endpoint

### POST `/analyze/`

Upload an image to analyze and get:
- **blur_score**: Laplacian variance score (higher = clearer)
- **is_blurry**: Boolean indicating if image is blurry
- **embedding_length**: Length of feature vector
- **predicted_class_id**: Image classification prediction

**Example Request:**
```bash
curl -X POST "http://localhost:8000/analyze/" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

**Example Response:**
```json
{
  "filename": "image.jpg",
  "blur_score": 245.67,
  "is_blurry": false,
  "embedding_length": 512,
  "predicted_class_id": 285
}
```

## 📌 Future Improvements
- Image similarity comparison (cosine similarity)
- Custom-trained authentication model
- Edge device optimization
- Database integration for fingerprint storage
- Web UI dashboard

## 📄 License
MIT