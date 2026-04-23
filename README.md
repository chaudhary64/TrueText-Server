# TruePixel Node.js Inference Server

This server exposes HTTP endpoints to run predictions on an input image using the ONNX model in `models/fake_detector.onnx`.

## Prerequisites

- Node.js 18+ recommended

## Setup

```bash
npm install
```

## Run

```bash
npm start
```

Server starts at:

- `http://localhost:3000` (default)

You can override the port with:

```bash
PORT=5000 npm start
```

## Endpoints

### 1) Health Check

- **Method:** `GET`
- **URL:** `/health`
- **Purpose:** Verifies server and model loading status.

Example:

```bash
curl http://localhost:3000/health
```

---

### 2) Predict From Image File (multipart/form-data)

- **Method:** `POST`
- **URL:** `/predict`
- **Content-Type:** `multipart/form-data`
- **Field Name:** `image`

Example:

```bash
curl -X POST http://localhost:3000/predict \
  -F "image=@path/to/your/image.jpg"
```

PowerShell example:

```powershell
curl.exe -X POST http://localhost:3000/predict -F "image=@C:/path/to/your/image.jpg"
```

---

### 3) Predict From Base64 Image (JSON)

- **Method:** `POST`
- **URL:** `/predict`
- **Content-Type:** `application/json`
- **Body:**

```json
{
  "imageBase64": "<base64-encoded-image>"
}
```

You can also send a data URL format:

```json
{
  "imageBase64": "data:image/png;base64,iVBORw0KGgo..."
}
```

## Response Format

Example response:

```json
{
  "predictedClassId": "1",
  "predictedClass": "AI-Generated (Fake)",
  "confidence": 0.9821,
  "probabilities": [
    {
      "classId": "0",
      "label": "Real",
      "probability": 0.0179
    },
    {
      "classId": "1",
      "label": "AI-Generated (Fake)",
      "probability": 0.9821
    }
  ],
  "model": {
    "name": "google/siglip-base-patch16-512",
    "imageSize": 512
  }
}
```

## Notes

- Max upload size is 10 MB.
- Input image is resized to `512x512`, normalized using model metadata (`mean/std`), and converted to CHW float tensor.
- If no image is supplied, API returns `400`.
