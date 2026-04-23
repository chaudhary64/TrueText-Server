# TruePixel Node.js Inference Server

This server exposes HTTP endpoints to run predictions on input images using multiple ONNX models. It automatically discovers all model subdirectories and runs inference on all of them concurrently, returning results only when all models complete their predictions.

## Architecture

The server supports a multi-model architecture where:

- **Shared Configuration**: A single `model_meta.json` and ONNX model files at the root of the `models/` directory
- **Model Identifiers**: Each subdirectory (e.g., `siglip`, `someother_models`) represents a model identifier
- **Automatic Discovery**: The server automatically discovers and loads all model subdirectories on startup

### Directory Structure

```
models/
├── model_meta.json          # Shared metadata (image_size, classes, etc.)
├── fake_detector.onnx       # Shared ONNX model
├── fake_detector.onnx.data  # ONNX data file
├── siglip/                  # Model identifier 1
└── someother_models/        # Model identifier 2
```

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

Or with development mode (auto-reload on changes):

```bash
npm run dev
```

Server starts at:

- `http://localhost:3000` (default)

You can override the port with:

```bash
PORT=5000 npm start
```

## Endpoints

### 1) Health Check / Model Status

- **Method:** `GET`
- **URL:** `/`
- **Purpose:** Verifies server and all model loading status.

Example:

```bash
curl http://localhost:3000/
```

Response:

```json
{
  "status": "ok",
  "models": {
    "siglip": {
      "status": "ok",
      "model": {
        "name": "google/siglip-base-patch16-512",
        "path": "C:\\...\\models\\fake_detector.onnx",
        "inputNames": ["pixel_values"],
        "outputNames": ["logits"]
      }
    },
    "someother_models": {
      "status": "ok",
      "model": {
        "name": "model_name",
        "path": "C:\\...\\models\\fake_detector.onnx",
        "inputNames": ["pixel_values"],
        "outputNames": ["logits"]
      }
    }
  },
  "summary": {
    "totalModels": 2,
    "healthyModels": 2
  }
}
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

### Multi-Model Prediction Response

The server runs inference on **all discovered models** and returns predictions from each:

```json
{
  "status": "success",
  "predictions": [
    {
      "modelName": "siglip",
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
      "modelMetadata": {
        "name": "google/siglip-base-patch16-512",
        "imageSize": 512
      }
    },
    {
      "modelName": "someother_models",
      "predictedClassId": "0",
      "predictedClass": "Real",
      "confidence": 0.8954,
      "probabilities": [
        {
          "classId": "0",
          "label": "Real",
          "probability": 0.8954
        },
        {
          "classId": "1",
          "label": "AI-Generated (Fake)",
          "probability": 0.1046
        }
      ],
      "modelMetadata": {
        "name": "another_model_name",
        "imageSize": 512
      }
    }
  ],
  "summary": {
    "totalModels": 2,
    "timestamp": "2026-04-24T10:30:45.123Z"
  }
}
```

**Important:** The response is sent **only after all models complete their predictions**. If any model fails, the entire request fails with an error response.

## Adding New Models

To add a new model to the system:

1. **Create a subdirectory** in `models/`:

   ```
   mkdir models/your_new_model
   ```

2. **Ensure the shared files exist** at the root `models/` level:
   - `model_meta.json` (shared configuration)
   - `fake_detector.onnx` (ONNX model file)
   - `fake_detector.onnx.data` (ONNX data file, if needed)

3. **Restart the server** - it will automatically discover the new model directory:

   ```
   npm start
   ```

4. **Verify** by checking the `/` endpoint - your new model should appear in the response.

## Configuration

### model_meta.json

This shared configuration file defines:

```json
{
  "model_name": "google/siglip-base-patch16-512",
  "onnx_path": "fake_detector.onnx",
  "image_size": 512,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5],
  "classes": {
    "0": "Real",
    "1": "AI-Generated (Fake)"
  }
}
```

### Environment Variables

- `PORT` - Server port (default: 3000)
- `CORS_ORIGIN` - CORS origin header (default: "\*")

Example:

```bash
PORT=5000 CORS_ORIGIN=http://localhost:3000 npm start
```

## Notes

- Max upload size is 10 MB.
- Input images are resized to the configured `image_size` (default: 512x512).
- Images are normalized using model metadata (`image_mean` and `image_std`).
- All images are converted to CHW float tensors.
- If no image is supplied, API returns `400`.
- Server requires all models to complete predictions before responding - if one fails, the entire request fails.
- Models are loaded lazily on first use and cached for subsequent requests.
