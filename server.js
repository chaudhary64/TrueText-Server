const fs = require("fs");
const path = require("path");
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const sharp = require("sharp");
const ort = require("onnxruntime-node");

const app = express();
const allowedOrigin = process.env.CORS_ORIGIN || "*";

function log(level, message, details = {}) {
  const extras = Object.keys(details).length
    ? ` | ${JSON.stringify(details)}`
    : "";
  const text = `[${level.toUpperCase()}] ${message}${extras}`;

  if (level === "error") {
    console.error(text);
    return;
  }

  if (level === "warn") {
    console.warn(text);
    return;
  }

  console.log(text);
}

const corsOptions = {
  origin: allowedOrigin,
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
};

app.use(cors(corsOptions));
app.options("*", cors(corsOptions));

app.use((req, _res, next) => {
  log("info", "Request received", {
    method: req.method,
    path: req.originalUrl,
  });
  next();
});

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

app.use(express.json({ limit: "10mb" }));

app.use((error, _req, res, next) => {
  if (error instanceof SyntaxError && "body" in error) {
    log("warn", "Invalid JSON body received");
    return res.status(400).json({
      message: "Invalid JSON body.",
    });
  }
  return next(error);
});

const modelsDir = path.resolve(__dirname, "models");
const metaPath = path.join(modelsDir, "model_meta.json");

if (!fs.existsSync(metaPath)) {
  throw new Error(`Missing model metadata file at ${metaPath}`);
}

const meta = JSON.parse(fs.readFileSync(metaPath, "utf8"));
const imageSize = meta.image_size || 512;
const imageMean = Array.isArray(meta.image_mean)
  ? meta.image_mean
  : [0.5, 0.5, 0.5];
const imageStd = Array.isArray(meta.image_std)
  ? meta.image_std
  : [0.5, 0.5, 0.5];

const relativeOnnxPath = meta.onnx_path || "./fake_detector.onnx";
const modelPath = path.resolve(modelsDir, relativeOnnxPath);

if (!fs.existsSync(modelPath)) {
  throw new Error(`Missing ONNX model file at ${modelPath}`);
}

let sessionPromise;
function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(modelPath);
  }
  return sessionPromise;
}

function softmax(values) {
  const maxVal = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxVal));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function getImageBufferFromRequest(req) {
  if (req.file && req.file.buffer) {
    return req.file.buffer;
  }

  if (typeof req.body.imageBase64 === "string" && req.body.imageBase64.trim()) {
    const cleaned = req.body.imageBase64.replace(
      /^data:image\/[a-zA-Z0-9+.-]+;base64,/,
      "",
    );
    return Buffer.from(cleaned, "base64");
  }

  return null;
}

async function preprocessImage(buffer) {
  const { data, info } = await sharp(buffer)
    .rotate()
    .removeAlpha()
    .resize(imageSize, imageSize, { fit: "fill" })
    .raw()
    .toBuffer({ resolveWithObject: true });

  const channels = info.channels;
  if (channels < 3) {
    throw new Error(
      "Unsupported image format: expected at least 3 channels (RGB).",
    );
  }

  const tensorData = new Float32Array(1 * 3 * imageSize * imageSize);

  for (let y = 0; y < imageSize; y += 1) {
    for (let x = 0; x < imageSize; x += 1) {
      const pixelBase = (y * imageSize + x) * channels;
      const pixelIndex = y * imageSize + x;

      for (let channel = 0; channel < 3; channel += 1) {
        const normalized = data[pixelBase + channel] / 255;
        tensorData[channel * imageSize * imageSize + pixelIndex] =
          (normalized - imageMean[channel]) / imageStd[channel];
      }
    }
  }

  return tensorData;
}

app.get("/health", async (_req, res) => {
  try {
    const session = await getSession();
    log("info", "Health check passed");
    res.json({
      status: "ok",
      model: {
        name: meta.model_name,
        path: modelPath,
        inputNames: session.inputNames,
        outputNames: session.outputNames,
      },
    });
  } catch (error) {
    log("error", "Health check failed", { error: error.message });
    res.status(500).json({
      status: "error",
      message: error.message,
    });
  }
});

app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    const imageBuffer = getImageBufferFromRequest(req);

    if (!imageBuffer || imageBuffer.length === 0) {
      log("warn", "Prediction request missing image payload");
      return res.status(400).json({
        message:
          "No image found. Send multipart/form-data with field 'image' or JSON with 'imageBase64'.",
      });
    }

    const session = await getSession();
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const inputData = await preprocessImage(imageBuffer);
    const inputTensor = new ort.Tensor("float32", inputData, [
      1,
      3,
      imageSize,
      imageSize,
    ]);

    const result = await session.run({ [inputName]: inputTensor });
    const outputTensor = result[outputName] || result[Object.keys(result)[0]];

    if (!outputTensor || !outputTensor.data) {
      log("error", "Model returned empty output tensor");
      return res.status(500).json({
        message: "Model returned no output data.",
      });
    }

    const rawScores = Array.from(outputTensor.data);
    let probabilities;

    if (rawScores.length === 1) {
      const fakeProb = sigmoid(rawScores[0]);
      probabilities = [1 - fakeProb, fakeProb];
    } else {
      probabilities = softmax(rawScores);
    }

    const classIds = Object.keys(meta.classes || {}).sort(
      (a, b) => Number(a) - Number(b),
    );
    const labels = classIds.length
      ? classIds.map((id) => meta.classes[id])
      : probabilities.map((_value, index) => `Class ${index}`);

    const bestIndex = probabilities.reduce(
      (best, current, index, arr) => (current > arr[best] ? index : best),
      0,
    );

    const responseProbabilities = probabilities.map((probability, index) => ({
      classId: classIds[index] || String(index),
      label: labels[index] || `Class ${index}`,
      probability,
    }));

    log("info", "Prediction completed", {
      predictedClass: labels[bestIndex] || `Class ${bestIndex}`,
      confidence: Number(probabilities[bestIndex].toFixed(4)),
    });

    return res.json({
      predictedClassId: classIds[bestIndex] || String(bestIndex),
      predictedClass: labels[bestIndex] || `Class ${bestIndex}`,
      confidence: probabilities[bestIndex],
      probabilities: responseProbabilities,
      model: {
        name: meta.model_name,
        imageSize,
      },
    });
  } catch (error) {
    log("error", "Prediction failed", { error: error.message });
    return res.status(500).json({
      message: "Prediction failed.",
      error: error.message,
    });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  log("info", "Inference server started", {
    url: `http://localhost:${port}`,
  });
});
