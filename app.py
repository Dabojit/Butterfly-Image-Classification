

import base64
import csv
import io
import json
import os
from typing import List

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = [
    os.path.join(BASE_DIR, "best_butterfly_model.keras"),
    os.path.join(BASE_DIR, "butterfly_model.h5"),
]
CLASSES_FILE = os.path.join(BASE_DIR, "class_names.json")
TRAIN_CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "Training_set.csv"),
    os.path.join(BASE_DIR, "butterfly_data", "Training_set.csv"),
]

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model = None
classes = []
model_error = None


def load_classes() -> List[str]:
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    for path in TRAIN_CSV_CANDIDATES:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            labels = sorted({row["label"] for row in rows if row.get("label")})
            return labels

    return [
        "ADONIS",
        "MONARCH",
        "PAINTED LADY",
        "PEACOCK",
    ]


def load_model():
    global model, classes, model_error
    classes = load_classes()
    model_error = None

    for model_path in MODEL_FILES:
        if not os.path.exists(model_path):
            continue
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Model Loaded: {model_path}")
            return True
        except Exception as exc:
            model_error = str(exc)
            print(f"⚠️ Could not load {model_path}: {exc}")

    model = None
    print("❌ Model not found")
    return False


def prepare_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def build_prediction_payload(pred_probs: np.ndarray, top_k: int = 5) -> dict:
    top_k = max(1, min(int(top_k), len(classes)))
    top_indices = np.argsort(pred_probs)[::-1][:top_k]
    top_conf = pred_probs[top_indices]

    top_predictions = []
    for rank, (idx, conf) in enumerate(zip(top_indices, top_conf), start=1):
        top_predictions.append({
            "rank": rank,
            "class": classes[idx] if idx < len(classes) else f"class_{idx}",
            "percentage": round(float(conf * 100), 2),
        })

    best_idx = int(top_indices[0])
    confidence = float(top_conf[0])
    return {
        "success": True,
        "predicted_class": classes[best_idx] if best_idx < len(classes) else f"class_{best_idx}",
        "confidence_percent": round(confidence * 100, 2),
        "predictions": top_predictions,
    }


load_model()


@app.route("/")
def home():
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "num_classes": len(classes),
        "model_error": model_error,
    })


@app.route("/info")
def info():
    return jsonify({
        "model_loaded": model is not None,
        "num_classes": len(classes),
        "classes": classes[:10] if classes else [],
        "model_error": model_error,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    file_storage = request.files.get("file") or request.files.get("image")
    if file_storage is None:
        return jsonify({"success": False, "error": "No image received"}), 400

    try:
        image = Image.open(file_storage)
        image_array = prepare_image(image)
        probs = model.predict(image_array, verbose=0)[0]
        return jsonify(build_prediction_payload(probs, request.args.get("top_k", 5)))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    try:
        data = request.get_json(silent=True) or {}
        image_b64 = data.get("image")
        if not image_b64:
            return jsonify({"success": False, "error": "No image data received"}), 400

        image_bytes = base64.b64decode(image_b64.split(",")[-1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = prepare_image(image)
        probs = model.predict(image_array, verbose=0)[0]
        return jsonify(build_prediction_payload(probs, data.get("top_k", 5)))
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "No images received"}), 400

    results = []
    for file_storage in files:
        try:
            image = Image.open(file_storage)
            image_array = prepare_image(image)
            probs = model.predict(image_array, verbose=0)[0]
            payload = build_prediction_payload(probs, request.args.get("top_k", 5))
            result_item = {
                "filename": file_storage.filename,
                "predicted_class": payload["predicted_class"],
                "confidence_percent": payload["confidence_percent"],
                "predictions": payload["predictions"],
            }
            results.append(result_item)
        except Exception as exc:
            results.append({
                "filename": file_storage.filename,
                "error": str(exc),
            })

    return jsonify({
        "success": True,
        "processed": len(results),
        "results": results,
    })


@app.route("/reload", methods=["POST"])
def reload_model():
    loaded = load_model()
    return jsonify({
        "success": True,
        "model_loaded": loaded,
        "num_classes": len(classes),
        "model_error": model_error,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)