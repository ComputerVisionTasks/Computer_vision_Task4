from base64 import b64decode
import json
import os
from pathlib import Path
import pickle
import warnings

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / 'Frontend'
MODEL_DIR = BASE_DIR / 'face_recognition_pca' / 'model'

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR / 'assets'),
    static_url_path='/assets',
)
CORS(app)

with open(MODEL_DIR / 'pca_model.pkl', 'rb') as model_file:
    pca_model = pickle.load(model_file)

with open(MODEL_DIR / 'knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open(MODEL_DIR / 'image_size.json', 'r', encoding='utf-8') as config_file:
    image_size_config = json.load(config_file)
    image_size = tuple(image_size_config['image_size'])

with open(MODEL_DIR / 'label_map.json', 'r', encoding='utf-8') as labels_file:
    raw_label_map = json.load(labels_file)
    label_map = {int(key): value for key, value in raw_label_map.items()}


def _load_image_from_request() -> np.ndarray | None:
    if 'image' in request.files:
        file_bytes = np.frombuffer(request.files['image'].read(), np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_data = None
    if request.form.get('imageData'):
        image_data = request.form.get('imageData')
    elif request.is_json:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get('imageData')

    if not image_data:
        return None

    if image_data.startswith('data:image') and ',' in image_data:
        image_data = image_data.split(',', 1)[1]

    file_bytes = b64decode(image_data)
    file_buffer = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(file_buffer, cv2.IMREAD_COLOR)


def _predict_person(image: np.ndarray) -> tuple[str, float, float]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, image_size)
    normalized_image = resized_image.astype(np.float32) / 255.0
    feature_vector = normalized_image.flatten().reshape(1, -1)

    pca_features = pca_model.transform(feature_vector)
    predicted_label = int(knn_model.predict(pca_features)[0])
    person_name = label_map.get(predicted_label, f'unknown_{predicted_label}')

    distance = 0.0
    if hasattr(knn_model, 'kneighbors'):
        distances, _ = knn_model.kneighbors(pca_features, n_neighbors=1)
        distance = float(distances[0][0])

    confidence = max(0.0, min(100.0, 100.0 - distance * 50.0))
    return person_name, confidence, distance


@app.route('/')
def home() -> object:
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/segmentation')
def segmentation_placeholder() -> object:
    return send_from_directory(FRONTEND_DIR / 'pages', 'segmentation.html')


@app.route('/api/recognize', methods=['POST'])
def recognize_face() -> object:
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        person_name, confidence, distance = _predict_person(image)
        return jsonify(
            {
                'success': True,
                'person': person_name,
                'confidence': round(confidence, 2),
                'distance': round(distance, 4),
            }
        )
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/labels', methods=['GET'])
def get_labels() -> object:
    ordered_labels = [label_map[index] for index in sorted(label_map)]
    return jsonify({'success': True, 'labels': ordered_labels, 'count': len(ordered_labels)})


@app.route('/api/health', methods=['GET'])
def health() -> object:
    return jsonify(
        {
            'success': True,
            'status': 'running',
            'models_loaded': True,
            'labels_count': len(label_map),
        }
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
