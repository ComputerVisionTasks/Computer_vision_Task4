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
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / 'Frontend'
MODEL_DIR = BASE_DIR / 'face_recognition_pca' / 'model'
DATA_DIR = BASE_DIR / 'face_recognition_pca' / 'data'

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

# Load test data for metrics
X_test = None
y_test = None
X_train = None
y_train = None

try:
    X_test = np.load(DATA_DIR / 'X_test.npy')
    y_test = np.load(DATA_DIR / 'y_test.npy')
    X_train = np.load(DATA_DIR / 'X_train.npy')
    y_train = np.load(DATA_DIR / 'y_train.npy')
except Exception as e:
    print(f"Warning: Could not load test data: {e}")


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


def _compute_model_metrics() -> dict:
    """Compute evaluation metrics using test data from PCA notebook"""
    if X_test is None or y_test is None or X_train is None or y_train is None:
        return {
            'success': False,
            'error': 'Test data not available',
            'available': False
        }
    
    try:
        # Data is already normalized and flattened from the PCA notebook
        # No need to divide by 255 or reshape
        X_test_data = X_test if len(X_test.shape) == 2 else X_test.reshape(X_test.shape[0], -1)
        X_train_data = X_train if len(X_train.shape) == 2 else X_train.reshape(X_train.shape[0], -1)
        
        # Transform with PCA
        X_test_pca = pca_model.transform(X_test_data)
        X_train_pca = pca_model.transform(X_train_data)
        
        # Get predictions
        y_train_pred = knn_model.predict(X_train_pca)
        y_test_pred = knn_model.predict(X_test_pca)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Per-class metrics
        class_metrics = {}
        unique_labels = np.unique(y_test)
        
        for label in sorted(unique_labels):
            mask = y_test == label
            if np.sum(mask) > 0:
                class_acc = accuracy_score(y_test[mask], y_test_pred[mask])
                class_metrics[str(int(label))] = {
                    'label': label_map.get(int(label), f'unknown_{label}'),
                    'samples': int(np.sum(mask)),
                    'accuracy': round(float(class_acc), 4)
                }
        
        # Model info
        model_info = {
            'n_components': int(pca_model.n_components_),
            'n_neighbors': int(knn_model.n_neighbors),
            'total_classes': len(label_map)
        }
        
        return {
            'success': True,
            'available': True,
            'train_accuracy': round(float(train_accuracy), 4),
            'test_accuracy': round(float(test_accuracy), 4),
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'class_metrics': class_metrics,
            'model_info': model_info
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'available': False
        }


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


@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics() -> object:
    """Get model evaluation metrics computed from PCA notebook test data"""
    metrics = _compute_model_metrics()
    return jsonify(metrics)


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
