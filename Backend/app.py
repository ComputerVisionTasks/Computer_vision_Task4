"""
app.py  –  EigenVision Suite Backend (Full)
Segmentation (K‑Means, Region Growing, Agglomerative, Mean Shift)
Thresholding (Optimal, Otsu, Spectral)
Face Detection (Haar Cascade) + Face Recognition (PCA / KNN)
"""

from base64 import b64decode, b64encode
import json
import os
from pathlib import Path
import pickle
import warnings
import time
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.metrics import accuracy_score, classification_report
from segmentation import (
    run_kmeans, run_region_growing, run_agglomerative, run_mean_shift,
    encode_jpeg_b64 as encode_bgr_to_b64_jpeg,
)
import sys

# Add Thresholding module to path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'Thresholding'))
from optimal_thresholding import OptimalThresholding
from otsu_thresholding import OtsuThresholding
from spectral_thresholding import SpectralThresholding

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / 'Frontend'
MODEL_DIR = BASE_DIR / 'face_recognition_pca' / 'model'
DATA_DIR = BASE_DIR / 'face_recognition_pca' / 'data'

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ── Flask app ──────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR / 'assets'),
    static_url_path='/assets',
)
CORS(app)

# ── Load PCA / KNN models ──────────────────────────────────────────────
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

# ── Load Haar Cascade ──────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar Cascade from: {HAAR_CASCADE_PATH}")

# ── Load test data for metrics ─────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════
#  Private helpers – image loading / detection / recognition
# ═══════════════════════════════════════════════════════════════════════

def _load_image_from_request() -> np.ndarray | None:
    """Read an image from multipart/form-data or base64 JSON payload."""
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


def _detect_faces(image: np.ndarray) -> tuple[list[dict], np.ndarray, str]:
    """
    Detect faces with Haar Cascade. Draws bounding rectangles only – no text.
    Returns:
        faces: list of {x, y, w, h}
        annotated: BGR image with rectangles
        mode: 'color' or 'grayscale'
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image.copy()
    mode = 'color' if is_color else 'grayscale'

    detected = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    annotated = image.copy()
    faces = []

    for (x, y, w, h) in detected:
        faces.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (65, 84, 241), 2)

    return faces, annotated, mode


def _preprocess_face_region(face_roi_gray: np.ndarray) -> np.ndarray:
    """
    Pad a possibly non-square face region to match the exact image_size while
    preserving aspect ratio – exactly as the training faces were prepared.
    Returns a normalized, flattened feature vector.
    """
    h, w = face_roi_gray.shape
    target_w, target_h = image_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(face_roi_gray, (new_w, new_h))

    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    canvas = canvas / 255.0
    return canvas.flatten().reshape(1, -1)


def _predict_person(image: np.ndarray) -> tuple[str, float, float]:
    """
    Project image onto PCA space and classify with KNN.
    Returns 'Unknown' when the nearest‑neighbour distance exceeds UNKNOWN_THRESHOLD.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    normed = resized.astype(np.float32) / 255.0
    vector = normed.flatten().reshape(1, -1)

    pca_feat = pca_model.transform(vector)
    pred_lbl = int(knn_model.predict(pca_feat)[0])

    distance = 0.0
    if hasattr(knn_model, 'kneighbors'):
        distances, _ = knn_model.kneighbors(pca_feat, n_neighbors=1)
        distance = float(distances[0][0])

    UNKNOWN_THRESHOLD = 1.5   # tunable; increase to be more permissive
    if distance > UNKNOWN_THRESHOLD:
        return 'Unknown', 0.0, distance

    name = label_map.get(pred_lbl, 'Unknown')
    # Confidence based on distance (0-100)
    confidence = max(0.0, min(100.0, 100.0 - distance * 50.0))
    return name, confidence, distance


def _recognize_face_region(face_roi_gray: np.ndarray) -> tuple[str, float, float, bool]:
    """
    Recognize a detected face ROI using PCA+KNN.
    Returns (person_name, confidence, distance, is_known).
    """
    feature_vector = _preprocess_face_region(face_roi_gray)
    pca_features = pca_model.transform(feature_vector)
    predicted_label = int(knn_model.predict(pca_features)[0])

    distances, _ = knn_model.kneighbors(pca_features, n_neighbors=1)
    distance = float(distances[0][0])

    UNKNOWN_THRESHOLD = 1.5
    if distance > UNKNOWN_THRESHOLD:
        return 'Unknown', 0.0, distance, False

    person_name = label_map.get(predicted_label, 'Unknown')
    confidence = max(0.0, min(100.0, 100.0 - distance * 50.0))
    return person_name, confidence, distance, True


def _compute_model_metrics() -> dict:
    """Compute evaluation metrics using test data from PCA notebook."""
    if X_test is None or y_test is None or X_train is None or y_train is None:
        return {'success': False, 'error': 'Test data not available', 'available': False}

    try:
        def _flat(X):
            return X if len(X.shape) == 2 else X.reshape(X.shape[0], -1)

        X_tr_pca = pca_model.transform(_flat(X_train))
        X_te_pca = pca_model.transform(_flat(X_test))
        y_tr_pred = knn_model.predict(X_tr_pca)
        y_te_pred = knn_model.predict(X_te_pca)

        class_metrics = {}
        for lbl in sorted(np.unique(y_test)):
            mask = y_test == lbl
            class_metrics[str(int(lbl))] = {
                'label': label_map.get(int(lbl), f'unknown_{lbl}'),
                'samples': int(mask.sum()),
                'accuracy': round(float(accuracy_score(y_test[mask], y_te_pred[mask])), 4),
            }

        return {
            'success': True,
            'available': True,
            'train_accuracy': round(float(accuracy_score(y_train, y_tr_pred)), 4),
            'test_accuracy': round(float(accuracy_score(y_test, y_te_pred)), 4),
            'train_samples': int(len(y_train)),
            'test_samples': int(len(y_test)),
            'class_metrics': class_metrics,
            'model_info': {
                'n_components': int(pca_model.n_components_),
                'n_neighbors': int(knn_model.n_neighbors),
                'total_classes': len(label_map),
            },
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'available': False}


# ═══════════════════════════════════════════════════════════════════════
#  Routes – static pages
# ═══════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/segmentation')
def segmentation_page():
    return send_from_directory(FRONTEND_DIR / 'pages', 'segmentation.html')

@app.route('/faceRecognition')
def face_recognition_page():
    return send_from_directory(FRONTEND_DIR / 'pages', 'faceRecognition.html')

@app.route('/thresholding')
def thresholding_page():
    return send_from_directory(FRONTEND_DIR / 'pages', 'Thresholding.html')


# ═══════════════════════════════════════════════════════════════════════
#  Routes – API (Face Detection & Recognition)
# ═══════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'success': True,
        'status': 'running',
        'models_loaded': True,
        'labels_count': len(label_map),
    })

@app.route('/api/labels', methods=['GET'])
def get_labels():
    ordered_labels = [label_map[i] for i in sorted(label_map)]
    return jsonify({'success': True, 'labels': ordered_labels, 'count': len(ordered_labels)})

@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    return jsonify(_compute_model_metrics())

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """
    POST /api/detect
    Returns bounding boxes only (no text labels) + annotated image as base64.
    """
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        faces, annotated, mode = _detect_faces(image)
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result_b64 = b64encode(buffer).decode('utf-8')
        result_src = f'data:image/jpeg;base64,{result_b64}'

        return jsonify({
            'success': True,
            'face_count': len(faces),
            'faces': faces,
            'result_image': result_src,
            'mode': mode,
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """
    POST /api/recognize
    Recognizes a single face (whole image is assumed to be a face).
    Returns 'Unknown' if confidence is low.
    """
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        started_at = time.perf_counter()
        person_name, confidence, distance = _predict_person(image)
        inference_time_ms = (time.perf_counter() - started_at) * 1000.0
        return jsonify({
            'success': True,
            'person': person_name,
            'confidence': round(confidence, 2),
            'distance': round(distance, 4),
            'inference_time_ms': round(inference_time_ms, 2),
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/detect-recognize', methods=['POST'])
def detect_and_recognize():
    """
    Detect all faces in the image, recognize each, draw bounding boxes + names,
    and return annotated image + per‑face results.
    """
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        result_img = image.copy()
        face_results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            person_name, confidence, distance, recognized = _recognize_face_region(face_roi)

            color = (65, 84, 241)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            label = f"{person_name} ({confidence:.0f}%)" if recognized else "Unknown"
            cv2.putText(result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            face_results.append({
                'id': len(face_results) + 1,
                'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                'person': person_name if recognized else 'Unknown',
                'confidence': round(confidence, 2),
                'distance': round(distance, 4),
                'recognized': recognized
            })

        _, buffer = cv2.imencode('.jpg', result_img)
        result_b64 = b64encode(buffer).decode('utf-8')
        result_src = f'data:image/jpeg;base64,{result_b64}'

        return jsonify({
            'success': True,
            'face_count': len(faces),
            'faces': face_results,
            'result_image': result_src,
            'mode': 'color' if len(image.shape) == 3 else 'grayscale'
        })

    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════
#  Routes – Segmentation (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════

def _load_seg_image() -> np.ndarray | None:
    """Load BGR image from multipart upload or base64 JSON/form payload."""
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
    buf = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

@app.route('/api/segment/kmeans', methods=['POST'])
def segment_kmeans():
    image = _load_seg_image()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        k = int(request.form.get('k') or (request.get_json(silent=True) or {}).get('k', 3))
        max_iter = int(request.form.get('max_iter') or (request.get_json(silent=True) or {}).get('max_iter', 20))
        k = max(2, min(k, 8))
        result = run_kmeans(image, k=k, max_iter=max_iter)
        return jsonify({
            'success': True,
            'result_image': encode_bgr_to_b64_jpeg(result['result_image']),
            'elapsed_ms': result['elapsed_ms'],
            'k': result['k'],
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/segment/region-growing', methods=['POST'])
def segment_region_growing():
    image = _load_seg_image()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        payload = request.get_json(silent=True) or {}
        seeds_raw = request.form.get('seeds') or payload.get('seeds')
        if seeds_raw:
            seeds = json.loads(seeds_raw) if isinstance(seeds_raw, str) else seeds_raw
        else:
            sx = int(request.form.get('seed_x') or payload.get('seed_x', image.shape[1] // 2))
            sy = int(request.form.get('seed_y') or payload.get('seed_y', image.shape[0] // 2))
            seeds = [[sx, sy]]
        if not seeds:
            return jsonify({'success': False, 'error': 'No seed points provided'}), 400
        threshold = int(request.form.get('threshold') or payload.get('threshold', 25))
        threshold = max(1, min(threshold, 200))
        result = run_region_growing(image, seeds=seeds, threshold=threshold)
        return jsonify({
            'success': True,
            'result_image': encode_bgr_to_b64_jpeg(result['result_image']),
            'elapsed_ms': result['elapsed_ms'],
            'seeds': result['seeds'],
            'threshold': result['threshold'],
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/segment/agglomerative', methods=['POST'])
def segment_agglomerative():
    image = _load_seg_image()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        payload = request.get_json(silent=True) or {}
        n_clusters = int(request.form.get('n_clusters') or payload.get('n_clusters', 4))
        n_clusters = max(2, min(n_clusters, 8))
        result = run_agglomerative(image, n_clusters=n_clusters)
        return jsonify({
            'success': True,
            'result_image': encode_bgr_to_b64_jpeg(result['result_image']),
            'elapsed_ms': result['elapsed_ms'],
            'n_clusters': result['n_clusters'],
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/segment/meanshift', methods=['POST'])
def segment_meanshift():
    image = _load_seg_image()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        payload = request.get_json(silent=True) or {}
        bandwidth = int(request.form.get('bandwidth') or payload.get('bandwidth', 20))
        bandwidth = max(5, min(bandwidth, 100))
        result = run_mean_shift(image, bandwidth=bandwidth)
        return jsonify({
            'success': True,
            'result_image': encode_bgr_to_b64_jpeg(result['result_image']),
            'elapsed_ms': result['elapsed_ms'],
            'bandwidth': result['bandwidth'],
        })
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════
#  Routes – Thresholding (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════

def _apply_thresholding(image_array, method, scope, **kwargs):
    """Apply thresholding and return (binary_image, metadata)."""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array

    metadata = {'scope': scope}

    if method == 'optimal':
        optimal = OptimalThresholding(scope)
        block_size = int(kwargs.get('window_size', 15)) if scope == 'local' else None
        result = optimal.apply_thresholding(gray, block_size=block_size)
        threshold = optimal.compute_optimal_threshold(gray)
        metadata['threshold'] = round(float(threshold), 2)

    elif method == 'otsu':
        otsu = OtsuThresholding(scope)
        block_size = int(kwargs.get('window_size', 15)) if scope == 'local' else None
        if scope == 'global':
            result = otsu.apply_global_threshold(gray)
            threshold = otsu.compute_best_threshold(gray)
            metadata['threshold'] = int(threshold)
        else:
            result = otsu.apply_local_threshold(gray, block_size=block_size)
            metadata['threshold'] = f"local({block_size}x{block_size})"

    elif method == 'spectral':
        spectral = SpectralThresholding()
        num_classes = int(kwargs.get('classes', 3))
        sigma = float(kwargs.get('sigma', 1.0))
        window_size = int(kwargs.get('window_size', 15)) if scope == 'local' else None

        if scope == 'global':
            result = spectral.global_otsu_multithreshold(gray, num_classes=num_classes, smoothing_sigma=sigma)
        else:
            result = spectral.local_otsu_multithreshold(gray, num_classes=num_classes,
                                                        window_size=window_size, smoothing_sigma=sigma)

        metadata['classes'] = num_classes
        metadata['sigma'] = sigma
        if scope == 'local':
            metadata['window_size'] = window_size

        # Normalize and convert to uint8 for proper encoding
        result = (result * 255 / np.max(result)).astype(np.uint8) if np.max(result) > 1 else result.astype(np.uint8)

    return result, metadata

@app.route('/api/threshold/optimal', methods=['POST'])
def threshold_optimal():
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        start_time = time.perf_counter()
        scope = request.form.get('scope', 'global')
        window_size = request.form.get('window_size', 15)
        result, metadata = _apply_thresholding(image, 'optimal', scope, window_size=window_size)
        _, buffer = cv2.imencode('.png', result)
        result_b64 = b64encode(buffer).decode('utf-8')
        metadata['result_image'] = f'data:image/png;base64,{result_b64}'
        metadata['elapsed_ms'] = round((time.perf_counter() - start_time) * 1000, 2)
        metadata['success'] = True
        return jsonify(metadata)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/threshold/otsu', methods=['POST'])
def threshold_otsu():
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        start_time = time.perf_counter()
        scope = request.form.get('scope', 'global')
        window_size = request.form.get('window_size', 15)
        result, metadata = _apply_thresholding(image, 'otsu', scope, window_size=window_size)
        _, buffer = cv2.imencode('.png', result)
        result_b64 = b64encode(buffer).decode('utf-8')
        metadata['result_image'] = f'data:image/png;base64,{result_b64}'
        metadata['elapsed_ms'] = round((time.perf_counter() - start_time) * 1000, 2)
        metadata['success'] = True
        return jsonify(metadata)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500

@app.route('/api/threshold/spectral', methods=['POST'])
def threshold_spectral():
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        start_time = time.perf_counter()
        scope = request.form.get('scope', 'global')
        classes = int(request.form.get('classes', 3))
        sigma = float(request.form.get('sigma', 1.0))
        window_size = int(request.form.get('window_size', 15))

        result, metadata = _apply_thresholding(
            image, 'spectral', scope,
            window_size=window_size, classes=classes, sigma=sigma
        )

        # Apply JET colormap for better visualisation
        result_8u = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        result_colored = cv2.applyColorMap(result_8u, cv2.COLORMAP_JET)

        _, buffer = cv2.imencode('.png', result_colored)
        result_b64 = b64encode(buffer).decode('utf-8')
        metadata['result_image'] = f'data:image/png;base64,{result_b64}'
        metadata['elapsed_ms'] = round((time.perf_counter() - start_time) * 1000, 2)
        metadata['success'] = True
        return jsonify(metadata)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, port=5000)