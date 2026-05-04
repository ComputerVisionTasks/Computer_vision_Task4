from base64 import b64decode
import json
import os
from pathlib import Path
import pickle
import warnings
import base64
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

    confidence = _confidence_from_features(pca_features, predicted_label, distance)
    return person_name, confidence, distance


def _confidence_from_features(pca_features: np.ndarray, predicted_label: int, distance: float = 0.0) -> float:
    """Return a stable confidence score for the predicted label."""
    if hasattr(knn_model, 'predict_proba'):
        probabilities = knn_model.predict_proba(pca_features)[0]
        classes = getattr(knn_model, 'classes_', [])
        if len(classes) > 0:
            class_index = int(np.where(classes == predicted_label)[0][0]) if predicted_label in classes else int(np.argmax(probabilities))
            return float(np.clip(probabilities[class_index] * 100.0, 0.0, 100.0))

    # Fallback for classifiers without probability estimates.
    return float(np.clip((1.0 / (1.0 + distance)) * 100.0, 0.0, 100.0))


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
def segmentation_page() -> object:
    return send_from_directory(FRONTEND_DIR / 'pages', 'segmentation.html')


@app.route('/faceRecognition')
def face_recognition_page() -> object:
    return send_from_directory(FRONTEND_DIR / 'pages', 'faceRecognition.html')


@app.route('/thresholding')
def thresholding_page() -> object:
    return send_from_directory(FRONTEND_DIR / 'pages', 'Thresholding.html')


@app.route('/api/recognize', methods=['POST'])
def recognize_face() -> object:
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        started_at = time.perf_counter()
        person_name, confidence, distance = _predict_person(image)
        inference_time_ms = (time.perf_counter() - started_at) * 1000.0
        return jsonify(
            {
                'success': True,
                'person': person_name,
                'confidence': round(confidence, 2),
                'distance': round(distance, 4),
                'inference_time_ms': round(inference_time_ms, 2),
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

@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """Detect faces in an uploaded image using Haar cascade."""
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Load the pre-trained Haar cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles on a copy of the original image
        result_img = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # Encode result image to base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        result_src = f'data:image/jpeg;base64,{result_b64}'

        # Prepare bounding box list
        face_list = []
        for i, (x, y, w, h) in enumerate(faces):
            face_list.append({
                'id': i + 1,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            })

        return jsonify({
            'success': True,
            'face_count': len(faces),
            'faces': face_list,
            'result_image': result_src,
            'mode': 'grayscale' if len(image.shape) == 2 else 'color'
        })

    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500



@app.route('/api/detect-recognize', methods=['POST'])
def detect_and_recognize():
    """Detect faces and recognize each one using PCA/KNN."""
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        result_img = image.copy()
        face_results = []

        for (x, y, w, h) in faces:
            # Extract face region from grayscale image
            face_roi = gray[y:y+h, x:x+w]
            person_name, confidence, distance, recognized = _recognize_face_region(face_roi)

            # Draw bounding box and label
            color = (255, 0, 0)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            label = f"{person_name} ({confidence:.0f}%)" if recognized else "Unknown"
            cv2.putText(result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

            face_results.append({
                'id': len(face_results) + 1,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'person': person_name if recognized else 'Unknown',
                'confidence': round(confidence, 2),
                'distance': round(distance, 4),
                'recognized': recognized
            })

        # Encode result image to base64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        result_src = f'data:image/jpeg;base64,{result_b64}'

        return jsonify({
            'success': True,
            'face_count': len(faces),
            'faces': face_results,
            'result_image': result_src,
            'mode': 'grayscale' if len(image.shape) == 2 else 'color'
        })

    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


def _preprocess_face_region(face_roi_gray: np.ndarray) -> np.ndarray:
    """
    Pad a possibly non-square face region to match the exact image_size while
    preserving the aspect ratio - just like the training faces were prepared.
    
    Returns a normalized, flattened feature vector.
    """
    h, w = face_roi_gray.shape
    target_w, target_h = image_size  # from config (e.g., 92, 112)

    # --- Scale the face to fit inside the target size, keeping aspect ratio ---
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(face_roi_gray, (new_w, new_h))

    # --- Create a black canvas of exactly target size, place face in the centre ---
    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Normalise to [0,1] and flatten
    canvas = canvas / 255.0
    return canvas.flatten().reshape(1, -1)


def _recognize_face_region(face_roi_gray: np.ndarray) -> tuple[str, float, float, bool]:
    """
    Recognize a detected face ROI using the PCA+KNN model.
    Returns (person_name, confidence, distance, is_known).
    """
    feature_vector = _preprocess_face_region(face_roi_gray)

    # PCA projection
    pca_features = pca_model.transform(feature_vector)

    # Predict label
    predicted_label = int(knn_model.predict(pca_features)[0])
    person_name = label_map.get(predicted_label, f'unknown_{predicted_label}')

    # Distance and confidence
    distances, _ = knn_model.kneighbors(pca_features, n_neighbors=1)
    distance = float(distances[0][0])
    confidence = _confidence_from_features(pca_features, predicted_label, distance)

    # Threshold – adjust this value based on your dataset
    THRESHOLD = 50.0  # lower if too many known faces become Unknown
    recognized = confidence > THRESHOLD

    return person_name, confidence, distance, recognized
# ------------------------------------------------------------------ #
#  Segmentation API helpers                                           #
# ------------------------------------------------------------------ #

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
    """K-Means segmentation on the L channel of LUV color space."""
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
    """Region Growing segmentation from one or more user-supplied seed points."""
    image = _load_seg_image()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    try:
        payload = request.get_json(silent=True) or {}
        # Accept seeds as a JSON array: [[x1,y1],[x2,y2],...]
        seeds_raw = request.form.get('seeds') or payload.get('seeds')
        if seeds_raw:
            seeds = json.loads(seeds_raw) if isinstance(seeds_raw, str) else seeds_raw
        else:
            # Fallback: single seed from legacy x/y fields
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
    """Agglomerative Clustering segmentation (operates on a 20×20 downscale internally)."""
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

# ──────────────────────────────────────────────────────────────
#  Thresholding APIs
# ──────────────────────────────────────────────────────────────

def _apply_thresholding(image_array, method, scope, **kwargs):
    """
    Apply thresholding to an image and return the result with metadata.
    
    Args:
        image_array: Input image as numpy array (color or grayscale)
        method: 'optimal', 'otsu', or 'spectral'
        scope: 'global' or 'local'
        **kwargs: Additional parameters (window_size, classes, sigma, etc.)
    
    Returns:
        Tuple of (binary_image, threshold_value_or_label, metadata_dict)
    """
    # Convert to grayscale if needed
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
            result = spectral.local_otsu_multithreshold(gray, num_classes=num_classes, window_size=window_size, smoothing_sigma=sigma)
        
        metadata['classes'] = num_classes
        metadata['sigma'] = sigma
        if scope == 'local':
            metadata['window_size'] = window_size
        
        # Convert result to uint8 for proper encoding
        result = (result * 255 / np.max(result)).astype(np.uint8) if np.max(result) > 1 else result.astype(np.uint8)
    
    return result, metadata


@app.route('/api/threshold/optimal', methods=['POST'])
def threshold_optimal():
    """Apply optimal thresholding to an uploaded image."""
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    try:
        start_time = time.perf_counter()
        
        scope = request.form.get('scope', 'global')
        window_size = request.form.get('window_size', 15)
        
        result, metadata = _apply_thresholding(image, 'optimal', scope, window_size=window_size)
        
        # Encode result image to base64
        _, buffer = cv2.imencode('.png', result)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metadata['result_image'] = f'data:image/png;base64,{result_b64}'
        metadata['elapsed_ms'] = round(elapsed_ms, 2)
        metadata['success'] = True
        
        return jsonify(metadata)
    
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/segment/meanshift', methods=['POST'])
def segment_meanshift():
    """Mean Shift segmentation on the L channel of LUV color space."""
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
@app.route('/api/threshold/otsu', methods=['POST'])
def threshold_otsu():
    """Apply Otsu thresholding to an uploaded image."""
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    try:
        start_time = time.perf_counter()
        
        scope = request.form.get('scope', 'global')
        window_size = request.form.get('window_size', 15)
        
        result, metadata = _apply_thresholding(image, 'otsu', scope, window_size=window_size)
        
        # Encode result image to base64
        _, buffer = cv2.imencode('.png', result)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metadata['result_image'] = f'data:image/png;base64,{result_b64}'
        metadata['elapsed_ms'] = round(elapsed_ms, 2)
        metadata['success'] = True
        
        return jsonify(metadata)
    
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/threshold/spectral', methods=['POST'])
def threshold_spectral():
    """Apply spectral thresholding to an uploaded image with colormap."""
    image = _load_image_from_request()
    if image is None:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    
    try:
        start_time = time.perf_counter()
        
        scope = request.form.get('scope', 'global')
        classes = int(request.form.get('classes', 3))
        sigma = request.form.get('sigma', 1.0)
        window_size = request.form.get('window_size', 15)
        
        result, metadata = _apply_thresholding(
            image, 'spectral', scope,
            window_size=window_size,
            classes=classes,
            sigma=sigma
        )
        
        result_f = result.astype(np.float32)

        # 2. السر هنا: بنقسم على عدد الكلاسيس الفعلي اللي السلايدر واقف عليه
        # ده بيضمن إن الـ Label رقم 1 دايماً هياخد لون أخضر، والـ Label 2 ياخد أصفر وهكذا
        denom = max(1, classes - 1)
        result_scaled = (result_f / denom) * 255

        # 3. اتأكد إن مفيش قيم بره الـ 0-255 وحولها لـ uint8
        result_scaled = np.clip(result_scaled, 0, 255).astype(np.uint8)

        # 4. التلوين دلوقتي هيكون مظبوط جداً
        result_colored = cv2.applyColorMap(result_scaled, cv2.COLORMAP_JET)
        result_scaled = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Encode colored result image to base64
        _, buffer = cv2.imencode('.png', result_colored)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metadata['result_image'] = f'data:image/png;base64,{result_b64}'
        metadata['elapsed_ms'] = round(elapsed_ms, 2)
        metadata['success'] = True
        
        return jsonify(metadata)
    
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)
