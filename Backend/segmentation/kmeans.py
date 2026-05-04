"""
segmentation/kmeans.py
From-scratch K-Means segmentation on the LUV L-channel.
"""
import time
import numpy as np
from .utils import manual_bgr_to_luv_l, false_color


def run_kmeans(image_bgr: np.ndarray,
               k: int = 3,
               max_iter: int = 20,
               tol: float = 0.5) -> dict:
    """
    K-Means clustering on the LUV L-channel (from scratch).

    Algorithm
    ---------
    1. Extract the L* lightness channel from LUV colour space.
    2. Flatten the 2-D channel to a 1-D pixel array for simpler indexing.
    3. Randomly pick k pixels as the initial centroids (fixed seed=42).
    4. Assignment step: each pixel is assigned to the centroid with the
       smallest absolute distance |pixel - centroid|.
    5. Update step: each centroid moves to the mean of its assigned pixels.
       If a cluster is empty (no pixels assigned), keep the old centroid.
    6. Repeat steps 4-5 until either max_iter is reached or the largest
       centroid movement across all clusters drops below `tol`.
    7. Map every pixel's cluster label to a vivid false-colour palette.
    """
    h, w = image_bgr.shape[:2]

    # Step 1 & 2 — extract L channel and flatten
    L = manual_bgr_to_luv_l(image_bgr).astype(np.float32)
    pixels = L.flatten()        # shape: (H*W,)
    N = pixels.shape[0]

    # Step 3 — random centroid initialisation
    rng = np.random.default_rng(seed=42)
    centroids = pixels[rng.choice(N, size=k, replace=False)].copy()

    labels = np.zeros(N, dtype=np.int32)
    t0 = time.perf_counter()

    for _ in range(max_iter):
        # Step 4 — compute |pixel - centroid| for every (pixel, centroid) pair
        # Result shape: (N, k); argmin along axis=1 gives each pixel's cluster
        dists = np.abs(pixels[:, None] - centroids[None, :])
        new_labels = np.argmin(dists, axis=1).astype(np.int32)

        # Step 5 — update each centroid to the mean of its members
        new_centroids = np.array([
            pixels[new_labels == c].mean() if np.any(new_labels == c) else centroids[c]
            for c in range(k)
        ], dtype=np.float32)

        # Step 6 — convergence check: stop if centroids barely moved
        max_shift = float(np.max(np.abs(new_centroids - centroids)))
        centroids = new_centroids
        labels    = new_labels
        if max_shift < tol:
            break

    elapsed = (time.perf_counter() - t0) * 1000.0

    # Step 7 — assign each pixel a colour based on its cluster label
    return {
        'result_image': false_color(labels, (h, w)),
        'elapsed_ms':   round(elapsed, 1),
        'k':            k,
    }
