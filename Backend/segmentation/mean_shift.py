"""
segmentation/mean_shift.py
From-scratch Mean Shift segmentation on the LUV L-channel.
"""
import time
import numpy as np
from .utils import manual_bgr_to_luv_l, false_color

_MAX_SHIFT_ITERS = 50    # max iterations for each intensity bin's shift
_SHIFT_TOL       = 0.5   # stop shifting when movement < this value (intensity units)
_MERGE_RATIO     = 0.5   # two modes are merged if |m1 - m2| < bandwidth * ratio


def run_mean_shift(image_bgr: np.ndarray, bandwidth: int = 20) -> dict:
    """
    Mean Shift segmentation on the LUV L-channel (from scratch).

    Histogram trick
    ---------------
    Operating per-pixel on a full image would be very slow (O(N²) per point).
    Instead, we work on the 1-D intensity histogram (256 bins) and treat each
    non-empty bin as a data point weighted by its pixel count.

    Algorithm
    ---------
    1. Extract the L* lightness channel; build a weighted histogram (256 bins).
    2. Mode-finding loop — for each non-empty bin:
       a. Treat the bin's value as the current point.
       b. Collect all bins within a flat window of radius = bandwidth.
       c. Shift to the weighted mean of those bins (weighted by pixel count).
       d. Repeat until the shift is < _SHIFT_TOL or _MAX_SHIFT_ITERS reached.
       e. Record the converged value as this bin's "mode".
    3. Mode merging — assign bin labels:
       Bins whose modes are within bandwidth * _MERGE_RATIO of an existing
       cluster centre are merged into that cluster; otherwise a new cluster is
       created.
    4. Map every pixel's intensity value to its cluster label.
    5. Apply a false-colour palette for visualisation.
    """
    h, w = image_bgr.shape[:2]

    # Step 1 — L-channel + weighted histogram
    L = manual_bgr_to_luv_l(image_bgr)
    pixels_flat = L.flatten().astype(np.uint8)   # shape: (H*W,)
    bins = np.arange(256, dtype=np.float32)      # bin centres 0…255
    hist = np.bincount(pixels_flat, minlength=256).astype(np.float32)  # pixel count per bin

    t0 = time.perf_counter()

    # Step 2 — shift each non-empty bin to its local density mode
    modes = bins.copy()
    for i in range(256):
        if hist[i] == 0:
            continue   # skip empty bins
        val = float(bins[i])
        for _ in range(_MAX_SHIFT_ITERS):
            # Collect bins within the flat kernel window
            mask = np.abs(bins - val) <= bandwidth
            total_weight = float(np.sum(hist[mask]))
            if total_weight == 0:
                break
            # Weighted mean = new position for the current point
            new_val = float(np.dot(bins[mask], hist[mask])) / total_weight
            if abs(new_val - val) < _SHIFT_TOL:
                val = new_val
                break   # converged
            val = new_val
        modes[i] = val

    # Step 3 — merge nearby modes and assign cluster labels to each bin
    unique_modes: list[float] = []
    bin_labels = np.full(256, -1, dtype=np.int32)
    for i in range(256):
        if hist[i] == 0:
            continue
        m = modes[i]
        assigned = False
        for ci, cm in enumerate(unique_modes):
            if abs(m - cm) < bandwidth * _MERGE_RATIO:
                bin_labels[i] = ci   # merge into existing cluster
                assigned = True
                break
        if not assigned:
            bin_labels[i] = len(unique_modes)   # new cluster
            unique_modes.append(m)

    elapsed = (time.perf_counter() - t0) * 1000.0

    # Step 4 — map every pixel's intensity to its cluster label
    pixel_labels = bin_labels[pixels_flat]          # shape: (H*W,)
    pixel_labels = np.clip(pixel_labels, 0, None)   # treat any -1 as cluster 0

    # Step 5 — false-colour visualisation
    return {
        'result_image': false_color(pixel_labels, (h, w)),
        'elapsed_ms':   round(elapsed, 1),
        'bandwidth':    bandwidth,
    }
