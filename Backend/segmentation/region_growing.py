"""
segmentation/region_growing.py
From-scratch Region Growing segmentation.
Supports one or more seed points; each grows independently and results are combined.
"""
import time
import numpy as np
from .utils import manual_bgr_to_luv_l, manual_gray_to_bgr


def _grow_from_seed(L: np.ndarray, row: int, col: int,
                    threshold: int, visited: np.ndarray) -> np.ndarray:
    """
    BFS from a single seed (row, col) on the L-channel array.
    Accepts a neighbour if |neighbour_intensity - seed_intensity| <= threshold.
    Uses a shared `visited` mask so multiple seeds don't re-expand the same pixels.
    Returns a binary mask (0/255) of the grown region.
    """
    h, w = L.shape
    seed_val = int(L[row, col])  # reference intensity at the seed

    # Output mask for this seed's region
    region = np.zeros((h, w), dtype=np.uint8)
    visited[row, col] = True

    # 8-connected neighbour offsets (all directions around a pixel)
    OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
               ( 0, -1),          ( 0, 1),
               ( 1, -1), ( 1, 0), ( 1, 1)]

    # BFS queue — start at the seed pixel
    queue = [(row, col)]
    head = 0

    while head < len(queue):
        r, c = queue[head]
        head += 1
        region[r, c] = 255  # mark pixel as belonging to this region

        for dr, dc in OFFSETS:
            nr, nc = r + dr, c + dc
            # Bounds check + not-yet-visited check
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                visited[nr, nc] = True
                # Intensity similarity criterion
                if abs(int(L[nr, nc]) - seed_val) <= threshold:
                    queue.append((nr, nc))

    return region


def run_region_growing(image_bgr: np.ndarray,
                        seeds: list,
                        threshold: int = 25) -> dict:
    """
    Region Growing segmentation from one or more seed points (from scratch).

    Algorithm
    ---------
    1. Convert BGR to the LUV Lightness channel for intensity comparisons.
    2. For each seed point, run a BFS that expands to 8-connected neighbours
       whose intensity differs from the seed value by at most `threshold`.
    3. Combine the regions from all seeds with a logical OR (pixel union).
    4. Return a white-on-black binary result image.

    Parameters
    ----------
    seeds     : list of [seed_x, seed_y] pairs  (x = column, y = row)
    threshold : max intensity difference to include a neighbour [1-200]
    """
    h, w = image_bgr.shape[:2]
    L = manual_bgr_to_luv_l(image_bgr)  # work on lightness channel only

    # Shared visited mask — prevents double-expanding the same pixel
    visited  = np.zeros((h, w), dtype=bool)
    combined = np.zeros((h, w), dtype=np.uint8)

    seed_coords = []  # keep track of clamped coordinates for the response
    t0 = time.perf_counter()

    for seed_x, seed_y in seeds:
        # Clamp seed to image bounds
        col = max(0, min(int(seed_x), w - 1))
        row = max(0, min(int(seed_y), h - 1))
        seed_coords.append([col, row])

        # Grow from this seed and OR into the combined output
        region = _grow_from_seed(L, row, col, threshold, visited)
        combined = np.maximum(combined, region)

    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        'result_image': manual_gray_to_bgr(combined),   # grayscale → 3-channel BGR
        'elapsed_ms':   round(elapsed, 1),
        'seeds':        seed_coords,
        'threshold':    threshold,
    }
