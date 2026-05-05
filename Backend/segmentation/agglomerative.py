import numpy as np
import cv2
import time
 
 
# ──────────────────────────────────────────────────────────────
#  Label → deterministic RGB colour
# ──────────────────────────────────────────────────────────────
_PALETTE = [
    (220,  20,  60), (30, 144, 255), ( 50, 205,  50), (255, 165,   0),
    (138,  43, 226), (  0, 206, 209), (255,  20, 147), (255, 215,   0),
    (  0, 128,   0), (255,  69,   0), (100, 149, 237), (144, 238, 144),
    (255, 182, 193), (173, 255,  47), (135, 206, 235), (210, 105,  30),
]
 
def label_color(label: int) -> np.ndarray:
    """Return an RGB uint8 array [R, G, B] for the given cluster label."""
    return np.array(_PALETTE[label % len(_PALETTE)], dtype=np.uint8)
 
 
# ──────────────────────────────────────────────────────────────
#  Core algorithm
# ──────────────────────────────────────────────────────────────
def agglomerative_segment(img: np.ndarray, num_clusters: int = 4) -> dict:
    """
    Agglomerative (bottom-up) image segmentation.
 
    Parameters
    ----------
    img          : H×W×3 or H×W uint8 numpy array (RGB or grayscale).
    num_clusters : Target number of segments (clamped to [2, 16]).
 
    Returns
    -------
    dict with keys:
        'output'       – H×W×3 uint8 segmented image (numpy array, RGB).
        'num_clusters' – Final number of clusters.
        'iterations'   – Number of merge steps performed.
 
    Algorithm
    ---------
    1. Divide image into square blocks → one initial cluster each.
    2. Compute mean colour per block.
    3. Repeatedly merge the two closest clusters (squared Euclidean
       colour distance, average linkage via weighted mean update).
    4. Stop when num_clusters remain.
    5. Map every pixel to its nearest final cluster colour (avoids
       blocky output).
    """
 
    # ── 0. Validate / normalise input ────────────────────────
    if img.ndim == 2:                           # grayscale → fake 1-ch
        img = img[:, :, np.newaxis]
    rows, cols, ch = img.shape
    img_f = img.astype(np.float64)             # working copy in float
 
    num_clusters = int(np.clip(num_clusters, 2, 16))
 
    # ── 1. Adaptive block size (keep total blocks ≤ 500) ─────
    # Limit number of initial blocks to avoid O(N³) slowdown.
    bsize = 50
    while ((rows // bsize) * (cols // bsize) > 500) and (bsize < 64):
        bsize *= 2
 
    brows = (rows + bsize - 1) // bsize
    bcols = (cols + bsize - 1) // bsize
    nb    = brows * bcols                       # total initial clusters
 
    # ── 2. Compute mean colour per block ─────────────────────
    # colors[i]  – float64 array of shape (ch,), mean colour of cluster i
    # counts[i]  – int, number of original blocks in cluster i
    # blocks[i]  – list of original block indices belonging to cluster i
    colors = np.zeros((nb, ch), dtype=np.float64)
    counts = np.ones(nb, dtype=np.int64)
    blocks = [[i] for i in range(nb)]
 
    for by in range(brows):
        for bx in range(bcols):
            bi   = by * bcols + bx
            y0, y1 = by * bsize, min((by + 1) * bsize, rows)
            x0, x1 = bx * bsize, min((bx + 1) * bsize, cols)
            patch  = img_f[y0:y1, x0:x1, :]   # shape (h, w, ch)
            colors[bi] = patch.mean(axis=(0, 1))
 
    # ── 3. Greedy agglomerative merging ──────────────────────
    active       = [True] * nb                 # active[i] → cluster i exists
    active_count = nb
    iters        = 0
 
    while active_count > num_clusters:
        act_idx = [i for i in range(nb) if active[i]]
 
        # Find the pair with minimum squared Euclidean colour distance
        min_dist = float('inf')
        mi, mj   = -1, -1
 
        # Vectorised pairwise distance over active clusters
        act_arr = np.array(act_idx)            # shape (K,)
        C       = colors[act_arr]              # shape (K, ch)
 
        # Compute upper-triangle distances without a Python double loop
        # diff[i, j] = C[i] - C[j]  →  dist² = sum over channels
        #   We use broadcasting: (K,1,ch) - (1,K,ch)
        diff   = C[:, np.newaxis, :] - C[np.newaxis, :, :]  # (K, K, ch)
        dist2  = (diff ** 2).sum(axis=2)                     # (K, K)
 
        # Mask diagonal & lower triangle (look at upper triangle only)
        K = len(act_idx)
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        dist2[~mask] = float('inf')
 
        flat_idx = dist2.argmin()
        ii, jj   = divmod(flat_idx, K)
 
        mi = act_idx[ii]
        mj = act_idx[jj]
 
        if mi < 0:
            break
 
        # ── 4. Merge mj into mi (weighted average colour) ────
        total           = counts[mi] + counts[mj]
        colors[mi]      = (colors[mi] * counts[mi] + colors[mj] * counts[mj]) / total
        counts[mi]      = total
        blocks[mi]     += blocks[mj]
        active[mj]      = False
        active_count   -= 1
        iters          += 1
 
    # ── 5. Collect final cluster colours ─────────────────────
    final_colors = colors[[i for i in range(nb) if active[i]]]  # (K, ch)
 
    # ── 6. Per-pixel nearest-cluster assignment ───────────────
    # Build pixel matrix: shape (H*W, ch)
    pixels = img_f.reshape(-1, ch)            # (N, ch)
 
    # Squared distances to each final cluster: (N, K)
    diff2 = pixels[:, np.newaxis, :] - final_colors[np.newaxis, :, :]
    dist2_px = (diff2 ** 2).sum(axis=2)       # (N, K)
    labels = dist2_px.argmin(axis=1)          # (N,)
 
    # ── 7. Build output image ─────────────────────────────────
    palette = np.stack([label_color(l) for l in range(len(final_colors))])
    output  = palette[labels].reshape(rows, cols, 3).astype(np.uint8)
 
    return {
        "output":       output,
        "num_clusters": num_clusters,
        "iterations":   iters,
    }
 

# ──────────────────────────────────────────────────────────────
#  API Wrapper for the Flask Backend
# ──────────────────────────────────────────────────────────────
def run_agglomerative(image_bgr: np.ndarray, n_clusters: int = 4) -> dict:
    """
    Adapter function that connects the new agglomerative implementation
    to the existing Flask API endpoints.
    """
    t0 = time.perf_counter()
    
    # Convert incoming OpenCV BGR image to RGB as expected by the new algorithm
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Execute the user's agglomerative clustering implementation
    res = agglomerative_segment(image_rgb, num_clusters=n_clusters)
    
    # Convert the RGB output back to BGR for the Flask JPEG encoder
    result_bgr = cv2.cvtColor(res['output'], cv2.COLOR_RGB2BGR)
    
    elapsed = (time.perf_counter() - t0) * 1000.0
    
    return {
        'result_image': result_bgr,
        'elapsed_ms':   round(elapsed, 1),
        'n_clusters':   res['num_clusters'],
    }
