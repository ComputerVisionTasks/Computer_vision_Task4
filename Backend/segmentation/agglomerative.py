"""
segmentation/agglomerative.py
Custom implementation of Agglomerative Hierarchical Clustering 
using Centroid Linkage for execution efficiency in pure Python.
"""
import time
import cv2
import numpy as np
from .utils import false_color, manual_resize_nn

def agglomerative_clustering_scratch(data, n_clusters):
    """
    Custom implementation of Agglomerative Hierarchical Clustering 
    using Centroid Linkage for execution efficiency in pure Python.
    """
    n_samples = len(data)
    
    # Initialize each data point as its own cluster
    # Format: {cluster_id: list_of_data_indices}
    clusters = {i: [i] for i in range(n_samples)}
    
    # Track cluster centroids to avoid redundant calculations
    centroids = {i: data[i].astype(float) for i in range(n_samples)}
    
    # Track the next available cluster ID
    next_cluster_id = n_samples
    
    # Iteratively merge the closest clusters until n_clusters is reached
    while len(clusters) > n_clusters:
        min_dist = float('inf')
        closest_pair = None
        
        cluster_ids = list(clusters.keys())
        n_current_clusters = len(cluster_ids)
        
        # Calculate pairwise centroid distances to find the closest clusters
        for i in range(n_current_clusters):
            for j in range(i + 1, n_current_clusters):
                c1 = cluster_ids[i]
                c2 = cluster_ids[j]
                
                # Euclidean distance between centroids
                dist = np.linalg.norm(centroids[c1] - centroids[c2])
                
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (c1, c2)
        
        # Merge the closest pair
        c1, c2 = closest_pair
        merged_indices = clusters[c1] + clusters[c2]
        
        # Calculate the new centroid
        new_centroid = np.mean(data[merged_indices], axis=0)
        
        # Store the new cluster
        clusters[next_cluster_id] = merged_indices
        centroids[next_cluster_id] = new_centroid
        
        # Delete the old merged clusters
        del clusters[c1]
        del clusters[c2]
        del centroids[c1]
        del centroids[c2]
        
        next_cluster_id += 1
        
    # Reconstruct the 1D label array mapped to the original data points
    labels = np.zeros(n_samples, dtype=int)
    for final_label, (cid, indices) in enumerate(clusters.items()):
        for idx in indices:
            labels[idx] = final_label
            
    return labels

def run_agglomerative(image_bgr: np.ndarray, n_clusters: int = 4) -> dict:
    """
    Wrapper function that adapts the custom agglomerative implementation
    for the web backend API.
    """
    t0 = time.perf_counter()
    orig_h, orig_w = image_bgr.shape[:2]
    
    # Target size is bumped to 25x25. 
    # Any higher causes extreme delays because the provided algorithm is O(N^3).
    target_size = (25, 25)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Resize
    img = cv2.resize(gray, target_size)
    
    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    img_sharpened = cv2.filter2D(img, -1, kernel)
    
    # Extract spatial coordinates to force spatial coherence in the clusters
    h, w = img_sharpened.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Scale coordinates so they balance with intensity values (0-255)
    spatial_weight = 255.0 / max(h, w)
    
    # Create feature vector [intensity, y, x]
    features = np.column_stack((
        img_sharpened.flatten(),
        y_coords.flatten() * spatial_weight,
        x_coords.flatten() * spatial_weight
    ))
    
    # Execute custom AHC
    labels = agglomerative_clustering_scratch(features, n_clusters)
    
    # Rebuild 2D segmented image at the small scale
    segmented_img_small = labels.reshape(img_sharpened.shape)
    
    # Upscale label image to original resolution using Nearest-Neighbour
    label_full = manual_resize_nn(segmented_img_small, orig_w, orig_h)

    # Vivid false-colour mapping: cluster index -> distinct palette colour
    result_bgr = false_color(label_full.flatten(), (orig_h, orig_w))
    
    elapsed = (time.perf_counter() - t0) * 1000.0

    return {
        'result_image': result_bgr,
        'elapsed_ms':   round(elapsed, 1),
        'n_clusters':   n_clusters,
    }
