"""
segmentation/__init__.py
Public API for the segmentation package.
Import all algorithm entry-points and the shared encode helper from here.
"""
from .kmeans          import run_kmeans
from .region_growing  import run_region_growing
from .agglomerative   import run_agglomerative
from .mean_shift      import run_mean_shift
from .utils           import encode_jpeg_b64

__all__ = [
    'run_kmeans',
    'run_region_growing',
    'run_agglomerative',
    'run_mean_shift',
    'encode_jpeg_b64',
]
