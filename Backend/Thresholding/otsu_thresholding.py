import cv2
import numpy as np
# from classes.image import Image  # Not needed for Flask API
# from PyQt5.QtWidgets import QApplication, QFileDialog  # Not needed for Flask API
# import sys  # Not needed for Flask API

class OtsuThresholding:
    """
    Otsu's thresholding for image segmentation.
    
    Supports two modes:
    - global
    - local
    """

    def __init__(self, mode):
        self.mode = mode.lower()

        if self.mode not in ['global', 'local']:
            raise ValueError("Mode must be 'global' or 'local'")
        
    def compute_best_threshold(self, image):
        """
        Calculate optimal threshold using Otsu's method.
        
        Args:
            image: Grayscale input image
            
        Returns:
            Optimal threshold value (0-255)
        """
        # Calculate image histogram (probability of each intensity)
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256]) #  256x1 array where each element represents the frequency of the corresponding intensity value in the image

        total_pixels = image.shape[0] * image.shape[1]

        # Normalize histogram to compute probabilities.
        probabilities = histogram / total_pixels
        
        best_threshold = 0
        best_variance = 0
        
        # Try all possible threshold values
        for threshold in range(256):
            # Split into background (below threshold) and foreground (above)
            background_probs = probabilities[:threshold]
            object_probs = probabilities[threshold:]
            
            # Skip if either class is empty
            if len(background_probs) == 0 or len(object_probs) == 0:
                continue
                
            # Calculate weights (sum of probabilities) for each class
            weight_background = np.sum(background_probs) # P1(k) as in slide 12
            weight_object = np.sum(object_probs) # P2(k) as in slide 12
            
            # Calculate mean intensity for background class
            if weight_background > 0:
                intensities_background = np.arange(threshold) #  generates an array of intensity values from 0 to threshold - 1
                weighted_background_sum = np.sum(intensities_background * background_probs.flatten()) 
                mean_background = weighted_background_sum / weight_background # m1(k) as in slide 12
            else:
                mean_background = 0
            
            # Calculate mean intensity for object class
            if weight_object > 0:
                intensities_object = np.arange(threshold, 256) #  generates an array of intensity values from threshold to 255
                weighted_object_sum = np.sum(intensities_object * object_probs.flatten())
                mean_foreground = weighted_object_sum / weight_object # m2(k)
            else:
                mean_foreground = 0
            
            # Calculate between-class variance
            # σ²_B = P₁(m₁ - m_G)² + P₂(m₂ - m_G)²
            # a simplified equivalent form of it:
            # P₁ * P₂ * (m₁ - m₂) ** 2
            class_variance = weight_background * weight_object * (mean_background - mean_foreground)**2 
            
            # Keep track of best threshold
            if class_variance > best_variance:
                best_variance = class_variance
                best_threshold = threshold
                
        return best_threshold

    def apply_global_threshold(self, image):
        """
        Apply global Otsu threshold to image.
        
        Args:
            image: Input image
            
        Returns:
            Binary thresholded image
        """
        # Ensure the image is 2D (grayscale)
        if len(image.shape) != 2:
            raise ValueError(f"Expected a 2D grayscale image, but got shape {image.shape}")

        # Compute the best threshold for the entire image
        threshold = self.compute_best_threshold(image)

        # Create binary image: pixels above threshold are object (255), others are background (0)
        binary_image = np.zeros_like(image, dtype=np.uint8)
        binary_image[image > threshold] = 255

        return binary_image

    def apply_local_threshold(self, image, block_size=None):
        """
        Apply local thresholding by dividing image into blocks.
        
        Args:
            image: Input image
            block_size (int): Size of blocks (must be odd)
            constant (int): Adjustment constant
            
        Returns:
            Binary thresholded image
        """
        # Ensure the image is 2D (grayscale)
        if len(image.shape) != 2:
            raise ValueError(f"Expected a 2D grayscale image, but got shape {image.shape}")

        # Get the dimensions of the image
        rows, cols = image.shape

        # Initialize the binary output image with zeros
        binary_image = np.zeros_like(image, dtype=np.uint8)

        # Process the image in non-overlapping blocks
        for row in range(0, rows, block_size):  # Step size is equal to block_size
            for col in range(0, cols, block_size):

                # Define the block boundaries
                row_end = min(row + block_size, rows)  # Ensure we don't go out of bounds
                col_end = min(col + block_size, cols)

                # Extract the block from the image
                block = image[row:row_end, col:col_end]

                # Skip empty blocks (unlikely in normal images)
                if block.size == 0:
                    continue

                # Compute the optimal threshold for the block
                threshold = self.compute_best_threshold(block)

                # Create a binary block by applying the threshold
                binary_block = np.zeros_like(block, dtype=np.uint8)
                binary_block[block > threshold] = 255  # Object pixels are set to 255

                # Place the binary block back into the corresponding region of the binary image
                binary_image[row:row_end, col:col_end] = binary_block

        # Return the final binary image
        return binary_image

    def apply_otsu(self, image, block_size=None):
        """
        Apply Otsu's thresholding based on the selected mode.

        Args:
            image (numpy.ndarray): Input grayscale image.
            block_size (int, optional): Block size for local thresholding.

        Returns:
            numpy.ndarray: Binary thresholded image.
        """
        # Ensure the image is 2D (grayscale)
        if len(image.shape) != 2:
            raise ValueError(f"Expected a 2D grayscale image, but got shape {image.shape}")

        if self.mode == 'global':
            return self.apply_global_threshold(image)
        elif self.mode == 'local':
            return self.apply_local_threshold(image, block_size=block_size)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")