import cv2
import numpy as np
# from classes.image import Image  # Not needed for Flask API
# from PyQt5.QtWidgets import QApplication, QFileDialog  # Not needed for Flask API
# import sys  # Not needed for Flask API

class OptimalThresholding:
    """
    A class to perform optimal thresholding on grayscale images using either global or local methods.

    - Global mode: Applies thresholding to the entire image.
    - Local mode: Divides the image into blocks, computes thresholds, and interpolates them.

    Attributes:
        thresholding_mode (str): The mode of thresholding, either "global" or "local".
        corner_size (int): Size of the corner patches (in pixels) used for initial background estimation.
        block_size (int): Size of blocks (in pixels) for local thresholding.
        convergence_threshold (float): Tolerance for convergence in the iterative thresholding process.
    """
    def __init__(self, thresholding_mode):

        self.corner_size = 1  # size = 1 means single corner pixels not corner patches
        self.block_size = 64  # block/patch/window size in local thresholding
        self.convergence_threshold = 1e-3
        mode = thresholding_mode.lower()
        if mode not in ["global", "local"]:
            raise ValueError("thresholding_mode must be 'global' or 'local'")
        self.thresholding_mode = mode

    def apply_thresholding(self, input_image, block_size=None):
        """
        Apply optimal thresholding to a grayscale image in either global or local mode.

        Args:
            input_image: Grayscale image as a 2D NumPy array.
            block_size (int, optional): Block size for local thresholding.

        Returns:
            np.ndarray: Binary image (0 for background, 255 for foreground).

        Raises:
            ValueError: If the input image is not a 2D grayscale array.
        """
        # Ensure the input image is valid
        if input_image is None or len(input_image.shape) != 2:
            raise ValueError("Input image is not loaded or not a valid grayscale image.")

        # Apply thresholding based on the specified mode
        if self.thresholding_mode == "global":
            binary_image = self.apply_global_thresholding(input_image)
        else:  # Local mode
            binary_image = self.apply_local_thresholding(input_image, block_size=block_size)

        return binary_image

    def get_corner_pixels(self, image):
        """
        Estimate initial background pixels by sampling the four corner pixels of the image.

        Args:
            image: Grayscale image as a 2D NumPy array.

        Returns:
            np.ndarray: Flattened array of the four corner pixel values.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rows, cols = image.shape
        corners = np.array([
            image[0, 0],              # Top-left corner pixel
            image[0, cols-1],         # Top-right corner pixel
            image[rows-1, 0],         # Bottom-left corner pixel
            image[rows-1, cols-1]     # Bottom-right corner pixel
        ])
        return corners

    def compute_optimal_threshold(self, image):
        """
        Compute the optimal threshold for a grayscale image.

        Algorithm:
        1. Start with an initial guess (four corners as background, rest as object).
        2. Compute mean gray-levels of background and object.
        3. Update the threshold as the average of the means.
        4. Repeat until the threshold converges.

        Args:
            image: Grayscale image as a 2D NumPy array.

        Returns:
            The optimal threshold value (float).
        """
        # Step 1: Initial guess for background and object pixels
        background_pixels = self.get_corner_pixels(image)
        
        # Object pixels are all pixels not in the corners
        all_pixels = image.flatten()
        object_pixels = all_pixels[~np.isin(all_pixels, background_pixels)]  # Exclude corner pixels

        # Step 2-4: Iterative threshold computation
        previous_threshold = 0.0
        while True:
            # Compute mean gray-levels of background 
            if len(background_pixels) > 0:
                background_mean = np.mean(background_pixels)
            else:
                background_mean = 0

            # Compute mean gray-levels of object 
            if len(object_pixels) > 0:
                object_mean = np.mean(object_pixels)
            else:
                object_mean = 0

            # Update threshold as the average of the means
            current_threshold = (background_mean + object_mean) / 2

            # Check for convergence
            if abs(current_threshold - previous_threshold) < self.convergence_threshold:
                # Print only for global thresholding
                if self.thresholding_mode == "global":
                    print(f"Optimal Threshold for Global Thresholding reached = {current_threshold}")
                break

            # If didn't reach the final threshold, Update again the pixel classifications based on the last threshold
            background_pixels = image[image <= current_threshold].flatten()
            object_pixels = image[image > current_threshold].flatten()
            previous_threshold = current_threshold

        return current_threshold

    def apply_global_thresholding(self, image):
        """
        Apply global optimal thresholding to the entire image.

        Args:
            image: Grayscale image as a 2D NumPy array.

        Returns:
            Binary image (0 for background, 255 for object).

        Raises:
            ValueError: If the image is not a 2D grayscale array.
        """
        # Ensure the image is 2D (grayscale)
        if len(image.shape) != 2:
            raise ValueError(f"Expected a 2D grayscale image, but got shape {image.shape}")

        # Compute the optimal threshold for the entire image
        threshold = self.compute_optimal_threshold(image)

        # Create binary image: pixels above threshold are foreground (255), others are background (0)
        binary_image = np.zeros_like(image, dtype=np.uint8)
        binary_image[image > threshold] = 255
        return binary_image

    def apply_local_thresholding(self, image, block_size=64):
        """
        Apply local optimal thresholding by dividing the image into non-overlapping blocks
        and applying thresholding to each block independently.

        Args:
            image: Grayscale image as a 2D NumPy array.
            block_size (int): Size of blocks for local thresholding.

        Returns:
            Binary image (0 for background, 255 for foreground).

        Raises:
            ValueError: If the image is not a 2D grayscale array.
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
                threshold = self.compute_optimal_threshold(block)

                # Create a binary block by applying the threshold
                binary_block = np.zeros_like(block, dtype=np.uint8)
                binary_block[block > threshold] = 255  # ForObject pixels are set to 255

                # Place the binary block back into the corresponding region of the binary image
                binary_image[row:row_end, col:col_end] = binary_block

        # Return the final binary image
        return binary_image