import numpy as np
import cv2
from scipy import ndimage

class SpectralThresholding:
    
    def __init__(self):
        pass
        
    def global_otsu_multithreshold( self,
                                    image: np.ndarray, 
                                    num_classes: int = 3,
                                    smoothing_sigma: float = 1.0) -> np.ndarray:

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute global histogram
        hist_global = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist_global = hist_global / np.sum(hist_global)  # Normalize histogram
        
        # Apply Gaussian smoothing to histogram
        smoothed_hist = ndimage.gaussian_filter1d(hist_global, smoothing_sigma)
        pixel_range = np.arange(256)
        
        # Global mean
        mg = np.sum(pixel_range * smoothed_hist)
        
        # For 2 classes
        if num_classes == 2:
            max_variance = 0
            optimal_threshold = 0
            
            for t in range(256):

                # calc probabilities for each class 
                p1 = np.sum(smoothed_hist[:t+1])
                p2 = np.sum(smoothed_hist[t+1:])
                
                if p1 == 0 or p2 == 0:
                    continue
                
                # calc means for each class
                m1 = np.sum(pixel_range[:t+1] * smoothed_hist[:t+1]) / p1
                m2 = np.sum(pixel_range[t+1:] * smoothed_hist[t+1:]) / p2
                
                # calc variance
                variance = p1 * p2 * ((m1 - m2) ** 2)
                
                # Check if this is the best threshold
                if variance > max_variance:
                    max_variance = variance
                    optimal_threshold = t
            
            return self.segment_image(image, [optimal_threshold])
        
        # For multiple classes
        max_variance = 0
        final_thresholds = []
        
        def calculate_variance(t_list):
            t_list = sorted(t_list)
            variance = 0
            
            # get ranges for each class based on thresholds
            ranges = [(0, t_list[0])] + \
                    [(t_list[i], t_list[i+1]) for i in range(len(t_list)-1)] + \
                    [(t_list[-1], 256)]
                    
            for start, end in ranges:
                # calc probabilities for each class
                p_k = np.sum(smoothed_hist[start:end])
                if p_k == 0:
                    continue
                # calc means for each class
                m_k = np.sum(pixel_range[start:end] * smoothed_hist[start:end]) / p_k
                # calc variance
                variance += p_k * ((m_k - mg) ** 2)
                
            return variance
        
        # Initialize with peaks from smoothed histogram as promising thresholds
        peaks = []
        for i in range(1, 255):
            # Find local maxima (peaks) in the smoothed histogram
            if smoothed_hist[i] > smoothed_hist[i-1] and smoothed_hist[i] > smoothed_hist[i+1]:

                # Filters peaks by their prominence (height relative to surrounding minima) to avoid minor fluctuations
                prominence = min(
                    smoothed_hist[i] - np.min(smoothed_hist[max(0, i-10):i]),
                    smoothed_hist[i] - np.min(smoothed_hist[i+1:min(256, i+11)])
                )
                if prominence > 0.01 * np.max(smoothed_hist):
                    peaks.append(i)
        
        # Use top num_classes-1 peaks as initial thresholds
        if len(peaks) >= num_classes-1:
            # Sort peaks by height
            peak_heights = [smoothed_hist[p] for p in peaks]
            indices = np.argsort(peak_heights)[-(num_classes-1):]
            thresholds = sorted([peaks[i] for i in indices])
        else:
            # Fallback to evenly spaced thresholds
            step = 256 // num_classes
            thresholds = [i * step for i in range(1, num_classes)]
        
        # Refine thresholds for 5 iterations
        for _ in range(5):
            for i in range(len(thresholds)):
                # Calculate variance for current threshold
                best_t = thresholds[i]
                # searching nearby values (±10) that maximize the between-class variance
                for t in range(max(0, best_t-10), min(255, best_t+10)):
                    temp_thresholds = thresholds.copy()
                    temp_thresholds[i] = t
                    variance = calculate_variance(temp_thresholds)
                    if variance > max_variance:
                        max_variance = variance
                        best_t = t
                        final_thresholds = temp_thresholds.copy()
                thresholds[i] = best_t
        
        return self.segment_image(image, final_thresholds)
    

    def local_otsu_multithreshold(  self,
                                    image, 
                                    num_classes = 3,
                                    window_size= 3,  
                                    smoothing_sigma = 1.0,):

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        if window_size < 3:
            raise ValueError("Window size must be at least 3")
            
        result = np.zeros_like(image, dtype=np.float32)
        # Calculate padding size
        pad_size = window_size // 2
        
        # Pad image for window analysis with reflection to avoid edge effects
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size,
                                cv2.BORDER_REFLECT)
        
        # Process each pixel with its local window
        for i in range(pad_size, padded.shape[0] - pad_size):
            for j in range(pad_size, padded.shape[1] - pad_size):
                # Extract local window centered at (i, j) our pixel of interest
                window = padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
                
                # Compute histogram for window
                hist = cv2.calcHist([window], [0], None, [256], [0, 256]).flatten()
                smoothed_hist = ndimage.gaussian_filter1d(hist, smoothing_sigma)
                
                # Find peaks
                peaks = []
                peak_heights = []
                
                for k in range(1, 255):
                    # Find local maxima (peaks) in the smoothed histogram
                    if smoothed_hist[k] > smoothed_hist[k-1] and smoothed_hist[k] > smoothed_hist[k+1]:
                        peaks.append(k)
                        peak_heights.append(smoothed_hist[k])
                
                # Select strongest peaks and find thresholds
                if len(peaks) >= num_classes:
                    # Sort peaks by height
                    indices = np.argsort(peak_heights)[-num_classes:]
                    peaks = sorted([peaks[idx] for idx in indices])
                    
                    # Find minima between peaks to determine thresholds
                    thresholds = []
                    for k in range(len(peaks)-1):
                        start, end = peaks[k], peaks[k+1]
                        minima_between_peaks_idx = start + np.argmin(smoothed_hist[start:end+1])
                        thresholds.append(minima_between_peaks_idx)
                    
                    # Assign class label based on thresholds
                    pixel_value = image[i-pad_size, j-pad_size]
                    label = 0
                    for k, t in enumerate(thresholds):
                        if pixel_value > t:
                            label = k + 1
                    result[i-pad_size, j-pad_size] = label
                else:
                    # If not enough peaks, use mean value
                    result[i-pad_size, j-pad_size] = np.mean(window)
        
        return result

    def segment_image(self, image, thresholds):

        segmented = np.zeros_like(image)
        thresholds = sorted(thresholds)
        
        # Iterate through thresholds and assign class labels
        # 0 for background, 1 for first class, ..., len(thresholds) for last class
        for i, t in enumerate(thresholds):
            if i == 0:
                segmented[image <= t] = 0
            else:
                segmented[(image > thresholds[i-1]) & (image <= t)] = i
        
        # Assigns the highest class label to pixels above the last threshold
        segmented[image > thresholds[-1]] = len(thresholds)
        
        return segmented
    

