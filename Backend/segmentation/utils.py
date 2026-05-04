"""
segmentation/utils.py
Shared manual image processing primitives — only NumPy.
cv2 is used ONLY for final JPEG encoding (not for any algorithm logic).
"""
import base64
import numpy as np
import cv2  # Only used for final JPEG encoding for the browser

# Vivid false-colour palette (RGB order) for labelling cluster regions
_PALETTE_RGB = np.array([
    [255,  56,  56],   # red
    [ 56, 220,  56],   # green
    [ 56,  56, 255],   # blue
    [255, 220,  40],   # yellow
    [220,  56, 220],   # magenta
    [ 40, 220, 220],   # cyan
    [255, 150,   0],   # orange
    [130,  40, 220],   # violet
], dtype=np.uint8)


def manual_bgr_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the luminosity weights:
      Gray = 0.114*B + 0.587*G + 0.299*R
    These weights match the standard ITU-R BT.601 luma coefficients.
    """
    gray = (img_bgr[:, :, 0] * 0.114 +   # Blue channel
            img_bgr[:, :, 1] * 0.587 +   # Green channel
            img_bgr[:, :, 2] * 0.299)    # Red channel
    return gray.astype(np.uint8)


def manual_bgr_to_luv_l(img_bgr: np.ndarray) -> np.ndarray:
    """
    Compute the CIE L* (Lightness) channel from a BGR image.

    Steps:
      1. Normalise BGR to [0, 1].
      2. Compute Y (relative luminance) via the RGB-to-XYZ matrix row for Y:
           Y = 0.2126*R + 0.7152*G + 0.0722*B
      3. Apply the CIE L* transfer function:
           L* = 116 * Y^(1/3) - 16   if Y > 0.008856
           L* = 903.3 * Y             otherwise
      4. Scale L* from [0,100] to [0,255] for uint8 storage.
    """
    img  = img_bgr.astype(np.float32) / 255.0
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Relative luminance (Y component of XYZ)
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b

    # CIE L* piecewise formula
    l_star = np.where(y > 0.008856,
                      116.0 * np.power(y, 1.0 / 3.0) - 16.0,
                      903.3 * y)

    # Map [0, 100] → [0, 255]
    return np.clip(l_star * (255.0 / 100.0), 0, 255).astype(np.uint8)


def manual_resize_nn(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Resize `img` to (new_h, new_w) using nearest-neighbour interpolation.
    Each output pixel is assigned the value of the nearest input pixel.
    Works for both grayscale (H, W) and colour (H, W, C) images.
    """
    old_h, old_w = img.shape[:2]
    # Map each output row/col index back to the nearest input row/col
    row_idx = np.clip((np.arange(new_h) * (old_h / new_h)).astype(np.int32), 0, old_h - 1)
    col_idx = np.clip((np.arange(new_w) * (old_w / new_w)).astype(np.int32), 0, old_w - 1)
    return img[np.ix_(row_idx, col_idx)]


def manual_gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    """
    Convert a single-channel grayscale image to a 3-channel BGR image
    by stacking the same channel three times: BGR = [G, G, G].
    """
    return np.stack([gray, gray, gray], axis=-1)


def false_color(labels: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Map a flat integer label array to a vivid BGR visualisation image.

    Parameters
    ----------
    labels : ndarray of shape (N,)  — cluster index for each pixel
    shape  : (height, width)        — spatial dimensions to reshape into
    """
    # Index palette by label (mod palette size so we never go out of bounds)
    rgb = _PALETTE_RGB[labels % len(_PALETTE_RGB)]   # (N, 3) in RGB order
    rgb_img = rgb.reshape(shape[0], shape[1], 3)
    # Reverse channel order RGB → BGR for OpenCV / browser compatibility
    return rgb_img[:, :, ::-1].copy()


def jet_colorize_manual(gray: np.ndarray) -> np.ndarray:
    """
    Apply a manual JET-like false-colour ramp to a grayscale image.

    The JET colour map goes: Blue → Cyan → Green → Yellow → Red
    as intensity increases from 0 to 255.
    Each channel is a piecewise-linear ramp defined by:
      R = clip(min(4v - 1.5,  -4v + 4.5), 0, 1)
      G = clip(min(4v - 0.5,  -4v + 3.5), 0, 1)
      B = clip(min(4v + 0.5,  -4v + 2.5), 0, 1)
    where v = pixel / 255.
    Output is in BGR order (B is first channel).
    """
    v = gray.astype(np.float32) / 255.0   # normalise to [0, 1]

    r = np.clip(np.minimum(4 * v - 1.5, -4 * v + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * v - 0.5, -4 * v + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * v + 0.5, -4 * v + 2.5), 0, 1)

    # Stack as BGR (b first) and scale to [0, 255]
    return (np.stack([b, g, r], axis=-1) * 255).astype(np.uint8)


def encode_jpeg_b64(image_bgr: np.ndarray) -> str:
    """
    Encode a BGR ndarray to a base64 JPEG data-URL string so it can be
    sent directly to the browser.
    cv2.imencode is used here purely for JPEG compression — not for any
    image-processing algorithm.
    """
    _, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
