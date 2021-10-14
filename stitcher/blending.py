"""Different types of image blending for rendering a final composite"""

import numpy as np
import scipy.ndimage
import cv2
from stitcher.compositing import PanoramaCompositor

def blend_linearly(composite: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Utility function for linearly blending a composite image

    Args:
        composite: A composite image with the shape (images, height, width, 3)
        weights: An array of image weights with the shape (images, height, width, 1)
    """
    img = composite * weights
    img = np.sum(img, axis=0)
    w = np.sum(weights, axis=0)
    w[np.isclose(w, 0.0)] = 1.0  # Avoid division by 0 where there is no weight
    return img / w


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize a floating point image with arbitrary values to a uint8 image with values [0, 255]"""
    res = 255 * (img - img.min()) / (img.max() - img.min())
    return res.astype("uint8")


class NoBlender:
    """Renders images without blending, using the painter's algorithm"""

    def __init__(self, compositor: PanoramaCompositor) -> None:
        """Initialize the blender from a `PanoramaCompositor`"""
        self.composite = compositor.composite
        self.weights = compositor.weights
        self.width = compositor.width
        self.height = compositor.height

    def render(self) -> None:
        """Renders and returns a final panorama image"""
        img = np.zeros((self.height, self.width, 3), dtype="int")
        for i, layer in enumerate(self.composite):
            w = (self.weights[i] > 0).squeeze(2)
            img[w] = layer[w]
        return normalize_image(img)


class LinearBlender:
    """Blends images in a composite linearly using the weights associated with each pixel value"""

    def __init__(self, compositor: PanoramaCompositor) -> None:
        """Initialize a linear blender from a `PanoramaCompositor`"""
        self.composite = compositor.composite
        self.weights = compositor.weights

    def render(self, p=1.0) -> np.ndarray:
        """Renders and returns a final panorama image using exponential weighting"""
        exp_weight = self.weights ** p
        img = blend_linearly(self.composite, exp_weight)
        return normalize_image(img)


class MultiBandBlender:
    """Renders panorama images using multi-band blending"""

    def __init__(self, compositor: PanoramaCompositor) -> None:
        """Initialize a multi-band blender from a `PanoramaCompositor`"""
        self.composite = compositor.composite
        argmax_weights = np.argmax(compositor.weights, axis=0)
        imgs = np.unique(argmax_weights)
        self.mask = (argmax_weights == imgs).astype("float32")
        self.mask = np.transpose(self.mask, [2, 0, 1])[..., np.newaxis]
        self.hard_mask = (compositor.weights > 0.0)
        self.mask *= self.hard_mask

    def render(self, bands) -> np.ndarray:
        """
        Renders and returns the final panorama image
        
        Args:
            bands: The number of bands to use for rendering
        """
        N = self.composite.shape[0]  # Number of images
        I = [np.copy(self.composite).astype("float32")]  # Gaussians
        L = []  # Laplacians
        M = [np.copy(self.mask)]  # Masks

        # Reduction step, move down the pyramid and compute Laplacians
        for k in range(1, bands):
            up_height, up_width = I[k-1].shape[1:3]
            up_size = (up_width, up_height)  # OpenCV makes me sad sometimes :'(
            I_k = [cv2.pyrDown(I[k-1][n, ...]) for n in range(N)]
            M_k = [cv2.pyrDown(M[k-1][n, ...]) for n in range(N)]
            I.append(np.stack(I_k))
            M.append(np.stack(M_k)[..., np.newaxis])
            L_k = [I[k-1][n, ...] - cv2.resize(cv2.pyrUp(I[k][n, ...]), up_size) for n in range(N)]
            L.append(np.stack(L_k))

        L.append(I[bands-1])  # Append the peak of the pyramid, just the Gaussian
        
        # Fusion step, blend the images in each band layer
        blended = [blend_linearly(l, m) for l, m in zip(L, M)]
        B = blended[-1]

        # Expansion step, move up the pyramid
        for k in range(bands-1, 0, -1):
            up_height, up_width = blended[k-1].shape[:2]
            up_size = (up_width, up_height)  # OpenCV makes me even more sad sometimes :'(
            B = cv2.resize(cv2.pyrUp(B), up_size) + blended[k-1]

        return normalize_image(B) * np.any(self.hard_mask, axis=0)
        