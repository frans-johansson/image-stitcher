"""Different types of image blending for rendering a final composite"""

import numpy as np
import scipy.ndimage
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
        img = np.zeros((self.height, self.width, 3), dtype="uint8")
        for i, layer in enumerate(self.composite):
            w = (self.weights[i] > 0).squeeze(2)
            img[w] = layer[w]
        return img


class LinearBlender:
    """Blends images in a composite linearly using the weights associated with each pixel value"""

    def __init__(self, compositor: PanoramaCompositor) -> None:
        """Initialize a linear blender from a `PanoramaCompositor`"""
        self.composite = compositor.composite
        self.weights = compositor.weights

    def render(self) -> np.ndarray:
        """Renders and returns a final panorama image"""
        img = blend_linearly(self.composite, self.weights)
        return img.astype("uint8")


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

    def render(self, bands, sigma=1.0) -> np.ndarray:
        """
        Renders and returns the final panorama image
        
        Args:
            bands: The number of bands to use for rendering
            sigma: The initial value of sigma for the first band
        """
        # img = np.zeros((*self.composite.shape[1:3], 3), dtype="float32")
        N = self.composite.shape[0]
        
        M = np.zeros((bands+1, *self.composite.shape[:3], 1))  # Mask, (bands, image, height, width, 1)
        I = np.zeros((bands+1, *self.composite.shape))  # Gaussian, (bands, image, height, width, RGB)
        L = np.zeros_like(I)  # Laplacian, (bands, image, height, width, RGB)

        # NOTE: The 0th band refers to the original images here
        M[0] = np.copy(self.mask)
        I[0] = np.copy(self.composite)
        L[0] = np.copy(self.composite)

        # Compute Gaussian pyramid for images and masks
        for k in range(1, bands+1):
            # Compute the next band
            sigma_k = np.sqrt(2*k+1)*sigma

            for n in range(N):  # Iterate over all images separately
                I[k, n, ..., 0] = scipy.ndimage.gaussian_filter(I[k-1, n, ..., 0], sigma_k)
                I[k, n, ..., 1] = scipy.ndimage.gaussian_filter(I[k-1, n, ..., 1], sigma_k)
                I[k, n, ..., 2] = scipy.ndimage.gaussian_filter(I[k-1, n, ..., 2], sigma_k)
                M[k, n, ...] = scipy.ndimage.gaussian_filter(M[k-1, n, ...], sigma_k)

            M[k, ...] *= self.hard_mask
            I[k, ...] *= self.hard_mask

        # Compute Laplacians
        L[1:bands] = I[1:bands, ...] - I[0:bands-1, ...]
        L[bands] = I[bands]

        # Images need to come first for blending
        M = np.transpose(M[1:], [1, 0, 2, 3, 4])
        L = np.transpose(L[1:], [1, 0, 2, 3, 4])
        
        # Blend, aggregate and normalize
        img = blend_linearly(L, M)
        img = np.sum(img, axis=0)
        return normalize_image(img)
