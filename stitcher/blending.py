"""Different types of image blending for rendering a final composite"""

import numpy as np
from stitcher.compositing import PanoramaCompositor

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
        img = self.composite * self.weights
        img = np.sum(img, axis=0)
        w = np.sum(self.weights, axis=0)
        w[np.isclose(w, 0.0)] = 1.0  # Avoid division by 0 where there is no weight
        img /= w
        img = np.nan_to_num(img)
        return img.astype("uint8")

