"""Different types of image blending for rendering a final composite"""

import numpy as np
from stitcher.compositing import PanoramaCompositor

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

