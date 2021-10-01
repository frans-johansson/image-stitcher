"""Handles loading and processing images in memory before and during the stitching process."""

from pathlib import Path
import numpy as np
import cv2

def rescale_image(image: np.ndarray, scale: float) -> np.ndarray:
        """
        Utility function for uniformly rescaling a single image

        Args:
            image: An image to rescale, expected to be either single channel or triple channel
            scale: The scaling factor applied to the width and height for the output image
        """
        (height, width) = image.shape[:2]
        new_dims = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_dims, interpolation=cv2.INTER_CUBIC)


class ImageCollection:
    """
    Reads and processes a collection of images in memory to be used for stitching. Note that the images are
    expected to be .png files.
    
    Members:
        high_res_images: A list of RGB images in high resolution to be used for compositing
        low_res_images: A list of grayscale images in a lower resolution to be used for transform estimation  
    """

    def __init__(self, path: Path, high_res_scale: float, low_res_scale: float) -> None:
        """
        Initializes an image collection from a path to the images. Reads and pre-processes all
        images upon initialization.

        Args:
            path: A path object to the location of the images
            high_res_scale: Scaling factor for the high-resolution color images
            high_res_scale: Scaling factor for the low-resolution grayscale images
        """
        # FIXME: Handle both png and jpg here
        image_files = [str(file) for file in path.rglob("*.jpg")]
        color_images = [cv2.imread(image) for image in image_files]
        grayscale_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in color_images]

        self.high_res_images = [rescale_image(image, high_res_scale) for image in color_images]
        self.low_res_images = [rescale_image(image, low_res_scale) for image in grayscale_images]
