"""Handles transformations and compositing of multiple images onto a single, seamless panorama"""

from typing import Tuple
from stitcher.features import FeatureHandler
from stitcher.images import ImageCollection
import numpy as np

class PanoramaCompositor:
    """
    Drives the image compositing process utilizing submodules for applying and estimating
    transformations before writing to a temporary, multi-layer composite image. Which will
    then be refined into a seamless, single-layer panorama. The supplied images are expected
    to form a single cohesive panorama, not several disjoint panoramas.

    Members:
        num_images: The number of images given to composite
        matches: Matches extracted from the supplied FeatureHandler.
        features: Features extracted from the supplied FeatureHandler.
        overlaps: Book-keeping dictionary for what images can be incorporated into the panorama next.
            Structured like {image: [overlapping images], ...}. Is updated throughout the compositing process.
        reference_img: Index for the selected reference image
        offset: A two-dimensional array representing the [x, y] offset that should be applied to each image
        width: The width of the final composite in pixels.
        height: The height of the final composite in pixels.
        composite: The final composite RGB image. Strucutured like a NxWxHx3 array, where N is the number of
            input images, W is the width and H is the height.
    """

    def __init__(self, images: ImageCollection, feature_handler: FeatureHandler) -> None:
        """
        Initializes and runs the image compositing procss for creating a panorama image.

        Args:
            images: An `ImageCollection` of images to composite
            feature_handler: A `FeatureHandler` which has been supplied the same image collection
                (TODO: Maybe just give this class the image collection and let it handle setting up the feature handler) 
        """
        self.num_images = len(images.low_res_images)
        self.matches = feature_handler.feature_matches
        self.features = feature_handler.image_features

        self.overlaps = {i: [j for j in self.matches[i].keys()] for i in self.matches.keys()}
        self.reference_img = self._find_reference_img()
        self.width, self.height, self.offset = self._compute_bounding_box()
        self.composite = np.full([self.num_images, self.width, self.height, 3], -1)



    def _find_reference_img(self) -> int:
        """
        Finds the index of the reference image for the composite based on the
        overlap graph

        Returns:
            reference_img: The index of the reference image in the ImageCollection
        """
        # TODO: Implement
        return 0

    def _compute_bounding_box(self) -> Tuple[int, int, np.ndarray]:
        """
        Computes the dimensions and offset for the final composite

        Returns:
            (width, height, offset): The width and height given as integers. The offset given as
                a two-dimensional array with [offset x, offset y].
        """
        # TODO: Implement
        return (200, 100, np.array([50, 50]))

    def _extend(self, img: int) -> None:
        """
        Pastes an image onto the composite.
        
        Args:
            img: The index of the image to incorporate.
        """
        # TODO: Implement
        pass

    def _run(self) -> None:
        """Driver function for running the compositing process"""     
        num_pasted = 0

        while num_pasted <= self.num_images:
            # 1. Find the next image to extend with
            #    using the overlap graph

            # 2. Extend with that image

            # 3. Update the overlap graph
            pass
