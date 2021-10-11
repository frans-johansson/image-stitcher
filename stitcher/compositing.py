"""Handles transformations and compositing of multiple images onto a single, seamless panorama"""

from typing import Tuple

import cv2
from numpy.lib.shape_base import expand_dims
from stitcher.features import FeatureHandler
from stitcher.images import ImageCollection
import numpy as np
import matplotlib.pyplot as plt

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
            Structured like {image: {overlapping images}, ...}, i.e. a dictionary of sets. Is updated
            throughout the compositing process.
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

        self.images = images.high_res_images

        self.overlaps = {i: {j for j in self.matches[i].keys()} for i in self.matches.keys()}
        self.reference_img = self._find_reference_img()
        self.height, self.width, self.offset = self._compute_bounding_box()
        self.composite = np.full([self.num_images, self.height, self.width, 3], -1)

        self.composite[self.reference_img] = self._perspective_transformation(self.images[self.reference_img], np.eye(3))

        self._run()


    def _find_reference_img(self) -> int:
        """
        Finds the index of the reference image for the composite based on the
        overlap graph

        Returns:
            reference_img: The index of the reference image in the ImageCollection
        """
        ref = max([(len(self.matches[i]), i) for i in self.matches])[1]

        print(f"Starting at image {ref}")
        return ref

    def _compute_bounding_box(self) -> Tuple[int, int, np.ndarray]:
        """
        Computes the dimensions and offset for the final composite

        Returns:
            (width, height, offset): The width and height given as integers. The offset given as
                a two-dimensional array with [offset x, offset y].
        """
        # The corners are indexed as (x, y)
        ref_im = self.images[self.reference_img]
        ref_ul = [0, 0]
        ref_ur = [ref_im.shape[1], 0]
        ref_ll = [0, ref_im.shape[0]]
        ref_lr = [ref_im.shape[1], ref_im.shape[0]]
        ref_corners = [ref_ul, ref_ur, ref_ll, ref_lr]

        img_height = [ref_ul[1], ref_lr[1]]
        img_width = [ref_ul[0], ref_lr[0]]
        
        for ind in self.overlaps[self.reference_img]:
            h = self._compute_homography(ind)         

            for u, v in ref_corners:
                coord = h @ np.array([[u], [v], [1]])
                u_1 = (coord[0] / coord[2]).round().astype("int")
                v_1 = (coord[1] / coord[2]).round().astype("int")

                img_height.append(v_1)
                img_width.append(u_1)

        min_v = min(img_height)
        min_u = min(img_width)
        max_v = max(img_height)
        max_u = max(img_width)

        height = max_v - min_v
        width = max_u - min_u

        offset = np.array([min_u, min_v])
            
        return (height.item(), width.item(), offset)

    def _extend(self, img: int) -> None:
        """
        Pastes an image onto the composite.
        
        Args:
            img: The index of the image to incorporate.
        """
        h = self._compute_homography(img)
        self.composite[img] = self._perspective_transformation(self.images[img], h)

        print(f"Added image {img}")

    def _run(self) -> None:
        """Driver function for running the compositing process"""
        pasted_images = [self.reference_img]     
        
        while len(self.overlaps[self.reference_img]) > 0:
            # 1. Find the next image to extend with
            #    using the overlap graph
            next_img = self.overlaps[self.reference_img].pop()

            # 2. Extend with that image
            self._extend(next_img)
            pasted_images += [next_img]

            # 3. Update the overlap graph
            # TODO: Skip this step to only get images overlapping with reference img
            #self.overlaps[self.reference_img] = self.overlaps[self.reference_img].union({
            #    neighbor for neighbor in self.overlaps[next_img]
            #    if neighbor not in pasted_images
            #})
            pass

    def _compute_homography(self, img_ind: int) -> np.ndarray:
        """Computes the homography given an image and the reference image. Return the homography.
        
        Args:
            img_ind: Index for an image.
        """
        ref_kp = self.features[self.reference_img]
        img_kp = self.features[img_ind]

        ref_points = np.array([ref_kp.getKeypoints()[m.queryIdx].pt for m in self.matches[self.reference_img][img_ind]])
        img_points = np.array([img_kp.getKeypoints()[m.trainIdx].pt for m in self.matches[self.reference_img][img_ind]])

        h, _ = cv2.findHomography(img_points, ref_points, method=cv2.RANSAC)

        return h

    def _perspective_transformation(self, img: np.ndarray, h: np.ndarray):
        """Performs perspective transformation of an image onto the reference image, given the homography matrix between the image pair.
            Adds the transformed image to a image with (height, width) as the composite, given an offset based on the reference image.

        Args:
            img: Image that will be transformed.
            h: Homography matrix
        """
        warped_im = np.full((self.height, self.width, img.shape[2]), -1)   
        h_inv = np.linalg.inv(h)
        for u in range(warped_im.shape[1]):
            for v in range(warped_im.shape[0]):
                coord = h_inv @ np.array(
                [
                    u + self.offset[0],
                    v + self.offset[1],
                    1
                ]
                )
                u_1 = (coord[0] / coord[2]).round().astype("int")
                v_1 = (coord[1] / coord[2]).round().astype("int")
                if u_1 >= 0 and v_1 >= 0 and u_1 < img.shape[1] and v_1 < img.shape[0]:
                    warped_im[v, u] = img[v_1, u_1]

        return warped_im

