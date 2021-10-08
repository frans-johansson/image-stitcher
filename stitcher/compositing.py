"""Handles transformations and compositing of multiple images onto a single, seamless panorama"""

from typing import Tuple

import cv2
from numpy.lib.shape_base import expand_dims
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

        self.overlaps = {i: {j for j in self.matches[i].keys()} for i in self.matches.keys()}
        self.reference_img = self._find_reference_img()
        self.width, self.height, self.offset = self._compute_bounding_box()
        self.composite = np.full([self.num_images, self.width, self.height, 3], -1)
        
        self.images = images.high_res_images #NOTE: Should this be high res?

        self.composite = np.expand_dims(self.images[self.reference_img], axis = 0)

        self._run()


    def _find_reference_img(self) -> int:
        """
        Finds the index of the reference image for the composite based on the
        overlap graph

        Returns:
            reference_img: The index of the reference image in the ImageCollection
        """
        max_len = 0
        ref = 0
        for i in self.matches:
            numb_pairs = len(self.matches[i])
            if numb_pairs > max_len:
                max_len = numb_pairs
                ref = i

        #b = [(len(self.matches[i]), i) for i, m in enumerate(self.matches)]

        #b = [i for i in self.matches if len(self.matches[i]) > max_len ]

        print(f"Starting at image {ref}")
        return 0

    def _compute_bounding_box(self) -> Tuple[int, int, np.ndarray]:
        """
        Computes the dimensions and offset for the final composite

        Returns:
            (width, height, offset): The width and height given as integers. The offset given as
                a two-dimensional array with [offset x, offset y].
        """
        # TODO: Implement
        return (1000, 2000, np.array([50, 50]))

    def _extend(self, img: int) -> None:
        """
        Pastes an image onto the composite.
        
        Args:
            img: The index of the image to incorporate.
        """
        # TODO: Implement
        h = self._compute_homography(img)
        a = self._perspective_transformation(img, h)
        a = np.expand_dims(a, axis = 0)
        np.concatenate((self.composite, a), axis = 0)

        image = self.composite.astype(np.uint8)

        # cv2.imshow("Composite image", image[0])
        # cv2.waitKey(0)

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
            self.overlaps[self.reference_img] = self.overlaps[self.reference_img].union({
                neighbor for neighbor in self.overlaps[next_img]
                if neighbor not in pasted_images
            })
            pass

    def _compute_homography(self, img_ind: int) -> np.ndarray: #TODO: np.ndarray correct return type?
        """Computes the homography given an image and the reference image. Return the homography.
        
        Args:
            img_ind: Index for an image.
        """

        if(img_ind in self.matches[self.reference_img]):
            ref_im = self.reference_img
        else:
            ref_im = next(iter(self.matches[img_ind]))

        ref_kp = self.features[ref_im]
        img_kp = self.features[img_ind]

        ref_points = np.array([ref_kp.getKeypoints()[m.queryIdx].pt for m in self.matches[ref_im][img_ind]])
        img_points = np.array([img_kp.getKeypoints()[m.trainIdx].pt for m in self.matches[ref_im][img_ind]])

        h, _ = cv2.findHomography(ref_points, img_points, method=cv2.RANSAC)

        return h

    def _perspective_transformation(self, img_ind: int, h: np.ndarray):
        """Performs perspective transformation of an image onto the reference image, given the homography matrix between the image pair.
        
        Args:
            img_ind: Index for the image that will be transformed.
            h: Homography matrix
        """

        ref_shape = self.images[self.reference_img].shape
        ref_ul = [0, 0]
        ref_lr = [ref_shape[0], ref_shape[1]]
        ref_ur = [0, ref_shape[1]]
        ref_ll = [ref_shape[0], 0]
        ref_corners = [ref_ul, ref_ur, ref_ll, ref_lr]

        # Project image corners
        h_inv = np.linalg.inv(h)
        proj_im_r = []
        proj_im_c = []
        for c in ref_corners:
            coord = h_inv @ np.array([[c[1]], [c[0]], [1]])
            u_1 = int(coord[0] / coord[2])
            v_1 = int(coord[1] / coord[2])
            proj_im_c.append(u_1)
            proj_im_r.append(v_1)

        v0_ul = min(proj_im_r)
        u0_ul = min(proj_im_c)
        v0_lr = max(ref_shape[0] - 1, max(proj_im_r))
        u0_lr = max(ref_shape[1] - 1, max(proj_im_c))
        new_r = v0_lr - v0_ul + 1
        new_c = u0_lr - u0_ul + 1

        r0 = ref_shape[0]
        c0 = ref_shape[1]

        warped_im = np.full((new_r, new_c, 3), -1)
        h_inv = np.linalg.inv(h)
        for u in range(warped_im.shape[0]):
            for v in range(warped_im.shape[1]):
                coord = h_inv @ np.array([[u], [v], [1]])
                u_1 = int(coord[0] / coord[2])
                v_1 = int(coord[1] / coord[2])
                try:
                    warped_im[v, u] = self.images[img_ind][v_1, u_1]
                    #warped_im[v, u] = img2[v, u]
                except IndexError:
                    pass

        image = warped_im.astype(np.uint8)

        cv2.imshow(f"Warped im {(img_ind)}", image)
        cv2.waitKey(0)

        return warped_im

        

