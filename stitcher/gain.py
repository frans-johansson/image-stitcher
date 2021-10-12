"""Calculates and applies a gain to all images."""

import numpy as np
import itertools
from scipy.optimize import least_squares
from stitcher.compositing import PanoramaCompositor

class GainCompensator:
    """
    Takes a composite of images and calculates and applies the gain for each one.

    Members:
        composite: An array of images that has been perspective projected.
        overlap_data: A dictionary holding the values of all overlapping pixels between the images.
    """

    def __init__(self, compositor: PanoramaCompositor):
        """
        Initializes the gain compensator.

        Args:
            composite: A NxRxCx3 array, where N is the number of images, R is the number of rows in each image
            and C is the number of columns in each image.
        """
        self.compositor = compositor
        self.composite = compositor.composite
        assert self.composite.ndim == 4, "Projected images have incorrect dimensions"
        self.overlap_data = self._find_overlap()

    def _find_overlap(self):
        """
        Finds where overlap occurs and updates the mask accordingly.

        Returns:
            mask: A RxC array with lists containting the indicies of all images having a valid pixel 
        """
        overlap_data = {}

        for i in range(self.composite.shape[0]):
            overlap_data[i] = {}

        mask = np.all(self.composite >= 0, axis=3)

        # Loop through all combinations of image pairs
        for i, j in list(itertools.combinations(range(mask.shape[0]), 2)):
                # Find indices where overlaps occur between the two images
                temp_mask = np.logical_and(mask[i], mask[j])
                idx = np.argwhere(temp_mask)

                overlap_data[i][j] = []
                overlap_data[j][i] = []

                # Add indices of pixel and overlapping image to the dictionary
                for r, c in idx:
                    overlap_data[i][j].append(self.composite[i, r, c, :])
                    overlap_data[j][i].append(self.composite[j, r, c, :])

        self.mask = mask

        return overlap_data


    def _calculate_gain_error(self, g):
        """
        Calculate the gain errror.

        Args:
            g: A matrix of the approximated gain values.

        Returns:
            The calculated error value.
        """
        error = 0
        sigma_n = 10
        sigma_g = 0.1

        for i, j in list(itertools.permutations(range(self.composite.shape[0]), 2)):
            if len(self.overlap_data[i][j]) != 0:

                mean_ij = np.mean(np.sum(self.overlap_data[i][j], axis=1)/3)
                mean_ji = np.mean(np.sum(self.overlap_data[j][i], axis=1)/3)

                error += len(self.overlap_data[i][j]) * ((g[i] * mean_ij - g[j] * mean_ji)**2 / sigma_n**2 + (1 - g[i])**2 / sigma_g**2)

        return error * 0.5

    def _calculate_gain(self):
        """
        Calculate the gain using a least squares solver.

        Returns:
            A matrix with the approxiamted gain values.
        """
        # Initialize the gain values to ones.
        g = np.ones((self.composite.shape[0]))

        # Run minimizer
        res = least_squares(
            self._calculate_gain_error,
            g,
            ftol=1e-3,
            method="trf",
        )

        return res.x


    def gain_compensate(self):
        """
        Perform the gain compensation by calculating the gain values, multiplying them with the images and
        map them back to range 0-255, with values -1 where there are no pixel values.

        Returns:
            The gain compensated images, same size as self.composite.
        """
        # Calculate the gain values
        g = self._calculate_gain()

        # Multiply with the gain in pixels that do not have value -1, let those pixels keep the value -1
        imgs = np.where(self.composite != -1, g[:, np.newaxis, np.newaxis, np.newaxis] * self.composite, -1)

        # Rescale the values to be integers between 0-255, except for non-pixel values that are -1
        imgs = (imgs / np.max(imgs)) * 255
        imgs[imgs < 0] = - 1
        imgs = np.floor(imgs)

        self.compositor.composite = imgs

        return self.compositor