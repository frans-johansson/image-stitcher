"""Finds a mask for the overlapping areas between the images in the panorama."""

import numpy as np
import itertools

class OverlapMask:
    """
    Takes a projected panorama and creates a mask for all areas where there is an overlap between the individual images.

    Members:
        projected_images: An array of images that has been projected onto the panorama.
        mask: A mask containing the imformation about where overlaps occur.
    """

    def __init__(self, projected_images):
        """
        Initializes the overlap mask.

        Args:
            projected_images: A NxRxCxM array, where N is the number of images, R is the number of rows in each image,
            C is the number of columns in each image and M is the number of color chanels.
        """
        self.projected_images = projected_images
        assert self.projected_images.ndim == 4, "Projected images have incorrect dimensions"
        #self.index_mask, self.bool_mask = self._find_overlap_mask()
        self.overlap_data, self.mask = self._find_overlap()


    def get_mask(self):
        """
        Returns the overlap mask.
        """
        return np.copy(self.mask)

    def get_overlap(self, idx):
        """
        Returns a list of all pixels where there is an overlap with projected_image[idx]

        Args:
            idx: Image index of the image to find overlaps for.
        """
        assert idx < self.projected_images.shape[0], "There is no image " + str(idx)
        return self.overlap_data[idx]

    def _if_overlap(self, arr):
        """
        Checks if there is an overlap between any number of images in the current position.

        Args:
            Arr: An 1D bool array with the length of the number of images.

        Returns:
            A bool, returns True if there are at least two True values in the bool array, otherwise False.
        """
        return np.sum(arr) >= 2

    def _find_overlap(self):
        """
        Finds where overlap occurs and updates the mask accordingly.

        Returns:
            mask: A RxC array with lists containting the indicies of all images having a valid pixel 
        """
        overlap_data = {}

        for i in range(self.projected_images.shape[0]):
            overlap_data[i] = []

        mask = np.all(self.projected_images >= 0, axis=3)

        # Loop through all combinations of image pairs
        for i, j in list(itertools.combinations(range(mask.shape[0]), 2)):  # This loop will have length of the nunmber of combinations between images
                # Find indices where overlaps occur between the two images
                temp_mask = np.logical_and(mask[i], mask[j])
                idx = np.argwhere(temp_mask)

                # Add indices of pixel and overlapping image to the dictionary
                for r, c in idx:           # This for-loop will have length of the number of overlapping pixels
                    overlap_data[i].append([r, c, j])
                    overlap_data[j].append([r, c, i])

        # Apply the if_overlap function to each pixel, value becomes True if
        # there are at least two images that has values in that pixel, otherwise False
        mask = np.apply_along_axis(self._if_overlap, 0, mask)

        return overlap_data, mask


    def add_nonoverlap_to_image(self, image):
        """
        Adds the non-overlapping parts of all images to the input image

        Args:
            image: A RxCx3 array, where RxC are the same dimensions as the mask.

        Returns:
            image: Updated version of image with non-overlapping image data added.
        """
        for r, c in np.ndindex(self.mask.shape):
            if self.num_overlap(r,c) == 1:
                image[r, c, :] = self.projected_images[self.mask[r,c][0], r, c, :]

        return image
