"""Finds a mask for the overlapping areas between the images in the panorama."""

import numpy as np

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
        self.overlap_dict, self.mask = self._find_overlap()


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
        assert idx < self.projected_images.shape[0]
        return self.overlap_dict[idx]


    def _find_overlap(self):
        """
        Finds where overlap occurs and updates the mask accordingly.

        Returns:
            mask: A RxC array with lists containting the indicies of all images having a valid pixel 
        """
        overlap_dict = {}

        for i in range(self.projected_images.shape[0]):
            overlap_dict[i] = []

        mask = np.all(self.projected_images >= 0, axis=3)

        for i in range(mask.shape[0]):
            for j in range(i+1,mask.shape[0]):
                temp_mask = np.logical_and(mask[i], mask[j])
                idx = np.argwhere(temp_mask)
                for k in range(idx.shape[0]):
                    overlap_dict[i].append([idx[k, 0], idx[k, 1], j])
                    overlap_dict[j].append([idx[k, 0], idx[k, 1], i])

        mask = np.any(mask, axis=0)

        return overlap_dict, mask


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
