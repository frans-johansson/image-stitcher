"""Handles feature detection and matching for all images."""

import itertools
from typing import List
import cv2
import numpy as np

class FeatureHandler:
    """
    Takes a list of images and detects and stores the features for all images. 
    Matches the features to other iamges and stores matches in a dictionary.

    Members:
        image_features: a list with a lists of features, where the list index corresponds to the image the features belong to.
        feature_matches: a dictionary containing feature matches between pairs of images. 
        match_conf: a matching confidence threshold for best-of-2 matching.
        overlap_thresh: Minimum number of feature matches between two image pairs allowed.
    """

    def __init__(self, images: List[np.ndarray], match_conf: float = 0.5, overlap_thresh: int = 20):
        """
        Initializes a features handler by computing and matching features in a list of images.
        Detects features and matches and stores them upon initialization.

        Args:
            images: list of images
            match_conf: the matching confidence threshold for best-of-2 matching, defaults to 0.5
            overlap_thresh: the amount of matches required between two images for them to overlap, defaults to 20
        """
        self.match_conf = match_conf
        self.overlap_thresh = overlap_thresh
        self.image_features = self.detect_features(images)
        self.feature_matches = self.match_features()

    def detect_features(self, images: List[np.ndarray]):
        """
        Detects image features for all images in a list. 
        Stores all features in a list.

        Args:
            images: list of images
        """
        sift = cv2.SIFT_create()
        return cv2.detail.computeImageFeatures(sift, images) # NOTE: img_idx is zero for all images in feature_matches

    def match_features(self):
        """
        Matches features in a set of images to each other and stores matches in a dictionary.

        The dictionary of matches is a nested dictionary. The first key in the dictionary is an image index i. The item in the dictionary is another dictionary, 
        where the key is image index j making up image pair i->j. Connected to key j is a list of match objects of matches between i->j.
        
        Example showing the structure of the dictionary:
            {0: {1: [...]}, 1: {0: [...], 2: [...]}, 2: {1: [...]}}
        """
        match_dict = {}

        bo2nm = cv2.detail.BestOf2NearestMatcher_create(match_conf = self.match_conf)

        image_pairs = itertools.combinations(range(len(self.image_features)), 2)

        for i, j in image_pairs:
            i_matches = bo2nm.apply(self.image_features[i], self.image_features[j]).getMatches()
            j_matches = bo2nm.apply(self.image_features[j], self.image_features[i]).getMatches()
            
            if (len(i_matches) + len(j_matches)) < self.overlap_thresh:
                continue
            
            match_dict.setdefault(i, {})
            match_dict.setdefault(j, {})
            match_dict[i].setdefault(j, i_matches)
            match_dict[j].setdefault(i, j_matches)

        return match_dict