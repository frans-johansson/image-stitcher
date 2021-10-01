"""Handles feature detection and matching for all images."""

import cv2
import itertools

class FeatureHandler:
    """
    Takes a list of images and detects and stores the features for all images. 
    Then matches the features to other images.

    Members:
        image_features: A list with a lists of features, where the list index corresponds to the image the features belong to.
        feature_matches: A dictionary containing feature matches between pai
    """

    def __init__(self, images: list):
        """
        Initializes a features handler by computing and matching features in a list of images.
        Detects features and matches and stores them upon initialization.

        Args:
            images: list of images
        """
        self.image_features = self.detect_features(images)
        self.feature_matches = self.match_features()

    def detect_features(self, images: list):
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

        """

        match_dict = {}

        bo2nm = cv2.detail.BestOf2NearestMatcher_create(match_conf = 0.5)

        image_pairs = itertools.permutations(range(len(self.image_features)), 2)

        for i, j in image_pairs:
            matches = [bo2nm.apply(self.image_features[i], self.image_features[j])]
            match_dict[(i, j)] = matches

        return match_dict