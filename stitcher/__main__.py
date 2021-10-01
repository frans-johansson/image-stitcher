"""Main entry point for the image stitching program"""

from argparse import ArgumentParser
from pathlib import Path
import cv2
import itertools
from stitcher.images import ImageCollection
from stitcher.features import FeatureHandler

if __name__ == "__main__":
    # Handle command line arguments
    argparser = ArgumentParser(
        prog="Panorama image stitcher",
        description=(
            "Stitches together images from a given directory into a single, seamless panorama image. "
            "The input images are expected to be .png files, but does not have to be ordered or named "
            "in any particular way."
        )
    )

    argparser.add_argument(
        "--path", help="Path to directory with input images", type=Path
    )
    argparser.add_argument(
        "--output", help="Name of the output panorama image", type=Path
    )
    argparser.add_argument(
        "--rescale-factor", help="Rescale factor applied to each image in the output",
        type=float, default=0.8
    )

    args = argparser.parse_args()

    # Load image collection from the filesystem
    image_collection = ImageCollection(
        path=args.path,
        high_res_scale=args.rescale_factor,
        low_res_scale=args.rescale_factor / 2
    )

    # TEST: Display all the images
    cv2.namedWindow("Test image collection")

    for image in image_collection.low_res_images:
        cv2.imshow("Test image collection", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # END TEST

    # Detect and match features
    feature_handler = FeatureHandler(image_collection.low_res_images)

    # TEST: Visualize some correspondences
    matches = feature_handler.feature_matches
    image_pairs = itertools.permutations(range(len(image_collection.low_res_images)), 2)

    cv2.namedWindow("Test features")

    for pair in image_pairs:
        ms = matches[pair][0].getMatches()
        sorted(ms, key=lambda x: x.distance)
        
        kp1 = feature_handler.image_features[pair[0]].getKeypoints()
        kp2 = feature_handler.image_features[pair[1]].getKeypoints()
        img1 = image_collection.low_res_images[pair[0]]
        img2 = image_collection.low_res_images[pair[1]]

        print(f"Showing {len(ms)} matches")

        match_img = cv2.drawMatches(img1, kp1, img2, kp2, ms, img2, flags=2)
        cv2.imshow("Test features", match_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # END TEST

    # ms = matches.getMatches()
    # sorted(ms, key=lambda x: x.distance)
    # kp1, kp2 = [f.getKeypoints() for f in features]

    # match_img = cv2.drawMatches(img1, kp1, img2, kp2, ms[:20], img2, flags=2)
    # cv2.imwrite("test.png", match_img)

    

