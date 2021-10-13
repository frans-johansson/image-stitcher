"""Main entry point for the image stitching program"""

import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
import cv2
from stitcher.blending import LinearBlender, NoBlender
from stitcher.compositing import PanoramaCompositor
from stitcher.images import ImageCollection
from stitcher.features import FeatureHandler
from stitcher.gain import GainCompensator

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
        "--output", help="Name of the output panorama image. Will have a timestamp appended", type=Path
    )
    argparser.add_argument(
        "--rescale-factor", help="Rescale factor applied to each image in the output",
        type=float, default=0.8
    )

    args = argparser.parse_args()

    # Load image collection from the filesystem
    image_collection = ImageCollection(
        path=args.path,
        rescale_factor=args.rescale_factor
    )

    # TEST: Display all the images
    # cv2.namedWindow("Test image collection")

    # for image in image_collection.low_res_images:
    #     cv2.imshow("Test image collection", image)
    #     cv2.waitKey(0)

    # cv2.destroyAllWindows()
    # END TEST

    # Detect and match features
    feature_handler = FeatureHandler(image_collection.low_res_images)

    # TEST: Visualize some correspondences
    # matches = feature_handler.feature_matches

    # for i, i_matches in matches.items():
    #     for j, ij_matches in i_matches.items():
    #         ms = ij_matches
    #         sorted(ms, key=lambda x: x.distance)

    #         kp1 = feature_handler.image_features[i].getKeypoints()
    #         kp2 = feature_handler.image_features[j].getKeypoints()
    #         img1 = image_collection.low_res_images[i]
    #         img2 = image_collection.low_res_images[j]

    #         print(f"Showing {len(ms)} matches")

    #         match_img = cv2.drawMatches(img1, kp1, img2, kp2, ms, img2, flags=2)
    #         cv2.imshow(f"Test features, images: {(i, j)}", match_img)
    #         cv2.waitKey(0)

    # cv2.destroyAllWindows()
    # END TEST

    compositor = PanoramaCompositor(image_collection, feature_handler)
    compositor = GainCompensator(compositor).gain_compensate()

    no_blend_result = NoBlender(compositor).render()
    linear_result = LinearBlender(compositor).render()

    timestamp = dt.datetime.now().strftime("%d%m%y_%H%M")
    cv2.imwrite(f"img/{args.output}_{timestamp}_linear.jpg", linear_result)
    cv2.imwrite(f"img/{args.output}_{timestamp}_no_blend.jpg", no_blend_result)
