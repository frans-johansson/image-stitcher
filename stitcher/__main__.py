"""Main entry point for the image stitching program"""

import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
import cv2
from stitcher.blending import LinearBlender, MultiBandBlender, NoBlender
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
    argparser.add_argument(
        "--bands", help="The number of bands to compute for multi-band blending",
        type=int, default=3
    )

    args = argparser.parse_args()

    # Load image collection from the filesystem
    image_collection = ImageCollection(
        path=args.path,
        rescale_factor=args.rescale_factor
    )

    # Detect and match features
    feature_handler = FeatureHandler(image_collection.low_res_images)

    # Compositing and gain compensation
    compositor = PanoramaCompositor(image_collection, feature_handler)
    compositor = GainCompensator(compositor).gain_compensate()

    # Render results with blending
    multi_band_result = MultiBandBlender(compositor).render(bands=args.bands)
    no_blend_result = NoBlender(compositor).render()
    linear_result = LinearBlender(compositor).render()

    # Save the results
    timestamp = dt.datetime.now().strftime("%d%m%y_%H%M")
    cv2.imwrite(f"img/{args.output}_{timestamp}_linear.jpg", linear_result)
    cv2.imwrite(f"img/{args.output}_{timestamp}_no_blend.jpg", no_blend_result)
    cv2.imwrite(f"img/{args.output}_{timestamp}_multi_band.jpg", multi_band_result)
