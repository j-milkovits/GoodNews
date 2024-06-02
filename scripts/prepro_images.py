from __future__ import absolute_import, division, print_function

import argparse
import json
import os
from random import seed

import h5py
import numpy as np
from imageio import imread
from PIL import Image


def main(params):
    with open(params["input_json"], "r") as f:
        imgs = json.load(f)["images"]

    seed(123)  # Make reproducible

    # Create output H5 file
    N = len(imgs)
    with h5py.File(params["output_h5"] + "_image.h5", "w") as f:
        dset = f.create_dataset(
            "images", (N, 3, 256, 256), dtype="uint8"
        )  # Space for resized images
        for i, img in enumerate(imgs):
            # Load the image
            img_path = os.path.join(params["images_root"], img["file_path"])
            I = imread(img_path)
            try:
                Ir = np.array(Image.fromarray(I).resize((256, 256), Image.ANTIALIAS))
            except Exception as e:
                print(
                    f'Failed resizing image {img["file_path"]} - see http://git.io/vBIE0',
                    e,
                )
                raise

            # Handle grayscale input images
            if len(Ir.shape) == 2:
                Ir = np.stack([Ir] * 3, axis=-1)

            # Transpose the image array to (3, 256, 256)
            Ir = Ir.transpose(2, 0, 1)
            dset[i] = Ir

            if i % 1000 == 0:
                print(f"Processing {i}/{N} ({i * 100.0 / N:.2f}% done)")

    print("Wrote", params["output_h5"] + "_image.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input JSON
    parser.add_argument(
        "--input_json",
        default="../data/data_news.json",
        help="Input JSON file to process into HDF5",
    )
    parser.add_argument(
        "--output_h5", default="../data/data_news", help="Output HDF5 file"
    )

    # Options
    parser.add_argument(
        "--images_root",
        default="../resized/",
        help="Root location in which images are stored, to be prepended to file_path in input JSON",
    )

    args = parser.parse_args()
    params = vars(args)  # Convert to ordinary dict
    print("Parsed input parameters:")
    print(json.dumps(params, indent=2))

    main(params)
