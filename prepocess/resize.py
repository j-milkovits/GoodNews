import argparse
import os

from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    # return image.resize(size, Image.ANTIALIAS)
    # removed ANTIALIAS option here, seems to be deprecated
    return image.resize(size)


def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_dir, image_name)
        try:
            with open(image_path, "rb") as f:
                with Image.open(f) as img:
                    img = resize_image(img, size)
                    img = img.convert("RGB")
                    output_path = os.path.join(output_dir, image_name)
                    img.save(output_path, "JPEG")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
        print(f"Processing images: {i + 1}/{num_images} images processed...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images to a specified size.")
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="The size to which images will be resized.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="../images/",
        help="Root directory containing images and where resized images will be saved.",
    )
    args = parser.parse_args()

    image_size = [args.img_size, args.img_size]
    resize_images(
        os.path.join(args.root, "images"),
        os.path.join(args.root, "resized"),
        size=image_size,
    )
