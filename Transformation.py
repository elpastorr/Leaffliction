#!/usr/bin/env python3
import os
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import rembg
from plantcv import plantcv as pcv


def create_roi(image, masked, filled):
    """
    Create an image with the Region of Interest (ROI) rectangle
    on the image and the mask.
    """
    # Define the dimensions of the ROI rectangle
    height, width, _ = image.shape

    roi = pcv.roi.rectangle(img=masked, x=0, y=0, h=height, w=width)

    kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type='partial')

    roi_image = image.copy()
    roi_image[kept_mask != 0] = (0, 255, 0)  # Green overlay for kept area

    x, y, w, h = cv2.boundingRect(kept_mask)

    cv2.rectangle(roi_image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return roi_image, kept_mask


def create_pseudolandmarks_image(image, kept_mask):
    """
    Create an image with pseudolandmarks drawn.
    """
    pseudo_landmarks = image.copy()
    top_x, bottom_x, center = pcv.homology.x_axis_pseudolandmarks(
        img=pseudo_landmarks, mask=kept_mask, label="default")

    for i in range(len(top_x)):
        top_point = (int(top_x[i][0][0]), int(top_x[i][0][1]))
        bottom_point = (int(bottom_x[i][0][0]), int(bottom_x[i][0][1]))
        center_point = (int(center[i][0][0]), int(center[i][0][1]))
        cv2.circle(pseudo_landmarks, top_point, 5, (255, 0, 0), -1)
        cv2.circle(pseudo_landmarks, bottom_point, 5, (255, 0, 255), -1)
        cv2.circle(pseudo_landmarks, center_point, 5, (0, 100, 255), -1)

    return pseudo_landmarks


def render_plot(image_path, images):
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(16, 9))

    fig.suptitle(f"Transformation of {image_path}")

    for (label, img), axe in zip(images.items(), ax.flat):
        axe.imshow(img)
        axe.set_title(label)
        axe.set(xticks=[], yticks=[])
        axe.label_outer()

    plt.show()
    plt.close()


def plot_stat(key, val):
    y = pcv.outputs.observations['default_1'][key]['value']
    x = [i * val for i in pcv.outputs.observations['default_1'][key]['label']]

    if key == "hue_frequencies":
        x = x[:int(255/2)]
        y = y[:int(255/2)]
    if (key == "blue-yellow_frequencies" or
       key == "green-magenta_frequencies"):
        x = [x + 128 for x in x]
    plt.plot(x, y, label=key)


def plot_histogram(image, kept_mask):
    """
    Plot the histogram of the color image
    """

    dict_labels = {
        "blue_frequencies": 1,
        "green_frequencies": 1,
        "green-magenta_frequencies": 1,
        "lightness_frequencies": 2.55,
        "red_frequencies": 1,
        "blue-yellow_frequencies": 1,
        "hue_frequencies": 1,
        "saturation_frequencies": 2.55,
        "value_frequencies": 2.55
    }

    label, _ = pcv.create_labels(mask=kept_mask)
    pcv.analyze.color(rgb_img=image,
                      colorspaces="all", labeled_mask=label, label="default")

    plt.subplots(figsize=(16, 9))

    for key, val in dict_labels.items():
        plot_stat(key, val)

    plt.legend()

    plt.title("Color Histogram")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of pixels (%)")

    plt.show()
    plt.close()


def process_directory(src, dst, max_workers=None):
    """
    Transform all the images in src directory and save them in dst directory
    """

    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.is_dir():
        print(f"Error: Source directory '{src}' does not exist.")
        sys.exit(1)
    dst_path.mkdir(parents=True, exist_ok=True)

    GREEN = "\033[92m"
    RESET = "\033[0m"
    print(f"{GREEN}Transformation phase, creating {dst} from {src}:\n{RESET}")

    tasks = []
    for directory in src_path.iterdir():
        if not directory.is_dir():
            continue
        target_dir = dst_path / directory.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_path in directory.glob("*.jpg"):
            tasks.append((str(file_path), str(target_dir)))

    if not tasks:
        print(f"No JPG images found in '{src}'.")
        return

    total = len(tasks)
    max_workers = max(1, min(total, max_workers or (os.cpu_count() or 1)))

    if max_workers == 1:
        for idx, (image_path, target_dir) in enumerate(tasks, 1):
            try:
                process_file(image_path, target_dir)
            except Exception as exc:
                print(f"\nError while processing '{image_path}': {exc}",
                      file=sys.stderr)
            finally:
                print(f"\rProcessed {idx}/{total} images", end="", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_file, image_path, target_dir): image_path
                for image_path, target_dir in tasks
            }

            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as exc:
                    print(f"\nError while processing '{futures[future]}': {exc}",
                          file=sys.stderr)
                finally:
                    print(f"\rProcessed {idx}/{total} images", end="", flush=True)

    print("\nDone.")


def process_file(image_path, dst):
    image_path = Path(image_path)
    dst_path = Path(dst) if dst is not None else None

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file '{image_path}' does not exist.")

    # Read the image from the given path
    image, _, _ = pcv.readimage(str(image_path), mode='rgb')

    image_without_bg = rembg.remove(image)

    gray_scale = pcv.rgb2gray_lab(rgb_img=image_without_bg, channel='l')

    thresh = pcv.threshold.binary(gray_img=gray_scale,
                                  threshold=35, object_type='light')

    filled = pcv.fill(bin_img=thresh, size=200)

    # Apply Gaussian blur to the image
    gaussian_blur = pcv.gaussian_blur(img=filled, ksize=(3, 3))

    # Remove background of image using the mask
    masked = pcv.apply_mask(img=image, mask=gray_scale, mask_color='white')

    # Define Region of Interest (ROI) on the image
    roi_image, kept_mask = create_roi(image, masked, filled)

    # Analyze the image to extract shape and color information
    analyzed_image = pcv.analyze.size(img=image, labeled_mask=kept_mask)

    # Create an image with pseudolandmarks drawn
    pseudo_landmarks = create_pseudolandmarks_image(image, kept_mask)

    images = {
        "Original": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        "Gaussian Blur": cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB),
        "Mask": cv2.cvtColor(masked, cv2.COLOR_BGR2RGB),
        "ROI": cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB),
        "Analyzed": cv2.cvtColor(analyzed_image, cv2.COLOR_BGR2RGB),
        "Pseudolandmarks": cv2.cvtColor(pseudo_landmarks, cv2.COLOR_BGR2RGB),
    }

    if dst_path is None:
        render_plot(image_path, images)
        plot_histogram(image, kept_mask)

    else:
        for label, img in images.items():
            output_name = f"{label}_{image_path.name}"
            cv2.imwrite(str(dst_path / output_name),
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    """
    This program is designed to perform transformations on image(s).
    It accepts an image path or source directory containing images
    and a destination directory where the transformed images will be saved.
    Usage:
        Transformation.py (-src) path/to/image(s) (-dst path/to/destination)
    Arguments:
        -src,  Path to source directory containing images to be transformed.
        -dst,  Path to directory where transformed images will be saved.
    """

    parser = argparse.ArgumentParser(description="Apply Transformation to" +
                                     " an image or a directory of images.")

    parser.add_argument("-src", "--source", help="Source directory of images")

    parser.add_argument("-dst", "--destination",
                        help="Destination directory for transformed images")

    parser.add_argument("-j", "--jobs", type=int, default=0,
                        help="Number of parallel workers to use (default: cpu count)")

    parser.add_argument("image", nargs='?', help="Path to image file")

    args = parser.parse_args()

    try:
        if args.source and args.destination:
            jobs = args.jobs if args.jobs > 0 else None
            process_directory(args.source, args.destination, jobs)
        elif args.image and not args.source and not args.destination:
            process_file(args.image, None)
        else:
            parser.print_help()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
