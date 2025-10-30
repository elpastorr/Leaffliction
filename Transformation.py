#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import sys
import argparse
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


def process_directory(src, dst):
    """
    Transform all the images in src directory and save them in dst directory
    """

    if not os.path.isdir(src):
        print(f"Error: Source directory '{src}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(dst):
        os.makedirs(dst)

    GREEN = "\033[92m"
    RESET = "\033[0m"
    print(f"{GREEN}Transformation phase, creating {dst} from {src}:\n{RESET}")

    for file in os.listdir(src):
        if file.lower().endswith(('.jpg')):
            process_file(os.path.join(src, file), dst)


def process_file(image_path, dst=None):
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)

    # Read the image from the given path
    image, _, _ = pcv.readimage(image_path, mode='rgb')

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

    if dst is None:
        render_plot(image_path, images)
        plot_histogram(image, kept_mask)

    else:
        for label, img in images.items():
            cv2.imwrite(os.path.join(
                        dst, (label + '_' +
                              image_path[image_path.rfind('/')+1:])),
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
                                     "an image or a directory of images.")

    parser.add_argument("-src", "--source", help="Source directory of images")

    parser.add_argument("-dst", "--destination",
                        help="Destination directory for transformed images")

    parser.add_argument("image", nargs='?', help="Path to image file")

    args = parser.parse_args()

    if args.source and args.destination:
        process_directory(args.source, args.destination)
    elif args.image and not args.source and not args.destination:
        process_file(args.image)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
