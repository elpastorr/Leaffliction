#!/usr/bin/env python3
import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
import argparse


def get_new_filename(filename: str, suffix: str) -> str:
    split_filename = filename.split(".")
    split_filename[0] = split_filename[0] + suffix
    return ".".join(split_filename)


def augment_dir(src_dir: str, aug_dir: str):
    if not os.path.exists(aug_dir):
        os.makedirs(aug_dir)

    for file in os.listdir(src_dir):
        img_path = os.path.join(src_dir, file)
        if os.path.isfile(img_path):
            augment(img_path, aug_dir)


def augment(img_path: str, aug_dir: str):
    try:
        original_img, path, filename = pcv.readimage(img_path)
    except Exception as e:
        print(e.__class__.__name__, e)

    if not os.path.exists(aug_dir):
        os.makedirs(aug_dir)

    rotate(original_img, filename, aug_dir, rot_angle=45, crop=True)

    skew(original_img, filename, aug_dir)

    shear(original_img, filename, aug_dir)

    crop(original_img, filename, aug_dir, x=80, y=80, h=100, w=100)

    color_invertion(original_img, filename, aug_dir)

    distortion(original_img, filename, aug_dir)

    # flip(original_img, filename, aug_dir, direction="vertical")


def rotate(img: np.ndarray, filename, aug_dir, rot_angle=45, crop=True):
    rotate_filename = get_new_filename(filename, "_Rotate")

    rotated = pcv.transform.rotate(img=img, rotation_deg=rot_angle, crop=crop)
    pcv.print_image(rotated, filename=os.path.join(aug_dir, rotate_filename))


def skew(img: np.ndarray, filename, aug_dir, skew_factor=0.3):
    skew_filename = get_new_filename(filename, "_Skew")

    skew_matrix = np.array([[1, skew_factor, 0], [-skew_factor, 1, 0]])

    rows, cols, _ = img.shape

    skewed = cv2.warpAffine(img, skew_matrix, (cols, rows))
    pcv.print_image(skewed, filename=os.path.join(aug_dir, skew_filename))


def shear(img: np.ndarray, filename, aug_dir, shear_factor=0.3):
    shear_filename = get_new_filename(filename, "_Shear")

    shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])

    rows, cols, _ = img.shape

    sheared = cv2.warpAffine(img, shear_matrix, (cols, rows))
    pcv.print_image(sheared, filename=os.path.join(aug_dir, shear_filename))


def crop(img: np.ndarray, filename, aug_dir, x, y, h, w):
    crop_filename = get_new_filename(filename, "_Crop")

    rows, cols, _ = img.shape

    cropped = pcv.crop(img=img, x=x, y=y, h=h, w=w)
    resized = cv2.resize(cropped, dsize=(rows, cols))
    pcv.print_image(resized, filename=os.path.join(aug_dir, crop_filename))


def color_invertion(img: np.ndarray, filename, aug_dir):
    invert_filename = get_new_filename(filename, "_Inverted")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, :] = 255 - hsv_img[:, :, :]

    inverted = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    pcv.print_image(inverted, filename=os.path.join(aug_dir, invert_filename))


def distortion(img: np.ndarray, filename, aug_dir):
    distortion_filename = get_new_filename(filename, "_Distortion")

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = hsv_img[:, :, 1] * 2

    distorted = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    pcv.print_image(distorted, filename=os.path.join(aug_dir,
                                                     distortion_filename))


# def flip(img: np.ndarray, filename, aug_dir, direction="vertical"):
#     flip_filename = get_new_filename(filename, "_Flip")

#     flipped = pcv.flip(img=img, direction=direction)
#     pcv.print_image(flipped, filename=os.path.join(aug_dir, flip_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Augmentation to" +
                                     "an image or a directory of images.")

    parser.add_argument("image", help="Path to image file")

    parser.add_argument("-dst", "--destination", default="augmented_directory",
                        help="Destination dir for augmented images")

    args = parser.parse_args()

    try:
        if not os.path.exists(args.image):
            raise FileNotFoundError(args.image + " does not exist.")
        if os.path.isfile(args.image):
            augment(args.image, args.destination)
        elif os.path.isdir(args.image):
            augment_dir(args.image, args.destination)
    except Exception as e:
        print(e.__class__.__name__, e)
        exit(0)
