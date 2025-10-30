#!/usr/bin/env python3
import argparse as ap
import tensorflow as tf
import os
import numpy as np
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import rembg
import cv2


class colors:
    GREEN = '\033[92m'
    ENDC = '\033[0m'


def render_plot(image_path: str, image, prediction: str):

    text = "===    DL classification    ===\n\nClass predicted : "

    fig, ax = plt.subplots(ncols=2, figsize=(10, 6), facecolor='black')

    fig.suptitle(f"Transformation of {image_path}")

    for i, (title, img) in enumerate(image.items()):
        ax[i].imshow(img)
        ax[i].set_title(title)
        ax[i].axis('off')
    for i in range(2):
        ax[i].set_position([0.05 + i * 0.5, 0.2, 0.4, 0.7])
    plt.figtext(0.5, 0.1, text, ha='center',
                fontsize=16, color='white', weight='bold')
    plt.figtext(0.5, 0.05, prediction, ha='center',
                fontsize=16, color='green', weight='bold')
    plt.show()
    plt.close()


def render(img_path: str, prediction: str):
    image, _, _ = pcv.readimage(img_path, mode='rgb')
    image_without_bg = rembg.remove(image)
    gray_scale = pcv.rgb2gray_lab(rgb_img=image_without_bg, channel='l')
    gaussian_blur = pcv.gaussian_blur(img=gray_scale, ksize=(3, 3))
    masked = pcv.apply_mask(img=image, mask=gaussian_blur, mask_color='white')
    render_plot(img_path, {"Original": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                "Masked": cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)},
                prediction)


def load_model(path: str):
    model: tf.keras.Model = tf.keras.models.load_model(path)
    return model


def main(model_path: str, image_path: str, classes: list[str], dorender: bool):
    if not os.path.isfile(model_path):
        raise Exception("Invalid model file.")

    if not os.path.exists(image_path):
        raise Exception("Invalid image path.")

    model: tf.keras.Model = load_model(model_path)

    image = None
    prediction = []
    if os.path.isfile(image_path):
        image = tf.keras.preprocessing.image.load_img(image_path,
                                                      target_size=(256, 256))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, axis=0)
        prediction.append(model.predict(image))
        if dorender:
            render(image_path, classes[prediction[0].argmax()])

    if os.path.isdir(image_path):
        image = []
        for img_file in os.listdir(image_path):
            image_full = os.path.join(image_path, img_file)
            if os.path.isdir(image_full):
                raise Exception("Image path :" + image_full +
                                " is a directory.")
            img = tf.keras.preprocessing.image.load_img(
                image_full, target_size=(256, 256))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.expand_dims(img, axis=0)
            image.append(img)
        image = tf.concat(image, axis=0)
        prediction.append(model.predict(image))

    print(f"\t{'\t'.join(classes)}")
    for i, predictions in enumerate(prediction):
        for j, predict in enumerate(predictions):
            print(f"Image {j}: [{'\t'.join(map(str, predict))}]")
            values = np.array(list(map(float, predict)))
            index = values.argmax()
            print(f"Predicted class: {classes[index]}")


if __name__ == "__main__":
    clas = ['Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab',
            'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']
    parser = ap.ArgumentParser(description="Predict image type with model")
    parser.add_argument("image", help="Path of image")
    parser.add_argument("model", help="Path of model save")
    parser.add_argument("-c", "--classes", default=clas, help="Classes Types")
    parser.add_argument("-r", "--dorender", default=True, help="Render image")

    args = parser.parse_args()
    try:
        main(args.model, args.image, args.classes, args.dorender)
    except Exception as e:
        print(e.__class__.__name__, e)
        exit(1)
