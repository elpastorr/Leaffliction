#!/usr/bin/env python3
import argparse
import tensorflow as tf
import os
import numpy as np


def load_model(path: str):
    model: tf.keras.Model = tf.keras.models.load_model(path)
    return model


def main(model_path: str, image_path: str, classes: list[str]):
    if (not os.path.isfile(model_path)):
        print("Invalid model file.")
        return
    
    if (not os.path.exists(image_path)):
        print("Invalid image path.")
        return
    
    model: tf.keras.Model = load_model(model_path)
    
    image = None
    prediction = []
    if (os.path.isfile(image_path)):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.expand_dims(image, axis=0)
        prediction.append(model.predict(image))

    if (os.path.isdir(image_path)):
        image = []
        for img_file in os.listdir(image_path):
            img = tf.keras.preprocessing.image.load_img(os.path.join(image_path, img_file), target_size=(256, 256))
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
    parser = argparse.ArgumentParser(description="Predict an image type with a given model.")
    parser.add_argument("image", help="Path of image")
    parser.add_argument("model", help="Path of model save")
    parser.add_argument("-c", "--classes", default=['Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab', 'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot'], help="Classes Types")
    args = parser.parse_args()
    main(args.model, args.image, args.classes)
