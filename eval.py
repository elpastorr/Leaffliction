import os
import argparse
from typing import List, Dict, Optional
import sys

import numpy as np
import tensorflow as tf


def get_true_class(image_path: str) -> str:
    """Extract true class from image path (parent directory name)"""
    return os.path.basename(os.path.dirname(image_path))


def get_images_with_classes(test_dir: str) -> List[tuple]:
    """Get list of (image_path, true_class) from test directory"""
    image_paths = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                true_class = get_true_class(image_path)
                image_paths.append((image_path, true_class))
    return image_paths


_MODEL_CACHE: Dict[str, tf.keras.Model] = {}


def get_or_load_model(model_path: str) -> tf.keras.Model:
    """Load a TensorFlow model once and reuse subsequent calls."""
    cached = _MODEL_CACHE.get(model_path)
    if cached is not None:
        return cached

    try:
        print(f"Loading model from {model_path}...", flush=True)
        model = tf.keras.models.load_model(model_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load model '{model_path}': {exc}") from exc

    _MODEL_CACHE[model_path] = model
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and convert an image to a model-ready numpy array."""
    try:
        image = tf.keras.utils.load_img(image_path, target_size=(256, 256))
        array = tf.keras.utils.img_to_array(image)
        return array
    except Exception as exc:
        raise RuntimeError(f"Unable to preprocess image '{image_path}': {exc}") from exc


def predict_batch(image_paths: List[str], model: tf.keras.Model) -> np.ndarray:
    """Run model predictions on a batch of image paths."""
    if not image_paths:
        return np.empty((0,))

    batch = np.stack([preprocess_image(path) for path in image_paths], axis=0)
    predictions = model.predict(batch, verbose=0)
    return predictions


def run_pred(image_path: str, model_path: str, classes: List[str]) -> Optional[str]:
    """Predict class for a single image using an in-process model cache."""
    try:
        model = get_or_load_model(model_path)
        prediction = predict_batch([image_path], model)
        if prediction.size == 0:
            return None

        predicted_index = int(np.argmax(prediction[0]))
        return classes[predicted_index]
    except Exception as exc:
        print(f"Error processing {image_path}: {exc}", file=sys.stderr)
        return None


def evaluate_model(test_dir: str, model_path: str, classes: List[str],
                   batch_size: int = 32) -> Dict:
    """Evaluate model accuracy on test directory"""
    print(f"\nEvaluating model on {test_dir}...")

    # Get all test images and their true classes
    test_images = get_images_with_classes(test_dir)
    if not test_images:
        print(f"No images found in {test_dir}")
        return None

    total = len(test_images)
    correct = 0
    class_correct: Dict[str, int] = {cls: 0 for cls in classes}
    class_total: Dict[str, int] = {cls: 0 for cls in classes}
    batch_size = max(1, batch_size)

    try:
        model = get_or_load_model(model_path)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return None

    # Process each image
    processed = 0
    for start in range(0, total, batch_size):
        batch = test_images[start:start + batch_size]
        image_paths = [image_path for image_path, _ in batch]
        true_classes = [true_class for _, true_class in batch]

        print(f"\rProcessing image {processed + len(batch)}/{total}...",
              end="", flush=True)

        try:
            predictions = predict_batch(image_paths, model)
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            processed += len(batch)
            continue

        for true_class, prediction in zip(true_classes, predictions):
            class_total[true_class] = class_total.get(true_class, 0) + 1

            predicted_index = int(np.argmax(prediction))
            predicted_class = classes[predicted_index]

            if predicted_class == true_class:
                correct += 1
                class_correct[true_class] = class_correct.get(true_class, 0) + 1

        processed += len(batch)

    print("\n")

    accuracy = (correct / total) * 100 if total > 0 else 0
    per_class_accuracy = {
        cls: (class_correct[cls] / class_total[cls] * 100)
        if class_total[cls] > 0 else 0
        for cls in classes
    }

    return {
        "total_images": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy
    }


def print_results(results: Dict) -> None:
    """Print evaluation results in a formatted way"""
    print("\n=== Evaluation Results ===")
    print(f"Total images evaluated: {results['total_images']}")
    print(f"Correct predictions: {results['correct_predictions']}")
    print(f"Overall accuracy: {results['accuracy']:.2f}%")

    print("\nPer-class accuracy:")
    for cls, acc in results['per_class_accuracy'].items():
        print(f"  {cls}: {acc:.2f}%")


def main():
    classes = ['Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab',
               'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy on a test set")
    parser.add_argument("test_dir", help="Directory containing" +
                        "test images in class subdirectories")
    parser.add_argument("model", help="Path to the trained model file")
    parser.add_argument("-c", "--classes", nargs="+", default=classes,
                        help="List of class names")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="Number of images to predict at once")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        sys.exit(1)

    # Run evaluation
    results = evaluate_model(args.test_dir, args.model,
                             args.classes, args.batch_size)
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
