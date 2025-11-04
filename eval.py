import os
import argparse
import subprocess
from typing import List, Dict
import sys


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


def run_pred(image_path: str, model_path: str, classes: List[str]) -> str:
    """Run predict.py on a single image and return predicted class"""
    try:
        # Capture the output of predict.py
        cmd = ["python3", "Predict.py", image_path, model_path, "-r"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse output to find predicted class
        lines = res.stdout.split('\n')
        for line in lines:
            if line.startswith("Predicted class: "):
                return line.replace("Predicted class: ", "").strip()

        raise ValueError("Could not find prediction in output")
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction on {image_path}:", e, file=sys.stderr)
        print(f"predict.py stderr: {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return None


def evaluate_model(test_dir: str, model_path: str, classes: List[str]) -> Dict:
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

    # Process each image
    for i, (image_path, true_class) in enumerate(test_images, 1):
        print(f"\rProcessing image {i}/{total}...", end="", flush=True)

        predicted_class = run_pred(image_path, model_path, classes)
        if predicted_class is None:
            continue

        class_total[true_class] = class_total.get(true_class, 0) + 1

        if predicted_class == true_class:
            correct += 1
            class_correct[true_class] = class_correct.get(true_class, 0) + 1

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
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' does not exist")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        sys.exit(1)

    # Run evaluation
    results = evaluate_model(args.test_dir, args.model, args.classes)
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
