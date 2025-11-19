import os
import shutil
import random
import sys
from Augmentation import augment_dir
from Transformation import process_directory
import argparse


def split_data(source_dir):
    """
    Split images from src dir into training (90%) and validation (10%) sets.

    Args:
        source_dir (str): Path to the source directory containing images
    """
    # Create destination directories
    split_dir = "splited_images"
    train_dir = os.path.join(split_dir, "training")
    val_dir = os.path.join(split_dir, "validation")

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    if not os.path.isdir(source_dir):
        raise Exception("Input is not a directory")

    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_files.append(os.path.join(root, file))

    random.shuffle(image_files)

    split_index = int(len(image_files) * 0.9)

    # Split files into training and validation sets
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Copy files to respective directories
    for file in train_files:
        dst_path = os.path.join(train_dir, os.path.basename(file))

        class_name = os.path.basename(os.path.dirname(file))

        # Create class subdirectory in training directory
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        dst_path = os.path.join(class_dir, os.path.basename(file))
        shutil.copy2(file, dst_path)

    for file in val_files:
        dst_path = os.path.join(val_dir, os.path.basename(file))

        class_name = os.path.basename(os.path.dirname(file))

        # Create class subdirectory in validation directory
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        dst_path = os.path.join(class_dir, os.path.basename(file))
        shutil.copy2(file, dst_path)

    print("Séparation terminée : ~90% in training, ~10% in validation :")
    print(f"{len(train_files)} files in training, {len(val_files)}" +
          " files in validation.")


def balance_data(dataset):
    """
    Équilibrer le nombre d'images par classe dans prepared_dataset.
    Pour chaque sous-dossier 'training' et 'validation', supprime aléatoirement
    des images (.jpg) dans les classes qui ont plus d'images afin que toutes
    les classes aient le même nombre (le minimum).
    """
    for subset in ("training", "validation"):
        sub_dir = os.path.join(dataset, subset)
        if not os.path.isdir(sub_dir):
            continue

        # récupérer les dossiers de classes
        class_dirs = [os.path.join(sub_dir, d) for d in os.listdir(sub_dir)
                      if os.path.isdir(os.path.join(sub_dir, d))]
        if not class_dirs:
            continue

        files_per_class = {}
        for d in class_dirs:
            files = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
            files_per_class[d] = files

        min_count = min(len(files) for files in files_per_class.values())

        for d, files in files_per_class.items():
            excess = len(files) - min_count
            if excess > 0:
                to_remove = random.sample(files, excess)
                for fname in to_remove:
                    path = os.path.join(d, fname)
                    try:
                        os.remove(path)
                    except OSError:
                        # Ignorer les erreurs de suppression et continuer
                        pass
        print(f"Balanced '{subset}': every class now has {min_count} images.")


def main():
    """
    This program is designed to perform split, augmentation and/or
     transformations on a dataset of images.
    It accepts a dataset directory containing images
    and creates a prepared dataset with training and validation sets.
    Usage:
        prepare_data.py dir/to/dataset [-a] [-t]
    Arguments:
        -a,  Specify to Augment the dataset.
        -t,  Specify to Transform the dataset.
    """

    parser = argparse.ArgumentParser(description="Apply split, Augmentation" +
                                     " and/or Transformation to a dir of img")

    parser.add_argument("dataset", help="Path to the dataset directory")

    parser.add_argument("-a", action="store_true", help="Specify to Augment")

    parser.add_argument("-t", action="store_true",
                        help="Specify to Transform")
    parser.add_argument("-b", action="store_true",
                        help="Specify to Balance the dataset instead of" +
                        " spliting or augmenting or transforming")

    args = parser.parse_args()
    try:
        if args.b:
            balance_data(args.dataset)
            print("Balanced dataset in 'prepared_dataset' directory.")

        else:
            split_data(args.dataset)
            src_dataset = ["splited_images/training",
                           "splited_images/validation"]
            dst_dataset = ["prepared_dataset/training",
                           "prepared_dataset/validation"]
            if args.a or args.t:
                if not os.path.exists(dst_dataset[0]):
                    os.makedirs(dst_dataset[0])

                if args.a:
                    augment_dir(src_dataset[0], dst_dataset[0])
                if args.t:
                    process_directory(src_dataset[0], dst_dataset[0])

                shutil.copytree(src_dataset[1], dst_dataset[1])
                shutil.rmtree("splited_images")
                print("Created prepared dataset in 'prepared_dataset' dir.")

    except FileNotFoundError as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
