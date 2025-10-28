import os
import shutil
import random
import sys

def split_data(source_dir):
    """
    Split images from source directory into training (80%) and validation (20%) sets.
    
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

    if not os.path.isdir(source_directory):
        raise Exception("Input is not a directory")

    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_files.append(os.path.join(root, file))

    # Shuffle files randomly
    random.shuffle(image_files)
    
    # Calculate split index
    split_index = int(len(image_files) * 0.8)
    
    # Split files into training and validation sets
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Copy files to respective directories
    for file in train_files:
        dst_path = os.path.join(train_dir, os.path.basename(file))
        shutil.copy2(file, dst_path)

    for file in val_files:
        dst_path = os.path.join(val_dir, os.path.basename(file))
        print(file, dst_path)
        shutil.copy2(file, dst_path)


if __name__ == "__main__":
    source_directory = sys.argv[1]
    split_data(source_directory)