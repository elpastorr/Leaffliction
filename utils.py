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

    files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                files.append(os.path.join(root, file))
    # files = [os.path.basename(f) for f in src_dirs]
    # print(src_dirs)
    # source_dir = os.path.dirname(src_dirs[0]) if src_dirs else source_dir

    # Get list of image files
    # files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]

    # Shuffle files randomly
    random.shuffle(files)
    
    # Calculate split index
    split_index = int(len(files) * 0.8)
    
    # Split files into training and validation sets
    train_files = files[:split_index]
    val_files = files[split_index:]
    print(files)
    return
    # Copy files to respective directories
    for file in train_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(train_dir, file)
        shutil.copy2(src_path, dst_path)
    
    for file in val_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(val_dir, file)
        shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    source_directory = sys.argv[1]
    split_data(source_directory)