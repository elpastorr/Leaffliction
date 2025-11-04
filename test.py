import shutil
from pathlib import Path
import sys
import random

SRC = Path("splited_images/validation")
DST = Path("test")
MAX_FILES = 150

if not SRC.exists() or not SRC.is_dir():
    print(f"Source directory not found: {SRC}", file=sys.stderr)
    sys.exit(1)

# Get all files recursively, including those in subdirectories
files = [p for p in SRC.rglob("*") if p.is_file()]
if not files:
    print(f"No files found in {SRC}", file=sys.stderr)
    sys.exit(1)

# Randomly select files up to MAX_FILES
count = min(MAX_FILES, len(files))
selected_files = random.sample(files, count)
copied = 0

for i, src_path in enumerate(selected_files, start=1):
    # Get the relative path from the source directory
    rel_path = src_path.relative_to(SRC)
    # Create the destination path maintaining folder structure
    dst_path = DST / rel_path

    # Create parent directories if they don't exist
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy the file
    shutil.copy2(src_path, dst_path)
    copied += 1
    print(f"[{i}/{count}] {rel_path} -> {dst_path}")

print(f"Finished copying {copied} randomly selected file(s) to '{DST}'" +
      " maintaining folder structure.")
