import shutil
from pathlib import Path
import sys
import argparse
import random

def main(src: Path, dst: Path, nb_files: int = 150) -> None:
    if not src.exists() or not src.is_dir():
        print(f"Source directory not found: {src}", file=sys.stderr)
        sys.exit(1)

    # Get all files recursively, including those in subdirectories
    files = [p for p in src.rglob("*") if p.is_file()]
    if not files:
        print(f"No files found in {src}", file=sys.stderr)
        sys.exit(1)

    # Randomly select files up to nb_files
    count = min(nb_files, len(files))
    selected_files = random.sample(files, count)
    copied = 0

    for i, src_path in enumerate(selected_files, start=1):
        # Get the relative path from the source directory
        rel_path = src_path.relative_to(src)
        # Create the destination path maintaining folder structure
        dst_path = dst / rel_path

        # Create parent directories if they don't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(src_path, dst_path)
        copied += 1
        print(f"[{i}/{count}] {rel_path} -> {dst_path}")

    print(f"Finished copying {copied} randomly selected file(s) to '{dst}'" +
        " maintaining folder structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Test Data Preparer",
        description="Randomly select and copy files from source to destination maintaining folder structure.",
        epilog="By Tamigore and Elpastor"
    )

    parser.add_argument(
        "-s",
        "--src",
        type=Path,
        default=Path("splited_images/validation"),
        help="Path to the source directory"
    )

    parser.add_argument(
        "-d",
        "--dst",
        type=Path,
        default=Path("test"),
        help="Path to the destination directory"
    )

    parser.add_argument(
        "-n",
        "--nb_files",
        type=int,
        default=0,
        help="Number of files to randomly select and copy (default: 150)"
    )

    args = parser.parse_args()
    src = args.src
    dst = args.dst
    nb_files = args.nb_files
    if nb_files == 0:
        nb_files = len([p for p in src.rglob("*") if p.is_file()])
    main(src, dst, nb_files)
