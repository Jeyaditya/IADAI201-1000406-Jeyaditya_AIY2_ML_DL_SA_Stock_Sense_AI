"""
StockSense Pro - Dataset Splitter
Partitions raw images and label text files into 70% Train, 15% Val, 15% Test splits
guaranteeing zero data leakage.
"""

import math
import random
import shutil
from pathlib import Path

RAW_IMAGES = Path("data/raw_subset/images")
RAW_LABELS = Path("data/raw_subset/labels")
OUTPUT_BASE = Path("data/dataset")

# Fixed random seed for reproducible splits
RANDOM_SEED = 42


def split_data():
    random.seed(RANDOM_SEED)

    # Find all images that have a matching label file
    all_images = sorted([
        img for img in RAW_IMAGES.glob("*.*")
        if (RAW_LABELS / f"{img.stem}.txt").exists()
    ])

    total_count = len(all_images)
    print(f"Total verified image-label pairs: {total_count}")

    # Shuffle deterministically
    random.shuffle(all_images)

    # Exact mathematical derivation: 70% train, 15% val, 15% test
    n_train = math.floor(0.70 * total_count)
    n_val = math.floor(0.15 * total_count)
    n_test = total_count - (n_train + n_val)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:],
    }

    print(f"Splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    for split_name, img_list in splits.items():
        split_img_dir = OUTPUT_BASE / "images" / split_name
        split_lbl_dir = OUTPUT_BASE / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            # Copy image
            shutil.copy2(img_path, split_img_dir / img_path.name)
            # Copy label
            lbl_path = RAW_LABELS / f"{img_path.stem}.txt"
            shutil.copy2(lbl_path, split_lbl_dir / lbl_path.name)

    print("\nDataset split completed successfully into data/dataset/")


if __name__ == "__main__":
    split_data()
