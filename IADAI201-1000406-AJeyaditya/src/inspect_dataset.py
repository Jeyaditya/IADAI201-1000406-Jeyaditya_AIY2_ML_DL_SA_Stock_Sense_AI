"""
StockSense Pro - RPC Dataset Inspector
Reads COCO JSON annotations to report image counts, supercategories,
and instance distributions without loading heavy image files.
"""

import json
from collections import Counter
from pathlib import Path

# Path matching your local machine structure
DATASET_ROOT = Path(r"D:\Datasets\ML_SA_Dataset\archive")
TRAIN_JSON_PATH = DATASET_ROOT / "instances_train2019.json"


def inspect_dataset():
    if not TRAIN_JSON_PATH.exists():
        print(f"Error: Could not find annotation file at: {TRAIN_JSON_PATH}")
        print("Please check your D: drive path.")
        return

    print(f"Loading annotations from: {TRAIN_JSON_PATH.name} ...")
    with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Inspect top-level keys
    print(f"\n--- TOP-LEVEL KEYS ---")
    print(list(data.keys()))

    # 2. Inspect overall counts
    num_images = len(data.get("images", []))
    num_annotations = len(data.get("annotations", []))
    num_categories = len(data.get("categories", []))

    print(f"\n--- OVERVIEW STATS ---")
    print(f"Total Images: {num_images}")
    print(f"Total Object Annotations: {num_annotations}")
    print(f"Total Unique Classes: {num_categories}")

    # 3. Supercategory breakdown
    categories = data.get("categories", [])
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_id_to_super = {c["id"]: c.get("supercategory", "none") for c in categories}

    super_counts = Counter()
    for c in categories:
        super_counts[c.get("supercategory", "unknown")] += 1

    print(f"\n--- SUPERCATEGORIES (Count of Subclasses) ---")
    for super_name, count in super_counts.most_common():
        print(f"  • {super_name}: {count} product types")

    # 4. Instance counts per category in train annotations
    annotations = data.get("annotations", [])
    cat_instance_counts = Counter(ann["category_id"] for ann in annotations)

    print(f"\n--- TOP 15 MOST FREQUENT CLASSES IN TRAIN ---")
    print(f"{'Class ID':<10} | {'Supercategory':<20} | {'Class Name':<30} | {'Instances':<10}")
    print("-" * 78)
    for cat_id, count in cat_instance_counts.most_common(15):
        c_name = cat_id_to_name.get(cat_id, "unknown")
        s_name = cat_id_to_super.get(cat_id, "unknown")
        print(f"{cat_id:<10} | {s_name:<20} | {c_name:<30} | {count:<10}")


if __name__ == "__main__":
    inspect_dataset()
