"""
StockSense Pro - Inspect Val/Cluttered Scenes
Verifies multi-product scene image counts and object density per image.
"""

import json
from pathlib import Path

DATASET_ROOT = Path(r"D:\Datasets\ML_SA_Dataset\archive")
VAL_JSON_PATH = DATASET_ROOT / "instances_val2019.json"


def inspect_val():
    if not VAL_JSON_PATH.exists():
        print(f"Error: {VAL_JSON_PATH} not found.")
        return

    print("Loading instances_val2019.json ...")
    with open(VAL_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_images = len(data.get("images", []))
    num_annotations = len(data.get("annotations", []))

    print(f"\n--- VAL SPLIT STATS ---")
    print(f"Total Multi-Product Images: {num_images}")
    print(f"Total Bounding Boxes: {num_annotations}")
    if num_images > 0:
        print(f"Average Items Per Image: {num_annotations / num_images:.2f}")


if __name__ == "__main__":
    inspect_val()
