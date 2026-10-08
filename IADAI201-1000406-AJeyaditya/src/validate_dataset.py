"""
StockSense Pro - Standalone Dataset Validator
"""
from pathlib import Path
import yaml

DATASET_ROOT = Path("data/dataset")
YAML_PATH = DATASET_ROOT / "data.yaml"


def validate():
    print("=" * 60)
    print("StockSense Pro: Validating Dataset Integrity")
    print("=" * 60)

    if not YAML_PATH.exists():
        print(f"Error: {YAML_PATH} not found.")
        return

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    num_classes = config.get("nc", 10)
    print(f"Loaded schema from data.yaml: {num_classes} classes configured.")

    total_images = 0
    total_boxes = 0
    total_errors = 0

    for split in ["train", "val", "test"]:
        img_dir = DATASET_ROOT / "images" / split
        lbl_dir = DATASET_ROOT / "labels" / split

        images = list(img_dir.glob("*.*"))
        split_boxes = 0
        split_errors = 0

        for img_p in images:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                split_errors += 1
                continue

            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    split_boxes += 1
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    if not (0 <= cls_id < num_classes):
                        split_errors += 1
                    for val in coords:
                        if not (0.0 <= val <= 1.0):
                            split_errors += 1

        print(f"[{split:<5}] Images: {len(images):<3} | Boxes: {split_boxes:<5} | Errors: {split_errors}")
        total_images += len(images)
        total_boxes += split_boxes
        total_errors += split_errors

    print("-" * 60)
    if total_errors == 0:
        print(f"Validation passed successfully! {total_images} images, {total_boxes:,} boxes across {num_classes} classes.")
    else:
        print(f"Validation failed with {total_errors} errors.")


if __name__ == "__main__":
    validate()
