"""
StockSense Pro - Subset Extractor & YOLO Converter
Extracts a lightweight subset from the 27 GB RPC dataset.
Maps 200 fine-grained SKUs into 6 retail supercategories and converts
COCO bboxes to YOLO normalized format.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

# Paths
SOURCE_ROOT = Path(r"D:\Datasets\ML_SA_Dataset\archive")
DEST_IMAGES = Path("data/raw_subset/images")
DEST_LABELS = Path("data/raw_subset/labels")

# Target 6 Supercategories
TARGET_SUPERCATS = [
    "drink",
    "puffed_food",
    "instant_noodles",
    "dessert",
    "canned_food",
    "stationery",
]

SUPERCAT_TO_ID = {name: idx for idx, name in enumerate(TARGET_SUPERCATS)}


def load_category_mapping(train_json_path):
    """Maps RPC's 200 category IDs to our 6 supercategory target IDs."""
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_target = {}
    for cat in data.get("categories", []):
        supercat = cat.get("supercategory")
        if supercat in SUPERCAT_TO_ID:
            cat_id_to_target[cat["id"]] = SUPERCAT_TO_ID[supercat]

    return cat_id_to_target


def process_dataset(max_val_images=250, max_train_images=150):
    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_LABELS.mkdir(parents=True, exist_ok=True)

    cat_mapping = load_category_mapping(SOURCE_ROOT / "instances_train2019.json")
    print(f"Mapped {len(cat_mapping)} SKUs to {len(TARGET_SUPERCATS)} Supercategories.")

    copied_images = 0
    total_boxes = 0
    class_counter = Counter()

    # Sources: val2019 (multi-item scenes) and train2019 (single-item canonical examples)
    sources = [
        ("val2019", "instances_val2019.json", max_val_images),
        ("train2019", "instances_train2019.json", max_train_images),
    ]

    for split_folder, json_file, max_count in sources:
        json_path = SOURCE_ROOT / json_file
        img_dir = SOURCE_ROOT / split_folder

        if not json_path.exists():
            print(f"Skipping {json_file}: not found.")
            continue

        print(f"\nProcessing {split_folder} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_id_to_meta = {img["id"]: img for img in data.get("images", [])}
        img_id_to_anns = {}
        for ann in data.get("annotations", []):
            if ann["category_id"] in cat_mapping:
                img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Select images containing our target classes
        selected_ids = list(img_id_to_anns.keys())[:max_count]

        for img_id in selected_ids:
            img_meta = img_id_to_meta[img_id]
            file_name = img_meta["file_name"]
            img_w = img_meta["width"]
            img_h = img_meta["height"]

            src_img_path = img_dir / file_name
            if not src_img_path.exists():
                continue

            # Copy image file
            dst_img_path = DEST_IMAGES / file_name
            shutil.copy2(src_img_path, dst_img_path)
            copied_images += 1

            # Write YOLO format label file (.txt)
            label_stem = Path(file_name).stem
            label_path = DEST_LABELS / f"{label_stem}.txt"

            with open(label_path, "w", encoding="utf-8") as lf:
                for ann in img_id_to_anns[img_id]:
                    target_cls = cat_mapping[ann["category_id"]]
                    x_min, y_min, w, h = ann["bbox"]

                    # Mathematical normalization derived above
                    x_center = (x_min + (w / 2.0)) / img_w
                    y_center = (y_min + (h / 2.0)) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h

                    # Clamp bounds to [0.0, 1.0] to safeguard against rounding edge cases
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    lf.write(f"{target_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    class_counter[TARGET_SUPERCATS[target_cls]] += 1
                    total_boxes += 1

    print("\n" + "=" * 60)
    print("SUBSET EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total Images Copied: {copied_images}")
    print(f"Total Bounding Boxes Generated: {total_boxes}")
    print("\nInstance Distribution Across Classes:")
    for cat_name, count in class_counter.items():
        print(f"  • {cat_name:<20}: {count} instances")


if __name__ == "__main__":
    process_dataset()
