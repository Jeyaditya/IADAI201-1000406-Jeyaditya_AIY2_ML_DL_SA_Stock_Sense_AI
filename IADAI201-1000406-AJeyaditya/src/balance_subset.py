"""
StockSense Pro - Subset Balancer
Targets underrepresented categories (stationery, puffed_food, instant_noodles)
from val2019 to ensure all 6 classes exceed 100+ instances for rubric compliance.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path(r"D:\Datasets\ML_SA_Dataset\archive")
DEST_IMAGES = Path("data/raw_subset/images")
DEST_LABELS = Path("data/raw_subset/labels")

TARGET_SUPERCATS = [
    "drink",
    "puffed_food",
    "instant_noodles",
    "dessert",
    "canned_food",
    "stationery",
]
SUPERCAT_TO_ID = {name: idx for idx, name in enumerate(TARGET_SUPERCATS)}


def load_category_mapping():
    with open(SOURCE_ROOT / "instances_train2019.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        cat["id"]: SUPERCAT_TO_ID[cat["supercategory"]]
        for cat in data.get("categories", [])
        if cat.get("supercategory") in SUPERCAT_TO_ID
    }


def balance_classes():
    cat_mapping = load_category_mapping()
    val_json_path = SOURCE_ROOT / "instances_val2019.json"
    val_img_dir = SOURCE_ROOT / "val2019"

    with open(val_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_id_to_meta = {img["id"]: img for img in data.get("images", [])}
    img_id_to_anns = {}
    for ann in data.get("annotations", []):
        if ann["category_id"] in cat_mapping:
            img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Count current boxes in our raw_subset
    current_counts = Counter()
    for lbl_file in DEST_LABELS.glob("*.txt"):
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    current_counts[TARGET_SUPERCATS[cls_id]] += 1

    print("Current counts before balancing:")
    for cat in TARGET_SUPERCATS:
        print(f"  • {cat:<20}: {current_counts[cat]}")

    # Identify classes below 120 instances
    target_threshold = 120
    added_images = 0

    existing_files = {p.name for p in DEST_IMAGES.glob("*.*")}

    for img_id, anns in img_id_to_anns.items():
        # Check if this image has any deficient classes
        has_deficient_class = any(
            current_counts[TARGET_SUPERCATS[cat_mapping[a["category_id"]]]] < target_threshold
            for a in anns
        )

        img_meta = img_id_to_meta[img_id]
        file_name = img_meta["file_name"]

        if has_deficient_class and file_name not in existing_files:
            src_img = val_img_dir / file_name
            if not src_img.exists():
                continue

            shutil.copy2(src_img, DEST_IMAGES / file_name)
            existing_files.add(file_name)
            added_images += 1

            img_w, img_h = img_meta["width"], img_meta["height"]
            label_stem = Path(file_name).stem
            with open(DEST_LABELS / f"{label_stem}.txt", "w", encoding="utf-8") as lf:
                for ann in anns:
                    t_cls = cat_mapping[ann["category_id"]]
                    x_min, y_min, w, h = ann["bbox"]
                    x_c = max(0.0, min(1.0, (x_min + (w / 2.0)) / img_w))
                    y_c = max(0.0, min(1.0, (y_min + (h / 2.0)) / img_h))
                    w_n = max(0.0, min(1.0, w / img_w))
                    h_n = max(0.0, min(1.0, h / img_h))
                    lf.write(f"{t_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    current_counts[TARGET_SUPERCATS[t_cls]] += 1

        # Stop if all classes are satisfied
        if all(current_counts[c] >= target_threshold for c in TARGET_SUPERCATS):
            break

    print("\n" + "=" * 50)
    print(f"Added {added_images} supplementary images.")
    print("Balanced Instance Distribution:")
    for cat in TARGET_SUPERCATS:
        print(f"  • {cat:<20}: {current_counts[cat]} instances")


if __name__ == "__main__":
    balance_classes()
