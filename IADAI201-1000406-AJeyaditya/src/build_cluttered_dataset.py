"""
StockSense Pro - 10-Class Cluttered Multi-Product Dataset Engine
Samples exclusively from RPC multi-product scene imagery (val2019).
Guarantees schema synchronization, dynamic data.yaml generation, and zero validation errors.
"""

import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
import yaml

# Source paths
SOURCE_ROOT = Path(r"D:\Datasets\ML_SA_Dataset\archive")
VAL_JSON_PATH = SOURCE_ROOT / "instances_val2019.json"
TRAIN_JSON_PATH = SOURCE_ROOT / "instances_train2019.json"
VAL_IMAGE_DIR = SOURCE_ROOT / "val2019"

OUTPUT_DIR = Path("data/dataset")

# 10 Target Retail Supercategories
TARGET_SUPERCATS = [
    "drink",            # 0
    "puffed_food",      # 1
    "instant_noodles",  # 2
    "dessert",          # 3
    "canned_food",      # 4
    "stationery",       # 5
    "dried_food",       # 6
    "chocolate",        # 7
    "candy",            # 8
    "milk",             # 9
]
SUPERCAT_TO_ID = {name: idx for idx, name in enumerate(TARGET_SUPERCATS)}

RANDOM_SEED = 42
TARGET_TOTAL_IMAGES = 450


def load_category_mapping():
    """Maps fine-grained SKU categories to target supercategory IDs."""
    with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for cat in data.get("categories", []):
        supercat = cat.get("supercategory")
        if supercat in SUPERCAT_TO_ID:
            mapping[cat["id"]] = SUPERCAT_TO_ID[supercat]
    return mapping


def build_and_validate():
    random.seed(RANDOM_SEED)

    print("=" * 65)
    print("StockSense Pro: Assembling 10-Class Cluttered Scene Dataset")
    print("=" * 65)

    if not VAL_JSON_PATH.exists() or not VAL_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing RPC files at {SOURCE_ROOT}. Verify drive path.")

    cat_mapping = load_category_mapping()
    num_classes = len(TARGET_SUPERCATS)
    print(f"Loaded schema: {len(cat_mapping)} SKUs mapped to {num_classes} target supercategories.")

    print("Parsing instances_val2019.json...")
    with open(VAL_JSON_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    img_meta_by_id = {img["id"]: img for img in val_data.get("images", [])}
    anns_by_img_id = {}

    for ann in val_data.get("annotations", []):
        cat_id = ann["category_id"]
        if cat_id in cat_mapping:
            anns_by_img_id.setdefault(ann["image_id"], []).append(ann)

    candidate_img_ids = list(anns_by_img_id.keys())
    random.shuffle(candidate_img_ids)

    selected_img_ids = []
    selected_counts = Counter()

    # Priority selection: balance underrepresented classes
    for img_id in candidate_img_ids:
        if len(selected_img_ids) >= TARGET_TOTAL_IMAGES:
            break
        anns = anns_by_img_id[img_id]
        has_priority = any(
            TARGET_SUPERCATS[cat_mapping[a["category_id"]]]
            in ["puffed_food", "dessert", "instant_noodles", "stationery", "dried_food", "chocolate", "candy", "milk"]
            for a in anns
        )
        if has_priority:
            selected_img_ids.append(img_id)
            for a in anns:
                selected_counts[TARGET_SUPERCATS[cat_mapping[a["category_id"]]]] += 1

    # Fill remaining slots
    for img_id in candidate_img_ids:
        if len(selected_img_ids) >= TARGET_TOTAL_IMAGES:
            break
        if img_id not in selected_img_ids:
            selected_img_ids.append(img_id)
            for a in anns_by_img_id[img_id]:
                selected_counts[TARGET_SUPERCATS[cat_mapping[a["category_id"]]]] += 1

    print("\nInstance Distribution Across 450 Cluttered Scenes:")
    for cat_name in TARGET_SUPERCATS:
        print(f"  • {cat_name:<20}: {selected_counts[cat_name]:,} instances")

    # Clean old dataset folder
    if OUTPUT_DIR.exists():
        print(f"\nPurging legacy dataset directory: {OUTPUT_DIR} ...")
        shutil.rmtree(OUTPUT_DIR)

    # 70% Train, 15% Val, 15% Test split
    total_imgs = len(selected_img_ids)
    n_train = math.floor(0.70 * total_imgs)
    n_val = math.floor(0.15 * total_imgs)

    split_map = {
        "train": selected_img_ids[:n_train],
        "val": selected_img_ids[n_train:n_train + n_val],
        "test": selected_img_ids[n_train + n_val:],
    }

    total_boxes_written = 0

    for split_name, img_ids in split_map.items():
        img_out_dir = OUTPUT_DIR / "images" / split_name
        lbl_out_dir = OUTPUT_DIR / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_id in img_ids:
            meta = img_meta_by_id[img_id]
            file_name = meta["file_name"]
            img_w, img_h = meta["width"], meta["height"]

            src_img = VAL_IMAGE_DIR / file_name
            if not src_img.exists():
                continue

            shutil.copy2(src_img, img_out_dir / file_name)

            label_file = lbl_out_dir / f"{Path(file_name).stem}.txt"
            with open(label_file, "w", encoding="utf-8") as lf:
                for ann in anns_by_img_id[img_id]:
                    t_cls = cat_mapping[ann["category_id"]]
                    x_min, y_min, w, h = ann["bbox"]

                    # Affine bounding box normalization
                    x_c = max(0.0, min(1.0, (x_min + (w / 2.0)) / img_w))
                    y_c = max(0.0, min(1.0, (y_min + (h / 2.0)) / img_h))
                    w_n = max(0.0, min(1.0, w / img_w))
                    h_n = max(0.0, min(1.0, h / img_h))

                    lf.write(f"{t_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    total_boxes_written += 1

    # Dynamic data.yaml writer
    names_dict = {idx: name for idx, name in enumerate(TARGET_SUPERCATS)}
    data_yaml_config = {
        "path": "../data/dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": num_classes,
        "names": names_dict,
    }

    yaml_target = OUTPUT_DIR / "data.yaml"
    with open(yaml_target, "w", encoding="utf-8") as yf:
        yaml.dump(data_yaml_config, yf, default_flow_style=False, sort_keys=False)

    print(f"\n[OK] data.yaml written with nc = {num_classes} to {yaml_target}")

    # Immediate self-validation pass
    print("\n" + "=" * 65)
    print("Executing Integrated Verification Pass...")
    print("=" * 65)

    verification_errors = 0
    boxes_verified = 0

    for split in ["train", "val", "test"]:
        lbl_dir = OUTPUT_DIR / "labels" / split
        img_dir = OUTPUT_DIR / "images" / split
        split_boxes = 0
        split_errs = 0

        for img_p in img_dir.glob("*.*"):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                split_errs += 1
                continue

            with open(lbl_p, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    split_boxes += 1
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    # Schema validation: 0 <= cls_id < 10
                    if not (0 <= cls_id < num_classes):
                        split_errs += 1
                    for val in coords:
                        if not (0.0 <= val <= 1.0):
                            split_errs += 1

        print(f"[{split:<5}] Images: {len(list(img_dir.glob('*.*'))):<3} | Boxes: {split_boxes:<5} | Errors: {split_errs}")
        boxes_verified += split_boxes
        verification_errors += split_errs

    print("-" * 65)
    if verification_errors == 0:
        print(f"VERIFICATION SUCCESSFUL: All {boxes_verified:,} boxes across {total_imgs} images pass validation!")
    else:
        print(f"VERIFICATION FAILED: {verification_errors} errors detected.")


if __name__ == "__main__":
    build_and_validate()
