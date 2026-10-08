"""
StockSense Pro - Ground-Truth Visualizer
Draws bounding boxes and labels onto sample images to verify annotations.
"""

from pathlib import Path
import cv2

DATA_DIR = Path("data/dataset")
OUT_DIR = Path("data/sanity_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["drink", "puffed_food", "instant_noodles", "dessert", "canned_food", "stationery"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def visualize_samples(sample_count=5):
    img_dir = DATA_DIR / "images" / "train"
    lbl_dir = DATA_DIR / "labels" / "train"

    sample_images = list(img_dir.glob("*.*"))[:sample_count]

    for img_path in sample_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img, _ = img.shape
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])

                        # Convert normalized back to absolute pixels
                        box_w = int(w * w_img)
                        box_h = int(h * h_img)
                        x1 = int((xc * w_img) - (box_w / 2))
                        y1 = int((yc * h_img) - (box_h / 2))
                        x2 = x1 + box_w
                        y2 = y1 + box_h

                        color = COLORS[cls_id % len(COLORS)]
                        label_name = CLASS_NAMES[cls_id]

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(img, label_name, (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out_path = OUT_DIR / f"vis_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    visualize_samples()
