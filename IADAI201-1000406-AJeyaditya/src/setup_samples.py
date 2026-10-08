"""
StockSense Pro - Sample Image Setup
Copies 3 representative images from data/dataset/images/test/
to data/sample_images/ for instant dashboard demonstration.
"""

from pathlib import Path
import shutil

TEST_DIR = Path("data/dataset/images/test")
SAMPLE_DIR = Path("data/sample_images")


def setup_samples():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    if not TEST_DIR.exists():
        print(f"Warning: {TEST_DIR} does not exist. Ensure dataset was split.")
        return

    test_images = list(TEST_DIR.glob("*.*"))[:3]
    if not test_images:
        print("No images found in test split directory.")
        return

    for idx, img_path in enumerate(test_images, 1):
        dest_path = SAMPLE_DIR / f"shelf_sample_{idx}{img_path.suffix}"
        shutil.copy2(img_path, dest_path)
        print(f"Preset image copied -> {dest_path}")

    print("Sample presets ready for dashboard demonstration.")


if __name__ == "__main__":
    setup_samples()
