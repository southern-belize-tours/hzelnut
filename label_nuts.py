#!/usr/bin/env python3
"""
Nut Labeling Pipeline for Production Dataset

This script:
1. Segments individual nuts from real_nuts images
2. Exports crops to a staging folder for manual labeling
3. After labeling, generates training-ready dataset

Usage:
    # Step 1: Extract nut crops for labeling
    python label_nuts.py extract

    # Step 2: Manually move crops into good/ or defect/ folders
    #         (use your file manager)

    # Step 3: Generate training dataset
    python label_nuts.py finalize
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, disk, opening
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


# Configuration
REAL_NUTS_DIR = Path("data/real_nuts")
STAGING_DIR = Path("data/labeling_staging")
LABELED_DIR = Path("data/labeled_nuts")
OUTPUT_DIR = Path("data/production_dataset")

MIN_NUT_AREA = 5000  # Minimum pixel area for a nut region
MAX_NUT_AREA = 500000  # Maximum pixel area
CROP_SIZE = 256  # Output crop size
CROP_PADDING = 1.2  # Padding factor around detected nut


def load_image(path):
    """Load image as numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def segment_nuts(img):
    """
    Segment individual nuts from an image using watershed.

    Returns:
        List of (bbox, mask) tuples for each nut
    """
    # Convert to grayscale
    gray = np.mean(img, axis=2).astype(np.float32) / 255.0

    # Smooth and threshold
    smooth = gaussian(gray, sigma=2)
    thresh = threshold_otsu(smooth)
    binary = smooth < thresh  # Nuts are darker than background

    # Clean up
    binary = remove_small_objects(binary, min_size=MIN_NUT_AREA // 10)
    binary = remove_small_holes(binary, area_threshold=1000)
    binary = opening(binary, disk(3))

    # Watershed segmentation
    distance = ndi.distance_transform_edt(binary)
    local_max = peak_local_max(distance, min_distance=50, labels=binary)
    markers = np.zeros_like(binary, dtype=int)
    for i, (y, x) in enumerate(local_max, start=1):
        markers[y, x] = i
    markers = ndi.label(markers)[0]
    labels = watershed(-distance, markers, mask=binary)

    # Extract regions
    nuts = []
    for region in regionprops(labels):
        if MIN_NUT_AREA <= region.area <= MAX_NUT_AREA:
            minr, minc, maxr, maxc = region.bbox
            mask = labels == region.label
            nuts.append({
                'bbox': (minr, minc, maxr, maxc),
                'mask': mask,
                'area': region.area,
                'centroid': region.centroid,
            })

    return nuts


def crop_nut(img, bbox, size=CROP_SIZE, padding=CROP_PADDING):
    """
    Extract a square crop around a nut bounding box.
    """
    minr, minc, maxr, maxc = bbox
    h, w = maxr - minr, maxc - minc

    # Make square with padding
    side = int(max(h, w) * padding)
    cy, cx = (minr + maxr) // 2, (minc + maxc) // 2

    y1 = max(0, cy - side // 2)
    x1 = max(0, cx - side // 2)
    y2 = min(img.shape[0], y1 + side)
    x2 = min(img.shape[1], x1 + side)

    crop = img[y1:y2, x1:x2]

    # Resize to standard size
    crop_pil = Image.fromarray(crop)
    crop_pil = crop_pil.resize((size, size), Image.BILINEAR)

    return np.array(crop_pil)


def extract_nuts(args):
    """Step 1: Extract nut crops from all images for labeling."""
    print("=" * 60)
    print("STEP 1: Extracting nut crops for labeling")
    print("=" * 60)

    # Create staging directory
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    unlabeled_dir = STAGING_DIR / "unlabeled"
    unlabeled_dir.mkdir(exist_ok=True)

    # Also create labeled folders
    (STAGING_DIR / "good").mkdir(exist_ok=True)
    (STAGING_DIR / "defect").mkdir(exist_ok=True)

    # Find all images
    image_paths = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        image_paths.extend(REAL_NUTS_DIR.glob(ext))

    print(f"Found {len(image_paths)} images in {REAL_NUTS_DIR}")

    # Process each image
    manifest_rows = []
    total_nuts = 0

    for img_path in tqdm(image_paths, desc="Processing images"):
        img = load_image(img_path)
        nuts = segment_nuts(img)

        for i, nut in enumerate(nuts):
            crop = crop_nut(img, nut['bbox'])

            # Save crop
            crop_name = f"{img_path.stem}_nut_{i:02d}.jpg"
            crop_path = unlabeled_dir / crop_name
            Image.fromarray(crop).save(crop_path, quality=95)

            manifest_rows.append({
                'crop_name': crop_name,
                'source_image': img_path.name,
                'bbox': str(nut['bbox']),
                'area': nut['area'],
                'label': '',  # To be filled during labeling
            })
            total_nuts += 1

    # Save manifest
    manifest_path = STAGING_DIR / "labeling_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print(f"\nExtracted {total_nuts} nut crops to: {unlabeled_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print()
    print("=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print(f"1. Open {unlabeled_dir} in your file manager")
    print(f"2. Review each crop and move to:")
    print(f"   - {STAGING_DIR / 'good'} for good nuts")
    print(f"   - {STAGING_DIR / 'defect'} for defective nuts")
    print(f"3. Run: python label_nuts.py finalize")


def finalize_labels(args):
    """Step 3: Generate training-ready dataset from labeled crops."""
    print("=" * 60)
    print("STEP 3: Finalizing labeled dataset")
    print("=" * 60)

    good_dir = STAGING_DIR / "good"
    defect_dir = STAGING_DIR / "defect"
    unlabeled_dir = STAGING_DIR / "unlabeled"

    # Count labels
    good_crops = list(good_dir.glob("*.jpg"))
    defect_crops = list(defect_dir.glob("*.jpg"))
    unlabeled_crops = list(unlabeled_dir.glob("*.jpg"))

    print(f"Good: {len(good_crops)}")
    print(f"Defect: {len(defect_crops)}")
    print(f"Unlabeled (remaining): {len(unlabeled_crops)}")

    if len(unlabeled_crops) > 0:
        print(f"\nWARNING: {len(unlabeled_crops)} crops still unlabeled!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Please finish labeling first.")
            return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_good = OUTPUT_DIR / "good"
    out_defect = OUTPUT_DIR / "defect"
    out_good.mkdir(exist_ok=True)
    out_defect.mkdir(exist_ok=True)

    # Copy labeled crops
    manifest_rows = []

    for crop_path in good_crops:
        shutil.copy(crop_path, out_good / crop_path.name)
        manifest_rows.append({
            'crop_path': f"data/production_dataset/good/{crop_path.name}",
            'label': 'good',
            'labeled_date': datetime.now().isoformat(),
        })

    for crop_path in defect_crops:
        shutil.copy(crop_path, out_defect / crop_path.name)
        manifest_rows.append({
            'crop_path': f"data/production_dataset/defect/{crop_path.name}",
            'label': 'defect',
            'labeled_date': datetime.now().isoformat(),
        })

    # Save manifest
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print()
    print("=" * 60)
    print("DATASET READY")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  good/: {len(good_crops)} crops")
    print(f"  defect/: {len(defect_crops)} crops")
    print(f"Manifest: {manifest_path}")
    print()
    print("Use with PyTorch:")
    print("  from torchvision.datasets import ImageFolder")
    print("  dataset = ImageFolder('data/production_dataset')")


def show_stats(args):
    """Show current labeling progress."""
    print("=" * 60)
    print("LABELING PROGRESS")
    print("=" * 60)

    if not STAGING_DIR.exists():
        print("No staging directory found. Run 'python label_nuts.py extract' first.")
        return

    good_dir = STAGING_DIR / "good"
    defect_dir = STAGING_DIR / "defect"
    unlabeled_dir = STAGING_DIR / "unlabeled"

    good = len(list(good_dir.glob("*.jpg"))) if good_dir.exists() else 0
    defect = len(list(defect_dir.glob("*.jpg"))) if defect_dir.exists() else 0
    unlabeled = len(list(unlabeled_dir.glob("*.jpg"))) if unlabeled_dir.exists() else 0

    total = good + defect + unlabeled
    labeled = good + defect

    print(f"Total crops: {total}")
    print(f"Labeled: {labeled} ({100*labeled/total:.1f}%)" if total > 0 else "No crops found")
    print(f"  - Good: {good}")
    print(f"  - Defect: {defect}")
    print(f"Remaining: {unlabeled}")

    if unlabeled == 0 and labeled > 0:
        print("\nAll crops labeled! Run 'python label_nuts.py finalize' to create dataset.")


def main():
    parser = argparse.ArgumentParser(description="Nut labeling pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract nut crops for labeling')
    extract_parser.set_defaults(func=extract_nuts)

    # Finalize command
    finalize_parser = subparsers.add_parser('finalize', help='Generate training dataset')
    finalize_parser.set_defaults(func=finalize_labels)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show labeling progress')
    stats_parser.set_defaults(func=show_stats)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
