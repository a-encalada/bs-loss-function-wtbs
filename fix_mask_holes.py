import os
import cv2
import numpy as np
from tqdm import tqdm

def fill_internal_holes(mask):
    """Fill black holes completely enclosed by white areas, without altering borders."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Invert for finding background
    inv = cv2.bitwise_not(binary)

    # Label connected components
    num_labels, labels = cv2.connectedComponents(inv)

    # Find the component that touches the border (true background)
    h, w = binary.shape
    border_labels = np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    ]))

    # Create a mask of holes (components not touching border)
    holes_mask = np.isin(labels, border_labels, invert=True).astype(np.uint8) * 255

    # Fill the holes
    filled = cv2.bitwise_or(binary, holes_mask)
    return filled


def fix_mask_artifacts(mask):
    """Fix both internal holes and cracks connected to mask border."""
    filled = fill_internal_holes(mask)

    # --- Morphological closing to fix thin black cracks touching edges ---
    # Kernel size can be tuned slightly, but (5,5) is usually safe
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


def fix_masks(dataset_root):
    for folder_name in sorted(os.listdir(dataset_root)):
        mask_dir = os.path.join(dataset_root, folder_name, "mask")
        if not os.path.isdir(mask_dir):
            continue

        print(f"Processing folder: {folder_name}")
        mask_files = [f for f in os.listdir(mask_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

        for mask_file in tqdm(mask_files, desc=f"Folder {folder_name}", unit="mask"):
            path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️  Could not read: {path}")
                continue

            cleaned = fix_mask_artifacts(mask)
            cv2.imwrite(path, cleaned)

    print("✅ All masks processed and fixed.")


if __name__ == "__main__":
    dataset_root = r"/home/angel.encalada/Documents/WTBSegmentation/dataset/Dataset"
    fix_masks(dataset_root)
