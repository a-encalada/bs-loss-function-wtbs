import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
import torch

# def _surface_points(mask_np):
#     """Return boundary coordinates of a binary mask."""
#     if mask_np.ndim == 3:  # (C,H,W) -> merge channels
#         mask_np = mask_np[0]
#     mask_np = mask_np.astype(bool)
#     return cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
def _surface_points(mask_np):
    """
    Return boundary coordinates of a binary mask using OpenCV contours.
    This version is more robust than binary_erosion — it works even
    with 1-pixel-wide or noisy masks and preserves shape perfectly.
    """
    if mask_np.ndim == 3:  # (C,H,W)
        mask_np = mask_np[0]
    mask_np = mask_np.astype(np.uint8)

    # Ensure mask is binary 0/1 (not grayscale)
    _, mask_bin = cv2.threshold(mask_np, 0, 1, cv2.THRESH_BINARY)

    # Find contours of the white region(s)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return np.empty((0, 2), dtype=int)

    # Combine all contours into one coordinate list
    contour_points = np.vstack([c.reshape(-1, 2) for c in contours])

    # cv2 gives (x, y) → swap to (row, col) for compatibility with numpy indexing
    contour_points = contour_points[:, [1, 0]]

    return contour_points

def hd95(pred, target, voxel_spacing=1.0):
    """Batch-averaged 95% Hausdorff distance."""
    N = pred.shape[0]
    hd_list = []
    for i in range(N):
        pred_np = pred[i].detach().cpu().numpy().astype(bool)
        target_np = target[i].detach().cpu().numpy().astype(bool)

        pred_pts = _surface_points(pred_np)
        target_pts = _surface_points(target_np)

        if len(pred_pts) == 0 or len(target_pts) == 0:
            hd_list.append(np.nan)
            continue

        dist_pred_to_target = cdist(pred_pts, target_pts).min(axis=1)
        dist_target_to_pred = cdist(target_pts, pred_pts).min(axis=1)

        hd95 = np.percentile(np.hstack([dist_pred_to_target, dist_target_to_pred]), 95)
        hd_list.append(hd95 * voxel_spacing)
    return np.round(np.nanmean(hd_list), 2)

if __name__ == "__main__":
    # --- Load your mask (as grayscale) ---
    path = "/home/angel.encalada/Documents/WTBSegmentation/dataset/Dataset/13/mask/img230_15.png"  # path to the attached file
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    # --- Create tensors for pred & target ---
    mask_t = torch.tensor(mask_bin[None, ...])  # shape (1, H, W)

    # --- Test hd95() ---
    score = hd95(mask_t, mask_t)
    print(f"HD95 (identical masks): {score}")

    # --- Test small offset to verify sensitivity ---
    shifted = np.roll(mask_bin, shift=5, axis=1)
    shifted_t = torch.tensor(shifted[None, ...])
    score_shifted = hd95(mask_t, shifted_t)
    print(f"HD95 (5px shift): {score_shifted}")
