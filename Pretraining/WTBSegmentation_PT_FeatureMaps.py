import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import PixelUnshuffle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.distributed as dist
import cv2
from sklearn.model_selection import train_test_split
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.metrics.pairwise import pairwise_distances
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
import csv
from PIL import Image
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
import time
from thop import profile
from scipy.ndimage import distance_transform_edt
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import InterpolationMode
import scienceplots

plt.style.use(['science','no-latex'])

class HierarchicalSplitDepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.pointwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # Depthwise convolutions for c2, c3, and c4
        self.depthwise2 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, groups=in_channels // 4, bias=False)
        self.depthwise3 = nn.Conv2d(int(in_channels * 0.375), int(in_channels * 0.375), kernel_size=3, padding=1, groups=int(in_channels*0.375), bias=False)
        self.depthwise4 = nn.Conv2d(int(in_channels * 0.4375), int(in_channels * 0.4375), kernel_size=3, padding=1, groups=int(in_channels*0.4375), bias=False)
        
        self.pointwise2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.pointwise1(x)
        
        # Split the input feature map into four equal parts
        c1, c2, c3, c4 = torch.split(x, x.shape[1] // 4, dim=1)
        
        # Process c2 with depthwise convolution
        c2 = self.depthwise2(c2)
        c21, c22 = torch.split(c2, c2.shape[1] // 2, dim=1)  # Split c2 into c21 and c22
        
        # Concatenate c21 with c1
        c21c1 = torch.cat([c21, c1], dim=1)
        
        # Concatenate c22 with c3 and apply depthwise convolution
        c3c22 = torch.cat([c3,c22], dim=1)
        c3c22 = self.depthwise3(c3c22)
        c3c221, c3c222 = torch.split(c3c22, c3c22.shape[1] // 2, dim=1)  # Split into c3c221 and c3c222
        
        # Concatenate c21c1 with c3c221
        c3c221c21c1 = torch.cat([c3c221, c21c1], dim=1)
        
        # Concatenate c3c222 with c4
        c4c3c222 = torch.cat([c4, c3c222], dim=1)
        
        # Apply depthwise convolution to c4c3c222
        c4c3c222 = self.depthwise4(c4c3c222)
        
        # Final concatenation
        x = torch.cat([c4c3c222, c3c221c21c1], dim=1)
        
        # Apply final pointwise convolution
        x = self.pointwise2(x)
        
        return x + residual  # Residual connection

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        def conv_block(in_c, out_c):
            #print(in_c, out_c)
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                HierarchicalSplitDepthwiseSeparableConvBlock(out_c, out_c)
            )
        
        def pixel_unshuffle_block(in_c, out_c, scale=2):
            #print(in_c)
            return nn.Sequential(
                PixelUnshuffle(scale),  # Scale factor of 2
                nn.Conv2d(in_c*scale**2, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = nn.Sequential(
            conv_block(in_channels, 64),
            pixel_unshuffle_block(64, 128)
        )
        
        self.encoder2 = nn.Sequential(
            conv_block(128, 128),
            pixel_unshuffle_block(128, 256)
        )
        
        self.encoder3 = nn.Sequential(
            conv_block(256, 256),
            pixel_unshuffle_block(256, 512)
        )
        
        self.encoder4 = nn.Sequential(
            conv_block(512, 512),
            pixel_unshuffle_block(512, 1024)
        )
        
        # Bottleneck (after pixel unshuffle)
        self.bottleneck = nn.Sequential(
            conv_block(1024, 1024)
        )
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        enc1 = self.encoder1(x)
        #print(f"After encoder1: {enc1.shape}")
        enc2 = self.encoder2(enc1)
        #print(f"After encoder2: {enc2.shape}")
        enc3 = self.encoder3(enc2)
        #print(f"After encoder3: {enc3.shape}")
        enc4 = self.encoder4(enc3)
        #print(f"After encoder4: {enc4.shape}")
        
        bottleneck = self.bottleneck(enc4)
        
        dec4 = self.decoder4(torch.cat([self.upconv4(bottleneck), self.encoder4[0](enc3)], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec4), self.encoder3[0](enc2)], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec3), self.encoder2[0](enc1)], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), self.encoder1[0](x)], dim=1))
        
        return self.activation(self.final_conv(dec1))


# Dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, mask_paths, size, num_classes=2, augmentation=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes  # Number of segmentation classes
        self.size = size
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Load the mask in grayscale
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
        
        # Apply threshold to binarize the mask
        _, binary_mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)  # Convert to 0 and 255
        
        # Convert binary mask to tensor and normalize to {0,1}
        binary_mask = torch.tensor(binary_mask, dtype=torch.long) // 255  
        #binary_mask = 1-binary_mask
        
        # Convert binary mask to one-hot encoding (C, H, W)
        #one_hot_mask = F.one_hot(binary_mask, num_classes=self.num_classes).permute(2, 0, 1).float()
            
        image = TF.resize(image, (self.size, self.size))
        mask = TF.resize(binary_mask.unsqueeze(0).float(), (self.size, self.size), interpolation=InterpolationMode.NEAREST)  # Ensure mask resizes differently if needed
        
        if self.augmentation:
            choice = random.choice(["flip", "rotate", "rot+flip"])
            if choice == 'flip':
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            elif choice == 'rotate':
                angle = random.uniform(-45, 45)  # Random Rotation
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            else:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                angle = random.uniform(-45, 45)  # Random Rotation
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Convert image to tensor
        image = TF.to_tensor(image)
        mask.clamp_(0.0, 1.0)

        return image, mask

# def prepare_data(base_dir):
#     """
#     Prepare train, val, and test splits ensuring 70/10/20 per class
#     for 'clear contrast' and 'unclear contrast' blade folders.
#     """
#     # Define which folders are unclear contrast
#     unclear_folders = {'6', '13', '15', '25'}

#     clear_images, clear_masks = [], []
#     unclear_images, unclear_masks = [], []

#     # Iterate over all folder IDs
#     for folder_name in sorted(os.listdir(base_dir)):
#         folder_path = os.path.join(base_dir, folder_name)
#         if not os.path.isdir(folder_path):
#             continue
        
#         image_dir = os.path.join(folder_path, "image")
#         mask_dir = os.path.join(folder_path, "mask")

#         # Get all image and mask paths
#         images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
#         masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.png'))])

#         # Add to respective class lists
#         if folder_name in unclear_folders:
#             unclear_images.extend(images)
#             unclear_masks.extend(masks)
#         else:
#             clear_images.extend(images)
#             clear_masks.extend(masks)

#     def split_class(images, masks, seed=42):
#         """Split one class into train/val/test = 70/10/20"""
#         train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
#             images, masks, test_size=0.30, random_state=seed
#         )
#         val_imgs, test_imgs, val_masks, test_masks = train_test_split(
#             temp_imgs, temp_masks, test_size=(2/3), random_state=seed
#         )
#         return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks

#     # Split both classes separately
#     clr_train_i, clr_train_m, clr_val_i, clr_val_m, clr_test_i, clr_test_m = split_class(clear_images, clear_masks)
#     unc_train_i, unc_train_m, unc_val_i, unc_val_m, unc_test_i, unc_test_m = split_class(unclear_images, unclear_masks)

#     # Combine them
#     train_images = clr_train_i + unc_train_i
#     train_masks  = clr_train_m + unc_train_m
#     val_images   = clr_val_i + unc_val_i
#     val_masks    = clr_val_m + unc_val_m
#     test_images  = clr_test_i #+ unc_test_i #+ clr_test_i
#     test_masks   = clr_test_m #+ unc_test_m #+ clr_test_m

#     # Shuffle each split so classes mix
#     combined = list(zip(train_images, train_masks))
#     random.shuffle(combined)
#     train_images, train_masks = zip(*combined)

#     combined = list(zip(val_images, val_masks))
#     random.shuffle(combined)
#     val_images, val_masks = zip(*combined)

#     combined = list(zip(test_images, test_masks))
#     random.shuffle(combined)
#     test_images, test_masks = zip(*combined)

#     return (
#         list(train_images), list(train_masks),
#         list(val_images), list(val_masks),
#         list(test_images), list(test_masks)
#     )
    
def prepare_data(base_dir):
    """
    Prepare train, val, and test splits ensuring 70/10/20 per class
    """
  
    image_dir = os.path.join(base_dir, "blades")
    mask_dir = os.path.join(base_dir, "masks")

    # Get all image and mask paths
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.png'))])

    def split_class(images, masks, seed=42):
        """Split one class into train/val/test = 70/10/20"""
        train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
            images, masks, test_size=0.30, random_state=seed
        )
        val_imgs, test_imgs, val_masks, test_masks = train_test_split(
            temp_imgs, temp_masks, test_size=(2/3), random_state=seed
        )
        return train_imgs, train_masks, val_imgs, val_masks, test_imgs, test_masks

    # Split both classes separately
    train_i, train_m, val_i, val_m, test_i, test_m = split_class(images, masks)

    # Shuffle each split so classes mix
    combined = list(zip(train_i, train_m))
    random.shuffle(combined)
    train_images, train_masks = zip(*combined)

    combined = list(zip(val_i, val_m))
    random.shuffle(combined)
    val_images, val_masks = zip(*combined)

    combined = list(zip(test_i, test_m))
    random.shuffle(combined)
    test_images, test_masks = zip(*combined)

    return (
        list(train_images), list(train_masks),
        list(val_images), list(val_masks),
        list(test_images), list(test_masks)
    )    
    
# Dice Score Calculation
def dice(pred, target, smooth=1e-6):
    """
    Calculate the Dice score for the segmentation mask.
    
    Args:
        pred (Tensor): predicted mask (N, C, H, W) with logits.
        target (Tensor): ground truth mask (N, C, H, W).
        threshold (float): threshold for converting logits to binary mask.
        smooth (float): smoothing factor to avoid division by zero.
        
    Returns:
        float: Dice score.
    """
    # Apply sigmoid to get probabilities, then apply threshold
    # pred = pred > threshold  # Convert to binary (0 or 1)
    
    # Flatten for calculation
    pred_flat = pred
    target_flat = target
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return np.round(dice*100,2)
    
def precision(pred, target, smooth=1e-6):
    """
    Batch-averaged precision (%).
    """
    N = pred.shape[0]
    scores = []
    for i in range(N):
        pred_flat = pred[i].view(-1)
        target_flat = target[i].view(-1)
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        precision = (tp + smooth) / (tp + fp + smooth)
        scores.append(precision.item() * 100.0)
    return np.round(np.mean(scores),2)


def recall(pred, target, smooth=1e-6):
    """
    Batch-averaged recall (%).
    """
    N = pred.shape[0]
    scores = []
    for i in range(N):
        pred_flat = pred[i].view(-1)
        target_flat = target[i].view(-1)
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        recall = (tp + smooth) / (tp + fn + smooth)
        scores.append(recall.item() * 100.0)
    return np.round(np.mean(scores),2)

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
    """
    Batch-averaged 95% Hausdorff distance (mm).
    """
    N = pred.shape[0]
    hd_list = []
    for i in range(N):
        pred_np = pred[i].cpu().numpy().astype(bool)
        target_np = target[i].cpu().numpy().astype(bool)

        pred_pts = _surface_points(pred_np)
        target_pts = _surface_points(target_np)

        if len(pred_pts) == 0 or len(target_pts) == 0:
            hd_list.append(np.nan)
            continue

        dist_pred_to_target = cdist(pred_pts, target_pts).min(axis=1)
        dist_target_to_pred = cdist(target_pts, pred_pts).min(axis=1)

        hd95 = np.percentile(np.hstack([dist_pred_to_target, dist_target_to_pred]), 95)
        hd_list.append(hd95 * voxel_spacing)
    return np.round(np.nanmean(hd_list),2)

def assd(pred, target, voxel_spacing=1.0):
    """
    Batch-averaged ASSD (mm).
    """
    N = pred.shape[0]
    assd_list = []
    for i in range(N):
        pred_np = pred[i].cpu().numpy().astype(bool)
        target_np = target[i].cpu().numpy().astype(bool)

        pred_pts = _surface_points(pred_np)
        target_pts = _surface_points(target_np)

        if len(pred_pts) == 0 or len(target_pts) == 0:
            assd_list.append(np.nan)
            continue

        dist_pred_to_target = cdist(pred_pts, target_pts).min(axis=1)
        dist_target_to_pred = cdist(target_pts, pred_pts).min(axis=1)
        assd = (dist_pred_to_target.mean() + dist_target_to_pred.mean()) / 2.0
        assd_list.append(assd * voxel_spacing)
    return np.round(np.nanmean(assd_list),2)



import os
import re
import random
import math
import torch
import matplotlib.pyplot as plt

# ---- 1. Register hooks on key UNet stages ----
def register_unet_feature_hooks(model):
    """
    Returns a dict 'acts' that will be filled with activations
    for the main encoder/decoder stages of your UNet.
    """
    layers = {
        "enc1": model.encoder1,
        "enc2": model.encoder2,
        "enc3": model.encoder3,
        "enc4": model.encoder4,
        "bottleneck": model.bottleneck,
        "dec4": model.decoder4,
        "dec3": model.decoder3,
        "dec2": model.decoder2,
        "dec1": model.decoder1,
    }

    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            # out: (B, C, H, W)
            activations[name] = out.detach().cpu()
        return hook

    handles = []
    for name, module in layers.items():
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

    return activations, handles


# ---- 2. Run a model on one image and collect feature maps + prediction ----
def run_model_collect_features(model, image, device):
    """
    image: (1, C, H, W) tensor on 'device'
    Returns: activations dict, pred tensor on CPU
    """
    model.eval()
    model.to(device)

    acts, handles = register_unet_feature_hooks(model)

    with torch.no_grad():
        pred = model(image.to(device))

    # Remove hooks to avoid clutter
    for h in handles:
        h.remove()

    return acts, pred.detach().cpu()

def main():
    # --------- CONFIG ---------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #base_dir = "/home/angel.encalada/Documents/WTBSegmentation/dataset/Dataset"
    base_dir = "/home/angel.encalada/Documents/WTBSegmentation/dataset/Blade30"
    models_dir = "/home/angel.encalada/Documents/WTBSegmentation/Coding/Pretraining"  # <- change if needed
    size = 512
    num_classes = 2
    batch_size = 1  # we just need 1 sample for visualization
    # --------------------------

    # ---- 1. Build test_loader (simple, no DDP here) ----
    train_images, train_masks, val_images, val_masks, test_images, test_masks = prepare_data(base_dir)

    test_dataset = ImageDataset(test_images, test_masks, size=size, num_classes=num_classes, augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Pick a random index from test set
    rand_idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[rand_idx]

    # Handle if your dataset returns (image, mask) or (image, mask, path)
    if isinstance(sample, (tuple, list)) and len(sample) == 3:
        image, target_mask, img_path = sample
        img_title = os.path.basename(img_path)
    else:
        image, target_mask = sample
        img_title = f"IDx: {rand_idx}"

    # Add batch dimension
    image = image.unsqueeze(0).to(device)
    target_mask = target_mask.unsqueeze(0)  # stay on CPU for plotting later

    # ---- 2. Find all trained models in models_dir ----
    model_files = [
        f for f in os.listdir(models_dir)
        if f.startswith("WTBSegmentationModel_PT_") and f.endswith(".pth") #and 'BS+C+FI' in f
    ]
    model_files.sort()  # for consistent order

    if not model_files:
        print("No model .pth files found in", models_dir)
        return

    # ---- 3. For each model, load weights and collect feature maps ----
    per_model_results = []  # list of dicts: { 'name':..., 'acts':..., 'pred':... }

    for fname in model_files:
        full_path = os.path.join(models_dir, fname)

        # Extract tag between 'PT_' and '_E' e.g. B+Dice from WTBSegmentationModel_PT_B+Dice_E9.pth
        m = re.search(r"WTBSegmentationModel_PT_(.+)_E\d+\.pth", fname)
        tag = m.group(1) if m else fname
        
        # NEW VALIDATION FOR EWSHM 2026
        if tag in ['Focal', 'BCE+Dice', 'Distance', 'HD+Dice', 'B+Dice', 'BS+C_b090']:

            print(f"Loading model: {fname} (tag: {tag})")
            
            #if 'BS+C+FI' in fname:
            model = UNet(in_channels=3, out_channels=1)
            #else:
                #model = UNet(in_channels=3, out_channels=2)
            checkpoint = torch.load(full_path, map_location=device)
    
            # If saved from DDP (keys start with 'module.')
            if any(k.startswith("module.") for k in checkpoint.keys()):
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    
            model.load_state_dict(checkpoint, strict=False)
    
            acts, pred = run_model_collect_features(model, image, device)
    
            per_model_results.append({
                "tag": tag,
                "acts": acts,
                "pred": pred,  # shape: (1, C, H, W)
            })

    # ---- 4. Prepare data for plotting ----
    # Move image + target_mask to numpy
    img_np = image[0].detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 3)

    # target_mask: (1, C, H, W) or (1, H, W)?
    target_np = target_mask[0].numpy()
    if target_np.ndim == 3:  # (C, H, W)
        # choose blade channel (assume channel 1 if 2 classes)
        gt_blade = target_np[0] if target_np.shape[0] > 1 else target_np[0]
    else:  # (H, W)
        gt_blade = target_np

    # ---- 5. Create subplot: one row per model, many columns ----
    stage_order2 = [
        "Enc. 1", "Enc. 2", "Enc. 3", "Enc. 4",
        "Bottleneck",
        "Dec. 4", "Dec. 3", "Dec. 2", "Dec. 1"
    ]
    stage_order = [
    "enc1", "enc2", "enc3", "enc4",
    "bottleneck",
    "dec4", "dec3", "dec2", "dec1"
    ]

    n_models = len(per_model_results)
    n_cols = 1 + len(stage_order) + 3   # img_org + stages + pred + org_mask

    fig, axes = plt.subplots(n_models, n_cols, figsize=(3*n_cols, 4*n_models))

    if n_models == 1:
        # Make axes 2D for unified indexing
        axes = axes.reshape(1, -1)
    
    fontsize = 28
    for row, res in enumerate(per_model_results):
        tag = res["tag"]
        acts = res["acts"]
        pred = res["pred"]  # (1, C, H, W)

        # ---- original image ----
        ax = axes[row, 0]
        if row == 0:
            ax.imshow(img_np)
            ax.set_title('Input', fontsize=fontsize)
        ax.axis("off")

        # ---- encoder/decoder stages (mean over channels) ----
        for ci, stage in enumerate(stage_order, start=1):
            ax = axes[row, ci]
            if stage not in acts:
                ax.axis("off")
                continue

            feat = acts[stage][0]  # (C, H, W)
            mean_fm = feat.mean(dim=0).numpy()

            ax.imshow(mean_fm, cmap="jet")
            if row == 0:
                ax.set_title(stage_order2[ci-1], fontsize=fontsize)
            ax.axis("off")

        # ---- predicted mask ----
        pred_np = pred[0].numpy()
        pred_blade = pred_np[0] if pred_np.ndim == 3 else pred_np
        axes[row, n_cols - 3].imshow(pred_blade>0.5, cmap="gray")
        if row == 0:
            axes[row, n_cols - 3].set_title("Pred. Mask", fontsize=fontsize)
        axes[row, n_cols - 3].axis("off")
    
        # ---- ground-truth mask ----
        axes[row, n_cols - 2].imshow(gt_blade>0.5, cmap="gray")
        if row == 0:
            axes[row, n_cols - 2].set_title("GT Mask", fontsize=fontsize)
        axes[row, n_cols - 2].axis("off")
        
        # ---- metrics panel ----
        dsc_val = dice((pred_blade > 0.5).astype(np.float32), gt_blade.astype(np.float32))
        prec_val = precision(torch.tensor(pred_blade>0.5).float().unsqueeze(0), 
                             torch.tensor(gt_blade).unsqueeze(0))
        rec_val  = recall(torch.tensor(pred_blade>0.5).float().unsqueeze(0), 
                          torch.tensor(gt_blade).unsqueeze(0))
        hd95_val = hd95(torch.tensor(pred_blade>0.5).float().unsqueeze(0), 
                        torch.tensor(gt_blade).unsqueeze(0))
        assd_val = assd(torch.tensor(pred_blade>0.5).float().unsqueeze(0), 
                        torch.tensor(gt_blade).unsqueeze(0))
    
        axm = axes[row, n_cols - 1]
        axm.axis("off")

        if tag=='Focal':
            txt = r'$\mathcal{L}_{Focal}$'
        elif tag=='Distance':
            txt = r'$\mathcal{L}_{Dist}$'
        elif tag=='BCE+Dice':
            txt = r'$\mathcal{L}_{BCE}+\mathcal{L}_{Dice}$'
        elif tag=='HD+Dice':
            txt = r'$\mathcal{L}_{HD}+\lambda\cdot \mathcal{L}_{Dice}$'
        elif tag=='B+Dice':
            txt = r'$\gamma\cdot \mathcal{L}_{B}+\mathcal{L}_{Dice}$'
        elif tag=='BS+C':
            txt = r'$\mathcal{L}_{BS}+\beta\cdot C_{Loc}$'
        elif tag=='BS+C+FI':
            txt = r'$\mathcal{L}_{BS}+\beta\cdot (C_{Loc}+FI)$'
        elif tag=='BCE+BS+C+FI':
            txt = r'$\mathcal{L}_{BCE}+\mathcal{L}_{BS}+\beta\cdot (C_{Loc}+FI)$'
        #for EWSHM only
        elif tag=='BS+C_b050':
            txt = r'$\beta \cdot \mathcal{L}_{BS} + (1-\beta) \cdot \frac{1}{\lambda} \cdot C_{Loc}$' + '\n' + r'$\beta=0.50$'
        elif tag=='BS+C_b090':
            txt = r'$\beta \cdot \mathcal{L}_{BS} + (1-\beta) \cdot \frac{1}{\lambda} \cdot C_{Loc}$' + '\n' + r'$\beta=0.90$'
            
        axm.text(
            0.05, 0.5,
            f"{txt}\n"
            f'\n'
            f"DSC:  {dsc_val:.2f} %\n"
            f"Prec.: {prec_val:.2f} %\n"
            f"Rec.:  {rec_val:.2f} %\n"
            f"HD95: {hd95_val:.2f} px\n"
            f"ASSD: {assd_val:.2f} px",
            fontsize=fontsize, va="center"
        )

    plt.tight_layout()
    out_name = "WTBSegmentation_PT_FeatureMaps.png"
    plt.savefig(out_name, bbox_inches="tight")
    plt.show()
    print(f"Saved visualization to {out_name}")


if __name__ == "__main__":
    main()
