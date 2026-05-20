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
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import InterpolationMode

torch.backends.cudnn.benchmark = True  # optimize kernel selection
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    def __init__(self, in_channels=3, out_channels=2):
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

class EarlyStopping:
    def __init__(self, patience=10, mode='min', delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.delta = delta
        self.best_epoch = 0
        self.best_lmbda = 1.0

    def __call__(self, model, current_score, epoch, lf_name, lmbda):
        if self.best_score is None:
            self.best_epoch = epoch
            self.best_score = current_score
            self.best_lmbda = lmbda
            
            if dist.get_rank() == 0:
                # remove any previous best model to keep only one
                for f in os.listdir('.'):
                    if f.startswith(f"WTBSegmentationModel_PT_{lf_name}") and f.endswith(".pth"):
                        try:
                            os.remove(f)
                        except:
                            pass
            
            torch.save(model.module.state_dict(), f"WTBSegmentationModel_PT_{lf_name}_E{epoch}.pth")
            print("Model saved successfully!")
        elif (self.mode == 'min' and current_score > self.best_score - self.delta) or \
             (self.mode == 'max' and current_score < self.best_score + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0
            self.best_epoch = epoch
            self.best_lmbda = lmbda
            
            if dist.get_rank() == 0:
                # remove any previous best model to keep only one
                for f in os.listdir('.'):
                    if f.startswith(f"WTBSegmentationModel_PT_{lf_name}") and f.endswith(".pth"):
                        try:
                            os.remove(f)
                        except:
                            pass
            
            torch.save(model.module.state_dict(), f"WTBSegmentationModel_PT_{lf_name}_E{epoch}.pth")
            print("Model saved successfully!")
    
    def get_best_model(self, lf_name):
        return f"WTBSegmentationModel_PT_{lf_name}_E{self.best_epoch}.pth"

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
#     test_images_c  = clr_test_i
#     test_masks_c = clr_test_m
#     test_images_u = unc_test_i
#     test_masks_u =  unc_test_m

#     # Shuffle each split so classes mix
#     combined = list(zip(train_images, train_masks))
#     random.shuffle(combined)
#     train_images, train_masks = zip(*combined)

#     combined = list(zip(val_images, val_masks))
#     random.shuffle(combined)
#     val_images, val_masks = zip(*combined)

#     combined = list(zip(test_images_c, test_masks_c))
#     random.shuffle(combined)
#     test_images_c, test_masks_c = zip(*combined)
    
#     combined = list(zip(test_images_u, test_masks_u))
#     random.shuffle(combined)
#     test_images_u, test_masks_u = zip(*combined)

#     return (
#         list(train_images), list(train_masks),
#         list(val_images), list(val_masks),
#         list(test_images_c), list(test_masks_c),
#         list(test_images_u), list(test_masks_u)
#     )

# Train, validation, test splits
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

class AdaptiveHD_DSC_Loss:
    def __init__(self, initial_lambda=1.0, eps=1e-8):
        self.lmbda = initial_lambda
        self.eps = eps
        self.mean_hd = 1.0
        self.mean_dsc = 1.0

    def __call__(self, pred, target):
        # Compute both losses
        L_hd = hausdorff_loss(pred, target)
        L_dsc = 1 - dice((pred > 0.5).float(), target) / 100.0  # convert Dice(%) → loss [0,1]
        
        total_loss = L_hd + self.lmbda * L_dsc
        return total_loss, L_hd.item(), L_dsc.item()

    def update_lambda(self, epoch_hd_loss, epoch_dsc_loss):
        """
        Update λ using mean loss values from the previous epoch.
        λ = mean(L_hd) / mean(L_dsc)
        """
        self.mean_hd = epoch_hd_loss
        self.mean_dsc = epoch_dsc_loss
        self.lmbda = (self.mean_hd / (self.mean_dsc + self.eps))

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
    N = pred.shape[0]
    scores = []
    for i in range(N):
        pred_flat = pred[i].view(-1)
        target_flat = target[i].view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        scores.append(dice.item() * 100.0)
    return np.round(np.mean(scores),2)

# Combined Loss function
def bce_dice_loss(pred, target, smooth=1e-6):
    """
    Batch-based combined Binary Cross Entropy (BCE) + Dice loss.
    
    Args:
        pred (Tensor): predicted mask (N, 1, H, W) with probabilities (0–1).
        target (Tensor): ground truth mask (N, 1, H, W) with binary values (0 or 1).
        smooth (float): smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: scalar loss averaged across the batch.
    """
    # Binary Cross Entropy averaged over batch
    bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')

    # Dice loss averaged per batch
    N = pred.shape[0]
    dice_scores = []
    for i in range(N):
        pred_flat = pred[i].view(-1)
        target_flat = target[i].view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_scores.append(dice)
    mean_dice = torch.stack(dice_scores).mean()

    # Combined loss
    loss = bce_loss + (1 - mean_dice)
    return loss

def focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-6):
    """
    Faster focal loss version for models that already apply sigmoid.
    Works on probabilities in [0, 1].
    """
    pred = pred.clamp(eps, 1.0 - eps)
    target = target.float()

    # BCE term (no logit)
    bce = -(target * torch.log(pred) + (1 - target) * torch.log1p(-pred))

    # Compute pt (prob of correctly classified class)
    pt = target * pred + (1 - target) * (1 - pred)

    # α and γ weights (no branching)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = (1 - pt) ** gamma

    # Combined loss
    loss = alpha_t * focal_weight * bce

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def compute_distance_map(mask, iterations=10):
    """
    Fast differentiable approximation of distance transform on GPU.
    mask: binary tensor (B,1,H,W)
    iterations: number of max-pooling steps
    """
    # invert to find background regions
    inv = 1 - mask
    dist = torch.zeros_like(mask)
    x = inv.clone()
    for i in range(iterations):
        x = F.max_pool2d(x, 3, 1, 1)
        dist += (x > 0).float()
    dist = dist / dist.max().clamp(min=1)
    return dist

def distance_map_loss(pred, target, reduction='mean'):
    target = target.float()
    pred = pred.float()

    # Fast GPU distance approximation
    dist_maps = compute_distance_map(target)

    loss = torch.abs(pred - target) * dist_maps
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def hausdorff_loss(pred, target, reduction='mean'):
    """
    Differentiable Hausdorff Distance Loss (Karimi et al., 2020)
    Args:
        pred: predicted probabilities (after sigmoid), shape (B,1,H,W)
        target: ground truth binary mask (0/1), same shape
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    pred = pred.float()
    target = target.float()

    loss_list = []
    for p, g in zip(pred, target):
        # Compute distance maps
        d_gt = compute_distance_map(g)
        d_pred = compute_distance_map((p > 0.5).float())  # binarized pred for distance

        # Combine distances and intensity differences
        diff = (p - g).abs()
        per_pixel_loss = diff**2 * (d_gt**2 + d_pred**2)

        loss_list.append(per_pixel_loss.mean())

    loss = torch.stack(loss_list)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def fast_signed_distance_map(mask, iterations=32):
    """
    Approximate signed distance map (SDF) using GPU operations only.
    Inside object: negative distances
    Outside object: positive distances
    """
    mask = mask.float()
    inv_mask = 1 - mask

    # Compute outward distance (background distance)
    dist_out = torch.zeros_like(mask)
    x = inv_mask.clone()
    for i in range(iterations):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        dist_out += (x > 0).float()

    # Compute inward distance (foreground distance)
    dist_in = torch.zeros_like(mask)
    x = mask.clone()
    for i in range(iterations):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        dist_in += (x > 0).float()

    # Signed distance (positive outside, negative inside)
    sdf = dist_out - dist_in

    # Normalize to [-1, 1]
    sdf = sdf / (sdf.abs().amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return sdf

def boundary_loss(pred, target, reduction='mean'):
    """
    Boundary Loss (Kervadec et al., 2019)
    Args:
        pred: predicted probabilities after sigmoid (B,1,H,W)
        target: ground truth binary mask (B,1,H,W)
    Returns:
        scalar boundary loss
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    pred = pred.float()
    target = target.float()

    # Compute batched SDFs entirely on GPU
    sdf = fast_signed_distance_map(target)

    # Weighted boundary loss (mean over batch)
    loss = (pred * sdf).mean()

    if reduction == 'sum':
        return loss * pred.numel()
    elif reduction == 'none':
        return loss
    else:
        return loss

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


# def _surface_points(mask_np):
#     """Return boundary coordinates of a binary mask."""
#     if mask_np.ndim == 3:  # (C,H,W) -> merge channels
#         mask_np = mask_np[0]
#     mask_np = mask_np.astype(bool)
#     return np.argwhere(mask_np ^ binary_erosion(mask_np))

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

# Training loop with Dice Score reporting (DDP-compatible)
def train(rank, world_size, model, train_loader, optimizer, epoch, batch_size, hd_dice_loss):
    model.train()
    running_loss = running_dice = running_precision = running_recall = 0.0
    running_hd95 = running_assd = total_infer_time = total_mem = 0.0
    running_hd = running_dsc = 0.0
    total_batches = 0

    train_loader.sampler.set_epoch(epoch)

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", disable=(rank != 0)):
        images, masks = images.to(rank, non_blocking=True), masks.to(rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()

        # ✅ AMP context
        # with autocast():
        outputs = model(images)
        #loss = boundary_loss(outputs, masks)
        loss, l_hd, l_dsc = hd_dice_loss(outputs, masks)

        torch.cuda.synchronize()
        infer_time = time.time() - t0

        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        total_mem += mem_gb
        total_infer_time += infer_time
        total_batches += 1

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward() 
        optimizer.step()

        preds = (outputs > 0.5).float()
        running_hd += l_hd
        running_dsc += l_dsc
        running_loss += loss.item()
        running_dice += dice(preds, masks)
        running_precision += precision(preds, masks)
        running_recall += recall(preds, masks)
        running_hd95 += hd95(preds, masks)
        running_assd += assd(preds, masks)
    
    # ⬇️ average across GPUs
    metrics = torch.tensor([
        running_hd / len(train_loader),
        running_dsc / len(train_loader),
        running_loss / len(train_loader),
        running_dice / len(train_loader),
        running_precision / len(train_loader),
        running_recall / len(train_loader),
        running_hd95 / len(train_loader),
        running_assd / len(train_loader),
        total_infer_time / total_batches if total_batches > 0 else 0.0,
        total_mem / total_batches if total_batches > 0 else 0.0
    ], device=rank)

    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= world_size
    avg_hd, avg_dsc, avg_loss, avg_dice, avg_prec, avg_rec, avg_hd95, avg_assd, avg_ipe, avg_mem = metrics.tolist()

    if rank == 0:
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f} - L_HD: {avg_hd:.4f} - L_DSC: {avg_dsc:.4f} - Lambda: {hd_dice_loss.lmbda:.2f} - Dice: {avg_dice:.2f}, Prec: {avg_prec:.2f}, Rec: {avg_rec:.2f}, HD95: {avg_hd95:.2f}, ASSD: {avg_assd:.2f}, IPE: {avg_ipe:.2f}, Mem: {avg_mem:.2f}")
    return avg_loss, avg_hd, avg_dsc, avg_dice, avg_prec, avg_rec, avg_hd95, avg_assd, avg_ipe, avg_mem

# Validation loop with Dice Score reporting (DDP-compatible)
def validate(rank, world_size, model, val_loader, batch_size, hd_dice_loss):
    model.eval()
    val_loader.sampler.set_epoch(0)
    
    running_loss = 0.0
    running_dice = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_hd95 = 0.0
    running_assd = 0.0
    total_infer_time = 0.0
    total_batches = 0
    total_mem = 0.0 
    running_hd = running_dsc = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation", disable=(dist.get_rank() != 0)):
            images, masks = images.to(rank), masks.to(rank)
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            infer_time = time.time() - t0

            # batch memory (GB)
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total_mem += mem_gb
            total_infer_time += infer_time
            total_batches += 1
            
            loss, l_hd, l_dsc = hd_dice_loss(outputs, masks)
            dice_score = dice((outputs>0.5).float(), masks)
            precision_score = precision((outputs>0.5).float(), masks)
            recall_score = recall((outputs>0.5).float(), masks)
            hd95_score = hd95((outputs>0.5).float(), masks)
            assd_score = assd((outputs>0.5).float(), masks)

            running_loss += loss.item()
            running_hd += l_hd
            running_dsc += l_dsc
            running_dice += dice_score
            running_precision += precision_score
            running_recall += recall_score
            running_hd95 += hd95_score
            running_assd += assd_score
    
    # ---------- Average across batches ----------
    metrics = torch.tensor([
        running_loss / len(val_loader),
        running_hd / len(val_loader),
        running_dsc / len(val_loader),
        running_dice / len(val_loader),
        running_precision / len(val_loader),
        running_recall / len(val_loader),
        running_hd95 / len(val_loader),
        running_assd / len(val_loader),
        total_infer_time / total_batches if total_batches > 0 else 0.0,
        total_mem / total_batches if total_batches > 0 else 0.0
    ], device=rank)

    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= world_size

    avg_loss, avg_hd, avg_dsc, avg_dice, avg_prec, avg_rec, avg_hd95, avg_assd, avg_ipe, avg_mem = metrics.tolist()

    if dist.get_rank() == 0:
        print(f"Val. Loss: {avg_loss:.4f}, L_HD: {avg_hd:.4f}, L_DSC: {avg_dsc:.4f}, Val. DSC: {avg_dice:.2f}, Val. Precision: {avg_prec:.2f}, Val. Recall: {avg_rec:.2f}, Val. HD95: {avg_hd95:.2f}, Val. ASSD: {avg_assd:.2f}, Val. IPE: {avg_ipe:.2f}, Val. Mem: {avg_mem:.2f}")
        
    return avg_loss, avg_hd, avg_dsc, avg_dice, avg_prec, avg_rec, avg_hd95, avg_assd, avg_ipe, avg_mem

# Testing function (DDP-compatible)
def test(rank, world_size, model, test_loader, batch_size, lmbda):
    model.eval()
    test_loader.sampler.set_epoch(0)
    
    running_loss = running_dice = running_precision = running_recall = 0.0
    running_hd95 = running_assd = 0.0
    running_hd = running_dsc = 0.0
    total_infer_time = 0.0
    total_batches = 0
    total_mem = 0.0

    # ---------- FLOPs (GFLOPs per batch) ----------
    if rank == 0:
        sample_input = torch.randn(1, *next(iter(test_loader))[0].shape[1:]).to(rank)
        flops, params = profile(model, inputs=(sample_input,), verbose=False)
        flops_g = flops / 1e9   # GFLOPs / batch
    dist.barrier()
    
    hd_dice_loss = AdaptiveHD_DSC_Loss(initial_lambda = lmbda)

    # ---------- Inference ----------
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f"Testing (Rank {rank})"):
            images, masks = images.to(rank), masks.to(rank)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            infer_time = time.time() - t0

            # batch memory (GB)
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            total_mem += mem_gb
            total_infer_time += infer_time
            total_batches += 1

            loss, l_hd, l_dsc = hd_dice_loss(outputs, masks)
            dice_score = dice((outputs>0.5).float(), masks)
            prec_score = precision((outputs>0.5).float(), masks)
            rec_score = recall((outputs>0.5).float(), masks)
            hd95_score = hd95((outputs>0.5).float(), masks)
            assd_score = assd((outputs>0.5).float(), masks)

            running_loss += loss.item()
            running_hd += l_hd
            running_dsc += l_dsc
            running_dice += dice_score
            running_precision += prec_score
            running_recall += rec_score
            running_hd95 += hd95_score
            running_assd += assd_score

    # ---------- Average across batches ----------
    metrics = torch.tensor([
        running_loss / len(test_loader),
        running_hd / len(test_loader),
        running_dsc / len(test_loader),
        running_dice / len(test_loader),
        running_precision / len(test_loader),
        running_recall / len(test_loader),
        running_hd95 / len(test_loader),
        running_assd / len(test_loader),
        total_infer_time / total_batches if total_batches > 0 else 0.0,
        total_mem / total_batches if total_batches > 0 else 0.0
    ], device=rank)

    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /= world_size

    avg_loss, avg_hd, avg_dsc, avg_dice, avg_prec, avg_rec, avg_hd95, avg_assd, avg_ipe, avg_mem = metrics.tolist()

    if dist.get_rank() == 0:
        print(
            f"Te. Loss: {avg_loss:.4f} | L_HD: {avg_hd:.4f} | L_DSC: {avg_dsc:.4f} | DSC: {avg_dice:.2f}% | "
            f"Precision: {avg_prec:.2f}% | Recall: {avg_rec:.2f}% | "
            f"HD95: {avg_hd95:.2f} px | ASSD: {avg_assd:.2f} px | "
            f"IPE: {avg_ipe:.2f} s/b | Mem: {avg_mem:.2f} GB/b | FLOPs: {flops_g:.2f} G/b"
        )

    if rank == 0:
        return (
            avg_loss, avg_hd, avg_dsc, avg_dice, avg_prec, avg_rec, avg_hd95, avg_assd,
            avg_ipe, avg_mem, flops_g
        )
    else:
        return (
            avg_loss, avg_hd, avg_dsc, avg_dice, avg_prec, avg_rec, avg_hd95, avg_assd,
            avg_ipe, avg_mem, None
        )

def visualize_random_samples(model, test_loader, rank, lf_name, num_samples=5):
    if dist.get_rank() != 0:
        return

    import os
    model.eval()
    ds = test_loader.dataset
    indices = random.sample(range(len(ds)), num_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(8, 3*num_samples))

    for i, idx in enumerate(indices):
        # Load sample directly by index (bypasses sampler order, fine for viz)
        image, target_mask = ds[idx]
        path = getattr(ds, "image_paths", None)
        title = '/'.join(path[idx].split('/')[-3:]) if path is not None else f"idx {idx}"

        image = image.unsqueeze(0).to(rank, non_blocking=True)
        target_mask = target_mask.unsqueeze(0).to(rank, non_blocking=True)

        with torch.no_grad():
            pred_mask = (model(image) > 0.5)

        target_np = target_mask.squeeze().cpu().numpy()
        pred_np   = pred_mask.squeeze().cpu().numpy()

        axes[i, 0].imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title(f"Input: {title}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(target_np, cmap='gray')
        axes[i, 1].set_title("Target")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"WTBSegmentation_PT_{lf_name}.png")
    plt.show()


def cleanup():
    dist.destroy_process_group()
    
def init_log_csv(log_path, headers):
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_log_csv(log_path, row):
    if dist.get_rank() == 0:
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

def write_test_result(test_log_path, test_loss, test_hd, test_dsc, best_lambda, test_dice, test_precision, test_recall, test_hd95, test_assd, test_ipe, test_mem, test_flops):
    if dist.get_rank() == 0:
        with open(test_log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['test_loss', 'test_hd', 'test_dsc', 'best_lambda', 'test_dice', 'test_precision', 'test_recall', 'test_hd95', 'test_assd', 'test_ipe', 'test_mem', 'test_flops'])
            writer.writerow([test_loss, test_hd, test_dsc, best_lambda, test_dice, test_precision, test_recall, test_hd95, test_assd, test_ipe, test_mem, test_flops])

def main():
    
    # Get the rank and world size from environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)

    # Initialize the process group
    dist.init_process_group(backend="nccl")

    # Transforms
    size = 512
    batch_size = 4
    lf_name = 'HD+Dice'

    # Load data (your prepare_data function)
    # train_images, train_masks, val_images, val_masks, test_images_c, test_masks_c, test_images_u, test_masks_u = prepare_data(
    #     r"/home/angel.encalada/Documents/WTBSegmentation/dataset/Dataset"
    # )
    
    train_images, train_masks, val_images, val_masks, test_images_c, test_masks_c = prepare_data(
        r"/home/angel.encalada/Documents/WTBSegmentation/dataset/Blade30"
    )
    
    train_dataset_org = ImageDataset(train_images, train_masks, size=size, num_classes=2)
    val_dataset_org = ImageDataset(val_images, val_masks, size=size, num_classes=2)
    test_dataset_c_org = ImageDataset(test_images_c, test_masks_c, size=size , num_classes=2)
    #test_dataset_u_org = ImageDataset(test_images_u, test_masks_u, size=size , num_classes=2)
    
    # train_dataset_aug = ImageDataset(train_images, train_masks, size=size, num_classes=2, augmentation=True)
    # val_dataset_aug = ImageDataset(val_images, val_masks, size=size, num_classes=2, augmentation=True)
    # test_dataset_aug = ImageDataset(test_images, test_masks, size=size , num_classes=2, augmentation=True)
    
    train_dataset = train_dataset_org #+ train_dataset_aug
    val_dataset = val_dataset_org #+ val_dataset_aug
    test_dataset_c = test_dataset_c_org #+ test_dataset_aug
    #test_dataset_u = test_dataset_u_org

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=42
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42
    )
    test_sampler_c = DistributedSampler(
        test_dataset_c,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42
    )
    # test_sampler_u = DistributedSampler(
    #     test_dataset_u,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=False,
    #     seed=42
    # )

    num_workers = 16  # adjust per GPU
    pin_memory = True
    prefetch_factor = 3

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    test_loader_c = DataLoader(
        test_dataset_c,
        batch_size=batch_size,
        sampler=test_sampler_c,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    # test_loader_u = DataLoader(
    #     test_dataset_u,
    #     batch_size=batch_size,
    #     sampler=test_sampler_u,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=prefetch_factor
    # )

    # --- Modelo y optimizador ---
    model = UNet(out_channels=1)

    # 🔹 Convertir BatchNorm a SyncBatchNorm (seguro para DDP)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(rank)

    # 🔹 Envolver en DDP una sola vez
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,  # Optimizer you're using (e.g., Adam, SGD)
        mode='min',  # Minimize validation loss
        factor=0.5,  # Reduce LR by half
        patience=5,  # Wait 5 epochs before reducing LR
        min_lr=1e-6  # Minimum LR value to avoid too much reduction
    )
    early_stopping = EarlyStopping(patience=10, mode='min')
    
    # --- Logs ---
    if rank == 0:
        train_log_path = f'logs/train_log_PT_{lf_name}.csv'
        init_log_csv(train_log_path, [
            'epoch', 'train_loss', 'train_hd', 'train_dsc', 'train_lmbda', 'train_dice', 'train_precision', 'train_recall',
            'train_hd95', 'train_assd', 'train_ipe', 'train_mem', 'val_loss', 'val_hd', 'val_dsc', 'val_dice', 'val_precision',
            'val_recall', 'val_hd95', 'val_assd', 'val_ipe', 'val_mem'
        ])
        
        test_log_path = f'logs/test_log_PT_{lf_name}.csv'
        init_log_csv(test_log_path, [
            'test_loss', 'test_hd', 'test_dsc', 'best_lambda', 'test_dice', 'test_precision', 'test_recall', 'test_hd95', 'test_assd', 'test_ipe', 'test_mem', 'test_flops'
            ])
        
        #writer = SummaryWriter(log_dir = f"runs/WTBS_PT_{lf_name}")

    # --- Entrenamiento ---
    num_epochs = 250
    hd_dice_loss = AdaptiveHD_DSC_Loss()
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss, train_hd, train_dsc, train_dice, train_precision, train_recall, train_hd95, train_assd, train_ipe, train_mem = train(
            rank, world_size, model, train_loader, optimizer, epoch, batch_size, hd_dice_loss
        )
        
        val_loss, val_hd, val_dsc, val_dice, val_precision, val_recall, val_hd95, val_assd, val_ipe, val_mem = validate(
            rank, world_size, model, val_loader, batch_size, hd_dice_loss
        )
        scheduler.step(val_loss)

        if rank == 0:
            append_log_csv(train_log_path, [
                epoch + 1, train_loss, train_hd, train_dsc, hd_dice_loss.lmbda, train_dice, train_precision, train_recall, train_hd95, train_assd, train_ipe, train_mem,
                val_loss, val_hd, val_dsc, val_dice, val_precision, val_recall, val_hd95, val_assd, val_ipe, val_mem
            ])
            
            # writer.add_scalar('Loss/Train', train_loss, epoch)
            # writer.add_scalar('Loss/Val', val_loss, epoch)
            # writer.add_scalar('Dice/Train', train_dice, epoch)
            # writer.add_scalar('Dice/Val', val_dice, epoch)
            # writer.add_scalar('Precision/Train', train_precision, epoch)
            # writer.add_scalar('Precision/Val', val_precision, epoch)
            # writer.add_scalar('Recall/Train', train_recall, epoch)
            # writer.add_scalar('Recall/Val', val_recall, epoch)
            # writer.add_scalar('HD95/Train', train_hd95, epoch)
            # writer.add_scalar('HD95/Val', val_hd95, epoch)
            # writer.add_scalar('ASSD/Train', train_assd, epoch)
            # writer.add_scalar('ASSD/Val', val_assd, epoch)
            # writer.add_scalar('IPE/Train', train_ipe, epoch)
            # writer.add_scalar('IPE/Val', val_ipe, epoch)
            # writer.add_scalar('Mem/Train', train_mem, epoch)
            # writer.add_scalar('Mem/Val', val_mem, epoch)

        early_stopping(model, val_loss, epoch+1, lf_name, hd_dice_loss.lmbda)
        hd_dice_loss.update_lambda(train_hd, train_dsc)
        if early_stopping.early_stop:
            if rank == 0:
                print("Early stopping triggered!")
            break
    
    best_model_name = early_stopping.get_best_model(lf_name)
    best_model = UNet(out_channels=1)
    best_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(best_model)
    
    # 1️⃣ Load checkpoint BEFORE wrapping in DDP
    checkpoint = torch.load(best_model_name, map_location=f"cuda:{rank}")
    
    # 2️⃣ If the model was saved from DDP (has "module." keys):
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    
    best_model.load_state_dict(checkpoint, strict=False)
    
    # 3️⃣ Now move to GPU and wrap in DDP
    best_model = best_model.to(rank)
    best_model = DDP(best_model, device_ids=[rank], output_device=rank)
    
    print('Testing on easy samples')
    test_loss, test_hd, test_dsc, test_dice, test_precision, test_recall, test_hd95, test_assd, test_ipe, test_mem, test_flops = test(
        rank, world_size, best_model.module, test_loader_c, batch_size, early_stopping.best_lmbda
    )
    if rank == 0:
        append_log_csv(test_log_path, [test_loss, test_hd, test_dsc, early_stopping.best_lmbda, test_dice, test_precision, test_recall,
                          test_hd95, test_assd, test_ipe, test_mem, test_flops])
    
    # print('Testing on hard samples')
    # test_loss, test_hd, test_dsc, test_dice, test_precision, test_recall, test_hd95, test_assd, test_ipe, test_mem, test_flops = test(
    #     rank, world_size, best_model.module, test_loader_u, batch_size, early_stopping.best_lmbda
    # )
    # if rank == 0:
    #     append_log_csv(test_log_path, [test_loss, test_hd, test_dsc, early_stopping.best_lmbda, test_dice, test_precision, test_recall,
    #                       test_hd95, test_assd, test_ipe, test_mem, test_flops])
    #     visualize_random_samples(best_model.module, test_loader_u, rank, lf_name, num_samples=10)

    # --- Limpieza ---
    dist.barrier()
    dist.destroy_process_group()

main()
