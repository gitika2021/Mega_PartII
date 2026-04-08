import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
def symmetry_aware_dice_loss(y_true, y_pred, epsilon=1e-6):
    # 1. Standard Orientation
    intersection1 = (y_true * y_pred).sum(dim=(1, 2))
    union1 = y_true.sum(dim=(1, 2)) + y_pred.sum(dim=(1, 2))
    dice1 = 1 - (2. * intersection1 + epsilon) / (union1 + epsilon)
    
    # 2. Flipped Orientation (Vertical Flip)
    y_true_flipped = torch.flip(y_true, dims=[1]) # Flip H (dim 1 is Height if shape is B,H,W)
    intersection2 = (y_true_flipped * y_pred).sum(dim=(1, 2))
    union2 = y_true_flipped.sum(dim=(1, 2)) + y_pred.sum(dim=(1, 2))
    dice2 = 1 - (2. * intersection2 + epsilon) / (union2 + epsilon)
    
    # Take the minimum loss (best alignment) for each item in batch
    loss = torch.min(dice1, dice2)
    return loss.mean()
def print_grad_stats(model, step):
    print(f"\n[Step {step}] Gradient Stats:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: min={param.grad.min():.4f}, max={param.grad.max():.4f}, norm={param.grad.norm():.4f}")

def add_noise_to_batch(lc_batch: torch.Tensor, snr: float):
    noise = torch.randn_like(lc_batch) * (1/snr)
    return lc_batch + noise

def symmetry_aware_bce(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    y_true = y_true.to(dtype=torch.float32, device=y_pred.device)
    y_true_flipped = torch.flip(y_true, dims=[1])

    bce_normal = F.binary_cross_entropy(y_pred, y_true, reduction="none").mean(dim=(1, 2))
    bce_flipped = F.binary_cross_entropy(y_pred, y_true_flipped, reduction="none").mean(dim=(1, 2))

    loss = torch.min(bce_normal, bce_flipped).mean()
    return loss


def symmetry_aware_mse(true_shape, predicted_shape):
    if isinstance(true_shape, np.ndarray):
        true_shape = torch.tensor(true_shape, dtype=torch.float32)
    if isinstance(predicted_shape, np.ndarray):
        predicted_shape = torch.tensor(predicted_shape, dtype=torch.float32)

    true_flipped = torch.flip(true_shape, dims=[1])

    mse_normal = ((true_shape - predicted_shape) ** 2).mean(dim=(1, 2))
    mse_flipped = ((true_flipped - predicted_shape) ** 2).mean(dim=(1, 2))

    min_mse = torch.min(mse_normal, mse_flipped).mean()
    return min_mse

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) metric for binary masks.
    """
    pred_mask = (pred_mask > threshold).float()
    true_mask = (true_mask > threshold).float()

    intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3)) - intersection
    
    # Handle division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def add_real_noise_to_batch(lc_batch: torch.Tensor, snr: float):
    noise = torch.randn_like(lc_batch) * (1/snr)
    return lc_batch + noise