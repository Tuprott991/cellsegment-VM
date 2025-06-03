import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from model_arch import InstanSegModel
from data_loader import get_loader
import numpy as np

# --- Losses ---
def dice_single(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2 * intersection + eps) / (union + eps)

def instance_dice_loss(pred_masks, gt_masks):
    # pred_masks: [M, H, W], gt_masks: [N, H, W]
    M, H, W = pred_masks.shape
    N = gt_masks.shape[0]
    dice_matrix = torch.zeros(M, N, device=pred_masks.device)
    for i in range(M):
        for j in range(N):
            pred = pred_masks[i]
            gt = gt_masks[j]
            if pred.shape != gt.shape:
                gt = gt.float()
                gt = torch.nn.functional.interpolate(
                    gt.unsqueeze(0).unsqueeze(0), size=pred.shape, mode='nearest'
                ).squeeze(0).squeeze(0)
            dice_matrix[i, j] = dice_single(pred, gt)
    # Hungarian matching (maximize dice)
    import scipy.optimize
    matched_pred, matched_gt = scipy.optimize.linear_sum_assignment(-dice_matrix.cpu().numpy())
    dice_scores = dice_matrix[matched_pred, matched_gt]
    return 1 - dice_scores.mean()

def bce_loss(pred, target):
    return nn.functional.binary_cross_entropy_with_logits(pred, target)

def l1_loss(pred, target):
    return nn.functional.l1_loss(pred, target)

# --- Validation metric ---
def f1_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)

# --- Training ---
def train():
    # Settings
    images_dir = "/kaggle/input/cell-segment/images"
    masks_dir = "/kaggle/input/cell-segment/masks"
    batch_size = 3
    num_epochs = 30
    pretrain_epochs = 10
    batches_per_epoch = 1000
    lr = 0.001
    val_split = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 50  # Max instances per image

    # Data
    full_loader = get_loader(images_dir, masks_dir, batch_size=1, shuffle=True, num_workers=2)
    dataset = full_loader.dataset
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = InstanSegModel(in_ch=3, base_ch=32, Dp=2, De=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0
    save_path = "best_model.pth"

    # --- Pretraining ---
    print("Pretraining for {} epochs...".format(pretrain_epochs))
    for epoch in range(pretrain_epochs):
        model.train()
        pbar = tqdm(train_loader, total=min(len(train_loader), batches_per_epoch), desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}")
        for i, (images, masks) in enumerate(pbar):
            if i >= batches_per_epoch:
                break
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            S, P, E, O = model(images)
            # Pretrain: BCE for S, Dice for instance masks
            loss_s = bce_loss(S, masks.unsqueeze(1).float())
            # For instance mask, use Dice loss on S as a proxy (since no instance mask yet)
            loss_inst = dice_single(S, masks.unsqueeze(1).float())
            loss = loss_s + loss_inst
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    # --- Main Training ---
    print("Main training for {} epochs...".format(num_epochs))
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, total=min(len(train_loader), batches_per_epoch), desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for i, (images, masks) in enumerate(pbar):
            if i >= batches_per_epoch:
                break
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            S, P, E, O = model(images)
            # L1 regression for S (distance map)
            loss_s = l1_loss(S, masks.unsqueeze(1).float())
            # Instance segmentation loss: Dice loss on up to K instances
            # For simplicity, use S as instance mask proxy (real use: segment_instances and compute loss per instance)
            batch_inst_losses = []
            for b in range(images.shape[0]):
                S_b = S[b:b+1]
                P_b = P[b:b+1]
                E_b = E[b:b+1]
                # O is [Dp, H, W], same for all
                predicted_masks = model.segment_instances(S_b, P_b, E_b, O, iou_threshold=0.6)
                if len(predicted_masks) == 0:
                    pred_masks = torch.zeros((1, *masks.shape[-2:]), device=masks.device)  # [1, H, W]
                else:
                    pred_masks = torch.stack(predicted_masks)  # [M, H, W]
                # Lấy ground truth instance masks cho ảnh này, giả sử masks[b, 1:] là các instance mask [N, H, W]
                gt_masks = masks[b, 1:]  # [N, H, W] hoặc [H, W] nếu chỉ có 1 instance
                if gt_masks.ndim == 2:
                    gt_masks = gt_masks.unsqueeze(0)  # [1, H, W]
                if gt_masks.shape[0] == 0:
                    continue
                gt_masks = gt_masks[gt_masks.sum(dim=(1,2)) > 0]
                if gt_masks.shape[0] == 0:
                    continue
                inst_loss = instance_dice_loss(pred_masks, gt_masks)
                batch_inst_losses.append(inst_loss)
            if len(batch_inst_losses) > 0:
                loss_inst = torch.stack(batch_inst_losses).mean()
            else:
                loss_inst = torch.tensor(0.0, device=images.device)
            loss = loss_s + loss_inst
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        # --- Validation ---
        model.eval()
        val_f1s = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                S, P, E, O = model(images)
                for b in range(images.shape[0]):
                    S_b = S[b:b+1]
                    P_b = P[b:b+1]
                    E_b = E[b:b+1]
                    # O is [Dp, H, W], same for all
                    predicted_masks = model.segment_instances(S_b, P_b, E_b, O, iou_threshold=0.6)
                    if len(predicted_masks) == 0:
                        pred_masks = torch.zeros((1, *masks.shape[-2:]), device=masks.device)  # [1, H, W]
                    else:
                        pred_masks = torch.stack(predicted_masks)  # [M, H, W]
                    gt_masks = masks[b, 1:]  # [N, H, W]
                    gt_masks = gt_masks[gt_masks.sum(dim=(1,2)) > 0]
                    if gt_masks.shape[0] == 0:
                        continue
                    # Tính F1 cho từng cặp matched (Hungarian matching)
                    M, N = pred_masks.shape[0], gt_masks.shape[0]
                    f1_matrix = torch.zeros(M, N, device=pred_masks.device)
                    for i in range(M):
                        for j in range(N):
                            f1_matrix[i, j] = f1_score(pred_masks[i], gt_masks[j])
                    import scipy.optimize
                    matched_pred, matched_gt = scipy.optimize.linear_sum_assignment(-f1_matrix.cpu().numpy())
                    f1_scores = f1_matrix[matched_pred, matched_gt]
                    val_f1s.append(f1_scores.mean().item())
        mean_f1 = np.mean(val_f1s) if len(val_f1s) > 0 else 0.0
        print(f"Epoch {epoch+1}: Val Instance F1 = {mean_f1:.4f}")

        # Save model for each epoch, no need to save best model
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()