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
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + eps) / (union + eps)

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
    images_dir = "prepared/images"
    masks_dir = "prepared/masks"
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
            loss_inst = dice_loss(S, masks.unsqueeze(1).float())
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
            predicted_masks = model.segment_instances(S, P, E, O, iou_threshold=0.6)  # Post-process to get instance masks
            if len(predicted_masks) == 0:
                pred_mask = torch.zeros_like(masks[0])  # [H, W]
            else:
                pred_mask = torch.clamp(torch.stack(predicted_masks).sum(dim=0), max=1.0)  # [H, W]
            loss_inst = dice_loss(pred_mask.unsqueeze(0).unsqueeze(0), masks[0].unsqueeze(0).unsqueeze(0).float())
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
                predicted_mask = model.segment_instances(S, P, E, O, iou_threshold=0.6)
                # Compute F1 score for validation
                val_f1 = f1_score(predicted_mask, masks.unsqueeze(1).float()).item()
                val_f1s.append(val_f1)
        mean_f1 = np.mean(val_f1s)
        print(f"Epoch {epoch+1}: Val F1 = {mean_f1:.4f}")

        # Save best model
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {epoch+1} with F1 {best_f1:.4f}")

if __name__ == "__main__":

    train()