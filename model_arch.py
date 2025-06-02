import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.coordinate_utils import make_coord_grid
from tqdm import tqdm
import torchsummary as summary


def conv_norm_act(in_channels, out_channels, sz, norm, act="ReLU", depthwise=False):
    """
    Convolution + Normalization + Activation block.
    If depthwise=True, uses depthwise conv.
    """
    if norm is None or norm == "None":
        norm_layer = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(out_channels, eps=1e-5, track_running_stats=False, affine=True)
    else:
        raise ValueError("Norm must be None, batch, or instance")

    if act is None or act == "None":
        act_layer = nn.Identity()
    elif act.lower() == "relu":
        act_layer = nn.ReLU(inplace=True)
    elif act.lower() == "relu6":
        act_layer = nn.ReLU6(inplace=True)
    elif act.lower() == "mish":
        act_layer = nn.Mish(inplace=True)
    else:
        raise ValueError("Act must be None, ReLU, ReLU6, or Mish")

    if depthwise:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2, groups=in_channels),
            norm_layer,
            act_layer,
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
            norm_layer,
            act_layer,
        )


class EncoderBlock(nn.Module):
    """
    Encoder block with optional pooling and residual-like shortcuts.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        pool=True,
        norm="BATCH",
        act="ReLU",
        shallow=False,
    ):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2) if pool else nn.Identity()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv2 = nn.Identity()
            self.conv3 = nn.Identity()

    def forward(self, x):
        x = self.maxpool(x)
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        return x


class DecoderBlock(nn.Module):
    """
    Decoder block: upsample, combine skip, and a series of convs with residual-like structure.
    """
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm="BATCH",
        act="ReLU",
        shallow=False,
    ):
        super().__init__()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv_skip = conv_norm_act(skip_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x + self.conv_skip(skip))
        x = x + self.conv4(self.conv3(x))
        return x


class UNetVariant(nn.Module):
    """
    U-Net variant using the provided EncoderBlock and DecoderBlock.
    """
    def __init__(self, in_channels=3, base_channels=32, norm="BATCH", act="ReLU"):
        super().__init__()
        # Encoder: 5 levels (including input level without pooling)
        self.enc1 = EncoderBlock(in_channels, base_channels, pool=False, norm=norm, act=act)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, pool=True, norm=norm, act=act)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, pool=True, norm=norm, act=act)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8, pool=True, norm=norm, act=act)
        self.enc5 = EncoderBlock(base_channels * 8, base_channels * 16, pool=True, norm=norm, act=act)

        # Decoder: mirror of encoder (skip connections)
        self.dec4 = DecoderBlock(base_channels * 16, base_channels * 8, base_channels * 8, norm=norm, act=act)
        self.dec3 = DecoderBlock(base_channels * 8, base_channels * 4, base_channels * 4, norm=norm, act=act)
        self.dec2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2, norm=norm, act=act)
        self.dec1 = DecoderBlock(base_channels * 2, base_channels, base_channels, norm=norm, act=act)

    def forward(self, x):
        # Encoder forward
        x1 = self.enc1(x)   # [B, base_channels, H, W]
        x2 = self.enc2(x1)  # [B, 2*base_channels, H/2, W/2]
        x3 = self.enc3(x2)  # [B, 4*base_channels, H/4, W/4]
        x4 = self.enc4(x3)  # [B, 8*base_channels, H/8, W/8]
        x5 = self.enc5(x4)  # [B, 16*base_channels, H/16, W/16]

        # Decoder forward
        d4 = self.dec4(x5, x4)  # [B, 8*base_channels, H/8, W/8]
        d3 = self.dec3(d4, x3)  # [B, 4*base_channels, H/4, W/4]
        d2 = self.dec2(d3, x2)  # [B, 2*base_channels, H/2, W/2]
        d1 = self.dec1(d2, x1)  # [B, base_channels,   H,   W]

        return d1  # final feature map


class HeadS(nn.Module):
    """
    Seed head for distance regression: outputs 1-channel map.
    """
    def __init__(self, in_ch, mid_ch=None):
        super().__init__()
        mid = mid_ch or in_ch
        self.conv1 = conv_norm_act(in_ch, mid, 3, norm="BATCH", act="ReLU")
        self.conv2 = nn.Conv2d(mid, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  # [B,1,H,W]


class HeadP(nn.Module):
    """
    Positional embedding head: outputs Dp-channel map.
    """
    def __init__(self, in_ch, Dp=2, mid_ch=None):
        super().__init__()
        mid = mid_ch or in_ch
        self.conv1 = conv_norm_act(in_ch, mid, 3, norm="BATCH", act="ReLU")
        self.conv2 = nn.Conv2d(mid, Dp, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  # [B,Dp,H,W]


class HeadE(nn.Module):
    """
    Conditional embedding head: outputs De-channel map.
    """
    def __init__(self, in_ch, De=4, mid_ch=None):
        super().__init__()
        mid = mid_ch or in_ch
        self.conv1 = conv_norm_act(in_ch, mid, 3, norm="BATCH", act="ReLU")
        self.conv2 = nn.Conv2d(mid, De, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  # [B,De,H,W]


class PhiHead(nn.Module):
    """
    Instance segmentation MLP head. Takes (Qij - Qk) and Eij, outputs 1-channel logit.
    """
    def __init__(self, Dp, De, hidden_ch=64):
        super().__init__()
        self.fc1  = nn.Conv2d(Dp + De, hidden_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2  = nn.Conv2d(hidden_ch, 1, kernel_size=1)

    def forward(self, offset_embed, cond_embed):
        x = torch.cat([offset_embed, cond_embed], dim=1)  # [B, Dp+De, H, W]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # [B,1,H,W]


class InstanSegModel(nn.Module):
    """
    InstanSeg model using the provided U-Net variant.
    """
    def __init__(self, in_ch=3, base_ch=32, Dp=2, De=4):
        super().__init__()
        # UNet variant backbone
        self.backbone = UNetVariant(in_channels=in_ch, base_channels=base_ch, norm="BATCH", act="ReLU")
        feat_ch = base_ch  # final backbone channels

        # Heads
        self.head_s = HeadS(feat_ch, mid_ch=feat_ch)
        self.head_p = HeadP(feat_ch, Dp=Dp, mid_ch=feat_ch)
        self.head_e = HeadE(feat_ch, De=De, mid_ch=feat_ch)

        # Phi head for instance probability
        self.phi = PhiHead(Dp=Dp, De=De, hidden_ch=feat_ch)

        self.Dp = Dp
        self.De = De

    def forward(self, x):
        """
        Forward pass: returns S, P, E, O.
        x: [B,3,H,W]
        """
        B, _, H, W = x.shape
        feat = self.backbone(x)      # [B, base_ch, H, W]
        S    = self.head_s(feat)     # [B, 1, H, W]
        P    = self.head_p(feat)     # [B, Dp, H, W]
        E    = self.head_e(feat)     # [B, De, H, W]
        O    = make_coord_grid(H, W, device=x.device, Dp=self.Dp)  # [Dp, H, W]
        return S, P, E, O

    def segment_instances(self, S, P, E, O, iou_threshold=0.6):
        """
        Post-processing to obtain instance masks from S, P, E, O.

        Steps:
          1. Local maxima on S → seed coords.
          2. Compute Q = P + O.
          3. For each seed: Qk = Pk + Ok, offset = Q - Qk.
             Predict logit via phi(offset, E) → prob → binarize.
          4. Merge masks with IoU > threshold.
        """
        B, _, H, W = S.shape
        assert B == 1, "Batch size >1 not supported in segment_instances"

        # 1. Local maxima to find seeds
        s_map = S[0, 0]  # [H, W]
        neigh_max = F.max_pool2d(s_map.unsqueeze(0).unsqueeze(0),
                                 kernel_size=3, stride=1, padding=1)  # [1,1,H,W]
        mask_local_max = (s_map.unsqueeze(0).unsqueeze(0) == neigh_max).float()
        foreground = (s_map > 0).float().unsqueeze(0).unsqueeze(0)
        seeds_mask = mask_local_max * foreground  # [1,1,H,W]
        seed_coords = torch.nonzero(seeds_mask[0, 0], as_tuple=False)  # [[i,j], ...]

        # 2. Compute Q = P + O (broadcast O over batch)
        Q = P + O.unsqueeze(0)  # P: [1,Dp,H,W], O: [Dp,H,W]

        instance_masks = []
        for (i, j) in seed_coords:
            i = int(i)
            j = int(j)
            # Seed embeddings
            Pk = P[0, :, i, j].view(self.Dp, 1, 1)  # [Dp,1,1]
            Ok = O[:, i, j].view(self.Dp, 1, 1)      # [Dp,1,1]
            Qk = Pk + Ok                            # [Dp,1,1]

            # Offset field: Q - Qk → [Dp, H, W]
            offset = Q[0] - Qk                      # [Dp,H,W]
            offset = offset.unsqueeze(0)            # [1,Dp,H,W]
            cond   = E[0].unsqueeze(0)              # [1,De,H,W]

            logit = self.phi(offset, cond)          # [1,1,H,W]
            prob  = torch.sigmoid(logit)[0, 0]      # [H, W]
            mask  = (prob > 0.5).float()            # binary [H, W]
            instance_masks.append(mask)

        # 3. Merge masks with high IoU
        merged_masks = []
        used = [False] * len(instance_masks)
        for idx, m in enumerate(instance_masks):
            if used[idx]:
                continue
            cur = m
            used[idx] = True
            for jdx in range(idx + 1, len(instance_masks)):
                if used[jdx]:
                    continue
                iou = compute_iou(cur, instance_masks[jdx])
                if iou > iou_threshold:
                    cur = torch.clamp(cur + instance_masks[jdx], max=1.0)
                    used[jdx] = True
            merged_masks.append(cur)

        return merged_masks


def compute_iou(mask1, mask2):
    """
    Compute Intersection-over-Union for two binary masks [H,W].
    """
    inter = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum() - inter + 1e-6
    return inter / union

import matplotlib.pyplot as plt
def show_npy_image(image, mask=None, title=""):
    plt.figure(figsize=(8,4) if mask is not None else (4,4))
    if mask is not None:
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(mask, cmap='jet')  # chỉ hiển thị mask
        plt.title("Predicted Mask")
        plt.axis("off")
    else:
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
import numpy as np

if __name__ == "__main__":
    # Example usage
    # print(torch.cuda.is_available())
    # model = InstanSegModel(in_ch=3, base_ch=32, Dp=2, De=4).to("cuda")
    # x = torch.randn(1, 3, 256, 256).to("cuda")  # Example input
    # S, P, E, O = model(x)

    # # Segment instances
    # masks = model.segment_instances(S, P, E, O)
    # print(f"Number of segmented instances: {len(masks)}")
    # for idx, mask in enumerate(masks):
    #     print(f"Instance {idx+1} mask shape: {mask.shape}")

    # Print model summary
    model = InstanSegModel(in_ch=3, base_ch=32, Dp=2, De=4).to("cuda")
    # summary.summary(model, (3, 256, 256), device="cuda", batch_size=1)

    # use image from prepared/images and show_npy_image to plot S
    import os
    import random
    images_dir = "prepared/images"
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".npy")]
    img_name = random.choice(image_files)
    img_path = os.path.join(images_dir, img_name)
    image = np.load(img_path)  # shape: (256,256,3)
    image_tensor = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0) / 255.0  # [1,3,256,256]
    image_tensor = image_tensor.to("cuda")
    with torch.no_grad():
        S, P, E, O = model(image_tensor)

    # Show result
    # Show result
    show_npy_image(image, S.cpu().squeeze().numpy(), title=f"Predicted Mask for {img_name}")
