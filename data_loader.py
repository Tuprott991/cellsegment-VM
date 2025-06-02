import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NPYCellDataset(Dataset):
    """
    Dataset for loading .npy images and masks for cell segmentation.
    Assumes:
      - images_dir: folder with (256,256,3) .npy images
      - masks_dir: folder with (256,256) .npy masks
      - Each image file has a corresponding mask file with the same name.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.filenames = [
            f for f in os.listdir(images_dir)
            if f.endswith('.npy') and os.path.isfile(os.path.join(masks_dir, f))
        ]
        self.filenames.sort()  # Optional: sort for reproducibility

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = np.load(img_path)  # shape: (256,256,3)
        mask = np.load(mask_path)  # shape: (256,256)

        # Convert to torch tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0  # [3,256,256], normalized
        mask = torch.from_numpy(mask).long()  # [256,256]

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

def get_loader(images_dir, masks_dir, batch_size=8, shuffle=True, num_workers=2, transform=None):
    dataset = NPYCellDataset(images_dir, masks_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Example usage:
if __name__ == "__main__":
    images_dir = "images"
    masks_dir = "masks"
    loader = get_loader(images_dir, masks_dir, batch_size=4)
    for images, masks in loader:
        print("Batch images:", images.shape)  # [B,3,256,256]
        print("Batch masks:", masks.shape)    # [B,256,256]
        break