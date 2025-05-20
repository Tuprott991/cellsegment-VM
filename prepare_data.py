import numpy as np
import pandas as pd
import os
import numpy as np
import cv2
from tqdm import tqdm

def rle_decode(mask_rle, shape):
    """
    Decode run-length encoded mask into binary mask.
    mask_rle: string of space-separated pairs
    shape: (height, width)
    Returns numpy array of shape (height, width)
    """
    s = list(map(int, mask_rle.strip().split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape((shape[0], shape[1]))  # Sửa lại chiều reshape

df = pd.read_csv("train.csv")
output_image_dir = "prepared/images"
output_mask_dir = "prepared/masks"
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Nhóm theo image ID
df['image_id'] = df['id'].apply(lambda x: x.split('_')[0])
grouped = df.groupby('image_id')

for image_id, group in tqdm(grouped):
    image_path = f"train/{image_id}.png"
    img = cv2.imread(image_path)
    height, width = group.iloc[0][['height', 'width']].astype(int)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for _, row in group.iterrows():
        m = rle_decode(row['annotation'], shape=(height, width))
        mask = np.maximum(mask, m)  # OR combine multiple cells

    # Resize (nếu muốn)
    img = cv2.resize(img, (256, 256))
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    np.save(f"{output_image_dir}/{image_id}.npy", img)
    np.save(f"{output_mask_dir}/{image_id}.npy", mask)
