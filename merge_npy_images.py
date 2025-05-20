import os
import numpy as np

image_dir = "prepared/images"
mask_dir = "prepared/masks"

# Lấy danh sách file và sort để đảm bảo đúng thứ tự
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

# Đảm bảo số lượng ảnh và mask bằng nhau
assert len(image_files) == len(mask_files), "Số lượng ảnh và mask không khớp!"

images = []
masks = []

for img_file, mask_file in zip(image_files, mask_files):
    assert img_file == mask_file, f"Tên file không khớp: {img_file} vs {mask_file}"
    img = np.load(os.path.join(image_dir, img_file))
    mask = np.load(os.path.join(mask_dir, mask_file))
    images.append(img)
    masks.append(mask)

images = np.stack(images, axis=0)
masks = np.stack(masks, axis=0)

np.save("data_train.npy", images)
np.save("mask_train.npy", masks)

print(f"Đã lưu {images.shape[0]} ảnh vào data_train.npy và mask_train.npy")