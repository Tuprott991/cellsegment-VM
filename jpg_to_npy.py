# Convert jpg images to npy format for cell segmentation dataset
import os
import numpy as np
import cv2
import pandas as pd
# import group

from tqdm import tqdm

# convert single jpg image to npy
def convert_jpg_to_npy(image_path, output_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    np.save(output_path, img)

convert_jpg_to_npy('1c48c0bd-1d27-4ef1-9246-d8dcd8da2776.jpg', '1c48c0bd-1d27-4ef1-9246-d8dcd8da2776.npy')