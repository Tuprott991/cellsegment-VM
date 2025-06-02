import sys
import numpy as np
import matplotlib.pyplot as plt

def show_npy_image(file_path):
    img = np.load(file_path)
    plt.imshow(img, cmap='gray')
    plt.title(f"Image: {file_path}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    file_path = "prepared/masks/0030fd0e6378.npy"  # Replace with your .npy file path
    show_npy_image(file_path)