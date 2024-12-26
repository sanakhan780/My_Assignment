import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class LandslideDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.file_names = f.read().splitlines()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image_path, mask_path = self.file_names[idx].split()
        image = preprocess_image(image_path, input_channels=14)
        mask = preprocess_mask(mask_path)
        return {'image': image, 'mask': mask}

def preprocess_image(image_path, input_channels):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    assert image.shape[2] == input_channels, f"Expected {input_channels} channels, got {image.shape[2]}"
    image = torch.tensor(image.transpose(2, 0, 1)) / 255.0  # Normalize to [0, 1]
    return image

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.long)
    return torch.tensor(mask)
