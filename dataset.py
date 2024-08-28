import sys
import os

# Initialisierung des PYTHONPATH
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from collections import defaultdict
import matplotlib.pyplot as plt

#from train.train import run
import numpy as np

import torch
import torch.utils
from torchvision import tv_tensors
from torchvision.transforms.v2 import CutMix, MixUp

from torch.cuda.amp import autocast, GradScaler
import time
import copy
#from models.UNetBatchNorm import UNetBatchNorm
from UNetBatchNorm import UNetBatchNorm
#from train import run

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Angepasste CustomDataset-Klasse, um auch ungelabelte Daten zu unterstützen
class CustomDataset(Dataset):
    def __init__(self, root_dir, dataset_name=None, transform=None, mask_transform=None, count=None, is_labeled=True):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.mask_transform = mask_transform
        self.count = count
        self.is_labeled = is_labeled

        # Pfade zu den Bildern und Masken
        self.image_folder = os.path.join(root_dir, 'grabs')
        if is_labeled:
            self.mask_folder = os.path.join(root_dir, 'masks')

        # Liste aller Bilddateien
        all_image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))

        self.image_files = []
        if is_labeled:
            self.mask_files = []

        for image_file in all_image_files:
            base_name = os.path.splitext(image_file)[0]
            self.image_files.append(image_file)

            if is_labeled:
                if dataset_name == "RetinaVessel":
                    mask_name = f"{base_name}.tiff"
                elif dataset_name == "Ölflecken":
                    mask_name = f"{base_name}_1.bmp"
                elif dataset_name == "circle_data":
                    mask_name = f"{base_name}1.png"
                else:
                    mask_name = f"{base_name}.tif"  # Gleicher Name wie das Bild

                if os.path.exists(os.path.join(self.mask_folder, mask_name)):
                    self.mask_files.append(mask_name)
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_name}")

        if self.count is not None:
            self.image_files = self.image_files[:self.count]
            if is_labeled:
                self.mask_files = self.mask_files[:self.count]

        print(f"Found {len(self.image_files)} images")
        if is_labeled:
            print(f"Found {len(self.mask_files)} masks")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.image_folder, self.image_files[idx])
            image = Image.open(img_name).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            if self.is_labeled:
                mask_name = os.path.join(self.mask_folder, self.mask_files[idx])
                mask = Image.open(mask_name).convert('L')

                if self.mask_transform is not None:
                    mask = self.mask_transform(mask)

                return image, mask
            else:
                return image

        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None if self.is_labeled else None


def get_dataloaders(root_dir, dataset_name=None, batch_size=None, transformations=None, split_size=0.8):
    transformations = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    transformations_mask = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=False),
    ])

    dataset = CustomDataset(root_dir=root_dir, dataset_name=dataset_name, transform=transformations, mask_transform=transformations_mask)

    dataset_size = len(dataset)
    train_size = int(split_size * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders, dataset

def show_image_and_mask(image, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy().astype(np.float32))
    ax[1].imshow(mask.squeeze().cpu().numpy().astype(np.float32), cmap='gray')
    plt.show()

# Beispielhafte Nutzung:
if __name__ == '__main__':
    train_dir = 'data/Dichtflächen_Cropped'
    dataset_name = 'Dichtflächen_Cropped'

    dataloaders,_ = get_dataloaders(root_dir=train_dir, dataset_name=dataset_name, batch_size=20)
    
    batch = next(iter(dataloaders['train']))
    images,masks = batch
    print(images.shape)
    print(masks.shape)

     

    for i in range(5,10):
        show_image_and_mask(images[i],masks[i])
        print(f"Image min: {images[i].min()}, max: {images[i].max()}")
        print(f"Mask min: {masks[i].min()}, max: {masks[i].max()}")