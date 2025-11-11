import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# class DiscDataset(Dataset):
#     def __init__(self, csv_path, crop_dir, img_size=(256, 256), augment=False):
#         self.df = pd.read_csv(csv_path)
#         self.crop_dir = os.path.join(crop_dir, "disc")
#         self.img_size = img_size
#         self.augment = augment
#         self.to_tensor = T.ToTensor()
#         self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_p = os.path.join(self.crop_dir, row['Image'])
#         mask_p = os.path.join(self.crop_dir, row['Disc_Mask'])
#         img = cv2.imread(img_p)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
#         mask = (mask > 127).astype(np.uint8)

#         if self.augment: 
#             if random.random() < 0.5: 
#                 img = np.fliplr(img).copy()
#                 mask = np.fliplr(mask).copy()
#             if random.random() < 0.5:
#                 img = np.flipud(img).copy()
#                 mask = np.flipud(mask).copy()

#         img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)

#         img_t = self.to_tensor(img).float()
#         img_t = self.normalize(img_t)
#         mask_t = torch.from_numpy(mask).float()

#         return img_t, mask_t

class DiscDataset(Dataset):
    def __init__(self, img_path, mask_path, img_size=(256, 256), augment=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.fnames = os.listdir(img_path)
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.img_path, fname)
        mask_path = os.path.join(self.mask_path, fname)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5: 
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
        
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)

        img_t = self.to_tensor(img).float()
        img_t = self.normalize(img_t)
        mask_t = torch.from_numpy(mask).float()

        return img_t, mask_t

class CombinedDiscDataset(Dataset):
    def __init__(self, img_path, strong_path, weak_path, img_size=(256, 256), augment=False):
        self.img_path = img_path 
        self.strong_path = strong_path
        self.weak_path = weak_path
        self.fnames = os.listdir(img_path)
        self.img_size = img_size
        self.augment = augment
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   

    def __len__(self):
        return len(self.fnames)   

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.img_path, fname)
        strong_path = os.path.join(self.strong_path, fname)
        weak_path = os.path.join(self.weak_path, fname)
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        strong = cv2.imread(strong_path, cv2.IMREAD_GRAYSCALE)
        weak = cv2.imread(weak_path, cv2.IMREAD_GRAYSCALE)
        strong = (strong > 127).astype(np.uint8)
        weak = (weak > 127).astype(np.uint8)

        if self.augment: 
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                strong = np.fliplr(strong).copy()
                weak = np.fliplr(weak).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy()
                strong = np.flipud(strong).copy()
                weak = np.flipud(weak).copy()

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        strong = cv2.resize(strong.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)
        weak = cv2.resize(weak.astype(np.uint8), self.img_size, interpolation=cv2.INTER_NEAREST)

        img_t = self.to_tensor(img).float()
        img_t = self.normalize(img_t)
        strong_t = torch.from_numpy(strong).float()
        weak_t = torch.from_numpy(weak).float()

        return img_t, strong_t, weak_t

class CupDataset(Dataset):
    def __init__(self, csv_path, crop_dir, split='train'):
        df = pd.read_csv(csv_path)

        # Full path for cup images/masks
        self.images = [os.path.join(crop_dir, x) for x in df['Cup_Image'].tolist()]
        self.masks = [os.path.join(crop_dir, x) for x in df['Cup_Mask'].tolist()]

        # 80/20 split
        n = len(self.images)
        idx = int(0.8 * n)
        if split == 'train':
            self.images, self.masks = self.images[:idx], self.masks[:idx]
        else:
            self.images, self.masks = self.images[idx:], self.masks[idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f'Missing {self.images[idx]} or {self.masks[idx]}')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 256x256
        img = cv2.resize(img, (256,256))
        mask = cv2.resize(mask, (256,256))

        # Normalize 
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        # HWC -> CHW
        img = torch.from_numpy(img).permute(2,0,1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask