import os
import cv2 as cv   
import torch
import numpy as np
from torch.utils.data import Dataset

CLASSES = ['Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox']

class MpoxDataset(Dataset):
    def __init__(self, base_path, fold = 1, split='Train', use_aug= True, img_size=100, mode= 'rgb'):
        """
        split   : 'Train', 'Valid', atau 'Test'
        use_aug : jika True dan split=='Train', gabungkan Original Train + AUG Train
        img_size: ukuran resize gambar (misal 100 atau 224)
        mode    : 'rgb', 'gray', atau 'clahe'
        """
        self.samples = []
        self.img_size = img_size
        self.mode = mode

        orig_dir = os.path.join(base_path, 'Original Images', 'Original Images', 'FOLDS', f'fold{fold}', split)
        dirs_to_load = [orig_dir]

        if split == 'Train' and use_aug:
            aug_dir = os.path.join(base_path, "Augmented Images", "Augmented Images", 'FOLDS_AUG', f'fold{fold}_AUG', 'Train' )
            dirs_to_load.append(aug_dir)

        for folder in dirs_to_load:
            for label_idx, cls_name in enumerate(CLASSES):
                cls_folder = os.path.join(folder,cls_name)
                if not os.path.exists(cls_folder):
                    continue
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    if os.path.isfile(img_path):
                        self.samples.append((img_path, label_idx))

    def __len__(self):# Batas 1 epoch
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv.imread(img_path)

        if self.mode == 'gray':
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (self.img_size, self.img_size))
            img = img[:, :, None]  # jadi (H, W, 1)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            if self.mode == 'clahe':# Tujuan CLAHE: Menonjolkan batas tepi & tekstur bintik/bopeng tanpa merusak warna aslinya
                lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)
                l, a, b = cv.split(lab) #Misahin Kecerahan(L) dan Warna(a,b)
                clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # tileGridSize = (8, 8) -> Membagi gambar menjadi kotak 8x8 untuk mengaplikasikan CLAHE
                # cliplimit = 2.0 -> Membatasi pengutan kontras di tiap kotak agar bintik noisy tidak terlalu menonjol
                lab = cv.merge((clahe.apply(l), a, b)) #Gabungin lagi dengan a,b
                img = cv.cvtColor(lab, cv.COLOR_LAB2RGB)

            elif self.mode == 'mask':
                ycrcb = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
                # Y(Luma) channel = kecerahan, Cr(Chromanic-Red) = merah, Cb(Chromanic-Blue) = biru
                # InRange: Membuat mask berdasarkan rentang nilai
                skin_mask = cv.inRange(ycrcb, np.array([0, 130, 75]), np.array([255, 185, 140])) # putih = kulit, hitam = backgorund
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
                skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel)
                if np.mean(skin_mask > 0) >= 0.05:
                    img = cv.bitwise_and(img, img, mask=skin_mask)
            img = cv.resize(img, (self.img_size, self.img_size))
            
        img = img.astype('float32') / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1)  # (H,W,C) -> FormatPyTorch(C,H,W)
        return img_tensor, torch.tensor(label, dtype=torch.long)