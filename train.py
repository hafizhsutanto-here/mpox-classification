import os
import gc
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from Utils.getData import MpoxDataset, CLASSES
from Utils.metrics import save_loss_curve
from Models.CNN import SimpleCNN
from Models.CNN2 import SimpleCNN2
from Models.Model3 import Model3Classifier

# Pilihan: "cnn1", "cnn2", "mobilevit_xxs", "convnextv2_atto", "deit_tiny_patch16_224"
DEFAULT_ARCH = "mobilevit_xxs"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

AUG_TF = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
])


def mixup_batch(x, y, num_classes=6, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_onehot = F.one_hot(y, num_classes).float()
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return mixed_x, mixed_y


class DatasetCache(Dataset):
    def __init__(self, dataset, desc="Loading"):
        self.x, self.y = [], []
        for i in tqdm(range(len(dataset)), desc=desc, leave=False):
            img, label = dataset[i]
            self.x.append(img.half())
            self.y.append(label)
        self.labels = np.array([int(L) for L in self.y])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].float(), self.y[idx]


def build_model(arch, num_classes, freeze_early, device):
    if arch == "cnn1":
        return SimpleCNN(input_c=3, output=num_classes).to(device)
    if arch == "cnn2":
        return SimpleCNN2(input_c=3, output=num_classes).to(device)
    return Model3Classifier(arch=arch, num_classes=num_classes, pretrained=True, freeze_early=freeze_early).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--arch", default=DEFAULT_ARCH,
                        choices=["cnn1", "cnn2", "mobilevit_xxs", "convnextv2_atto", "deit_tiny_patch16_224"])
    parser.add_argument("--prep", default=None, choices=["rgb", "clahe", "mask"])
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--only-fold", type=int, default=None, help="Isi 1 untuk uji cepat fold 1 saja")
    parser.add_argument("--start-fold", type=int, default=1, help="Mulai dari fold ke-berapa (1..5)")
    parser.add_argument("--freeze", action="store_true", help="Kunci layer awal Model 3 (default: semua layer dilatih)")
    parser.add_argument("--cooldown", type=int, default=0, help="Jeda detik antar fold (misal 900 = 15 menit)")
    parser.add_argument("--show", action="store_true", help="Tampilkan grafik loss di layar setelah tiap fold")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    is_cnn = args.arch in ("cnn1", "cnn2")
    if args.prep is None:
        args.prep = "rgb" if is_cnn else "clahe"
    if args.size is None:
        args.size = 100 if is_cnn else 224
    if args.arch == "deit_tiny_patch16_224" and args.size != 224:
        args.size = 224
    if args.epoch is None:
        if args.arch == "cnn1":
            args.epoch = 100
        elif args.arch == "cnn2":
            args.epoch = 20
        else:
            args.epoch = 10
    if args.lr is None:
        args.lr = 1e-3 if is_cnn else 1e-4

    if args.data is None:
        import kagglehub
        args.data = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    tag = f"{args.arch}_{args.prep}_{args.size}"
    ckpt_dir = os.path.join("checkpoints", tag)
    res_dir = os.path.join("results", tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    mean_t = IMAGENET_MEAN.to(device)
    std_t = IMAGENET_STD.to(device)
    folds = [args.only_fold] if args.only_fold else list(range(args.start_fold, 6))

    for k in folds:
        print(f"\n=== TRAIN {tag} | FOLD {k} ===")
        torch.manual_seed(42)
        np.random.seed(42)

        train_set = DatasetCache(MpoxDataset(args.data, fold=k, split="Train", use_aug=True, img_size=args.size, mode=args.prep), desc="Cache Train")
        valid_set = DatasetCache(MpoxDataset(args.data, fold=k, split="Valid", img_size=args.size, mode=args.prep), desc="Cache Valid")
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, pin_memory=(device == "cuda"))
        valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=False, pin_memory=(device == "cuda"))

        counts = np.bincount(train_set.labels, minlength=len(CLASSES))
        weights = torch.tensor(len(train_set) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32).to(device)

        model = build_model(args.arch, len(CLASSES), args.freeze, device)
        if is_cnn:
            criterion = nn.NLLLoss(weight=weights)
            optimizer = Adam(model.parameters(), lr=args.lr)
            scheduler = None
        else:
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)

        train_losses, valid_losses = [], []
        best_vloss, best_state = 1e9, None

        for ep in range(args.epoch):
            model.train()
            tloss = 0.0
            for src, trg in tqdm(train_loader, desc=f"Epoch {ep+1}/{args.epoch}", leave=False):
                src, trg = src.to(device), trg.to(device)
                optimizer.zero_grad()
                if not is_cnn:
                    src = (AUG_TF(src) - mean_t) / std_t
                    mx, my = mixup_batch(src, trg, len(CLASSES), alpha=0.2)
                    log_p = F.log_softmax(model(mx), dim=1)
                    loss = -(my * weights.unsqueeze(0) * log_p).sum(dim=1).mean()
                else:
                    loss = criterion(model(src), trg)
                loss.backward()
                optimizer.step()
                tloss += loss.item()
            if scheduler is not None:
                scheduler.step()

            model.eval()
            vloss, yv, pv = 0.0, [], []
            with torch.no_grad():
                for src, trg in valid_loader:
                    src, trg = src.to(device), trg.to(device)
                    if not is_cnn:
                        src = (src - mean_t) / std_t
                    out = model(src)
                    vloss += criterion(out, trg).item()
                    yv.extend(trg.cpu().numpy())
                    pv.extend(torch.argmax(out, dim=1).cpu().numpy())

            tloss /= len(train_loader)
            vloss /= len(valid_loader)
            train_losses.append(tloss)
            valid_losses.append(vloss)
            print(f"Epoch {ep+1}/{args.epoch} | train loss {tloss:.4f} | valid loss {vloss:.4f} | valid acc {accuracy_score(yv, pv):.4f}")

            if vloss < best_vloss:
                best_vloss = vloss
                best_state = copy.deepcopy(model.state_dict())

        torch.save(best_state, os.path.join(ckpt_dir, f"fold{k}.pt"))
        save_loss_curve(train_losses, valid_losses, os.path.join(res_dir, f"fold{k}_training.png"),
                        title=f"Loss Fold {k} ({tag})", show=args.show)
        del train_set, valid_set, train_loader, valid_loader, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.cooldown > 0 and k != folds[-1]:
            print(f"Jeda {args.cooldown} detik sebelum fold berikutnya...")
            time.sleep(args.cooldown)


if __name__ == "__main__":
    main()
