import os
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from Utils.getData import MpoxDataset, CLASSES
from Utils.metrics import compute_metrics, save_confusion_matrix
from Models.CNN import SimpleCNN
from Models.CNN2 import SimpleCNN2
from Models.Model3 import Model3Classifier

# Harus sama dengan DEFAULT_ARCH di train.py saat klik Run (▶) tanpa terminal
DEFAULT_ARCH = "mobilevit_xxs"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def save_roc(y_true, y_prob, title, save_path):
    y_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(CLASSES):
        if y_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def build_model(arch, num_classes, device):
    if arch == "cnn1":
        return SimpleCNN(input_c=3, output=num_classes).to(device)
    if arch == "cnn2":
        return SimpleCNN2(input_c=3, output=num_classes).to(device)
    return Model3Classifier(arch=arch, num_classes=num_classes, pretrained=False).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--arch", default=DEFAULT_ARCH,
                        choices=["cnn1", "cnn2", "mobilevit_xxs", "convnextv2_atto", "deit_tiny_patch16_224"])
    parser.add_argument("--prep", default=None, choices=["rgb", "clahe", "mask"])
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--no-tta", action="store_true", help="Matikan Test-Time Augmentation (default: aktif untuk Model 3)")
    parser.add_argument("--show", action="store_true", help="Tampilkan heatmap confusion matrix gabungan di layar")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg", force=True)

    is_cnn = args.arch in ("cnn1", "cnn2")
    if args.prep is None:
        args.prep = "rgb" if is_cnn else "clahe"
    if args.size is None:
        args.size = 100 if is_cnn else 224
    if args.arch == "deit_tiny_patch16_224":
        args.size = 224

    if args.data is None:
        import kagglehub
        args.data = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    tag = f"{args.arch}_{args.prep}_{args.size}"
    ckpt_dir = os.path.join("checkpoints", tag)
    res_dir = os.path.join("results", tag)
    os.makedirs(res_dir, exist_ok=True)

    mean_t = IMAGENET_MEAN.to(device)
    std_t = IMAGENET_STD.to(device)
    folds = [args.only_fold] if args.only_fold else [1, 2, 3, 4, 5]
    all_metrics, y_true_all, y_pred_all = [], [], []

    for k in folds:
        test_ds = MpoxDataset(args.data, fold=k, split="Test", img_size=args.size, mode=args.prep)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=(device == "cuda"))

        model = build_model(args.arch, len(CLASSES), device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{k}.pt"), map_location=device, weights_only=True))
        model.eval()

        yt, yp, yprob = [], [], []
        with torch.no_grad():
            for src, trg in tqdm(test_loader, desc=f"Test Fold {k}", leave=False):
                src = src.to(device)
                if is_cnn:
                    probs = torch.exp(model(src))
                else:
                    src = (src - mean_t) / std_t
                    probs = torch.softmax(model(src), dim=1)
                    if not args.no_tta:
                        probs_flip = torch.softmax(model(torch.flip(src, dims=[3])), dim=1)
                        probs = 0.5 * (probs + probs_flip)
                yt.extend(trg.numpy())
                yp.extend(torch.argmax(probs, dim=1).cpu().numpy())
                yprob.extend(probs.cpu().numpy())

        yt, yp, yprob = np.array(yt), np.array(yp), np.array(yprob)
        m = compute_metrics(yt, yp, yprob)
        m["fold"] = k
        all_metrics.append(m)
        print(f"TEST FOLD {k}: " + " | ".join(f"{key} {val:.4f}" for key, val in m.items() if key != "fold"))

        save_confusion_matrix(yt, yp, os.path.join(res_dir, f"fold{k}_cm.png"), title=f"CM Fold {k} ({tag})")
        save_roc(yt, yprob, f"ROC Fold {k} ({tag})", os.path.join(res_dir, f"fold{k}_roc.png"))
        y_true_all.extend(yt)
        y_pred_all.extend(yp)

    df = pd.DataFrame(all_metrics)[["fold", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    df.to_csv(os.path.join(res_dir, "hasil_per_fold.csv"), index=False)

    summary = df.drop(columns="fold").agg(["mean", "std"]).T
    summary.to_csv(os.path.join(res_dir, "ringkasan_rata_rata.csv"))
    print("\n=== RATA-RATA ===")
    print(summary.round(4))

    save_confusion_matrix(y_true_all, y_pred_all, os.path.join(res_dir, "cm_gabungan.png"),
                          title=f"CM Gabungan ({tag})", show=args.show)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(CLASSES))))
    recall_per_kelas = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    salah = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                salah.append((int(cm[i, j]), CLASSES[i], CLASSES[j]))
    salah.sort(reverse=True)
    with open(os.path.join(res_dir, "analisis_kesalahan.txt"), "w", encoding="utf-8") as f:
        f.write("Recall per kelas (gabungan semua fold):\n")
        for nama, r in zip(CLASSES, recall_per_kelas):
            f.write(f"  {nama}: {r:.4f}\n")
        f.write("\nKesalahan klasifikasi terbanyak (jumlah, kelas asli -> diprediksi sebagai):\n")
        for jml, asli, prediksi in salah[:10]:
            f.write(f"  {jml}x  {asli} -> {prediksi}\n")
    print("\n" + open(os.path.join(res_dir, "analisis_kesalahan.txt"), encoding="utf-8").read())

    zip_path = shutil.make_archive(os.path.join("results", f"{tag}_results"), "zip", res_dir)
    print(f"Hasil tersimpan & ter-zip di: {zip_path}")


if __name__ == "__main__":
    main()
