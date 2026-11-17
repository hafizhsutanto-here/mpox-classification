import os
import numpy as np
import pandas as pd 
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import(
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from Utils.getData import CLASSES

def compute_metrics(y_true, y_pred, y_prob):
    """
    y_true : list/array label asli (0..5)
    y_pred : list/array label prediksi kelas (0..5)
    y_prob : list/array probabilitas shape(N, 6) hasil softmax
    """
    return {
        'accuracy' :accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average ="macro", zero_division = 0),
        'recall': recall_score(y_true, y_pred, average="macro", zero_division = 0),
        'f1_score' :f1_score(y_true, y_pred, average='macro', zero_division = 0),
        'roc_auc' : roc_auc_score(y_true, y_prob, multi_class="ovr", average = 'macro'),
    }

def save_loss_curve(train_losses, valid_losses, save_path, title='Training & Validation Loss', show=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(train_losses) +1), train_losses, color="#931a00", label="Train")
    plt.plot(range(1, len(valid_losses) +1), valid_losses, color="#3399e6", label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def save_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix", show=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)

    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()