import kagglehub
from Utils.getData import MpoxDataset

path = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")
train_ds = MpoxDataset(path, fold=1, split='Train', use_aug=True)
valid_ds = MpoxDataset(path, fold=1, split='Valid')
test_ds  = MpoxDataset(path, fold=1, split='Test')

print("Train:", len(train_ds), "| Valid:", len(valid_ds), "| Test:", len(test_ds))
x, y = train_ds[0]
print("Shape 1 gambar:", x.shape, "| Label:", y.item())
