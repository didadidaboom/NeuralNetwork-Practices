import torch
from torch.utils.data import DataLoader, Dataset
from config import HP
import numpy as np

class BandnoteDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = np.loadtxt(data_path, delimiter=",")

    def __getitem__(self, idx):
        #[1.0 0.1, ..., 1]
        item = self.dataset[idx]
        x, y = item[:HP.in_features], item[HP.in_features:]
        return torch.Tensor(x).float().to(HP.device), torch.Tensor(y).squeeze().long().to(HP.device)

    def __len__(self):
        return self.dataset.shape[0]

trainset = BandnoteDataset(HP.trainset_path)
train_loader = DataLoader(trainset, batch_size=HP.batch_size, shuffle=True, drop_last=True)
devset = BandnoteDataset(HP.devset_path)
dev_loader = DataLoader(devset, batch_size=HP.batch_size, shuffle=True, drop_last=True)
# testset = BandnoteDataset(HP.testset_path)
# test_loader = DataLoader(testset, batch_size=HP.batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    bkdatase = BandnoteDataset(HP.testset_path)
    bkdataloader = DataLoader(bkdatase, batch_size=10, shuffle=True, drop_last=True)  # drop_last: drop the last one which size is not equal to batch size
    for batch in bkdataloader:
        print(" features: ", batch[0], " type: ", batch[1])
        break