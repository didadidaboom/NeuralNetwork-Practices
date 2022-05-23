import torch
from torch.utils.data import DataLoader
from config import HP
from utils import load_image, load_meta
from torchvision import transforms as T

hg_transform = T.Compose([
    T.Resize((112, 112)),                 # fixed the input shape
    T.RandomRotation(degrees=45),       # rotate the image
    T.GaussianBlur(kernel_size=(3, 3)), # increase bluring
    T.RandomHorizontalFlip(),           # flipping images
    T.ToTensor(),                        # regularization & float32 tensor
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  #
])

class HandGestureDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path):
        self.meta_path = meta_path
        self.dataset = load_meta(self.meta_path) #[(0, image_path), (), ...]

    def __getitem__(self, idx):
        item = self.dataset[idx]   # (0, image_path)
        cls_id, path = int(item[0]), item[1]
        image = load_image(path)
        return hg_transform(image).to(HP.device), cls_id

    def __len__(self):
        return len(self.dataset)

train_dataset = HandGestureDataset(HP.metadata_train_path)
train_dataloader = DataLoader(train_dataset, batch_size=HP.batch_size, shuffle=True, drop_last=True)
eval_dataset = HandGestureDataset(HP.metadata_eval_path)
eval_dataloader = DataLoader(eval_dataset, batch_size=HP.batch_size, shuffle=False, drop_last=False)
# test_dataset = HandGestureDataset(HP.metadata_test_path)
# test_dataloader = DataLoader(test_dataset, batch_size=HP.batch_size, shuffle=False, drop_last=False)

