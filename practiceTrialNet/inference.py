import torch
from torch.utils.data import DataLoader
from dataset_hg import HandGestureDataset
from model import FinalTrialNet
from config import HP

# new model instance
model = FinalTrialNet()
checkpoint = torch.load('./model_save/model_7_11000.pth', map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])

testset = HandGestureDataset(HP.metadata_eval_path)
test_loader = DataLoader(testset, batch_size=HP.batch_size, shuffle=False, drop_last=True)

model.eval()

total_cnt = 0
correct_cnt = 0

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        pred = model(x)
        # print(pred)
        total_cnt += pred.size(0)
        correct_cnt += (torch.argmax(pred, 1)==y).sum()
        # print(torch.argmax(pred, 1))
        # break
print('ACC: %.3f'%(correct_cnt/total_cnt))