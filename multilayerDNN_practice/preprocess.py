import numpy as np
from config import HP
import os

trainset_ratio = 0.7
devset_ratio = 0.2
testset_ratio = 0.1

np.random.seed(HP.seed)
dataset = np.loadtxt(HP.data_path, delimiter=",", skiprows=1)
np.random.shuffle(dataset)

# print(dataset[:10])
n_items = dataset.shape[0]
# print(n_items)

trainset_num = int(trainset_ratio*n_items)
devset_num = int(devset_ratio*n_items)
testset_num = int(testset_ratio*n_items)

np.savetxt(os.path.join(HP.data_dir, "train.txt"), dataset[:trainset_num], delimiter=",")
np.savetxt(os.path.join(HP.data_dir, "dev.txt"), dataset[trainset_num:devset_num+trainset_num], delimiter=",")
np.savetxt(os.path.join(HP.data_dir, "test.txt"), dataset[trainset_num+devset_num:], delimiter=",")
