import os
from config import HP
from utils import recursive_fetching
import random
import json

random.seed(HP.seed)

#construct a mapping from classification to id
cls_mapper = {
    "cls2id": {"A":0, "B":1, "C":2, "Five":3, "Point":4, "V":5},
    "id2cls": {0:"A", 1:"B", 2:"C", 3:"Five", 4:"Point", 5:"V"}
}
json.dump(cls_mapper, open(HP.cls_mapper_path, 'w'))  # save in json

# achieve train and test datasets, and then merge them together
train_items = recursive_fetching(HP.train_data_root, ['ppm']) #achieve dataset under train files
test_items = recursive_fetching(HP.test_data_root, ['ppm']) # achieve dataset under test files
dataset = train_items+test_items
dataset_num = len(dataset)
print("Total Items: %d"%dataset_num)
random.shuffle(dataset)

"""
# To get dataset format:
    0:["./data/...", "./data/..."],
    1:["./data/...", "./data/..."],
    ...
    ...
"""
dataset_dict = {}
for data in dataset:
    # print(data)
    data = data.split("|")
    cls_id = int(data[0])
    data_tmp = data[1]
    # print(data_tmp)
    if cls_id not in dataset_dict:
        dataset_dict[cls_id] = [data_tmp]
    else:
        dataset_dict[cls_id].append(data_tmp)

# split dataset into train/eval/test
# train_ratio, eval_ratio, test_ratio = 0.8, 0.1, 0.1
train_ratio, eval_ratio, test_ratio = 0.08, 0.01, 0.01
train_set, eval_set, test_set = [], [], []
for _, set_list in dataset_dict.items():
    length = len((set_list))
    train_num, eval_num = int(train_ratio*length), int(eval_ratio*length)
    test_num = length - train_num - eval_num
    random.shuffle(set_list)
    train_set.extend(set_list[:train_num])
    eval_set.extend(set_list[train_num:train_num+eval_num])
    test_set.extend(set_list[train_num+eval_num:])

# shuffle it
random.shuffle(train_set)
random.shuffle(eval_set)
random.shuffle(test_set)

print(" train set: eval set: test set -> %d %d %d "%(len(train_set), len(eval_set), len(test_set)))

with open(HP.metadata_train_path, 'w') as fw:
    for tmp in train_set:
        tmp = tmp.split("~")
        cls = tmp[0]
        tmp_data = tmp[1]
        cls_id = cls_mapper["cls2id"][cls]
        # print(cls_id)
        fw.write("%d|%s\n"%(cls_id, tmp_data))

with open(HP.metadata_eval_path, 'w') as fw:
    for tmp in eval_set:
        tmp = tmp.split("~")
        cls = tmp[0]
        tmp_data = tmp[1]
        cls_id = cls_mapper["cls2id"][cls]
        # print(cls_id)
        fw.write("%d|%s\n"%(cls_id, tmp_data))

with open(HP.metadata_test_path, 'w') as fw:
    for tmp in test_set:
        tmp = tmp.split("~")
        cls = tmp[0]
        tmp_data = tmp[1]
        cls_id = cls_mapper["cls2id"][cls]
        # print(cls_id)
        fw.write("%d|%s\n"%(cls_id, tmp_data))


from utils import load_meta, load_image
mode_set, size_set = [], []
for _, path in load_meta(HP.metadata_test_path):
    img = load_image(path)
    mode_set.append(img.mode)
    size_set.append(img.size)
print(set(mode_set), set(size_set))
