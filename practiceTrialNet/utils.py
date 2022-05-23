import glob
import os
import json
import random
from PIL import Image

from config import HP

def recursive_data(id, path, format, cls):
    sub_path = "/*"
    first_list = glob.glob(path+sub_path)
    datasets = []
    if len(first_list):
        for first in first_list:
            print("222--", first)
            first_end = first.split(".")[-1]
            if first_end and first_end==format[0]:
                tmp = str(id)+"|"+cls+"~"+first
                datasets.append(tmp)
            else:
                sub_dataset = recursive_data(id, first, format, cls)
                if sub_dataset:
                    datasets = datasets+sub_dataset
    return datasets

def recursive_fetching(path, format):
    first_list = glob.glob(path+"/*")
    datasets = []
    for first in first_list:
        print("111 -- ",first)
        cls = first.split("/")[-1]
        cls_mapper = json.load(open(HP.cls_mapper_path, 'r'))
        id = cls_mapper["cls2id"][cls]
        print("cls = ",cls, " id = ", id, " first = ", first)
        first_end = first.split(".")[-1]
        if first_end and first_end == format[0]:
            tmp = str(id) + "|" + cls + "~" + first
            datasets.append(tmp)
        else:
            sub_datasets = recursive_data(id, first, format, cls)
            if sub_datasets:
                datasets = datasets+sub_datasets
    return datasets

def load_meta(meta_path):
    with open(meta_path, 'r') as fr:
        return [line.strip().split("|") for line in fr.readlines()]

def load_image(image_path):
    return Image.open(image_path)


if __name__ == '__main__':
    re = recursive_fetching("./data/test", ["ppm"])
    print(len(re))
    print(re)
    # d1 = ["a", "b", "c"]
    # d2 = ["a2", "b2", "c2"]
    # c1 = d1+d2
    # random.shuffle(c1)
    # print(c1)