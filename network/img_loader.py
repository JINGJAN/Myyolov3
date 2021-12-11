import json
import os

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import shutil




# d = read_json("/mnt/Jan_data/Data/Object_detetction_data/self_driving_2000/train.json")
# print(d)

class coco_Image_loader(Dataset):
    def __init__(self, json_path, img_size, png_dir):
        self.json_path = json_path
        self.img_size = img_size
        self.png_dir = png_dir
        self.data = self.read_json(self.json_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_name = self.data[item][0]
        img_path = os.path.join(self.png_dir, img_name)
        img = Image.open(img_path)
        scale = [img.size[0] / self.img_size[0], img.size[1] / self.img_size[1]] * 2

        img_resize = (self.img_size[0], self.img_size[1])
        img = img.resize(img_resize)
        t = self.transformer_()
        img = t(img)

        class_bbox = self.data[item][1]

        label = self.get_loc_class(class_bbox, scale)
        return img, label

    def get_loc_class(self, infos, scale):
        class_plus_loc = 5
        out = np.zeros((100, class_plus_loc), dtype=int)
        length_info = len(infos)
        steps = int(length_info / class_plus_loc)
        for step in range(steps):
            info = infos[class_plus_loc * step:class_plus_loc * (step + 1)]
            out[step][-1] = int(info[0])
            for j in range(len(scale)):
                out[step][j] = round(info[j + 1] / scale[j])

        # for i, info in enumerate(infos):
        #     category = info[0]
        #     bbox = info[1:]
        #     for j in range(len(scale)):
        #         out[i][j] = round(bbox[j] / scale[j])
        #     out[i][-1] = int(category)
        return torch.tensor(out)

    def transformer_(self):
        T = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        return T

    def read_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        out = []
        for key, value in data.items():
            out.append([key, value])
        return out





