import torch
from icecream import ic
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np


class MyTargetTransform:

    def __init__(self):
        self.S = 7
        self.B = 2
        self.C = 20
        self.class_map = {
            "aeroplane": 0,
            "bicycle": 1,
            "bird": 2,
            "boat": 3,
            "bottle": 4,
            "bus": 5,
            "car": 6,
            "cat": 7,
            "chair": 8,
            "cow": 9,
            "diningtable": 10,
            "dog": 11,
            "horse": 12,
            "motorbike": 13,
            "person": 14,
            "pottedplant": 15,
            "sheep": 16,
            "sofa": 17,
            "train": 18,
            "tvmonitor": 19,
        }

    def __call__(self, target):
        ic(target)
        label = np.zeros((self.S, self.S, self.B * 5 + self.C))
        objects = target["annotation"]["object"]
        width = int(target["annotation"]["size"]["width"])
        height = int(target["annotation"]["size"]["height"])
        for obj in objects:
            cls_name = obj["name"]
            cls_id = self.class_map[cls_name]
            bndbox = obj["bndbox"]
            x1, x2 = float(bndbox["xmin"]), float(bndbox["xmax"])
            y1, y2 = float(bndbox["ymin"]), float(bndbox["ymax"])
            x_center = (x1 + x2) / 2.0 / width
            y_center = (y1 + y2) / 2.0 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            x_grid = self.S * x_center
            y_grid = self.S * y_center
            col = int(x_grid)
            row = int(y_grid)

            x_offset = x_grid - col
            y_offset = y_grid - row

            if label[row, col, 24] == 0:
                label[row, col, 24] = 1
                label[row, col, 29] = 1
                label[row, col, 20:24] = [x_offset, y_offset, w, h]
                label[row, col, 25:29] = [x_offset, y_offset, w, h]
                label[row, col, cls_id] = 1
        return torch.from_numpy(label).float()


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((448, 448)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # ),
    ]
)


vocPath = "data/voc2007/VOCtrainval_06-Nov-2007"
train_voc_datasets = torchvision.datasets.VOCDetection(
    root=vocPath,
    year="2007",
    image_set="train",
    transform=transform,
    target_transform=MyTargetTransform(),
    download=False,
)
ic(len(train_voc_datasets))
image, target = train_voc_datasets[0]
ic(image.shape, target.shape)


# image_t = transform(image)
# ic(image_t.shape)

# writer = SummaryWriter("logs")
# writer.add_image("img", image_t, global_step=0)
# writer.close()
