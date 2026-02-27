import torch
from icecream import ic
import torchvision
import kagglehub
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# path = kagglehub.dataset_download("zaraks/pascal-voc-2007")
# ic(path)

vocPath = "data/voc2007/VOCtrainval_06-Nov-2007"
train_voc_datasets = torchvision.datasets.VOCDetection(
    root=vocPath, year="2007", image_set="train", download=False
)


def plot_voc_sample(img, target):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    objs = target["annotation"]["object"]
    if not isinstance(objs, list):
        objs = [objs]
    for obj in objs:
        bndbox = obj["bndbox"]
        x1, y1 = int(bndbox["xmin"]), int(bndbox["ymin"])
        x2, y2 = int(bndbox["xmax"]), int(bndbox["ymax"])
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        plt.text(x1, y1, obj["name"], color="white", backgroundcolor="red")
    plt.show()


img, target = train_voc_datasets[0]
ic(img, target)
# plot_voc_sample(img, target)
