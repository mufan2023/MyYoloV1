import torch
from icecream import ic
import torchvision
import kagglehub
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


yoloDatasetRoot = "data/yoloVoc2007"
voc2007Root = "data/voc2007/VOCtrainval_06-Nov-2007"
annotations_path = "data/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations"
jpegImages_path = "data/voc2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"


