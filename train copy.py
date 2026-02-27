import torch
from icecream import ic
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from torchvision import transforms


class MyYoloV1(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1_3_64 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        layers = [
            self.conv1_3_64,
            nn.LeakyReLU(negative_slope=0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        return output


sourceImg = Image.open("./sorce_imgs/many_people.png")
myToTensor = transforms.ToTensor()
sourceImg_x = myToTensor(sourceImg)
ic(sourceImg_x.shape)
myYoloV1 = MyYoloV1()
output = myYoloV1(sourceImg_x)
ic(output.shape)

# writer = SummaryWriter(log_dir="logs")
# writer.add_image("sorceImg", sourceImg_x)

# imgs = output.unsqueeze(1)
# ic(imgs.shape)
# index = 1
# groupIndex = 0
# for img in imgs:
#     writer.add_image("outputImgs" + str(index), img, global_step=groupIndex)
#     if index % 8 == 0:
#         groupIndex += 1
#         index = 1
#     else:
#         index += 1

# writer.close()
