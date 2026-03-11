import torch
from icecream import ic
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from PIL import Image
from torchvision import transforms


class MyYoloV1(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        layers = [
            # conv1 448 -> 112
            nn.Conv2d(3, 192, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv2 112 -> 56
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3 56 -> 28
            nn.Conv2d(256, 128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        # conv4
        for i in range(4):
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        # conv5
        for i in range(2):
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            ]
        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
        ]
        # conv6
        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
        ]
        # linear
        layers += [
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.Linear(4096, 30 * self.S * self.S),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        output = self.model(x)
        output = torch.reshape(output, (x.size(dim=0), self.S, self.S, -1))

        output[..., :20] = torch.sigmoid(output[..., :20])
        output[..., 20:21] = torch.sigmoid(output[..., 20:21])
        output[..., 25:26] = torch.sigmoid(output[..., 25:26])

        output[..., 21:23] = torch.sigmoid(output[..., 21:23])
        output[..., 26:28] = torch.sigmoid(output[..., 26:28])

        return output


# sourceImg = Image.open("./sorce_imgs/many_people.png")
# myToTensor = transforms.ToTensor()
# sourceImg_x = myToTensor(sourceImg)
# ic(sourceImg_x.shape)
# sourceImg_x = sourceImg_x.unsqueeze(0)
# ic(sourceImg_x.shape)
# myYoloV1 = MyYoloV1()
# output = myYoloV1(sourceImg_x)
# ic(output.shape)


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
